# engine.py (updated: robust semi-supervised tracking + preserved unsup loss + safe CSV + per-class metrics)


import os
import torch
import torch.nn.functional as F
import numpy as np
import csv
from segm.metrics import gather_data
import segm.utils.torch as ptu
from segm.utils import distributed
import warnings
from segm.utils.logging_config import (
    init_history,
    append_history,
    write_csv,
    plot_losses,
    plot_metrics
)

LOSS_HISTORY = init_history()

IGNORE_LABEL = 255
EPS = 1e-6



# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def dice_loss_masked(pred_prob, gt_mask_onehot, valid_mask, smooth=1e-6):
    """
    pred_prob: tensor (N, H, W) probabilities for a class
    gt_mask_onehot: tensor (N, H, W) binary (0/1) mask for that class
    valid_mask: tensor (N, H, W) float mask where 1 = valid pixels to consider
    returns: 1 - dice (so 0 => perfect)
    """
    valid = valid_mask.view(-1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_prob.device)
    p = pred_prob.contiguous().view(-1)[valid]
    g = gt_mask_onehot.contiguous().view(-1)[valid]
    intersection = (p * g).sum()
    return 1 - (2. * intersection + smooth) / (p.sum() + g.sum() + smooth)


# ----------------------------
# TRAINING FUNCTION
# ----------------------------
def train_one_epoch(model, data_loader, optimizer, lr_scheduler, epoch, amp_autocast, loss_scaler=None, log_dir=None, class_weights=None, val_loader=None, teacher_model=None, unsup_weight=1.0):
    """
    Train model for one epoch.
    Unsupervised loss = CrossEntropy(student_outputs_unlabeled, pseudo_labels_from_teacher)
                        + Dice(student_probs_unlabeled, pseudo_labels_onehot)
    unsup_weight scales the whole unsupervised loss term.
    """
    model.train()
        
    C = getattr(data_loader.dataset, "n_cls", None)
    if C is None:
        #C = None
        raise ValueError("Dataset must define n_cls for class-wise metrics.")

    TP_labeled = np.zeros(C, dtype=np.float64)
    FP_labeled = np.zeros(C, dtype=np.float64)
    FN_labeled = np.zeros(C, dtype=np.float64)
    
    TP_all = np.zeros(C, dtype=np.float64)
    FP_all = np.zeros(C, dtype=np.float64)
    FN_all = np.zeros(C, dtype=np.float64)
    
    
    
    ce_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    weighted_ce_fn = ce_fn if class_weights is None else torch.nn.CrossEntropyLoss(weight=class_weights.to(ptu.device), ignore_index=IGNORE_LABEL)

    # ---------------Epoch accumulators---------------
    
    ce_epoch = weighted_ce_epoch = 0.0

    dice_sup_epoch = 0.0
    dice_unsup_epoch = 0.0
    
    sup_epoch = unsup_epoch = total_epoch = 0.0
    
    total_pixels_labeled = correct_pixels_labeled = total_pixels_all = correct_pixels_all = 0
    
    total_labeled_samples = 0
    total_unlabeled_samples = 0
    


    teacher_model_eval = None
    if teacher_model is not None:
        teacher_model_eval = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
        teacher_model_eval.to(ptu.device)
        teacher_model_eval.eval()

    printed_sample_param_flag = False

    print(f"\n[Epoch {epoch}] Starting training epoch with {len(data_loader)} batches...\n", flush=True)
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx % 20 == 0:
            print(f"[Epoch {epoch}] Processing batch {batch_idx}/{len(data_loader)}", flush=True)
            
    
        images = batch["image"].to(ptu.device)
        masks = batch.get("segmentation", batch.get("mask"))
        if masks is None:
            raise RuntimeError("Batch does not contain 'mask' or 'segmentation' key.")
        masks = masks.to(ptu.device).long()
        B, _, H, W = images.shape


        # detect which samples in the batch are labeled
        is_labeled = batch.get("is_labeled", None)
        if is_labeled is None:
            tmp = (masks != IGNORE_LABEL)
            #is_labeled_tensor = tmp.any(dim=1).any(dim=1)
            is_labeled_tensor = (masks != IGNORE_LABEL).any(dim=(1, 2))
        else:
            if isinstance(is_labeled, (list, tuple)):
                is_labeled_tensor = torch.tensor(is_labeled, dtype=torch.bool, device=ptu.device)
            elif isinstance(is_labeled, torch.Tensor):
                is_labeled_tensor = is_labeled.to(ptu.device).bool()
            else:
                is_labeled_tensor = torch.tensor(is_labeled, dtype=torch.bool, device=ptu.device)

        
        unlabeled_mask = ~is_labeled_tensor
        labeled_mask = is_labeled_tensor
        
        if batch_idx % 20 == 0:
            print(
                f"[Epoch {epoch}] Batch {batch_idx}: "
                f"Labeled={labeled_mask.sum().item()}, "
                f"Unlabeled={unlabeled_mask.sum().item()}",
                flush=True
            )

        optimizer.zero_grad()
        with amp_autocast():
            outputs = model(images)
            
            if batch_idx % 20 == 0:
                print(f"Student forward pass done", flush=True)
            
            probs = torch.softmax(outputs, dim=1)

            # --- Supervised Loss ---
            sup_loss = torch.tensor(0.0, device=ptu.device)
            weighted_ce_loss = torch.tensor(0.0, device=ptu.device)
            if labeled_mask.any():
                out_sup = outputs[labeled_mask]
                mask_sup = masks[labeled_mask]
                sup_loss = ce_fn(out_sup, mask_sup)
                weighted_ce_loss = weighted_ce_fn(out_sup, mask_sup) if class_weights is not None else sup_loss

            # --- Unsupervised Loss ---            
            unsup_loss = torch.tensor(0.0, device=ptu.device)

            dice_unsup = torch.tensor(0.0, device=ptu.device)

            if teacher_model_eval is not None and unlabeled_mask.any():
                imgs_unl = images[unlabeled_mask]
            
                with torch.no_grad():
                    teacher_logits = teacher_model_eval(imgs_unl)
                    
                    if batch_idx % 20 == 0:
                        print(f"Teacher forward pass done", flush=True)
                    
                    teacher_probs = torch.softmax(teacher_logits, dim=1)
                    
                    
                    pseudo_labels = torch.argmax(teacher_probs, dim=1)
            
                student_logits_unl = outputs[unlabeled_mask]
                student_probs_unl = torch.softmax(student_logits_unl, dim=1)
            
                
                #ce_unsup = F.cross_entropy(student_logits_unl, pseudo_labels)
                
                bs = images.size(0)
            
                # optional dice 
                probs_u = student_probs_unl
                                
                confidence = teacher_probs.max(dim=1)[0]
                valid_unl = (confidence > 0.7).float()
                                
                # ----------------UNSUP CE LOSS----------------
                ce_map = F.cross_entropy(
                    student_logits_unl,
                    pseudo_labels,
                    reduction='none'
                )  # (B, H, W)
                
                #ce_unsup = (ce_map * valid_unl.float()).sum() / (valid_unl.sum() + 1e-6)
                
                
                if valid_unl.sum() > 0:
                    ce_unsup = (ce_map * valid_unl).sum() / valid_unl.sum()
                else:
                    ce_unsup = torch.tensor(0.0, device=ptu.device)
                            
                # ----------------UNSUP DICE LOSS----------------
                per_class_dice_unl = []
                for c in range(C):
                    pred_prob = probs_u[:, c, :, :]
                    #gt_onehot = (torch.argmax(teacher_probs, dim=1) == c).float()
                    gt_onehot = (pseudo_labels == c).float()
                    d = dice_loss_masked(pred_prob, gt_onehot, valid_unl)
                    per_class_dice_unl.append(d)
            
                dice_unsup = torch.stack(per_class_dice_unl).mean()
            
                # ----------------FINAL UNSUP LOSS----------------                
                
                lambda_ce = 1.0
                lambda_dice = 0.5
                
                unsup_loss = (lambda_ce * ce_unsup + lambda_dice * dice_unsup)
                
                unsup_loss = unsup_loss / (lambda_ce + lambda_dice)
                
                
                
                
            # --- Dice over labeled pixels ---
            dice_val = torch.tensor(0.0, device=ptu.device)
            if labeled_mask.any():
                probs_l = probs[labeled_mask]
                mask_l = masks[labeled_mask]
                valid_l = (mask_l != IGNORE_LABEL).float()
                per_class_dice = []
                for c in range(C):
                    pred_prob = probs_l[:, c, :, :]
                    gt_onehot = (mask_l == c).float()
                    #gt_onehot = (pseudo_labels == c).float()
                    
                    d = dice_loss_masked(pred_prob, gt_onehot, valid_l)
                    per_class_dice.append(d)
                dice_val = torch.stack(per_class_dice).mean() if len(per_class_dice) > 0 else torch.tensor(0.0, device=ptu.device)

            
            supervised_total = sup_loss + dice_val
            total_loss = supervised_total + unsup_weight * unsup_loss

        if loss_scaler is not None:
            loss_scaler(total_loss, optimizer)
        else:
            if batch_idx % 20 == 0:
                print(f"Total loss ready: {total_loss.item():.6f}", flush=True)
        
            total_loss.backward()
            optimizer.step()
        
            if batch_idx % 20 == 0:
                print(f"Optimizer step complete\n", flush=True)
        
        # =====================================================
        # 
        # =====================================================        
        bs = images.size(0)

        ce_epoch += float(sup_loss.item()) * bs
        weighted_ce_epoch += float(weighted_ce_loss.item()) * bs
        #dice_epoch += float(dice_val.item()) * bs
        #dice_unsup_epoch += float(dice_unsup.item()) * bs
        
        num_unl = unlabeled_mask.sum().item()
        dice_unsup_epoch += float(dice_unsup.item()) * num_unl
        
        
        dice_sup_epoch += float(dice_val.item()) * bs
        sup_epoch += float(supervised_total.item()) * bs
        #unsup_epoch += float(unsup_loss.item()) * bs
        unsup_epoch += float(unsup_loss.item()) * num_unl
        
        total_epoch += float(total_loss.item()) * bs
        
        #total_epoch += float(supervised_total.item()) * bs + float(unsup_loss.item()) * num_unl
        
        # =====================================================

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch == 0 and batch_idx == 0 and not printed_sample_param_flag:
            mm = model.module if hasattr(model, "module") else model
            p = next(mm.parameters())
            try:
                print("DEBUG sample param (epoch0 batch0):", float(p.data.view(-1)[0]), flush=True)
            except Exception:
                pass
            printed_sample_param_flag = True
        
        
        num_labeled = labeled_mask.sum().item()
        num_unlabeled = unlabeled_mask.sum().item()
        
        total_labeled_samples += num_labeled
        if batch_idx % 50 == 0:
            print(f"Total labeled images: {total_labeled_samples}")
        total_unlabeled_samples += num_unlabeled
                
        preds_flat = torch.argmax(outputs, dim=1).view(B, -1).cpu().numpy()
        masks_flat = masks.view(B, -1).cpu().numpy()
        labeled_mask_np = labeled_mask.cpu().numpy()
        
        for b in range(B):
            mask_b = masks_flat[b]
            pred_b = preds_flat[b]
            #lbl_mask_b = labeled_mask_np[b]
        
            # =====================================================
            # LABELED METRICS (ONLY for truly labeled samples)
            # =====================================================            
            valid_pixels = (mask_b != IGNORE_LABEL)
            mask_valid = mask_b[valid_pixels]
            pred_valid = pred_b[valid_pixels]
            
            # ALWAYS compute ALL metrics from valid pixels
            total_pixels_all += mask_valid.size
            correct_pixels_all += int((mask_valid == pred_valid).sum())
            
            for c in range(C):
                TP_all[c] += int(np.sum((mask_valid == c) & (pred_valid == c)))
                FP_all[c] += int(np.sum((mask_valid != c) & (pred_valid == c)))
                FN_all[c] += int(np.sum((mask_valid == c) & (pred_valid != c)))
            
            # ONLY labeled images contribute to labeled metrics
            if labeled_mask_np[b].item():
            
                total_pixels_labeled += mask_valid.size
                correct_pixels_labeled += int((mask_valid == pred_valid).sum())
            
                for c in range(C):
                    TP_labeled[c] += int(np.sum((mask_valid == c) & (pred_valid == c)))
                    FP_labeled[c] += int(np.sum((mask_valid != c) & (pred_valid == c)))
                    FN_labeled[c] += int(np.sum((mask_valid == c) & (pred_valid != c)))
        
    total_samples = len(data_loader.dataset)
    total_unlabeled_samples = max(1, total_unlabeled_samples)
    
    ce_epoch /= total_samples
    weighted_ce_epoch /= total_samples
    
    dice_sup_epoch /= total_samples
    #dice_unsup_epoch /= total_samples
    sup_epoch /= total_samples
    total_epoch /= total_samples
    
    
    #bgcdf_epoch /= total_unlabeled_samples
    unsup_epoch /= total_unlabeled_samples
    dice_unsup_epoch /= total_unlabeled_samples


    Dice_labeled_val = np.mean(2 * TP_labeled / (2 * TP_labeled + FP_labeled + FN_labeled + EPS))
    Dice_all_val     = np.mean(2 * TP_all / (2 * TP_all + FP_all + FN_all + EPS))
    IoU_labeled_val  = np.mean(TP_labeled / (TP_labeled + FP_labeled + FN_labeled + EPS))
    IoU_all_val      = np.mean(TP_all / (TP_all + FP_all + FN_all + EPS))
        
    gt_freq = (TP_labeled + FN_labeled) / (np.sum(TP_labeled + FN_labeled) + EPS)
    FWIoU_value = float(np.sum(gt_freq * (TP_labeled / (TP_labeled + FP_labeled + FN_labeled + EPS))))
    

    PerClassDice = 2 * TP_labeled / (2 * TP_labeled + FP_labeled + FN_labeled + EPS)
    PerClassIoU  = TP_labeled / (TP_labeled + FP_labeled + FN_labeled + EPS)

    val_loss_epoch = None
    if val_loader is not None:
    
        print(f"\n[Epoch {epoch}] Starting validation...\n", flush=True)
    
        val_loss_epoch = compute_validation_loss(model, val_loader, ce_fn, weighted_ce_fn, amp_autocast)
        
    def safe_append(key, value):
        append_history(LOSS_HISTORY, key, value)

    safe_append("CE", ce_epoch)
    safe_append("Weighted_CE", weighted_ce_epoch)
    
    #safe_append("Dice", dice_epoch)
    safe_append("Dice_SupLoss", dice_sup_epoch)
    safe_append("Dice_UnsupLoss", dice_unsup_epoch)
    safe_append("DiceMetric", Dice_labeled_val)  
    
    
    safe_append("Sup", sup_epoch)
    safe_append("Unsup", unsup_epoch)
    safe_append("Total", total_epoch)
    #safe_append("BGCDF", bgcdf_epoch)
    safe_append("Validation", val_loss_epoch)
    
    safe_append("PixelAcc", correct_pixels_labeled / max(1, total_pixels_labeled))
    safe_append("MeanIoU", IoU_labeled_val)
    safe_append("FWIoU", FWIoU_value)

   
    print("\n" + "="*70)
    print(f"[DEBUG] Epoch {epoch} - Labeled Pixels / Per-Class Stats")
    print("-"*70)
    total_images = len(data_loader.dataset)
    print(f"Total images in data loader: {total_images}")
    for c in range(len(TP_labeled)):
        print(f"Class {c}: TP={TP_labeled[c]:.0f}, FP={FP_labeled[c]:.0f}, FN={FN_labeled[c]:.0f}, Dice={PerClassDice[c]:.4f}, IoU={PerClassIoU[c]:.4f}")
    print("="*70 + "\n")
    
    print("\n" + "="*70)
    print(f"[DEBUG] Epoch {epoch} - Overall Metrics")
    print("-"*70)
    print(f"PixelAcc (labeled): {correct_pixels_labeled / max(1, total_pixels_labeled):.4f}")
    print(f"PixelAcc (all): {correct_pixels_all / max(1, total_pixels_all):.4f}")
    print(f"Mean IoU (labeled): {IoU_labeled_val:.4f}")
    print(f"Mean IoU (all): {IoU_all_val:.4f}")
    print(f"Mean Dice (labeled): {Dice_labeled_val:.4f}")
    print(f"Mean Dice (all): {Dice_all_val:.4f}")
    print("="*70 + "\n")  
    
        
    if log_dir and ptu.dist_rank == 0:
            
        write_csv(log_dir, epoch, LOSS_HISTORY)
        plot_losses(log_dir, LOSS_HISTORY)
        plot_metrics(log_dir, LOSS_HISTORY)
    

    return {
        "CE": ce_epoch, 
        "Weighted_CE": weighted_ce_epoch, 
                
        "Dice_SupLoss": dice_sup_epoch,
        "Dice_UnsupLoss": dice_unsup_epoch,
        
        "Sup": sup_epoch, 
        "Unsup": unsup_epoch, 
        "Total": total_epoch,
        "Validation": val_loss_epoch, 

        
        "PixelAcc_Labeled": correct_pixels_labeled / max(1, total_pixels_labeled),
        "PixelAcc_All": correct_pixels_all / max(1, total_pixels_all),
        
        #"DiceMetric": Dice_labeled_val,
        #"IoU": IoU_labeled_val,
        #"BGCDF": bgcdf_epoch,
        #"IoU": MeanIoU,
        "IoU_Labeled": IoU_labeled_val,
        "IoU_All": IoU_all_val,
        
        "DiceMetric": Dice_labeled_val
    }
    

# ----------------------------
# Validation
# ----------------------------
def compute_validation_loss(model, val_loader, ce_fn, weighted_ce_fn, amp_autocast):
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()
    total_val = 0.0
    total_samples = 0
    
    n_batches = len(val_loader)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(ptu.device)
            masks = batch.get("segmentation", batch.get("mask"))
            if masks is None:
                continue

            masks = masks.to(ptu.device).long()

            with amp_autocast():
                outputs = model_eval(images)
                probs = torch.softmax(outputs, dim=1)

                ce_loss = ce_fn(outputs, masks)

                B, C, H, W = outputs.shape
                valid_mask = (masks != IGNORE_LABEL).float()

                per_class_dice = []
                for c in range(C):
                    d = dice_loss_masked(
                        probs[:, c, :, :],
                        (masks == c).float(),
                        valid_mask
                    )
                    per_class_dice.append(d)

                dice_val = torch.stack(per_class_dice).mean()

                total_loss = ce_loss + dice_val

            bs = images.size(0)
            total_val += float(total_loss.item()) * bs
            total_samples += bs

    return total_val / max(1, total_samples)    
    
       

# ----------------------------
# Evaluate
# ----------------------------
@torch.no_grad()
def evaluate(model, data_loader, val_seg_gt, window_size=None, window_stride=None, amp_autocast=None, log_dir=None, epoch=None):
    model_eval = model.module if hasattr(model, "module") else model
    seg_pred = {}
    skipped_gt_all_ignore = 0
    total_samples = 0
    for batch in data_loader:
        images = batch["image"].to(ptu.device)
        ids = batch.get("id", None)
        if ids is None:
            ids = [f"img_{i}" for i in range(images.shape[0])]
        with amp_autocast():
            outputs = model_eval(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        for i, file_id in enumerate(ids):
            total_samples += 1
            key = file_id
            if key not in val_seg_gt:
                key = os.path.splitext(file_id)[0]
                if key not in val_seg_gt:
                    warnings.warn(f"Prediction id {file_id} not found; skipping.")
                    continue
            gt = val_seg_gt[key]
            if np.all(gt == IGNORE_LABEL):
                skipped_gt_all_ignore += 1
                continue
            pred = preds[i]
            if pred.shape != gt.shape:
                import cv2
                pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            seg_pred[key] = pred
    if skipped_gt_all_ignore > 0:
        warnings.warn(f"Skipped {skipped_gt_all_ignore} validation samples (GT all IGNORE_LABEL).")
    if len(seg_pred) == 0:
        raise RuntimeError("No valid predictions to evaluate (all GTs were blank or missing).")
    seg_pred = gather_data(seg_pred)
    val_seg_gt_filtered = {k: np.asarray(val_seg_gt[k], dtype=np.int64) for k in seg_pred.keys()}
    n_cls = getattr(data_loader.dataset, "n_cls", 2)
    metrics = compute_segmentation_metrics(seg_pred, val_seg_gt_filtered, n_cls)

    if log_dir and epoch is not None:
        from segm.utils.logging_config import write_eval_csv
        write_eval_csv(log_dir, epoch, metrics)
            
    # ----------------------------
    # Debug print for total and predicted pixels per class
    # ----------------------------
    print("\n" + "="*70)
    print("[DEBUG] Per-class pixel counts")
    print("-"*70)
    for c in range(n_cls):
        print(f"Class {c}: GT pixels = {metrics['GT_pixels'][c]}, Pred pixels = {metrics['Pred_pixels'][c]}")
    print("="*70 + "\n")
            
    ###################################################################
    ###################################################################       
            
    return metrics

def compute_segmentation_metrics(preds, gts, n_cls):
    eps = 1e-6
    TP = np.zeros(n_cls, dtype=np.float64)
    FP = np.zeros(n_cls, dtype=np.float64)
    FN = np.zeros(n_cls, dtype=np.float64)
    GT = np.zeros(n_cls, dtype=np.float64)
    PRED = np.zeros(n_cls, dtype=np.float64)
    total_valid_pixels = 0
    total_correct_pixels = 0

    for k in preds.keys():
        pred = np.asarray(preds[k], dtype=np.int64).flatten()
        gt   = np.asarray(gts[k], dtype=np.int64).flatten()
        valid = (gt != IGNORE_LABEL)
        if valid.sum() == 0:
            continue
        pred_v = pred[valid]
        gt_v   = gt[valid]
        total_valid_pixels += int(valid.sum())
        total_correct_pixels += int((pred_v == gt_v).sum())
        for c in range(n_cls):
            pred_c = (pred_v == c)
            gt_c   = (gt_v == c)
            TP[c] += np.sum(pred_c & gt_c)
            FP[c] += np.sum(pred_c & (~gt_c))
            FN[c] += np.sum((~pred_c) & gt_c)
            GT[c] += np.sum(gt_c)
            PRED[c] += np.sum(pred_c)

    PerClassIoU   = TP / (TP + FP + FN + eps)
    PerClassDice  = 2 * TP / (2 * TP + FP + FN + eps)
    Precision     = TP / (PRED + eps)
    Recall        = TP / (GT + eps)
    F1            = 2 * (Precision * Recall) / (Precision + Recall + eps)
    PixelAcc      = total_correct_pixels / (total_valid_pixels + eps)
    MeanIoU       = float(np.mean(PerClassIoU))
    FWIoU         = float(np.sum((GT / total_valid_pixels) * PerClassIoU))
    

    metrics = {
        "PixelAcc": PixelAcc,
        "MeanIoU": MeanIoU,
        "IoU": PerClassIoU.astype(np.float32),
        "FWIoU": FWIoU,
        "PerClassDice": PerClassDice.astype(np.float32),
        "Precision": Precision.astype(np.float32),
        "Recall": Recall.astype(np.float32),
        "F1": F1.astype(np.float32),
        "GT_pixels": GT.astype(np.int64),
        "Pred_pixels": PRED.astype(np.int64),
    }
    
    #metrics["IoU_all"] = PerClassIoU
    
    # ----------------------------
    # Debug print for total and predicted pixels per class
    # ----------------------------    
    return metrics

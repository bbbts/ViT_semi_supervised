# -*- coding: utf-8 -*-

import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
import pandas as pd
from tabulate import tabulate

from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial.distance import cdist

import segm.utils.torch as ptu
from segm.data.utils import STATS, dataset_cat_description
from segm.model.factory import load_model
from segm.model.utils import inference

# Dataset imports
from segm.data.ade20k import ADE20KSegmentation
from segm.data.flame import FlameDataset
from segm.data.meta import MetaDataset

IGNORE_LABEL = 255

# ---------------------------------------------------------
# Boundary Metrics
# ---------------------------------------------------------
def extract_boundary(mask):
    """
    Binary boundary extraction.
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask)
    boundary = mask.astype(bool) ^ eroded
    return boundary



def chamfer_distance(pred_mask, gt_mask):

    pred_pts = np.argwhere(extract_boundary(pred_mask))
    gt_pts = np.argwhere(extract_boundary(gt_mask))

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    dists = cdist(pred_pts, gt_pts)

    forward = np.mean(np.min(dists, axis=1) ** 2)
    backward = np.mean(np.min(dists, axis=0) ** 2)

    return forward + backward


def assd(pred_mask, gt_mask):

    pred_pts = np.argwhere(extract_boundary(pred_mask))
    gt_pts = np.argwhere(extract_boundary(gt_mask))

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    dists = cdist(pred_pts, gt_pts)

    forward = np.mean(np.min(dists, axis=1))
    backward = np.mean(np.min(dists, axis=0))

    return (forward + backward) / 2.0


# ---------------------------------------------------------
# Convert segmentation map to RGB using dataset category colors
# ---------------------------------------------------------
def seg_to_rgb(seg, colors):
    """
    Convert HxW segmentation map into RGB visualization.
    """
    if torch.is_tensor(seg):
        seg = seg.cpu().numpy()

    if seg.ndim == 3 and seg.shape[0] == 1:
        seg = seg[0]

    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in np.unique(seg):
        if cls_id == IGNORE_LABEL:
            rgb[seg == cls_id] = (50, 50, 50)
            continue
        if cls_id >= len(colors):
            continue
        rgb[seg == cls_id] = colors[cls_id]

    return rgb


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
@click.command()
@click.option("--model-path", type=str, required=True)
@click.option("--input-dir", "-i", type=str, required=True)
@click.option("--output-dir", "-o", type=str, required=True)
@click.option("--gpu/--cpu", default=True, is_flag=True)
@click.option("--gt-dir", type=str, default=None)
def main(model_path, input_dir, output_dir, gpu, gt_dir):

    ptu.set_gpu_mode(gpu)

    print("\n Loading model...")
    model, variant = load_model(model_path)
    model.to(ptu.device)
    model.eval()

    # ---------------- Dataset selection ----------------
    dataset_name = (
        variant.get("dataset")
        or variant.get("dataset_kwargs", {}).get("dataset", "ade20k")
    )
    normalization_key = variant.get("dataset_kwargs", {}).get("normalization", "default")

    if dataset_name.lower() == "ade20k":
        from segm.data.ade20k import ADE20K_CATS_PATH as CATS_PATH
        DatasetClass = ADE20KSegmentation
    elif dataset_name.lower() == "flame":
        from segm.data.flame import FLAME_CATS_PATH as CATS_PATH
        DatasetClass = FlameDataset
    elif dataset_name.lower() == "meta":
        from segm.data.meta import META_CATS_PATH as CATS_PATH
        DatasetClass = MetaDataset
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    cat_names, cat_colors = dataset_cat_description(CATS_PATH)
    print(f"Loaded {len(cat_colors)} categories\n")

    # ---------------- Paths ----------------
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    gt_dir = Path(gt_dir) if gt_dir is not None else None

    list_dir = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    print(f"Found {len(list_dir)} images\n")

    preds, filenames = [], []

    # ---------------- Inference settings ----------------
    image_size = variant.get("inference_kwargs", {}).get("window_size", 512)
    window_stride = variant.get("inference_kwargs", {}).get("window_stride", image_size)
    batch_size = variant.get("inference_kwargs", {}).get("batch_size", 2)

    # Use vibrant colors for predictions (same as first script)
    vibrant_colors = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )

    # ---------------------------------------------------------
    # Inference loop
    # ---------------------------------------------------------
    for filename in tqdm(list_dir, ncols=80, desc="Running inference"):

        pil_im = Image.open(filename).convert("RGB")

        im = F.pil_to_tensor(pil_im).float() / 255.0
        im = F.normalize(
            im, STATS[normalization_key]["mean"], STATS[normalization_key]["std"]
        )
        im = im.to(ptu.device).unsqueeze(0)

        logits = inference(
            model,
            [im],
            [{"flip": False}],
            ori_shape=im.shape[2:4],
            window_size=image_size,
            window_stride=window_stride,
            batch_size=batch_size,
        )

        seg_map = logits.argmax(0)
        seg_np = seg_map.cpu().numpy()

        # -------- Prediction colored mask --------
        seg_rgb = seg_to_rgb(seg_np, vibrant_colors)
        pil_seg = Image.fromarray(seg_rgb)

        # -------- Overlay prediction --------
        pil_pred_overlay = Image.blend(pil_im, pil_seg, 0.5)

        # -------- GT handling  --------
        if gt_dir is not None:
            gt_path = gt_dir / (filename.stem + ".png")
            if gt_path.exists():
                gt_np = np.array(Image.open(gt_path))

                gt_rgb = seg_to_rgb(gt_np, vibrant_colors)
                pil_gt = Image.fromarray(gt_rgb)

                if pil_gt.size != pil_pred_overlay.size:
                    pil_gt = pil_gt.resize(pil_pred_overlay.size, Image.NEAREST)

                # Side-by-side GT | Prediction
                combined = Image.new(
                    "RGB",
                    (pil_gt.width + pil_pred_overlay.width, pil_gt.height),
                )
                combined.paste(pil_gt, (0, 0))
                combined.paste(pil_pred_overlay, (pil_gt.width, 0))
                combined.save(output_dir / filename.name)
            else:
                pil_pred_overlay.save(output_dir / filename.name)

        else:
            pil_pred_overlay.save(output_dir / filename.name)

        preds.append(seg_np)
        filenames.append(filename.name)

    print(f"\n Inference complete. Saved results to {output_dir}\n")

    # ---------------------------------------------------------
    # Evaluation 
    # ---------------------------------------------------------
    if gt_dir is None:
        return

    print("\n Running evaluation...")

    # Determine split name
    possible_splits = ["val", "validation"]
    split_name = None
    for s in possible_splits:
        if (
            (gt_dir.parents[1] / "images" / s).exists()
            and (gt_dir.parents[1] / "masks" / s).exists()
        ):
            split_name = s
            break
    if split_name is None:
        split_name = "val"

    val_dataset = DatasetClass(
        image_size=image_size,
        crop_size=image_size,
        split=split_name,
        normalization=normalization_key,
        root=gt_dir.parents[1],
    )
    n_cls = val_dataset.n_cls

    # Load GT maps
    gt_maps = {
        p.stem: np.array(Image.open(p)) for p in sorted(gt_dir.glob("*.png"))
    }

    intersection = np.zeros(n_cls)
    union = np.zeros(n_cls)
    gt_count = np.zeros(n_cls)
    pred_count = np.zeros(n_cls)
    tp = np.zeros(n_cls)
    fp = np.zeros(n_cls)
    fn = np.zeros(n_cls)

    total_pixels = 0
    correct_pixels = 0
    
    
    chamfer_scores = []
    assd_scores = []

    for fname, pred in zip(filenames, preds):
        key = Path(fname).stem
        if key not in gt_maps:
            continue

        gt = gt_maps[key]
        mask = gt != IGNORE_LABEL
        
        for cls in range(n_cls):

            pred_cls = (pred == cls)
            gt_cls = (gt == cls)
        
            if gt_cls.sum() == 0:
                continue
            
            cd = chamfer_distance(pred_cls, gt_cls)
            if not np.isnan(cd):
                chamfer_scores.append(cd)
            
            #asd_val = average_surface_distance(pred_cls, gt_cls)
            #if not np.isnan(asd_val):
                #asd_scores.append(asd_val)
            
            assd_val = assd(pred_cls, gt_cls)
            if not np.isnan(assd_val):
                assd_scores.append(assd_val)

        total_pixels += mask.sum()
        correct_pixels += (pred[mask] == gt[mask]).sum()

        for cls in range(n_cls):
            pcls = pred == cls
            gcls = gt == cls

            inter = np.logical_and(pcls, gcls).sum()
            uni = np.logical_or(pcls, gcls).sum()

            intersection[cls] += inter
            union[cls] += uni
            gt_count[cls] += gcls.sum()
            pred_count[cls] += pcls.sum()

            tp[cls] += inter
            fp[cls] += np.logical_and(pcls, ~gcls).sum()
            fn[cls] += np.logical_and(~pcls, gcls).sum()

    pixel_acc = correct_pixels / (total_pixels + 1e-6)
    per_class_iou = intersection / (union + 1e-6)
    mean_iou = np.nanmean(per_class_iou)
    
    gt_freq = gt_count / (total_pixels + 1e-6)
    fw_iou = np.sum(gt_freq * per_class_iou)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    dice = 2 * intersection / (2 * intersection + fp + fn + 1e-6)
    
    mean_chamfer = (
        np.mean(chamfer_scores)
        if len(chamfer_scores) > 0
        else np.nan
    )
    mean_assd = (
        np.mean(assd_scores)
        if len(assd_scores) > 0
        else np.nan
    )

    metrics = {
        "Pixel_Acc": pixel_acc,
        "Mean_IoU": mean_iou,
        "FWIoU": fw_iou,
        "Dice": np.mean(dice),
        "Precision": np.mean(precision),
        "Recall": np.mean(recall),
        "F1": np.mean(f1),
        
        "ChamferDistance": mean_chamfer,
        "ASSD": mean_assd,
    
        "PerClassDice": dice.tolist(),
        "PerClassIoU": per_class_iou.tolist(),
    }

    csv_path = output_dir / "eval_metrics.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    print(f"\n Saved evaluation metrics to {csv_path}\n")

    # ---- Per-class table ----
    table = []
    for i in range(n_cls):
        table.append(
            [
                i,
                cat_names[i] if i < len(cat_names) else f"class_{i}",
                round(per_class_iou[i], 4),
                round(dice[i], 4),
                round(precision[i], 4),
                round(recall[i], 4),
                round(f1[i], 4),
            ]
        )

    print("Per-class Metrics:\n")
    print(tabulate(table, headers=["ID", "Name", "IoU", "Dice", "Precision", "Recall", "F1"]))


if __name__ == "__main__":
    main()


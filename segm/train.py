#!/usr/bin/env python3
#SEMI_SUPERVISED

import sys
from pathlib import Path
import yaml
import torch
import click
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os
import shutil
import random
import numpy as np
from PIL import Image
from tabulate import tabulate

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config
from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params
#from segm.engine import LOSS_HISTORY
import segm.engine as engine

from segm.engine import evaluate
from contextlib import suppress
from timm.utils import NativeScaler



from segm.utils.logging_config import init_history
import segm.utils.logging_config as log_cfg


import pandas as pd
import matplotlib.pyplot as plt

IGNORE_LABEL = 255  # consistent ignore

#engine.LOSS_HISTORY = init_history()

if not hasattr(engine, "LOSS_HISTORY") or engine.LOSS_HISTORY is None:
    engine.LOSS_HISTORY = init_history()

# -------------------------------
# SEMI-SUPERVISED DATASET GENERATION
# -------------------------------
def create_semi_supervised_dataset(dataset_dir, labeled_ratio, base_output_dir="semi_supervised_dataset"):
    """
    Creates a semi-supervised version of a dataset:
    - Labeled masks are kept for a fraction of train images
    - Unlabeled masks are replaced with IGNORE_LABEL
    - Works safely on NFS mounts
    """
    dataset_dir = Path(dataset_dir)

    # Create a timestamped folder to avoid conflicts / NFS busy errors
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_output_dir}_{timestamp}")
    out_images_dir = output_dir / "images"
    out_masks_dir = output_dir / "masks"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    split_names = {
        "train": ["train", "training"],
        "val": ["val", "validation", "validate"],
        "test": ["test", "testing"],
    }

    summary_table = []

    for split_key, aliases in split_names.items():
        images_split_dir = None
        masks_split_dir = None
        for alias in aliases:
            tmp_img = images_dir / alias
            tmp_mask = masks_dir / alias
            if tmp_img.exists() and tmp_mask.exists():
                images_split_dir = tmp_img
                masks_split_dir = tmp_mask
                break
        if images_split_dir is None or masks_split_dir is None:
            continue

        out_img_split = out_images_dir / split_key
        out_mask_split = out_masks_dir / split_key
        out_img_split.mkdir(parents=True, exist_ok=True)
        out_mask_split.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(images_split_dir.glob("*.*")))
        mask_files = sorted(list(masks_split_dir.glob("*.*")))

        assert len(image_files) == len(mask_files), f"Images and masks mismatch in {split_key}"

        # Determine labeled indices
        if split_key == "train":
            n_total = len(image_files)
            n_labeled = int(n_total * labeled_ratio)
            labeled_indices = set(random.sample(range(n_total), n_labeled))
        else:
            labeled_indices = set(range(len(image_files)))

        n_labeled_count = 0
        n_blank_count = 0

        # Copy files safely
        for i, (img_path, mask_path) in enumerate(zip(image_files, mask_files)):
            shutil.copy(img_path, out_img_split / img_path.name)
            mask_out_path = out_mask_split / mask_path.name

            if i in labeled_indices:
                shutil.copy(mask_path, mask_out_path)
                n_labeled_count += 1
            else:
                mask_img = Image.open(mask_path)
                blank_mask = Image.fromarray(
                    np.full((mask_img.height, mask_img.width), IGNORE_LABEL, dtype=np.uint8)
                )
                blank_mask.save(mask_out_path)
                n_blank_count += 1

        summary_table.append([split_key, len(image_files), n_labeled_count, n_blank_count])

    print("\nSemi-Supervised Dataset Summary:")
    print(tabulate(summary_table, headers=["Split", "Total Images", "Labeled Masks", "Blank Masks"]))
    print(f"Semi-supervised dataset created at: {output_dir}\n")
    return str(output_dir)



# -------------------------------
# MAIN TRAIN FUNCTION
# -------------------------------
@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str, help="Dataset name (used in segm configs)")
@click.option("--dataset-dir", type=str, default=None, help="Original fully labeled dataset dir")
@click.option("--labeled-ratio", type=float, default=None, help="Fraction of train images to keep labeled")
@click.option("--teacher-dir", type=str, default=None, help="Path to pre-trained teacher model")
@click.option("--im-size", default=None, type=int)
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)


def main(
    log_dir, dataset, dataset_dir, labeled_ratio, teacher_dir,
    im_size, crop_size, window_size, window_stride,
    backbone, decoder, optimizer, scheduler, weight_decay,
    dropout, drop_path, batch_size, epochs, learning_rate,
    normalization, eval_freq, amp, resume
):
    ptu.set_gpu_mode(True)
    distributed.init_process()

    #if dataset_dir is not None and labeled_ratio is not None:
        #dataset_root = create_semi_supervised_dataset(dataset_dir, labeled_ratio)
    #else:
        #dataset_root = None
        
        
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    #load_history(log_dir)
    
    variant_file = log_dir / "variant.yml"
    checkpoint_path = log_dir / "checkpoint.pth"
    
    #if resume and variant_file.exists():
    if resume and checkpoint_path.exists() and variant_file.exists():
        print("Loading existing dataset split AND last saved checkpoint from variant.yml")
        print("Resuming: loading checkpoint ONLY (no dataset regeneration)")
    
        #load_history(log_dir)
            
        with open(variant_file, "r") as f:
            old_variant = yaml.full_load(f)
    
        dataset_root = old_variant["dataset_kwargs"]["root"]
    
        print(f"Reusing dataset root: {dataset_root}")
    
    else:
        if dataset_dir is not None and labeled_ratio is not None:
            dataset_root = create_semi_supervised_dataset(
                dataset_dir,
                labeled_ratio
            )
        else:
            dataset_root = None    
        
    

    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    decoder_cfg = cfg["decoder"]["mask_transformer"] if "mask_transformer" in decoder else cfg["decoder"][decoder]

    im_size = im_size or dataset_cfg["im_size"]
    crop_size = crop_size or dataset_cfg.get("crop_size", im_size)
    window_size = window_size or dataset_cfg.get("window_size", im_size)
    window_stride = window_stride or dataset_cfg.get("window_stride", im_size)

    model_cfg.update({
        #"image_size": (crop_size, crop_size),
        "image_size": [crop_size, crop_size],
        "backbone": backbone,
        "dropout": dropout,
        "drop_path_rate": drop_path,
    })
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    world_batch_size = batch_size or dataset_cfg["batch_size"]
    num_epochs = epochs or dataset_cfg["epochs"]
    lr = learning_rate or dataset_cfg["learning_rate"]
    eval_freq = eval_freq or dataset_cfg.get("eval_freq", 1)
    if normalization:
        model_cfg["normalization"] = normalization

    batch_size = max(1, world_batch_size // max(1, ptu.world_size))

    dataset_kwargs = dict(
        dataset=dataset,
        image_size=im_size,
        crop_size=crop_size,
        batch_size=batch_size,
        normalization=model_cfg.get("normalization", "vit"),
        split="train",
        num_workers=2,
        root=dataset_root or dataset_cfg.get("data_root", dataset_cfg.get("root", None)),
    )

    #print(f"Creating training dataset for {dataset}...")
    
    if resume and checkpoint_path.exists():
        print(f"Reloading existing dataset from: {dataset_kwargs['root']}")
    else:
        print(f"Creating training dataset for {dataset}...")
        
    train_dataset = create_dataset(dataset_kwargs)

    val_split = None
    for split_name in ["validation", "val"]:
        try:
            val_kwargs = dataset_kwargs.copy()
            val_kwargs["split"] = split_name
            val_kwargs["batch_size"] = 1
            val_dataset = create_dataset(val_kwargs)
            val_split = split_name
            print(f"Detected validation split: '{val_split}'")
            break
        except RuntimeError as e:
            if "No images or masks found" in str(e) or "No images" in str(e):
                continue
            raise

    if val_split is None:
        raise RuntimeError("No validation split found. Expected folder 'validation' or 'val'.")

    def make_loader(dataset_obj, batch_size, shuffle=True):
        sampler = DistributedSampler(dataset_obj, shuffle=shuffle) if ptu.distributed else None
        loader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=(sampler is None) and shuffle,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        )
        loader.sampler_obj = sampler
        return loader

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size=1, shuffle=False)

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Validation ('{val_split}') dataset length: {len(val_loader.dataset)}")

    n_cls = train_dataset.n_cls

    # ---- Student Model ----
    model_cfg["n_cls"] = n_cls
    student_model = create_segmenter(model_cfg).to(ptu.device)

    # ---- Teacher Model ----
    teacher_model = None
    if teacher_dir is not None:
        teacher_model = create_segmenter(model_cfg).to(ptu.device)
        checkpoint = torch.load(os.path.join(teacher_dir, "checkpoint.pth"), map_location="cpu")
        teacher_model.load_state_dict(checkpoint["model"])
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    # ---- Distributed wrap BEFORE optimizer creation (important) ----
    if ptu.distributed:
        student_model = DDP(student_model, device_ids=[ptu.device], find_unused_parameters=True)

    # ---- Optimizer & scheduler ----
    iter_max = len(train_loader) * num_epochs
    iter_warmup = 0.0

    optimizer_kwargs = dict(
        opt=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
        clip_grad=None,
        sched=scheduler,
        epochs=num_epochs,
        min_lr=1e-5,
        poly_power=0.9,
        poly_step_size=1,
        iter_max=iter_max,
        iter_warmup=iter_warmup,
    )
    opt_args = type('', (), {})()
    for k, v in optimizer_kwargs.items():
        setattr(opt_args, k, v)

    optimizer = create_optimizer(opt_args, student_model)
    lr_scheduler = create_scheduler(opt_args, optimizer)

    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()


    # ---- Resume ----
    
    start_epoch = 0    
        
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    

        if "loss_history" in checkpoint:
            for k in checkpoint["loss_history"]:
                if k in engine.LOSS_HISTORY:
                    engine.LOSS_HISTORY[k] = list(checkpoint["loss_history"][k])
    

        print("Rebuilding history from checkpoint + CSV...")
    
        csv_path = os.path.join(log_dir, "losses.csv")
    
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
    
            for k in engine.LOSS_HISTORY.keys():
                if k in df.columns:
                    engine.LOSS_HISTORY[k] = df[k].tolist()     
                    
                        
                    
        num_epochs = checkpoint.get("total_epochs", num_epochs)
    
        if hasattr(student_model, "module"):
            student_model.module.load_state_dict(checkpoint["model"])
        else:
            student_model.load_state_dict(checkpoint["model"])
    
        optimizer.load_state_dict(checkpoint["optimizer"])
    
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
    
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    
        start_epoch = checkpoint["epoch"] + 1
    
        print(f"Resuming from epoch {start_epoch}")

    variant = dict(
        world_batch_size=world_batch_size,
        dataset_kwargs=dataset_kwargs,
        net_kwargs=model_cfg,
        optimizer_kwargs=optimizer_kwargs,
        amp=amp,
        log_dir=str(log_dir),
        inference_kwargs=dict(im_size=im_size, window_size=window_size, window_stride=window_stride),
    )
        
    #if not (resume and variant_file.exists()):
    if not variant_file.exists():
        with open(log_dir / "variant.yml", "w") as f:
            #yaml.dump(variant, f)
            yaml.safe_dump(variant, f)

    print(f"Encoder parameters: {num_params(student_model.encoder)}")
    print(f"Decoder parameters: {num_params(student_model.decoder)}")

    # ---- Train loop ----
    
    #bgcdf_loss_fn = BGCDFLoss(num_classes=n_cls).to(ptu.device)
    
    #for epoch in range(num_epochs):
    for epoch in range(start_epoch, num_epochs):
        if hasattr(train_loader, "sampler_obj") and train_loader.sampler_obj is not None:
            if hasattr(train_loader.sampler_obj, "set_epoch"):
                train_loader.sampler_obj.set_epoch(epoch)
    
        
        train_logger = engine.train_one_epoch(
            student_model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            log_dir=str(log_dir),
            val_loader=val_loader,
            teacher_model=teacher_model,
            unsup_weight=1.0,
        )   


        print(f"[Epoch {epoch+1}/{num_epochs}] Losses: {train_logger}", flush=True)

        if ptu.dist_rank == 0:
            log_cfg.plot_eval_metrics_from_csv(log_dir)
            
            snapshot = {
                "model": student_model.module.state_dict() if hasattr(student_model, "module") else student_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "total_epochs": num_epochs,
                #"loss_history": {k: list(v) for k, v in LOSS_HISTORY.items()},
                "loss_history": {k: list(v) for k, v in engine.LOSS_HISTORY.items()},
            }
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            
            torch.save(snapshot, checkpoint_path)
                                        

        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            val_seg_gt = {}
            for idx in range(len(val_loader.dataset)):
                item = val_loader.dataset[idx]
                mask = item.get("segmentation", item.get("mask"))
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                file_id = item.get("id", f"img_{idx}")
                file_id = os.path.splitext(file_id)[0]
                val_seg_gt[file_id] = mask_np

            print(f"\n[Epoch {epoch}] Starting evaluation...\n", flush=True)

            eval_logger = evaluate(student_model, val_loader, val_seg_gt, window_size, window_stride, amp_autocast, log_dir=str(log_dir), epoch=epoch)
            

            print(f"Evaluation Metrics [Epoch {epoch}]: {eval_logger}", flush=True)
            print("")

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(0)

if __name__ == "__main__":
    main()

    

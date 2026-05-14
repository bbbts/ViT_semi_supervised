# segm/utils/logging_config.py

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. HISTORY MANAGEMENT
# =========================================================

def init_history():    
    return {
        "CE": [],
        "Weighted_CE": [],
    
        "Dice_SupLoss": [],
        "Dice_UnsupLoss": [],
    
        "Sup": [],
        "Unsup": [],
        "Total": [],
        "Validation": [],
        #"BGCDF": [],
    
        "PixelAcc": [],
        "MeanIoU": [],
        "DiceMetric": [],
        "FWIoU": [],
    }


def append_history(history, key, value):
    if key not in history:
        history[key] = []
    history[key].append(float(value) if value is not None else np.nan)


def serialize_history(history):
    return {k: list(v) for k, v in history.items()}


def restore_history_from_checkpoint(checkpoint, history):
    if "loss_history" not in checkpoint:
        return history

    saved = checkpoint["loss_history"]
    for k in history.keys():
        if k in saved:
            history[k] = list(saved[k])
    return history

# =========================================================
# 2. CSV LOGGING (SINGLE SOURCE OF TRUTH)
# =========================================================

def write_csv(log_dir, epoch, history, filename="losses.csv"):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, filename)

    row = {"epoch": epoch}
    for k, v in history.items():
        row[k] = v[-1] if len(v) > 0 else np.nan

    file_exists = os.path.exists(csv_path)

    fieldnames = ["epoch"] + list(history.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def write_eval_csv(log_dir, epoch, metrics, filename="evaluation_metrics.csv"):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, filename)

    row = {"epoch": int(epoch)}

    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            row[k] = str(v.tolist())
        else:
            row[k] = float(v)

    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

# =========================================================
# 3. PLOTTING (LOSS + METRICS)
# =========================================================

def plot_losses(log_dir, history):
    os.makedirs(log_dir, exist_ok=True)

    keys = ["CE", "Weighted_CE", "Sup", "Unsup", "Total"]

    plt.figure(figsize=(10, 6))

    for k in keys:
        if k in history and len(history[k]) > 0:
            plt.plot(history[k], label=k)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_losses.png"))
    plt.close()


def plot_metrics(log_dir, history):
    os.makedirs(log_dir, exist_ok=True)

    #keys = ["PixelAcc_Labeled", "MeanIoU", "DiceMetric", "FWIoU"]
    keys = ["PixelAcc", "MeanIoU", "DiceMetric", "FWIoU", "Dice_SupLoss", "Dice_UnsupLoss"]
    
    print("PLOTTING KEYS:", history.keys())

    plt.figure(figsize=(10, 6))

    for k in keys:
        if k in history and len(history[k]) > 0:
            plt.plot(history[k], label=k)

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Segmentation Metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_metrics.png"))
    plt.close()


def plot_eval_metrics_from_csv(log_dir, filename="evaluation_metrics.csv"):
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)

    plt.figure()
    
    plt.plot(df["epoch"], df["MeanIoU"], label="MeanIoU")
    plt.plot(df["epoch"], df["PixelAcc"], label="PixelAcc")
    plt.plot(df["epoch"], df["FWIoU"], label="FWIoU")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "eval_metrics.png"))
    plt.close()
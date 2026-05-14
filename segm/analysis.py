# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATHS = {
    0.4: "/home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META/segm/MODEL_FILE_0.4/evaluation_metrics.csv",
    0.5: "/home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META/segm/MODEL_FILE_0.5/evaluation_metrics.csv",
    0.6: "/home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META/segm/MODEL_FILE_0.6/evaluation_metrics.csv",
    0.7: "/home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META/segm/MODEL_FILE_0.7/evaluation_metrics.csv",
}

def compute_means(csv_path):
    df = pd.read_csv(csv_path)
    return (
        df["MeanIoU"].mean(),
        df["PixelAcc"].mean(),
        df["FWIoU"].mean()
    )

ratios = []
miou, pix, fw = [], [], []

for r, p in sorted(MODEL_PATHS.items()):
    m, pa, f = compute_means(p)
    ratios.append(r)
    miou.append(m)
    pix.append(pa)
    fw.append(f)

    print(f"Ratio {r}: mIoU={m:.8f}, PixelAcc={pa:.8f}, FWIoU={f:.8f}")

miou = np.array(miou)
pix = np.array(pix)
fw = np.array(fw)

# =========================================================
# RELATIVE IMPROVEMENT 
# =========================================================
miou_rel = (miou - miou.min()) * 1000
pix_rel  = (pix  - pix.min())  * 1000
fw_rel   = (fw   - fw.min())   * 1000

plt.figure(figsize=(8,5), dpi=200)

plt.plot(ratios, miou_rel, marker="o", linewidth=2, label="MeanIoU")
plt.plot(ratios, pix_rel, marker="o", linewidth=2, label="PixelAcc")
plt.plot(ratios, fw_rel, marker="o", linewidth=2, label="FWIoU")

plt.xlabel("Labeled Ratio")
plt.ylabel("Relative Improvement (scaled)")
plt.title("Performance Gain vs Labeled Ratio")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.savefig(
    "/home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META/segm/PLOTS/relative_trend.png",
    bbox_inches="tight"
)
plt.close()

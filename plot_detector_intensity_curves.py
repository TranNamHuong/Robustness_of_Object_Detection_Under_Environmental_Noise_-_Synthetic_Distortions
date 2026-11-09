#!/usr/bin/env python
"""Plot full raw intensity curves for two detectors with shared 320px AP protocol."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
YOLO_PATH = ROOT / "runs/yolo_custom_fullval_seed42/metrics.csv"
SSD_PATH = ROOT / "runs/ssdlite_fullval_seed42/metrics.csv"
OUT = ROOT / "runs/normalized_detector_comparison_fullval"
FAMILIES = {
    "Gaussian": [0.01, 0.03, 0.06],
    "Salt-and-Pepper": [0.01, 0.03, 0.06],
    "Fog": [0.15, 0.30, 0.50],
    "Low-light factor†": [0.25, 0.40, 0.60],
}
PREFIX = {"Gaussian": "gaussian", "Salt-and-Pepper": "salt_pepper", "Fog": "fog", "Low-light factor†": "low_light"}

def load(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}

def main():
    yolo, ssd = load(YOLO_PATH), load(SSD_PATH)
    records = []
    for family, levels in FAMILIES.items():
        for level in levels:
            level_text = str(level).rstrip("0").rstrip(".")
            condition = f"{PREFIX[family]}_{level_text}"
            for detector, data in (("YOLOv8n-320", yolo), ("SSDLite320-MobileNetV3-Large", ssd)):
                if condition not in data:
                    raise KeyError(f"Missing {condition} from {detector}")
                row = data[condition]
                records.append({"family": family, "level": level, "condition": condition,
                                "detector": detector, "images": int(row["images"]),
                                "mAP50": float(row["mAP50"]), "mAP50_95": float(row["mAP50_95"]),
                                "tp_iou50": int(row["tp_iou50"]), "fp_iou50": int(row["fp_iou50"]),
                                "fn_iou50": int(row["fn_iou50"])})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "raw_intensity_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"YOLOv8n-320": "#2563eb", "SSDLite320-MobileNetV3-Large": "#ea580c"}
    for ax, (family, levels) in zip(axes.flat, FAMILIES.items()):
        for detector in colors:
            vals = [next(r["mAP50"] for r in records if r["family"] == family and r["level"] == level and r["detector"] == detector) for level in levels]
            ax.plot(levels, vals, marker="o", linewidth=2, label=detector, color=colors[detector])
        ax.set_title(family)
        ax.set_xlabel("Intensity / transformation factor")
        ax.set_ylabel("mAP@0.5")
        ax.set_xticks(levels)
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Raw corruption intensity curves — full COCO val2017, shared AP protocol")
    fig.savefig(OUT / "raw_mAP_vs_intensity_both_detectors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    families = list(FAMILIES)
    x = np.arange(len(families))
    width = 0.34
    for i, detector in enumerate(colors):
        values = []
        for family in families:
            raw = [r["mAP50"] for r in records if r["family"] == family and r["detector"] == detector]
            values.append(min(raw))
        ax.bar(x + (i - 0.5) * width, values, width, label=detector, color=colors[detector])
    ax.set_xticks(x, ["Gaussian", "Salt-Pepper", "Fog", "Low-light"])
    ax.set_ylabel("Worst raw mAP@0.5 among tested levels")
    ax.set_title("Worst observed raw condition by corruption family")
    ax.grid(axis="y", alpha=0.25); ax.legend()
    fig.savefig(OUT / "worst_condition_by_family.png", dpi=180)
    plt.close(fig)
    print(f"Saved {len(records)} raw detector-level rows and curves in {OUT}")
    print("† Low-light factor: lower value means darker image.")

if __name__ == "__main__":
    main()

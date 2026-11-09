#!/usr/bin/env python
"""Compare raw and recovered mAP at all full-val conditions available for both detectors."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
YOLO = ROOT / "runs/yolo_custom_fullval_seed42/metrics.csv"
SSD = ROOT / "runs/ssdlite_fullval_seed42/metrics.csv"
OUT = ROOT / "runs/normalized_detector_comparison_fullval"
FAMILIES = {"Gaussian": [0.06], "Salt-and-Pepper": [0.01, 0.03, 0.06],
            "Fog": [0.15, 0.30, 0.50], "Low-light factor†": [0.25, 0.40, 0.60]}
PREFIX = {"Gaussian": "gaussian", "Salt-and-Pepper": "salt_pepper", "Fog": "fog", "Low-light factor†": "low_light"}
MODELS = [("YOLOv8n-320", YOLO, "#2563eb"), ("SSDLite320-MobileNetV3-Large", SSD, "#ea580c")]

def load(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}

def main():
    data = {name: load(path) for name, path, _ in MODELS}
    rows = []
    for family, levels in FAMILIES.items():
        for level in levels:
            level_text = str(level).rstrip("0").rstrip(".")
            raw_condition = f"{PREFIX[family]}_{level_text}"
            rec_condition = raw_condition + "_recovered"
            for detector, _, _ in MODELS:
                metrics = data[detector]
                if raw_condition not in metrics or rec_condition not in metrics:
                    raise KeyError(f"Missing raw/recovered {raw_condition} for {detector}")
                raw, rec = metrics[raw_condition], metrics[rec_condition]
                rows.append({"family": family, "level": level, "detector": detector,
                             "raw_condition": raw_condition, "recovered_condition": rec_condition,
                             "raw_mAP50": float(raw["mAP50"]), "recovered_mAP50": float(rec["mAP50"]),
                             "delta_mAP50": float(rec["mAP50"]) - float(raw["mAP50"]),
                             "raw_mAP50_95": float(raw["mAP50_95"]), "recovered_mAP50_95": float(rec["mAP50_95"]),
                             "raw_fp": int(raw["fp_iou50"]), "recovered_fp": int(rec["fp_iou50"]),
                             "raw_fn": int(raw["fn_iou50"]), "recovered_fn": int(rec["fn_iou50"])})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "recovery_intensity_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (family, levels) in zip(axes.flat, FAMILIES.items()):
        if len(levels) == 1:
            x = np.arange(len(MODELS)); width = 0.34
            for i, (detector, _, color) in enumerate(MODELS):
                row = next(r for r in rows if r["family"] == family and r["detector"] == detector)
                ax.bar(i - width/2, row["raw_mAP50"], width, color=color,
                       label=f"{detector} raw")
                ax.bar(i + width/2, row["recovered_mAP50"], width, color=color,
                       alpha=0.55, hatch="//", label=f"{detector} NLM")
            ax.set_xticks(x, [m[0] for m in MODELS], rotation=10, ha="right")
        else:
            for detector, _, color in MODELS:
                for key, style, suffix in (("raw_mAP50", "-", "raw"), ("recovered_mAP50", "--", "recovered")):
                    vals = [next(r[key] for r in rows if r["family"] == family and r["level"] == level and r["detector"] == detector) for level in levels]
                    ax.plot(levels, vals, marker="o", linestyle=style, linewidth=1.8,
                            color=color, label=f"{detector} {suffix}")
            ax.set_xticks(levels)
        ax.set_title(family)
        ax.set_ylabel("mAP@0.5")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Preprocessing recovery by intensity — full COCO val2017")
    fig.savefig(OUT / "recovery_curves_both_detectors.png", dpi=180)
    plt.close(fig)
    print(f"Saved {len(rows)} raw/recovered pairs to {OUT / 'recovery_intensity_metrics.csv'}")
    for family in FAMILIES:
        subset = [r for r in rows if r["family"] == family]
        print(f"{family}: " + "; ".join(f"{r['detector']} L{r['level']}: {r['raw_mAP50']:.4f}->{r['recovered_mAP50']:.4f} ({r['delta_mAP50']:+.4f})" for r in subset))

if __name__ == "__main__":
    main()

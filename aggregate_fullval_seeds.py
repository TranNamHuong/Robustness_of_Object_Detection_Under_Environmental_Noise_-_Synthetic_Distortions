#!/usr/bin/env python
"""Summarize selected full-val stochastic conditions over noise seeds 42 and 43."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "YOLOv8n-320": ROOT / "runs/yolo_custom_fullval_seed42/metrics.csv",
    "SSDLite320-MobileNetV3-Large": ROOT / "runs/ssdlite_fullval_seed42/metrics.csv",
}
SEEDS = [42, 43, 44]
PAIRS = [("Gaussian 0.06 raw", "gaussian_0.06"),
         ("Salt-and-Pepper 0.06 raw", "salt_pepper_0.06"),
         ("Salt-and-Pepper 0.06 median", "salt_pepper_0.06_recovered")]
METRICS = ["mAP50", "mAP50_95", "fp_iou50", "fn_iou50"]

def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}

def main():
    sources = {name: read(path) for name, path in SOURCES.items()}
    rows = []
    for detector, table in sources.items():
        for label, base_condition in PAIRS:
            values = {}
            for metric in METRICS:
                samples = []
                for seed in SEEDS:
                    condition = base_condition if seed == 42 else f"{base_condition}_seed{seed}"
                    value = float(table[condition][metric])
                    values[f"{metric}_seed{seed}"] = value
                    samples.append(value)
                values[f"{metric}_mean"] = float(np.mean(samples))
                values[f"{metric}_sd_sample"] = float(np.std(samples, ddof=1))
            rows.append({"detector": detector, "condition": label, "n_seeds": len(SEEDS), **values})
    out_dir = ROOT / "runs/normalized_detector_comparison_fullval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "seed_repeat_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    conditions = [p[0] for p in PAIRS]
    x = np.arange(len(conditions)); width = 0.34
    colors = ["#2563eb", "#ea580c"]
    for i, detector in enumerate(SOURCES):
        means = [next(r["mAP50_mean"] for r in rows if r["condition"] == c and r["detector"] == detector) for c in conditions]
        errs = [next(r["mAP50_sd_sample"] for r in rows if r["condition"] == c and r["detector"] == detector) for c in conditions]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=4,
               label=detector, color=colors[i])
    ax.set_xticks(x, conditions, rotation=10, ha="right")
    ax.set_ylabel("mAP@0.5 mean ± sample SD")
    ax.set_title("Selected full-val conditions over two corruption seeds")
    ax.grid(axis="y", alpha=0.25); ax.legend()
    fig.savefig(out_dir / "seed_repeat_map50.png", dpi=180)
    plt.close(fig)
    print(f"Saved summary to {out_dir / 'seed_repeat_summary.csv'}")
    for row in rows:
        print(f"{row['detector']} | {row['condition']}: mAP50={row['mAP50_mean']:.4f} ± {row['mAP50_sd_sample']:.4f}; mAP50-95={row['mAP50_95_mean']:.4f} ± {row['mAP50_95_sd_sample']:.4f}")

if __name__ == "__main__":
    main()

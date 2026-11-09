#!/usr/bin/env python
"""Compare raw Gaussian noise with bilateral and available NLM recovery results."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/normalized_detector_comparison_fullval"
LEVELS = [0.01, 0.03, 0.06]
SOURCES = {
    "YOLOv8n-320": ROOT / "runs/yolo_custom_fullval_seed42/metrics.csv",
    "SSDLite320-MobileNetV3-Large": ROOT / "runs/ssdlite_fullval_seed42/metrics.csv",
}

def load(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}

def main():
    data = {name: load(path) for name, path in SOURCES.items()}
    rows = []
    for detector, metrics in data.items():
        for level in LEVELS:
            level_text = str(level).rstrip("0").rstrip(".")
            raw = metrics[f"gaussian_{level_text}"]
            bilateral = metrics[f"gaussian_{level_text}_bilateral"]
            nlm = metrics.get(f"gaussian_{level_text}_recovered")
            rows.append({"detector": detector, "level": level,
                         "raw_mAP50": float(raw["mAP50"]),
                         "bilateral_mAP50": float(bilateral["mAP50"]),
                         "bilateral_delta_vs_raw": float(bilateral["mAP50"]) - float(raw["mAP50"]),
                         "nlm_mAP50": float(nlm["mAP50"]) if nlm else "",
                         "nlm_delta_vs_raw": float(nlm["mAP50"]) - float(raw["mAP50"]) if nlm else "",
                         "raw_mAP50_95": float(raw["mAP50_95"]),
                         "bilateral_mAP50_95": float(bilateral["mAP50_95"]),
                         "raw_fp": int(raw["fp_iou50"]), "bilateral_fp": int(bilateral["fp_iou50"]),
                         "raw_fn": int(raw["fn_iou50"]), "bilateral_fn": int(bilateral["fn_iou50"])})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "gaussian_denoiser_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, detector in zip(axes, SOURCES):
        subset = [r for r in rows if r["detector"] == detector]
        levels = [r["level"] for r in subset]
        ax.plot(levels, [r["raw_mAP50"] for r in subset], marker="o", label="Raw Gaussian")
        ax.plot(levels, [r["bilateral_mAP50"] for r in subset], marker="s", label="Bilateral")
        nlm = [r for r in subset if r["nlm_mAP50"] != ""]
        if nlm:
            ax.scatter([r["level"] for r in nlm], [r["nlm_mAP50"] for r in nlm],
                       marker="D", s=70, label="NLM (full-val result available)")
        ax.set_title(detector)
        ax.set_xlabel("Gaussian noise level (sigma / 255)")
        ax.set_ylabel("mAP@0.5")
        ax.set_xticks(levels)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Gaussian denoiser comparison — shared AP evaluator, 5,000 images")
    fig.savefig(OUT / "gaussian_denoiser_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Saved {OUT / 'gaussian_denoiser_comparison.csv'}")
    for row in rows:
        print(f"{row['detector']} level {row['level']}: raw={row['raw_mAP50']:.4f}, bilateral={row['bilateral_mAP50']:.4f} ({row['bilateral_delta_vs_raw']:+.4f}), NLM={row['nlm_mAP50'] if row['nlm_mAP50'] != '' else 'n/a'}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compare saved full-COCO clean metrics for YOLOv8n and YOLOv8s."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUTS = {
    "YOLOv8n": ROOT / "runs/full_val2017_raw_seed42/metrics.csv",
    "YOLOv8s": ROOT / "runs/yolov8s_fullval_clean/metrics.csv",
}
OUT = ROOT / "runs/yolo_model_size_comparison"


def clean_row(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["condition"] == "clean"]
    if len(rows) != 1:
        raise ValueError(f"Expected one clean row in {path}")
    return rows[0]


def main():
    latency_path = ROOT / "runs/yolo_model_size_comparison/single_image_latency.csv"
    with latency_path.open(newline="", encoding="utf-8-sig") as handle:
        latency_by_model = {row["model"]: row for row in csv.DictReader(handle)}
    rows = []
    for model_name, path in INPUTS.items():
        row = clean_row(path)
        rows.append({"model": model_name, "images": int(row["images"]), "imgsz": 640,
                     "mAP50": float(row["mAP50"]), "mAP50_95": float(row["mAP50_95"]),
                     "ultralytics_precision": float(row["precision_ultralytics"]),
                     "ultralytics_recall": float(row["recall_ultralytics"]),
                     "tp_iou50_conf025": int(row["tp_iou50"]),
                     "fp_iou50_conf025": int(row["fp_iou50"]),
                     "fn_iou50_conf025": int(row["fn_iou50"]),
                     "latency_mean_ms": float(latency_by_model[model_name]["latency_mean_ms"]),
                     "latency_median_ms": float(latency_by_model[model_name]["latency_median_ms"]),
                     "latency_p95_ms": float(latency_by_model[model_name]["latency_p95_ms"]),
                     "peak_cuda_allocated_gb": float(latency_by_model[model_name]["peak_cuda_allocated_gb"])})
    base, larger = rows
    out_rows = []
    for row in rows:
        out_rows.append({**row, "delta_mAP50_vs_yolov8n": row["mAP50"] - base["mAP50"],
                         "delta_mAP50_95_vs_yolov8n": row["mAP50_95"] - base["mAP50_95"],
                         "relative_mAP50_gain_pct": 100 * (row["mAP50"] / base["mAP50"] - 1)})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "yolov8n_vs_yolov8s_fullval.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    labels = [row["model"] for row in out_rows]
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    charts = ((axes[0, 0], "mAP50", "mAP@0.5", "mAP", 0.0),
              (axes[0, 1], "mAP50_95", "mAP@0.5:0.95", "mAP", 0.0),
              (axes[1, 0], "latency_mean_ms", "Single-image latency", "Milliseconds", 0.0),
              (axes[1, 1], "peak_cuda_allocated_gb", "Peak CUDA allocation", "GB", 0.0))
    for ax, metric, title, ylabel, _ in charts:
        values = [row[metric] for row in out_rows]
        bars = ax.bar(x, values, width, color=["#4776A8", "#22A58A"])
        ax.set_xticks(x, labels)
        ax.set_ylim(0, max(values) * 1.28)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            label = f"{value:.4f}\n{value*100:.2f}%" if metric.startswith("mAP") else f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, value + max(values) * 0.025,
                    label, ha="center", va="bottom", fontsize=9)
    chart = OUT / "yolov8n_vs_yolov8s_fullval.png"
    fig.savefig(chart, dpi=180)
    plt.close(fig)

    print(f"Saved comparison CSV: {path}")
    print(f"Saved chart: {chart}")
    print(f"mAP50 gain: {larger['mAP50']-base['mAP50']:+.4f} "
          f"({100*(larger['mAP50']/base['mAP50']-1):+.2f}% relative)")
    print(f"mAP50-95 gain: {larger['mAP50_95']-base['mAP50_95']:+.4f}")
    print(f"FP delta @conf.25: {larger['fp_iou50_conf025']-base['fp_iou50_conf025']:+d}; "
          f"FN delta: {larger['fn_iou50_conf025']-base['fn_iou50_conf025']:+d}")


if __name__ == "__main__":
    main()

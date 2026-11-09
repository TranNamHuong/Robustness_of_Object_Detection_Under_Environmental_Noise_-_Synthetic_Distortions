#!/usr/bin/env python
"""Evaluate YOLO size-stratified recall across three disjoint COCO subsets."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from analyze_size_robustness import CONDITIONS, SIZE_BINS, count_size_recall, load_image_targets

ROOT = Path(__file__).resolve().parent
SUBSETS = {
    "subset1": ROOT / "runs/noise_sweep_seed42/datasets",
    "subset2": ROOT / "runs/image_subset_set2/datasets",
    "subset3": ROOT / "runs/image_subset_set3/datasets",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/size_stratified_subsets")
    args = parser.parse_args()

    device = "0" if args.device == "0" and torch.cuda.is_available() else args.device
    model = YOLO(str(args.model))
    labels = {name: label for label, name in CONDITIONS}
    rows = []
    reference_names = None

    for subset, root in SUBSETS.items():
        subset_names = None
        for condition_label, condition_name in CONDITIONS:
            image_dir = root / condition_name / "images/val"
            files = sorted(image_dir.glob("*.jpg"))
            if not files:
                raise FileNotFoundError(f"No COCO images in {image_dir}")
            names = [path.name for path in files]
            if subset_names is None:
                subset_names = names
            elif names != subset_names:
                raise ValueError(f"Image sets differ in {subset}/{condition_name}")

            images, targets = zip(*(load_image_targets(path) for path in files))
            predictions = []
            for start in range(0, len(files), args.batch):
                batch = images[start:start + args.batch]
                results = model.predict(list(batch), imgsz=args.imgsz, device=device,
                                        conf=args.conf, iou=0.7, max_det=300, verbose=False)
                for result in results:
                    boxes = result.boxes
                    predictions.append({
                        "boxes": boxes.xyxy.detach().cpu().numpy().astype(np.float32),
                        "labels": boxes.cls.detach().cpu().numpy().astype(np.int64),
                        "scores": boxes.conf.detach().cpu().numpy().astype(np.float32),
                    })
                seen = min(start + len(batch), len(files))
                if seen % 100 == 0 or seen == len(files):
                    print(f"{subset} {condition_name}: {seen}/{len(files)}", flush=True)

            stats = count_size_recall(predictions, targets, args.iou)
            for size, _, _ in SIZE_BINS:
                item = stats[size]
                rows.append({"subset": subset, "condition": condition_label, "size": size,
                             "ground_truth": item["ground_truth"], "tp": item["tp"],
                             "fn": item["fn"], "recall": item["recall"],
                             "confidence": args.conf, "match_iou": args.iou,
                             "images": len(files)})
        if reference_names is None:
            reference_names = subset_names

    args.out.mkdir(parents=True, exist_ok=True)
    by_subset = args.out / "size_stratified_recall_by_subset.csv"
    with by_subset.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for condition_label, _ in CONDITIONS:
        for size, _, _ in SIZE_BINS:
            selected = [row for row in rows if row["condition"] == condition_label and row["size"] == size]
            values = np.asarray([row["recall"] for row in selected], dtype=np.float64)
            summary.append({"condition": condition_label, "size": size,
                            "recall_mean": float(np.mean(values)),
                            "recall_sd_between_subsets": float(np.std(values, ddof=1)),
                            "n_subsets": len(values),
                            "ground_truth_total": sum(row["ground_truth"] for row in selected),
                            "confidence": args.conf, "match_iou": args.iou,
                            "images_per_subset": selected[0]["images"]})
    summary_path = args.out / "size_stratified_recall_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    sizes = [entry[0] for entry in SIZE_BINS]
    condition_names = [label for label, _ in CONDITIONS]
    values = np.asarray([[next(row["recall_mean"] for row in summary
                               if row["condition"] == condition and row["size"] == size)
                          for size in sizes] for condition in condition_names])
    deviations = np.asarray([[next(row["recall_sd_between_subsets"] for row in summary
                                   if row["condition"] == condition and row["size"] == size)
                              for size in sizes] for condition in condition_names])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(sizes)), [name.capitalize() for name in sizes])
    ax.set_yticks(np.arange(len(condition_names)), condition_names)
    ax.set_title("YOLOv8n size-stratified recall across disjoint COCO subsets")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(col_idx, row_idx,
                    f"{values[row_idx, col_idx]:.3f}\n±{deviations[row_idx, col_idx]:.3f}",
                    ha="center", va="center",
                    color="black" if values[row_idx, col_idx] > 0.65 else "white", fontsize=8)
    fig.colorbar(image, ax=ax, pad=0.03, label="Recall mean (color); SD (text)")
    fig.tight_layout()
    chart = args.out / "size_stratified_recall_by_subset.png"
    fig.savefig(chart, dpi=180)
    plt.close(fig)

    print(f"Saved per-subset metrics: {by_subset}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved chart: {chart}")
    for row in summary:
        print(f"{row['condition']:20s} {row['size']:6s} "
              f"recall={row['recall_mean']:.4f} ± {row['recall_sd_between_subsets']:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compare YOLO recall by COCO object-size bin on existing COCO conditions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SEED_ROOTS = {
    42: ROOT / "runs/noise_sweep_seed42/datasets",
    43: ROOT / "runs/repeat_seed43/datasets",
    44: ROOT / "runs/repeat_seed44/datasets",
}
CONDITIONS = [
    ("Clean", "clean"),
    ("Gaussian 0.06 raw", "gaussian_0.06"),
    ("Gaussian 0.06 NLM", "gaussian_0.06_recovered"),
    ("S&P 0.06 raw", "salt_pepper_0.06"),
    ("S&P 0.06 median", "salt_pepper_0.06_recovered"),
    ("Fog 0.50 raw", "fog_0.5"),
    ("Fog 0.50 CLAHE", "fog_0.5_recovered"),
    ("Low-light 0.25 raw", "low_light_0.25"),
    ("Low-light 0.25 CLAHE", "low_light_0.25_recovered"),
]
SIZE_BINS = [("small", 0.0, 32.0**2), ("medium", 32.0**2, 96.0**2),
             ("large", 96.0**2, float("inf"))]


def load_image_targets(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    label_path = (image_path.parents[2] / "labels/val" / image_path.name).with_suffix(".txt")
    classes, boxes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, xc, yc, bw, bh = map(float, values[:5])
        classes.append(int(cls))
        boxes.append([(xc-bw/2)*width, (yc-bh/2)*height,
                      (xc+bw/2)*width, (yc+bh/2)*height])
    return image, (np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                   np.asarray(classes, dtype=np.int64))


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if not len(gt) or not len(pred):
        return np.zeros((len(gt), len(pred)), dtype=np.float32)
    lt = np.maximum(gt[:, None, :2], pred[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    area_gt = np.maximum(gt[:, 2]-gt[:, 0], 0) * np.maximum(gt[:, 3]-gt[:, 1], 0)
    area_pred = np.maximum(pred[:, 2]-pred[:, 0], 0) * np.maximum(pred[:, 3]-pred[:, 1], 0)
    return inter / np.maximum(area_gt[:, None] + area_pred[None, :] - inter, 1e-9)


def count_size_recall(predictions, targets, iou_threshold):
    stats = {name: {"ground_truth": 0, "tp": 0} for name, _, _ in SIZE_BINS}
    for pred, (gt_boxes, gt_classes) in zip(predictions, targets):
        area = np.maximum(gt_boxes[:, 2]-gt_boxes[:, 0], 0) * np.maximum(
            gt_boxes[:, 3]-gt_boxes[:, 1], 0)
        for name, lower, upper in SIZE_BINS:
            stats[name]["ground_truth"] += int(np.sum((area >= lower) & (area < upper)))
        if not len(gt_boxes) or not len(pred["boxes"]):
            continue
        overlaps = iou_matrix(gt_boxes, pred["boxes"])
        candidates = []
        for gt_idx in range(len(gt_boxes)):
            for pred_idx in range(len(pred["boxes"])):
                if (gt_classes[gt_idx] == pred["labels"][pred_idx]
                        and overlaps[gt_idx, pred_idx] >= iou_threshold):
                    candidates.append((float(overlaps[gt_idx, pred_idx]),
                                       float(pred["scores"][pred_idx]), gt_idx, pred_idx))
        candidates.sort(reverse=True)
        matched_gt, matched_pred = set(), set()
        for _, _, gt_idx, pred_idx in candidates:
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
        for gt_idx in matched_gt:
            for name, lower, upper in SIZE_BINS:
                if lower <= area[gt_idx] < upper:
                    stats[name]["tp"] += 1
                    break
    for values in stats.values():
        values["fn"] = values["ground_truth"] - values["tp"]
        values["recall"] = values["tp"] / values["ground_truth"] if values["ground_truth"] else float("nan")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5,
                        help="Ground-truth match IoU for size-stratified recall")
    parser.add_argument("--seeds", nargs="+", type=int, choices=sorted(SEED_ROOTS), default=[42],
                        help="Noise seeds available for this fixed COCO subset")
    parser.add_argument("--conditions", nargs="+", choices=[name for _, name in CONDITIONS],
                        default=[name for _, name in CONDITIONS])
    parser.add_argument("--out", type=Path, default=ROOT / "runs/size_stratified_analysis")
    args = parser.parse_args()

    device = "0" if args.device == "0" and torch.cuda.is_available() else args.device
    model = YOLO(str(args.model))
    all_rows = []
    label_by_name = {name: label for label, name in CONDITIONS}
    for seed in args.seeds:
        expected_names = None
        for condition_name in args.conditions:
            dataset_root = SEED_ROOTS[seed] / condition_name
            image_dir = dataset_root / "images/val"
            files = sorted(image_dir.glob("*.jpg"))
            if not files:
                raise FileNotFoundError(f"No COCO images in {image_dir}")
            current_names = [path.name for path in files]
            if expected_names is None:
                expected_names = current_names
            elif current_names != expected_names:
                raise ValueError(f"Image sets differ at seed {seed}, condition {condition_name}")

            images, targets = zip(*(load_image_targets(path) for path in files))
            raw_results = []
            for start in range(0, len(files), args.batch):
                batch = images[start:start + args.batch]
                results = model.predict(list(batch), imgsz=args.imgsz, device=device,
                                        conf=args.conf, iou=0.7, max_det=300, verbose=False)
                raw_results.extend(results)
                seen = min(start + len(batch), len(files))
                if seen % 100 == 0 or seen == len(files):
                    print(f"seed{seed} {condition_name}: {seen}/{len(files)}", flush=True)

            predictions = []
            for result in raw_results:
                boxes = result.boxes
                predictions.append({
                    "boxes": boxes.xyxy.detach().cpu().numpy().astype(np.float32),
                    "labels": boxes.cls.detach().cpu().numpy().astype(np.int64),
                    "scores": boxes.conf.detach().cpu().numpy().astype(np.float32),
                })
            size_stats = count_size_recall(predictions, targets, args.iou)
            for size_name, _, _ in SIZE_BINS:
                count = size_stats[size_name]["ground_truth"]
                tp = size_stats[size_name]["tp"]
                fn = size_stats[size_name]["fn"]
                recall = size_stats[size_name]["recall"]
                all_rows.append({"seed": seed, "condition": label_by_name[condition_name],
                                 "size": size_name, "ground_truth": count, "tp": tp,
                                 "fn": fn, "recall": recall, "confidence": args.conf,
                                 "match_iou": args.iou, "images": len(files)})

    args.out.mkdir(parents=True, exist_ok=True)
    by_seed_path = args.out / "size_stratified_recall_by_seed.csv"
    with by_seed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    for condition_name in [label_by_name[name] for name in args.conditions]:
        for size_name, _, _ in SIZE_BINS:
            rows = [row for row in all_rows if row["condition"] == condition_name and row["size"] == size_name]
            recalls = np.asarray([row["recall"] for row in rows], dtype=np.float64)
            summary_rows.append({
                "condition": condition_name, "size": size_name,
                "ground_truth_per_seed": rows[0]["ground_truth"],
                "recall_mean": float(np.mean(recalls)),
                "recall_sd": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
                "n_seeds": len(recalls), "confidence": args.conf,
                "match_iou": args.iou, "images_per_seed": rows[0]["images"],
            })
    metrics_path = args.out / "size_stratified_recall.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    sizes = [entry[0] for entry in SIZE_BINS]
    condition_labels = [label_by_name[name] for name in args.conditions]
    values = np.asarray([[next(row["recall_mean"] for row in summary_rows
                               if row["condition"] == condition and row["size"] == size)
                          for size in sizes] for condition in condition_labels])
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(sizes)), [name.capitalize() for name in sizes])
    ax.set_yticks(np.arange(len(condition_labels)), condition_labels)
    ax.set_title(f"YOLOv8n recall by COCO object size\nconf={args.conf:g}, match IoU={args.iou:g}")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center",
                    color="black" if value > 0.65 else "white", fontsize=9, weight="bold")
    colorbar = fig.colorbar(image, ax=ax, pad=0.03)
    colorbar.set_label("Recall")
    fig.tight_layout()
    figure_path = args.out / "size_stratified_recall.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"Saved per-seed metrics: {by_seed_path}")
    print(f"Saved mean/SD summary: {metrics_path}")
    print(f"Saved chart: {figure_path}")
    for row in summary_rows:
        print(f"{row['condition']:20s} {row['size']:6s} GT/seed={row['ground_truth_per_seed']:4d} "
              f"recall={row['recall_mean']:.4f} ± {row['recall_sd']:.4f} (n={row['n_seeds']})")


if __name__ == "__main__":
    main()

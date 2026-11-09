#!/usr/bin/env python
"""Measure COCO FP/FN sensitivity to confidence threshold on existing subsets."""
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

from analyze_size_robustness import CONDITIONS, load_image_targets

ROOT = Path(__file__).resolve().parent
SUBSETS = {
    "subset1": ROOT / "runs/noise_sweep_seed42/datasets",
    "subset2": ROOT / "runs/image_subset_set2/datasets",
    "subset3": ROOT / "runs/image_subset_set3/datasets",
}
THRESHOLDS = (0.05, 0.25, 0.50)
CONFUSION_CONDITIONS = [
    ("clean", "Clean"),
    ("gaussian_0.06", "Gaussian 0.06 raw"),
    ("gaussian_0.06_recovered", "Gaussian 0.06 NLM"),
    ("salt_pepper_0.06", "S&P 0.06 raw"),
    ("salt_pepper_0.06_recovered", "S&P 0.06 median"),
    ("fog_0.5", "Fog 0.50 raw"),
    ("fog_0.5_recovered", "Fog 0.50 CLAHE"),
    ("low_light_0.25", "Low-light 0.25 raw"),
    ("low_light_0.25_recovered", "Low-light 0.25 CLAHE"),
]


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


def counts_at_threshold(predictions, targets, nclasses: int, threshold: float, match_iou: float):
    matrix = np.zeros((nclasses + 1, nclasses + 1), dtype=np.int64)
    for pred, (gt_boxes, gt_classes) in zip(predictions, targets):
        keep = pred["scores"] >= threshold
        pred_boxes = pred["boxes"][keep]
        pred_classes = pred["labels"][keep]
        pred_scores = pred["scores"][keep]
        if len(gt_boxes) and len(pred_boxes):
            overlaps = iou_matrix(gt_boxes, pred_boxes)
            gi, pi = np.where(overlaps >= match_iou)
            candidates = sorted(zip(overlaps[gi, pi].tolist(), pred_scores[pi].tolist(),
                                    gi.tolist(), pi.tolist()), reverse=True)
        else:
            candidates = []
        matched_gt, matched_pred = set(), set()
        for _, _, gt_idx, pred_idx in candidates:
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            gt_cls, pred_cls = int(gt_classes[gt_idx]), int(pred_classes[pred_idx])
            if 0 <= gt_cls < nclasses and 0 <= pred_cls < nclasses:
                matrix[gt_cls, pred_cls] += 1
        for gt_idx, gt_cls in enumerate(gt_classes):
            if gt_idx not in matched_gt and 0 <= int(gt_cls) < nclasses:
                matrix[int(gt_cls), nclasses] += 1
        for pred_idx, pred_cls in enumerate(pred_classes):
            if pred_idx not in matched_pred and 0 <= int(pred_cls) < nclasses:
                matrix[nclasses, int(pred_cls)] += 1

    diagonal = int(np.trace(matrix[:nclasses, :nclasses]))
    misclassified = int(matrix[:nclasses, :nclasses].sum()) - diagonal
    fp = int(matrix[nclasses, :nclasses].sum()) + misclassified
    fn = int(matrix[:nclasses, nclasses].sum()) + misclassified
    gt_count = int(matrix[:nclasses, :].sum())
    pred_count = int(matrix[:, :nclasses].sum())
    return {"tp": diagonal, "fp": fp, "fn": fn, "ground_truth": gt_count,
            "predictions": pred_count,
            "precision": diagonal / (diagonal + fp) if diagonal + fp else 0.0,
            "recall": diagonal / (diagonal + fn) if diagonal + fn else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/confidence_threshold_analysis")
    args = parser.parse_args()

    device = "0" if args.device == "0" and torch.cuda.is_available() else args.device
    model = YOLO(str(args.model))
    import yaml
    data = yaml.safe_load((ROOT / "datasets/coco/full_val2017/data.yaml").read_text(encoding="utf-8"))
    raw_names = data["names"]
    nclasses = len(raw_names)
    rows = []

    for subset, root in SUBSETS.items():
        expected_names = None
        for condition_key, condition_label in CONFUSION_CONDITIONS:
            image_dir = root / condition_key / "images/val"
            files = sorted(image_dir.glob("*.jpg"))
            if not files:
                raise FileNotFoundError(f"No images at {image_dir}")
            names = [path.name for path in files]
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise ValueError(f"Image sets differ in {subset}/{condition_key}")
            images, targets = zip(*(load_image_targets(path) for path in files))
            predictions = []
            for start in range(0, len(files), args.batch):
                batch = images[start:start + args.batch]
                results = model.predict(list(batch), imgsz=args.imgsz, device=device,
                                        conf=0.001, iou=0.7, max_det=300, verbose=False)
                for result in results:
                    boxes = result.boxes
                    predictions.append({
                        "boxes": boxes.xyxy.detach().cpu().numpy().astype(np.float32),
                        "labels": boxes.cls.detach().cpu().numpy().astype(np.int64),
                        "scores": boxes.conf.detach().cpu().numpy().astype(np.float32),
                    })
                seen = min(start + len(batch), len(files))
                if seen % 100 == 0 or seen == len(files):
                    print(f"{subset} {condition_key}: {seen}/{len(files)}", flush=True)

            for threshold in THRESHOLDS:
                metrics = counts_at_threshold(predictions, targets, nclasses, threshold, args.match_iou)
                rows.append({"subset": subset, "condition": condition_label,
                             "confidence": threshold, "match_iou": args.match_iou,
                             "images": len(files), **metrics})

    args.out.mkdir(parents=True, exist_ok=True)
    by_subset_path = args.out / "confidence_counts_by_subset.csv"
    with by_subset_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for condition_key, condition_label in CONFUSION_CONDITIONS:
        for threshold in THRESHOLDS:
            selected = [row for row in rows if row["condition"] == condition_label
                        and row["confidence"] == threshold]
            row = {"condition": condition_label, "confidence": threshold,
                   "n_subsets": len(selected), "match_iou": args.match_iou,
                   "images_per_subset": selected[0]["images"]}
            for metric in ("tp", "fp", "fn", "ground_truth", "predictions", "precision", "recall"):
                values = np.asarray([item[metric] for item in selected], dtype=float)
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sd_between_subsets"] = float(np.std(values, ddof=1))
            row["fp_per_100_images_mean"] = row["fp_mean"] / row["images_per_subset"] * 100
            row["fn_per_100_images_mean"] = row["fn_mean"] / row["images_per_subset"] * 100
            summary.append(row)
    summary_path = args.out / "confidence_threshold_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    plot_families = [
        ("Gaussian", "Gaussian 0.06 raw", "Gaussian 0.06 NLM"),
        ("S&P", "S&P 0.06 raw", "S&P 0.06 median"),
        ("Fog", "Fog 0.50 raw", "Fog 0.50 CLAHE"),
        ("Low-light", "Low-light 0.25 raw", "Low-light 0.25 CLAHE"),
    ]
    colors = {"Gaussian": "tab:blue", "S&P": "tab:red", "Fog": "tab:green", "Low-light": "tab:purple"}
    for family, raw_label, recovered_label in plot_families:
        color = colors[family]
        for label, style in ((raw_label, "-"), (recovered_label, "--")):
            selected = [row for row in summary if row["condition"] == label]
            selected.sort(key=lambda row: row["confidence"])
            axes[0].plot([row["confidence"] for row in selected],
                         [row["fp_per_100_images_mean"] for row in selected],
                         linestyle=style, marker="o", color=color,
                         label=f"{family} {'raw' if style == '-' else 'recovered'}")
            axes[1].plot([row["confidence"] for row in selected],
                         [row["fn_per_100_images_mean"] for row in selected],
                         linestyle=style, marker="o", color=color,
                         label=f"{family} {'raw' if style == '-' else 'recovered'}")
    for ax, title, ylabel in ((axes[0], "False positives", "FP per 100 images"),
                              (axes[1], "Missed / incorrect objects", "FN per 100 images")):
        ax.set_title(title)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(THRESHOLDS)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    chart_path = args.out / "fp_fn_by_confidence_threshold.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    print(f"Saved per-subset counts: {by_subset_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved chart: {chart_path}")
    clean = {(row["confidence"]): row for row in summary if row["condition"] == "Clean"}
    for row in summary:
        if row["condition"] in {"S&P 0.06 raw", "S&P 0.06 median"}:
            print(f"{row['condition']:18s} conf={row['confidence']:.2f} "
                  f"FP/100={row['fp_per_100_images_mean']:.1f} "
                  f"FN/100={row['fn_per_100_images_mean']:.1f} "
                  f"clean FP/100={clean[row['confidence']]['fp_per_100_images_mean']:.1f}")


if __name__ == "__main__":
    main()

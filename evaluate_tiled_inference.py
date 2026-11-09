#!/usr/bin/env python
"""Compare full-frame and tiled YOLO inference on COCO val2017."""
from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from detection_inference import predict_image
from yolo_custom_eval import ap_for_class, confusion_counts, iou_matrix, read_targets


ROOT = Path(__file__).resolve().parent
AREA_RANGES = {"small": (0.0, 32.0 ** 2), "medium": (32.0 ** 2, 96.0 ** 2),
               "large": (96.0 ** 2, float("inf"))}


def recall_by_size(predictions, targets, conf: float = 0.25, iou_threshold: float = 0.5):
    totals = CounterLike()
    matched_totals = CounterLike()
    for prediction, (gt_boxes, gt_classes) in zip(predictions, targets):
        areas = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 0) * np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1], 0)
        for area in areas:
            group = area_group(float(area))
            totals[group] += 1
        keep = prediction["scores"] >= conf
        boxes = prediction["boxes"][keep]
        labels = prediction["labels"][keep]
        scores = prediction["scores"][keep]
        overlaps = iou_matrix(gt_boxes, boxes)
        candidates = [
            (float(overlaps[g, p]), float(scores[p]), g, p)
            for g in range(len(gt_boxes)) for p in range(len(boxes))
            if gt_classes[g] == labels[p] and overlaps[g, p] >= iou_threshold
        ]
        candidates.sort(reverse=True)
        matched_gt, matched_pred = set(), set()
        for _, _, gt_index, pred_index in candidates:
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            matched_totals[area_group(float(areas[gt_index]))] += 1
    return {group: matched_totals[group] / totals[group] if totals[group] else float("nan")
            for group in AREA_RANGES}


class CounterLike(defaultdict):
    def __init__(self):
        super().__init__(int)


def area_group(area: float) -> str:
    if area < AREA_RANGES["small"][1]:
        return "small"
    if area < AREA_RANGES["large"][0]:
        return "medium"
    return "large"


def evaluate(model, image_paths, names, imgsz: int, device: str, confidence: float,
             tiled: bool) -> dict:
    predictions, targets = [], []
    started = time.time()
    for index, image_path in enumerate(image_paths, start=1):
        image, target = read_targets(image_path)
        result = predict_image(model, image, imgsz, confidence, device, tiled=tiled)
        boxes = result.boxes
        predictions.append({
            "boxes": boxes.xyxy.detach().cpu().numpy().astype(np.float32),
            "labels": boxes.cls.detach().cpu().numpy().astype(np.int64),
            "scores": boxes.conf.detach().cpu().numpy().astype(np.float32),
        })
        targets.append(target)
        if index % 100 == 0 or index == len(image_paths):
            mode = "tiled" if tiled else "full-frame"
            print(f"{mode}: {index}/{len(image_paths)} images", flush=True)

    thresholds = [0.50 + 0.05 * i for i in range(10)]
    aps = np.asarray([ap_for_class(predictions, targets, cls, thresholds)
                      for cls in range(len(names))])
    matrix, tp, fp, fn = confusion_counts(predictions, targets, len(names), conf=0.25, iou_thr=0.5)
    size_recall = recall_by_size(predictions, targets, conf=0.25, iou_threshold=0.5)
    return {
        "mode": "tiled_640_overlap25" if tiled else "full_frame",
        "images": len(image_paths),
        "imgsz": imgsz,
        "mAP50": float(np.nanmean(aps[:, 0])),
        "mAP50_95": float(np.nanmean(aps)),
        "tp_iou50_conf025": tp,
        "fp_iou50_conf025": fp,
        "fn_iou50_conf025": fn,
        "recall_small_conf025_iou50": size_recall["small"],
        "recall_medium_conf025_iou50": size_recall["medium"],
        "recall_large_conf025_iou50": size_recall["large"],
        "elapsed_sec": round(time.time() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "yolov8s.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/coco/full_val2017/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional fixed-seed image sample; default evaluates all 5,000 images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Low threshold used for AP; confusion counts use 0.25")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/tiled_inference_eval")
    args = parser.parse_args()

    yaml_path = args.data.resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()
    image_dir = dataset_root / "images/val"
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if args.limit > 0:
        if args.limit > len(image_paths):
            parser.error(f"--limit exceeds image count ({len(image_paths)})")
        image_paths = sorted(random.Random(args.seed).sample(image_paths, args.limit))
    if not image_paths:
        raise FileNotFoundError(f"No validation images at {image_dir}")
    raw_names = config["names"]
    names = ([str(raw_names.get(i, raw_names.get(str(i), i))) for i in range(len(raw_names))]
             if isinstance(raw_names, dict) else [str(name) for name in raw_names])
    device = "0" if args.device == "0" and torch.cuda.is_available() else args.device
    model_path = args.model.resolve()
    model = YOLO(str(model_path))
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tiled in (False, True):
        result = evaluate(model, image_paths, names, args.imgsz, device, args.conf, tiled)
        result["model"] = model_path.name
        result["device"] = device
        result["conf_for_ap"] = args.conf
        result["conf_for_counts"] = 0.25
        rows.append(result)
        output = args.out / "summary.csv"
        with output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{result['mode']}: mAP50={result['mAP50']:.4f}; "
              f"small recall={result['recall_small_conf025_iou50']:.4f}; "
              f"FP={result['fp_iou50_conf025']}; elapsed={result['elapsed_sec']}s", flush=True)
    print(f"Saved comparison to {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()

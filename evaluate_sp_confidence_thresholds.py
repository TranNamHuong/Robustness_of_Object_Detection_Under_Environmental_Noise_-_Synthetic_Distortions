#!/usr/bin/env python
"""Compare confidence thresholds for baseline/recovery and noise-aware YOLOv8n."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
THRESHOLDS = (0.05, 0.25, 0.50)
CONDITIONS = {
    "baseline_raw": ROOT / "runs/full_val2017_raw_seed42/datasets/salt_pepper_0.06",
    "baseline_median": ROOT / "runs/full_val2017_recovered_seed42/datasets/salt_pepper_0.06_recovered",
    "noiseaware_raw": ROOT / "runs/full_val2017_raw_seed42/datasets/salt_pepper_0.06",
}


def read_targets(label_path: Path, height: int, width: int):
    classes, boxes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, xc, yc, bw, bh = map(float, values[:5])
        classes.append(int(cls))
        boxes.append([(xc - bw / 2) * width, (yc - bh / 2) * height,
                      (xc + bw / 2) * width, (yc + bh / 2) * height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=np.int64)


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if not len(gt) or not len(pred):
        return np.zeros((len(gt), len(pred)), dtype=np.float32)
    lt = np.maximum(gt[:, None, :2], pred[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    area_gt = np.maximum(gt[:, 2] - gt[:, 0], 0) * np.maximum(gt[:, 3] - gt[:, 1], 0)
    area_pred = np.maximum(pred[:, 2] - pred[:, 0], 0) * np.maximum(pred[:, 3] - pred[:, 1], 0)
    return inter / np.maximum(area_gt[:, None] + area_pred[None, :] - inter, 1e-9)


def counts(gt_boxes, gt_classes, pred_boxes, pred_classes, scores,
           threshold: float, iou_threshold: float = 0.5):
    keep = scores >= threshold
    pred_boxes, pred_classes, scores = pred_boxes[keep], pred_classes[keep], scores[keep]
    overlaps = iou_matrix(gt_boxes, pred_boxes)
    candidates = [(float(overlaps[g, p]), float(scores[p]), g, p)
                  for g in range(len(gt_boxes)) for p in range(len(pred_boxes))
                  if overlaps[g, p] >= iou_threshold]
    candidates.sort(reverse=True)
    matched_gt, matched_pred = set(), set()
    tp = fp = fn = 0
    for _, _, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        if gt_classes[gi] == pred_classes[pi]:
            tp += 1
        else:
            fp += 1
            fn += 1
    fn += len(gt_boxes) - len(matched_gt)
    fp += len(pred_boxes) - len(matched_pred)
    return tp, fp, fn, len(gt_boxes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--noiseaware", type=Path, default=ROOT / "runs/finetune_coco80/yolov8n_noiseaware_5k_pair_lr0001/weights/best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/noiseaware_yolov8n_eval_seed42")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    models = {"baseline": args.baseline.resolve(), "noiseaware": args.noiseaware.resolve()}
    rows = []
    for condition, dataset_dir in CONDITIONS.items():
        image_dir = dataset_dir / "images/val"
        label_dir = dataset_dir / "labels/val"
        images = sorted(image_dir.glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No validation images in {image_dir}")
        if condition in {"baseline_raw", "baseline_median"}:
            applicable_models = ["baseline"]
        else:
            applicable_models = ["noiseaware"]
        for model_name in applicable_models:
            model_path = models[model_name]
            model = YOLO(str(model_path))
            totals = {threshold: np.zeros(4, dtype=np.int64) for threshold in THRESHOLDS}
            print(f"{model_name}/{condition}: {len(images)} images", flush=True)
            for start in range(0, len(images), args.batch):
                paths = images[start:start + args.batch]
                batch_images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
                if any(image is None for image in batch_images):
                    raise RuntimeError(f"Unreadable image in batch starting at {start}")
                results = model.predict(source=batch_images, imgsz=args.imgsz,
                                        batch=args.batch, device=args.device, conf=0.001,
                                        iou=0.7, max_det=300, verbose=False)
                for path, image, result in zip(paths, batch_images, results):
                    h, w = image.shape[:2]
                    gt_boxes, gt_classes = read_targets(label_dir / path.with_suffix(".txt").name, h, w)
                    boxes = result.boxes
                    pred_boxes = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) else np.empty((0, 4))
                    pred_classes = boxes.cls.cpu().numpy().astype(np.int64) if boxes is not None and len(boxes) else np.empty(0, dtype=np.int64)
                    scores = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else np.empty(0)
                    for threshold in THRESHOLDS:
                        totals[threshold] += np.asarray(counts(gt_boxes, gt_classes, pred_boxes,
                                                               pred_classes, scores, threshold))
                seen = min(start + len(paths), len(images))
                if seen % 1000 == 0 or seen == len(images):
                    print(f"  {seen}/{len(images)}", flush=True)
            del model
            for threshold, (tp, fp, fn, gt) in totals.items():
                rows.append({"model": model_name, "condition": condition,
                             "confidence": threshold, "images": len(images),
                             "tp": int(tp), "fp": int(fp), "fn": int(fn),
                             "ground_truth": int(gt),
                             "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                             "recall": float(tp / gt) if gt else 0.0})

    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "salt_pepper_confidence_thresholds.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved threshold comparison: {output}", flush=True)


if __name__ == "__main__":
    main()

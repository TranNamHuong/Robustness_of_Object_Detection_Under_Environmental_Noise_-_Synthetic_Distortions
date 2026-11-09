#!/usr/bin/env python
"""Evaluate a YOLO checkpoint with the same 101-point AP implementation as SSDLite."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "runs/full_val2017_raw_seed42/datasets"
RECOVERY = [ROOT / "runs/full_val2017_recovered_seed42/datasets",
            ROOT / "runs/full_val2017_gaussian_recovered_seed42/datasets",
            ROOT / "runs/full_val2017_gaussian_bilateral_seed42/datasets"]
CONDITIONS = ["clean", "gaussian_0.06", "gaussian_0.06_recovered",
              "salt_pepper_0.06", "salt_pepper_0.06_recovered",
              "fog_0.5", "fog_0.5_recovered",
              "low_light_0.25", "low_light_0.25_recovered"]


def locate(condition: str) -> Path:
    if "_seed" in condition and condition.rsplit("_seed", 1)[1].isdigit():
        base, seed = condition.rsplit("_seed", 1)
        candidate = ROOT / f"runs/full_val2017_seed{seed}_selected/datasets" / base
        if (candidate / "data.yaml").exists():
            return candidate
    for root in [RAW, *RECOVERY]:
        path = root / condition
        if (path / "data.yaml").exists():
            return path
    raise FileNotFoundError(condition)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def ap_for_class(predictions, targets, cls: int, thresholds: list[float]) -> list[float]:
    gt_by_image = {}
    total_gt = 0
    detections = []
    for image_id, (boxes, labels) in enumerate(targets):
        gt = boxes[labels == cls]
        gt_by_image[image_id] = gt
        total_gt += len(gt)
        pred = predictions[image_id]
        keep = pred["labels"] == cls
        detections.extend((float(score), image_id, box) for box, score in zip(pred["boxes"][keep], pred["scores"][keep]))
    if total_gt == 0:
        return [float("nan")] * len(thresholds)
    detections.sort(key=lambda item: item[0], reverse=True)
    cached = []
    for _, image_id, box in detections:
        gt = gt_by_image[image_id]
        if len(gt):
            overlap = iou_matrix(box.reshape(1, 4), gt)[0]
            cached.append((image_id, [(int(j), float(overlap[j])) for j in np.argsort(overlap)[::-1]]))
        else:
            cached.append((image_id, []))
    recalls = np.linspace(0.0, 1.0, 101)
    result = []
    for threshold in thresholds:
        matched = {i: np.zeros(len(boxes), dtype=bool) for i, boxes in gt_by_image.items()}
        tp = np.zeros(len(detections), dtype=np.float64)
        fp = np.zeros(len(detections), dtype=np.float64)
        for i, (image_id, overlaps) in enumerate(cached):
            match = next((j for j, value in overlaps if value >= threshold and not matched[image_id][j]), None)
            if match is None:
                fp[i] = 1
            else:
                tp[i] = 1
                matched[image_id][match] = True
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        recall = ctp / total_gt
        precision = ctp / np.maximum(ctp + cfp, 1e-12)
        if len(precision):
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            samples = [precision[recall >= r].max() if np.any(recall >= r) else 0.0 for r in recalls]
            result.append(float(np.mean(samples)))
        else:
            result.append(0.0)
    return result


def confusion_counts(predictions, targets, nclasses, conf=0.25, iou_thr=0.5):
    matrix = np.zeros((nclasses + 1, nclasses + 1), dtype=np.int64)
    for pred, (gt_boxes, gt_classes) in zip(predictions, targets):
        keep = pred["scores"] >= conf
        boxes, classes, scores = pred["boxes"][keep], pred["labels"][keep], pred["scores"][keep]
        overlaps = iou_matrix(gt_boxes, boxes)
        candidates = [(overlaps[g, p], scores[p], g, p) for g in range(len(gt_boxes))
                      for p in range(len(boxes)) if overlaps[g, p] >= iou_thr]
        candidates.sort(reverse=True)
        mg, mp = set(), set()
        for _, _, g, p in candidates:
            if g in mg or p in mp:
                continue
            matrix[gt_classes[g], classes[p]] += 1
            mg.add(g); mp.add(p)
        for g, cls in enumerate(gt_classes):
            if g not in mg:
                matrix[cls, nclasses] += 1
        for p, cls in enumerate(classes):
            if p not in mp:
                matrix[nclasses, cls] += 1
    tp = int(np.trace(matrix[:nclasses, :nclasses]))
    mis = int(matrix[:nclasses, :nclasses].sum()) - tp
    fp = int(matrix[nclasses, :nclasses].sum()) + mis
    fn = int(matrix[:nclasses, nclasses].sum()) + mis
    return matrix, tp, fp, fn


def read_targets(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")
    h, w = image.shape[:2]
    label_path = (image_path.parents[2] / "labels/val" / image_path.name).with_suffix(".txt")
    classes, boxes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        cls, xc, yc, bw, bh = map(float, vals[:5])
        classes.append(int(cls))
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                      (xc + bw / 2) * w, (yc + bh / 2) * h])
    return image, (np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--model", default=str(ROOT / "yolov8n.pt"))
    parser.add_argument("--model-name", default=None,
                        help="Name recorded in the output CSV (defaults to the checkpoint filename).")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/yolo_custom_fullval_seed42",
                        help="Output directory for metrics and confusion matrices.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    device = "0" if args.device == "0" and torch.cuda.is_available() else args.device
    model = YOLO(args.model)
    model_name = args.model_name
    if model_name is None:
        model_name = Path(args.model).stem
        if model_name.lower().startswith("yolo"):
            model_name = "YOLO" + model_name[4:]
    data = yaml.safe_load((ROOT / "datasets/coco/full_val2017/data.yaml").read_text(encoding="utf-8"))
    raw_names = data["names"]
    names = [raw_names.get(i, raw_names.get(str(i), str(i))) for i in range(len(raw_names))] if isinstance(raw_names, dict) else raw_names
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.csv"
    existing = {}
    if metrics.exists():
        with metrics.open(newline="", encoding="utf-8-sig") as f:
            existing = {r["condition"]: r for r in csv.DictReader(f)}
    for condition in args.conditions:
        if condition in existing:
            print(f"Skipping existing {condition}", flush=True)
            continue
        images_dir = locate(condition) / "images/val"
        files = sorted(p for p in images_dir.iterdir() if p.is_file())
        if len(files) != 5000:
            raise ValueError(f"Expected 5000 images at {images_dir}, found {len(files)}")
        preds, targets = [], []
        started = time.time()
        for start in range(0, len(files), args.batch):
            batch_files = files[start:start + args.batch]
            batch_images, batch_targets = zip(*(read_targets(path) for path in batch_files))
            results = model.predict(list(batch_images), imgsz=args.imgsz, device=device,
                                    conf=0.001, iou=0.7, max_det=300, verbose=False)
            for result in results:
                boxes = result.boxes
                preds.append({"boxes": boxes.xyxy.detach().cpu().numpy().astype(np.float32),
                              "labels": boxes.cls.detach().cpu().numpy().astype(np.int64),
                              "scores": boxes.conf.detach().cpu().numpy().astype(np.float32)})
            targets.extend(batch_targets)
            seen = min(start + len(batch_files), len(files))
            if seen % 400 == 0 or seen == len(files):
                print(f"{condition}: {seen}/5000 images", flush=True)
        thresholds = [0.50 + 0.05 * i for i in range(10)]
        aps = np.asarray([ap_for_class(preds, targets, cls, thresholds) for cls in range(len(names))])
        map50, map5095 = float(np.nanmean(aps[:, 0])), float(np.nanmean(aps))
        matrix, tp, fp, fn = confusion_counts(preds, targets, len(names))
        np.savetxt(out / f"confusion_{condition}.csv", matrix, fmt="%d", delimiter=",")
        elapsed = time.time() - started
        row = {"model": model_name, "condition": condition, "images": len(files),
               "imgsz": args.imgsz, "ap_implementation": "101-point-interpolated-IoU50-95",
               "mAP50": map50, "mAP50_95": map5095, "tp_iou50": tp, "fp_iou50": fp,
               "fn_iou50": fn, "confusion_conf": 0.25, "confusion_iou": 0.5,
               "elapsed_sec": round(elapsed, 2)}
        existing[condition] = row
        with metrics.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader(); writer.writerows(existing.values())
        print(f"{condition}: mAP50={map50:.4f}, mAP50-95={map5095:.4f}, FP={fp}, FN={fn}, elapsed={elapsed:.1f}s", flush=True)
    print(f"Saved results to {metrics}")


if __name__ == "__main__":
    main()

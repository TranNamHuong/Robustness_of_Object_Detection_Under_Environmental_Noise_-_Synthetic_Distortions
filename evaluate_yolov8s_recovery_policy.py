#!/usr/bin/env python
"""Evaluate the product-default YOLOv8s on existing full-val recovery conditions."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/yolov8s_fullval_recovery_policy"
CONDITIONS = [
    ("clean", ROOT / "datasets/coco/full_val2017/data.yaml"),
    ("gaussian_0.06", ROOT / "runs/full_val2017_raw_seed42/datasets/gaussian_0.06/data.yaml"),
    ("gaussian_0.06_bilateral", ROOT / "runs/full_val2017_gaussian_bilateral_seed42/datasets/gaussian_0.06_bilateral/data.yaml"),
    ("salt_pepper_0.06", ROOT / "runs/full_val2017_raw_seed42/datasets/salt_pepper_0.06/data.yaml"),
    ("salt_pepper_0.06_recovered", ROOT / "runs/full_val2017_recovered_seed42/datasets/salt_pepper_0.06_recovered/data.yaml"),
    ("fog_0.50", ROOT / "runs/full_val2017_raw_seed42/datasets/fog_0.5/data.yaml"),
    ("fog_0.50_CLAHE", ROOT / "runs/full_val2017_recovered_seed42/datasets/fog_0.5_recovered/data.yaml"),
    ("low_light_0.25", ROOT / "runs/full_val2017_raw_seed42/datasets/low_light_0.25/data.yaml"),
    ("low_light_0.25_CLAHE", ROOT / "runs/full_val2017_recovered_seed42/datasets/low_light_0.25_recovered/data.yaml"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    partial = OUT / "metrics.partial.csv"
    completed = {}
    if partial.exists():
        with partial.open(newline="", encoding="utf-8-sig") as handle:
            completed = {row["condition"]: row for row in csv.DictReader(handle)}

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(ROOT / "yolov8s.pt"))
    for condition, data_yaml in CONDITIONS:
        if condition in completed:
            print(f"Skipping completed {condition}", flush=True)
            continue
        if not data_yaml.is_file():
            raise FileNotFoundError(data_yaml)
        started = time.perf_counter()
        metrics = model.val(data=str(data_yaml), imgsz=640, batch=16, device=device,
                            conf=0.001, iou=0.7, plots=False,
                            project=str(OUT / "ultralytics"), name=condition,
                            exist_ok=True, verbose=False)
        box = metrics.box
        speed = getattr(metrics, "speed", {}) or {}
        completed[condition] = {
            "condition": condition, "images": 5000, "imgsz": 640, "device": device,
            "mAP50": float(box.map50), "mAP50_95": float(box.map),
            "precision_ultralytics": float(box.mp), "recall_ultralytics": float(box.mr),
            "preprocess_ms_per_image": float(speed.get("preprocess", float("nan"))),
            "inference_ms_per_image": float(speed.get("inference", float("nan"))),
            "postprocess_ms_per_image": float(speed.get("postprocess", float("nan"))),
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
        with partial.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(completed.values()))))
            writer.writeheader()
            writer.writerows(completed.values())
        print(f"{condition}: mAP50={box.map50:.4f}, mAP50-95={box.map:.4f}; saved partial", flush=True)

    final = OUT / "metrics.csv"
    with final.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(completed.values()))))
        writer.writeheader()
        writer.writerows(completed.values())
    print(f"Saved YOLOv8s full-val recovery metrics: {final}")


if __name__ == "__main__":
    main()

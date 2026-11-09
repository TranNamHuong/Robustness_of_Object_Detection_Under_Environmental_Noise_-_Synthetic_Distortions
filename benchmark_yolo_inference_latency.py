#!/usr/bin/env python
"""Measure single-image YOLO latency on the same GPU used by the desktop app."""
from __future__ import annotations

import csv
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "datasets/coco/full_val2017/images/val"
MODELS = [("YOLOv8n", ROOT / "yolov8n.pt"), ("YOLOv8s", ROOT / "yolov8s.pt")]
OUT = ROOT / "runs/yolo_model_size_comparison"
N_IMAGES = 200
N_WARMUP = 20
IMGSZ = 640
CONF = 0.25
IOU = 0.7


def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    paths = sorted(IMAGE_DIR.glob("*.jpg"))[:N_IMAGES]
    if len(paths) < N_IMAGES:
        raise RuntimeError(f"Expected {N_IMAGES} COCO images in {IMAGE_DIR}")
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError("Could not decode an image in the latency sample")

    latency_samples = {name: [] for name, _ in MODELS}
    memory_samples = {name: [] for name, _ in MODELS}
    # Reverse execution order on the second pass to reduce GPU warm-up/order bias.
    for order in (MODELS, list(reversed(MODELS))):
        for name, weights in order:
            model = YOLO(str(weights))
            for image in images[:N_WARMUP]:
                model.predict(source=image, imgsz=IMGSZ, conf=CONF, iou=IOU,
                              device=device, verbose=False)
            if device != "cpu":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            latencies = []
            for image in images:
                start = time.perf_counter()
                model.predict(source=image, imgsz=IMGSZ, conf=CONF, iou=IOU,
                              device=device, verbose=False)
                if device != "cpu":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
            peak_memory_gb = (torch.cuda.max_memory_allocated() / 1024**3) if device != "cpu" else 0.0
            latency_samples[name].extend(latencies)
            memory_samples[name].append(peak_memory_gb)
            print(f"pass {len(memory_samples[name])} {name}: mean={np.mean(latencies):.2f} ms")
            del model
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()

    rows = []
    for name, weights in MODELS:
        latencies = latency_samples[name]
        peak_memory_gb = max(memory_samples[name])
        rows.append({"model": name, "weights": weights.name, "device": device,
                     "images_per_order": len(images), "orders_per_model": len(memory_samples[name]),
                     "imgsz": IMGSZ, "confidence": CONF, "nms_iou": IOU,
                     "warmup_images_per_order": N_WARMUP,
                     "latency_mean_ms": float(np.mean(latencies)),
                     "latency_median_ms": float(np.median(latencies)),
                     "latency_p95_ms": float(np.percentile(latencies, 95)),
                     "single_image_fps_mean": 1000 / float(np.mean(latencies)),
                     "peak_cuda_allocated_gb": peak_memory_gb})
        print(f"{name}: mean={np.mean(latencies):.2f} ms, median={np.median(latencies):.2f} ms, "
              f"p95={np.percentile(latencies,95):.2f} ms, peak CUDA={peak_memory_gb:.3f} GB")

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "single_image_latency.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved latency results: {output}")


if __name__ == "__main__":
    main()

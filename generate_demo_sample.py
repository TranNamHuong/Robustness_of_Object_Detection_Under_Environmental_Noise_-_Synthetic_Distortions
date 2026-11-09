"""Generate a reproducible YOLOv8s demo from an existing COCO val image."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app import annotate
from preprocessing import recover_image
from robustness_eval import Condition, perturb


ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "datasets" / "coco" / "val2017" / "000000000139.jpg"
WEIGHTS = ROOT / "yolov8s.pt"
OUT = ROOT / "runs" / "demo_sample"


def save(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(str(IMAGE), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(IMAGE)
    condition = Condition("gaussian_0.06", "gaussian", 0.06)
    noisy = perturb(source, condition, np.random.default_rng(42))
    recovered = recover_image(noisy, "gaussian", "bilateral")
    model = YOLO(str(WEIGHTS))
    device = "0" if torch.cuda.is_available() else "cpu"

    views = {}
    for name, image in (("baseline", source), ("noisy", noisy), ("recovered", recovered)):
        result = model.predict(source=image, imgsz=640, conf=0.25, iou=0.7,
                               device=device, verbose=False)[0]
        views[name] = annotate(image, result, model.names)
        save(OUT / f"{name}_detections.jpg", views[name].image)
    save(OUT / "noisy_input.jpg", noisy)
    save(OUT / "recovered_input.jpg", recovered)
    metadata = {
        "source": str(IMAGE.relative_to(ROOT)),
        "model": WEIGHTS.name,
        "device": device,
        "condition": condition.name,
        "recovery": "bilateral",
        "imgsz": 640,
        "confidence": 0.25,
        "nms_iou": 0.7,
        "counts": {name: view.count for name, view in views.items()},
        "classes": {name: view.classes for name, view in views.items()},
        "warning": "COCO demo sample; does not measure target-domain accuracy.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "device": device, "counts": metadata["counts"]}))


if __name__ == "__main__":
    main()

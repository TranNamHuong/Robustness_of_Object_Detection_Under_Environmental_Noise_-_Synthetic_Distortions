from __future__ import annotations

import base64
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = {"yolov8n": PROJECT_ROOT / "yolov8n.pt", "yolov8s": PROJECT_ROOT / "yolov8s.pt"}
MODEL_INFO = {
    "yolov8n": {"label": "YOLOv8n", "family": "YOLO", "input_size": 640},
    "yolov8s": {"label": "YOLOv8s", "family": "YOLO", "input_size": 640},
    "ssdlite": {"label": "SSD-MobileNet", "family": "Torchvision SSD", "input_size": 320},
}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 8000


def runtime_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def readiness() -> dict[str, Any]:
    device = runtime_device()
    available = {key: True for key in MODEL_INFO}
    for model_id, path in WEIGHTS.items():
        available[model_id] = path.is_file()
    try:
        import torchvision
        torchvision_version = torchvision.__version__
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        weights_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
        ssdlite_ready = weights_path.is_file()
        available["ssdlite"] = ssdlite_ready
        if not ssdlite_ready:
            available["ssdlite_error"] = f"Weights are not cached locally: {weights_path}"
    except Exception as exc:
        torchvision_version = None
        ssdlite_ready = False
        available["ssdlite"] = False
        available["ssdlite_error"] = str(exc)
    return {
        "status": "ok",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "ultralytics_version": __import__("ultralytics").__version__,
        "models": available,
        "ssdlite_ready": ssdlite_ready,
    }


@lru_cache(maxsize=3)
def load_model(model_id: str):
    if model_id not in MODEL_INFO:
        raise ValueError(f"Unsupported model: {model_id}")
    device = runtime_device()
    if model_id in WEIGHTS:
        path = WEIGHTS[model_id]
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = YOLO(str(path))
        return model, device
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"SSD-MobileNet weights are not cached locally ({weights_path}). "
            "The app will not download model weights during inference."
        )
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()
    return (model, weights), device


def validate_image_bytes(data: bytes, filename: str) -> np.ndarray:
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 20 MB upload limit.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("Upload a JPG, PNG, or WebP image.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded content is not a decodable image.")
    height, width = image.shape[:2]
    if max(height, width) > MAX_IMAGE_SIDE:
        raise ValueError(f"Image dimensions are limited to {MAX_IMAGE_SIDE} pixels per side.")
    return image


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def detect(model_id: str, image_bgr: np.ndarray, confidence: float, nms_iou: float) -> dict[str, Any]:
    model_bundle, device = load_model(model_id)
    _synchronize(device)
    started = time.perf_counter()
    if model_id in WEIGHTS:
        model = model_bundle
        result = model.predict(
            source=image_bgr,
            imgsz=MODEL_INFO[model_id]["input_size"],
            conf=confidence,
            iou=nms_iou,
            max_det=300,
            device=str(device),
            verbose=False,
        )[0]
        detections: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes):
            coords = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            names = model.names
            for bbox, class_id, score in zip(coords, classes, scores, strict=True):
                detections.append({
                    "class_id": int(class_id),
                    "label": str(names.get(int(class_id), class_id)),
                    "confidence": float(score),
                    "bbox_xyxy": [float(value) for value in bbox],
                })
    else:
        model, weights = model_bundle
        model.nms_thresh = nms_iou
        model.score_thresh = confidence
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(Image.fromarray(image_rgb)).to(device)
        with torch.inference_mode():
            result = model([tensor])[0]
        categories = weights.meta["categories"]
        detections = []
        for bbox, class_id, score in zip(result["boxes"], result["labels"], result["scores"], strict=True):
            confidence_value = float(score.detach().cpu())
            if confidence_value < confidence:
                continue
            label_id = int(class_id.detach().cpu())
            detections.append({
                "class_id": label_id,
                "label": categories[label_id] if label_id < len(categories) else str(label_id),
                "confidence": confidence_value,
                "bbox_xyxy": [float(value) for value in bbox.detach().cpu().tolist()],
            })
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"detections": detections, "latency_ms": round(elapsed_ms, 3), "device": str(device)}


def overlay(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    output = image_bgr.copy()
    for item in detections:
        class_id = int(item["class_id"])
        color = ((53 + class_id * 29) % 210 + 30, (109 + class_id * 47) % 210 + 30, (171 + class_id * 17) % 210 + 30)
        x1, y1, x2, y2 = (int(round(value)) for value in item["bbox_xyxy"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{item['label']} {item['confidence']:.2f}"
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1, height + 8)
        cv2.rectangle(output, (x1, top - height - 7), (x1 + width + 7, top + baseline - 3), color, -1)
        cv2.putText(output, label, (x1 + 3, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def encode_image(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Could not encode the preview image.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def result_images(image_bgr: np.ndarray, noisy_bgr: np.ndarray, restored_bgr: np.ndarray | None,
                  detection_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "original": encode_image(overlay(image_bgr, detection_results["original"]["detections"])),
        "noisy": encode_image(overlay(noisy_bgr, detection_results["noisy"]["detections"])),
        "restored": encode_image(overlay(restored_bgr, detection_results["restored"]["detections"])) if restored_bgr is not None else None,
    }

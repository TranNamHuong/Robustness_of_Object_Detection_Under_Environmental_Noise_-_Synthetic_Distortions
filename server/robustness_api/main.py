from __future__ import annotations

import threading
import csv
import json
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from denoise import LEVEL_AWARE_AUTO  # noqa: E402
from preprocessing import recover_image  # noqa: E402
from robustness_eval import Condition, perturb  # noqa: E402
from .benchmark_repository import EXPERIMENTS, filter_records, get_experiments, get_records  # noqa: E402
from .inference import (  # noqa: E402
    MAX_IMAGE_BYTES,
    MODEL_INFO,
    detect,
    readiness,
    result_images,
    validate_image_bytes,
)

app = FastAPI(
    title="Object Detection Robustness Lab API",
    version="0.1.0",
    description="Local inference and read-only benchmark data for the research lab.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
OPEN_IMAGES_ROOT = PROJECT_ROOT / "datasets" / "open_images_v7_test_extra"
app.mount("/data/open-images-300", StaticFiles(directory=OPEN_IMAGES_ROOT / "images" / "val"), name="open-images-gallery")
INFERENCE_LOCK = threading.BoundedSemaphore(value=1)
FAMILY_LEVELS = {
    "gaussian": (0.01, 0.03, 0.06),
    "salt_pepper": (0.01, 0.03, 0.06),
    "fog": (0.15, 0.30, 0.50),
    "low_light": (0.25, 0.40, 0.60),
}
CONDITIONS = [
    {"id": "clean", "label_vi": "Ảnh gốc", "label_en": "Clean", "family": "clean", "levels": []},
    {"id": "gaussian", "label_vi": "Nhiễu Gaussian", "label_en": "Gaussian noise", "family": "gaussian", "levels": list(FAMILY_LEVELS["gaussian"])},
    {"id": "gaussian_nlm", "label_vi": "Gaussian + NLM", "label_en": "Gaussian denoising", "family": "gaussian", "levels": list(FAMILY_LEVELS["gaussian"])},
    {"id": "salt_pepper", "label_vi": "Nhiễu muối tiêu", "label_en": "Salt-and-Pepper", "family": "salt_pepper", "levels": list(FAMILY_LEVELS["salt_pepper"])},
    {"id": "fog", "label_vi": "Sương mù mô phỏng", "label_en": "Synthetic fog", "family": "fog", "levels": list(FAMILY_LEVELS["fog"])},
    {"id": "low_light", "label_vi": "Thiếu sáng", "label_en": "Low light", "family": "low_light", "levels": list(FAMILY_LEVELS["low_light"])},
]
MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolov8s": "YOLOv8s",
    "ssdlite": "SSD-MobileNet",
}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return readiness()


@app.get("/api/models")
def models() -> list[dict[str, Any]]:
    state = readiness()
    return [
        {
            "id": model_id,
            **info,
            "ready": bool(state["models"].get(model_id, False)),
            "pretrained": True,
            "classes": 80 if model_id.startswith("yolov8") else 91,
        }
        for model_id, info in MODEL_INFO.items()
    ]


@app.get("/api/conditions")
def conditions() -> list[dict[str, Any]]:
    return CONDITIONS


def _validated_condition(condition: str, severity: float) -> tuple[str, float, bool]:
    if condition not in {item["id"] for item in CONDITIONS}:
        raise ValueError(f"Unsupported image condition: {condition}")
    if condition == "clean":
        return "clean", 0.0, False
    family = "gaussian" if condition == "gaussian_nlm" else condition
    supported = FAMILY_LEVELS[family]
    level = next((value for value in supported if abs(value - severity) < 1e-8), None)
    if level is None:
        levels = ", ".join(f"{value:g}" for value in supported)
        raise ValueError(f"Severity {severity:g} was not measured for {family}; choose {levels}.")
    return family, level, condition == "gaussian_nlm"


def _recovery_method(family: str, level: float, requested: str, force_nlm: bool) -> str | None:
    if family == "clean":
        return None
    if force_nlm:
        return "nlmeans"
    if requested == "none":
        return None
    if requested != "auto":
        if requested not in {"median", "nlmeans", "clahe", "bilateral"}:
            raise ValueError(f"Unsupported recovery method: {requested}")
        return requested
    choices = LEVEL_AWARE_AUTO.get(family, {})
    for measured_level, method in choices.items():
        if abs(float(measured_level) - level) < 1e-8:
            return None if method == "none" else method
    return None


def _build_pipeline(image: np.ndarray, condition: str, severity: float, seed: int,
                    recovery: str) -> tuple[str, float, str | None, np.ndarray, np.ndarray | None]:
    family, level, force_nlm = _validated_condition(condition, severity)
    if family == "clean":
        return family, level, None, image.copy(), None
    corrupted = perturb(
        image,
        Condition(f"{family}_{level:g}", family, level),
        np.random.default_rng(seed),
    )
    method = _recovery_method(family, level, recovery, force_nlm)
    restored = recover_image(corrupted, family, method) if method else None
    return family, level, method, corrupted, restored


def _run_models(image: np.ndarray, noisy: np.ndarray, restored: np.ndarray | None,
                models_to_run: list[str], confidence: float, nms_iou: float) -> dict[str, Any]:
    unknown = sorted(set(models_to_run) - MODEL_INFO.keys())
    if unknown:
        raise ValueError(f"Unknown model IDs: {', '.join(unknown)}")
    result: dict[str, Any] = {}
    for model_id in models_to_run:
        views: dict[str, Any] = {
            "original": detect(model_id, image, confidence, nms_iou),
            "noisy": detect(model_id, noisy, confidence, nms_iou),
        }
        if restored is not None:
            views["restored"] = detect(model_id, restored, confidence, nms_iou)
        result[model_id] = {
            "model": MODEL_LABELS[model_id],
            "views": views,
            "images": result_images(image, noisy, restored, {
                "original": views["original"],
                "noisy": views["noisy"],
                "restored": views.get("restored", views["noisy"]),
            } if restored is not None else {"original": views["original"], "noisy": views["noisy"], "restored": views["noisy"]}),
        }
        if restored is None:
            result[model_id]["images"]["restored"] = None
    return result


async def _run_request(
    file: UploadFile,
    model_ids: list[str],
    condition: str,
    severity: float,
    confidence: float,
    nms_iou: float,
    seed: int,
    recovery: str,
) -> dict[str, Any]:
    if not 0.01 <= confidence <= 0.99:
        raise HTTPException(status_code=422, detail="confidence must be between 0.01 and 0.99")
    if not 0.1 <= nms_iou <= 0.95:
        raise HTTPException(status_code=422, detail="nms_iou must be between 0.1 and 0.95")
    data = await file.read(MAX_IMAGE_BYTES + 1)
    try:
        image = validate_image_bytes(data, file.filename or "upload.jpg")
        family, level, method, noisy, restored = _build_pipeline(image, condition, severity, seed, recovery)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except cv2.error as exc:
        raise HTTPException(status_code=422, detail=f"Could not process the image: {exc}") from exc
    try:
        if not INFERENCE_LOCK.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Another inference request is running; retry shortly.")
        try:
            # Keep local CUDA inference on the request's owning thread. PyTorch's
            # CUDA context is already initialized by this process, and this
            # single-user local server deliberately serializes GPU work.
            results = _run_models(image, noisy, restored, model_ids, confidence, nms_iou)
        finally:
            INFERENCE_LOCK.release()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc
    return {
        "source_filename": Path(file.filename or "upload.jpg").name,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "condition": condition,
        "family": family,
        "severity": level,
        "seed": seed,
        "recovery_method": method,
        "confidence_threshold": confidence,
        "nms_iou_threshold": nms_iou,
        "has_ground_truth": False,
        "metrics": None,
        "models": results,
        "provenance": {
            "checkpoint_policy": "original COCO-pretrained weights; no fine-tuning",
            "upload_retention": "not stored by the API",
            "severity_note": "synthetic corruption parameter; measured values only",
        },
    }


@app.post("/api/detect")
async def api_detect(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "yolov8s",
    condition: Annotated[str, Form()] = "clean",
    severity: Annotated[float, Form()] = 0.0,
    confidence: Annotated[float, Form()] = 0.25,
    nms_iou: Annotated[float, Form()] = 0.7,
    seed: Annotated[int, Form()] = 42,
    recovery: Annotated[str, Form()] = "auto",
) -> dict[str, Any]:
    return await _run_request(file, [model], condition, severity, confidence, nms_iou, seed, recovery)


@app.post("/api/compare")
async def api_compare(
    file: Annotated[UploadFile, File()],
    models_to_run: Annotated[list[str], Form(alias="models")] = ["yolov8n", "yolov8s", "ssdlite"],
    condition: Annotated[str, Form()] = "clean",
    severity: Annotated[float, Form()] = 0.0,
    confidence: Annotated[float, Form()] = 0.25,
    nms_iou: Annotated[float, Form()] = 0.7,
    seed: Annotated[int, Form()] = 42,
    recovery: Annotated[str, Form()] = "auto",
) -> dict[str, Any]:
    if len(models_to_run) < 2 or len(models_to_run) != len(set(models_to_run)):
        raise HTTPException(status_code=422, detail="Choose at least two distinct models for comparison.")
    return await _run_request(file, models_to_run, condition, severity, confidence, nms_iou, seed, recovery)


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]:
    return get_experiments()


@app.get("/api/metrics")
def metrics(
    experiment_id: str,
    model: str | None = None,
    condition: str | None = None,
    family: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    experiment = next((item for item in EXPERIMENTS if item["id"] == experiment_id), None)
    if experiment is None and experiment_id != "user_reference_unverified":
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment_id}")
    records = filter_records(
        get_records(experiment_id), model=model, condition=condition, family=family, stage=stage
    )
    metadata = next((item for item in get_experiments() if item["id"] == experiment_id), None)
    return {
        "experiment": metadata,
        "filters": {
            "models": sorted({row["model"] for row in records}),
            "conditions": sorted({row["condition"] for row in records}),
            "families": sorted({row["family"] for row in records}),
            "stages": sorted({row["stage"] for row in records}),
        },
        "records": records,
        "record_count": len(records),
        "source_file": metadata["source"] if metadata else None,
        "reference_only": bool(metadata and metadata.get("kind") == "reference_only"),
    }


@app.get("/api/exports")
def export_metrics(experiment_id: str) -> Any:
    import csv
    import io
    from fastapi.responses import StreamingResponse

    experiment = next((item for item in get_experiments() if item["id"] == experiment_id), None)
    if experiment is None or not experiment.get("available"):
        raise HTTPException(status_code=404, detail="No measured result file is available for this experiment.")
    records = get_records(experiment_id)
    if not records:
        raise HTTPException(status_code=404, detail="This experiment has no exportable records.")
    buffer = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in records for key in row))
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    filename = f"{experiment_id}_metrics.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/recommendations")
def recommendations(experiment_id: str, criterion: str = "map", condition: str | None = None,
                    stage: str | None = None) -> dict[str, Any]:
    """Rank only records that share a measured dataset protocol."""
    if criterion not in {"map", "precision", "recall", "robustness", "recovery", "realtime"}:
        raise HTTPException(status_code=422, detail="Unsupported recommendation criterion.")
    experiment = next((item for item in get_experiments() if item["id"] == experiment_id), None)
    if experiment is None or experiment.get("kind") == "reference_only":
        return {"status": "insufficient_evidence", "reason": "Reference-only or unknown data cannot support recommendations.", "candidates": []}
    records = get_records(experiment_id)
    if criterion in {"map", "precision", "recall", "realtime"}:
        metric = {"map": "mAP50_95", "precision": "precision", "recall": "recall", "realtime": "latency_ms"}[criterion]
        eligible = [row for row in records if row.get("reference_only") is not True]
        if criterion == "map":
            eligible = [row for row in eligible if row.get("condition") == (condition or "clean") and row.get("stage") in {"clean", "raw", "restored"}]
            if stage:
                eligible = [row for row in eligible if row.get("stage") == stage]
            selected = [row for row in eligible if row.get("mAP50_95") is not None]
            if len({row["model"] for row in selected}) < 2:
                fallback = [row for row in eligible if row.get("mAP50") is not None]
                if len({row["model"] for row in fallback}) >= 2:
                    metric = "mAP50"
                    selected = fallback
        elif criterion in {"precision", "recall", "realtime"}:
            selected = [row for row in eligible if row.get(metric) is not None and row.get("condition") == (condition or "clean") and row.get("stage") in {"clean", "raw", "restored"}]
            if stage:
                selected = [row for row in selected if row.get("stage") == stage]
        if len({row["model"] for row in selected}) < 2:
            return {"status": "insufficient_evidence", "reason": "At least two models with the selected measured metric and shared condition are required.", "candidates": [], "experiment": experiment}
        reverse = criterion != "realtime"
        selected.sort(key=lambda row: row[metric], reverse=reverse)
        candidates = [{"model": row["model"], "value": row[metric], "metric": metric, "condition": row["condition"], "stage": row["stage"], "source_file": row["source_file"], "images": row["images"], "protocol": row["protocol"]} for row in selected]
    elif criterion == "robustness":
        if not condition or condition == "clean":
            raise HTTPException(status_code=422, detail="Select a measured noisy condition for robustness ranking.")
        noisy = [row for row in records if row["condition"] == condition and row["stage"] == "raw" and row.get("mAP50_95") is not None]
        clean = {row["model"]: row for row in records if row["condition"] == "clean" and row["stage"] in {"clean", "raw"} and row.get("mAP50_95") is not None}
        candidates = []
        for row in noisy:
            baseline = clean.get(row["model"])
            if baseline and baseline["mAP50_95"]:
                drop_pct = 100 * (baseline["mAP50_95"] - row["mAP50_95"]) / baseline["mAP50_95"]
                candidates.append({"model": row["model"], "value": drop_pct, "metric": "relative_mAP_drop_pct", "clean_mAP50_95": baseline["mAP50_95"], "noisy_mAP50_95": row["mAP50_95"], "condition": condition, "source_file": row["source_file"], "images": row["images"], "protocol": row["protocol"]})
        candidates.sort(key=lambda row: row["value"])
    else:
        if not condition:
            raise HTTPException(status_code=422, detail="Select a measured condition for recovery ranking.")
        restored = [row for row in records if row["condition"] == condition and row["stage"] == "restored" and row.get("mAP50_95") is not None]
        candidates = []
        for row in restored:
            raw = next((item for item in records if item["model"] == row["model"] and item["condition"] == condition and item["stage"] == "raw"), None)
            if raw and raw.get("mAP50_95") is not None:
                candidates.append({"model": row["model"], "value": row["mAP50_95"] - raw["mAP50_95"], "metric": "recovery_gain_mAP50_95", "raw_mAP50_95": raw["mAP50_95"], "restored_mAP50_95": row["mAP50_95"], "condition": condition, "source_file": row["source_file"], "images": row["images"], "protocol": row["protocol"]})
        candidates.sort(key=lambda row: row["value"], reverse=True)
    if len(candidates) < 2:
        return {"status": "insufficient_evidence", "reason": "Fewer than two comparable measured model results are available for this criterion.", "candidates": candidates, "experiment": experiment}
    metric_note = f" using {candidates[0]['metric']}" if criterion == "map" else ""
    return {"status": "ok", "criterion": criterion, "winner": candidates[0], "candidates": candidates, "experiment": experiment, "rule": f"Ranking{metric_note} is descriptive and limited to the selected dataset, evaluator, thresholds, condition, and recorded protocol."}


OPEN_CLASS_NAMES = ["Person", "Chair", "Couch", "Bed", "Remote control", "Mobile phone", "Clock", "Teddy bear"]
OPEN_CLASS_IDS = {"/m/01g317": 0, "/m/01mzpv": 1, "/m/02crq1": 2, "/m/03ssj5": 3, "/m/0qjjc": 4, "/m/050k8": 5, "/m/01x3z": 6, "/m/0kmg4": 7}


@app.get("/api/gallery")
def gallery(page: int = 1, page_size: int = 12, class_name: str | None = None, min_score: float = 0.25) -> dict[str, Any]:
    if page < 1 or not 1 <= page_size <= 48 or not 0 <= min_score <= 1:
        raise HTTPException(status_code=422, detail="Invalid gallery pagination or confidence filter.")
    data_root = OPEN_IMAGES_ROOT
    prediction_file = PROJECT_ROOT / "runs" / "open_images_v7_test_extra_eval" / "predictions_yolov8s_img320.json"
    if not prediction_file.is_file():
        return {"items": [], "page": page, "page_size": page_size, "total": 0, "classes": OPEN_CLASS_NAMES, "protocol": "Open Images V7 · 300-image disjoint labeled subset"}
    with (data_root / "subset_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["ImageID"]: row for row in csv.DictReader(stream)}
    ground_truth: dict[str, list[dict[str, Any]]] = {}
    with (data_root / "selected_annotations.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            local_id = OPEN_CLASS_IDS.get(row["LabelName"])
            if local_id is None:
                continue
            ground_truth.setdefault(row["ImageID"], []).append({"class_name": OPEN_CLASS_NAMES[local_id], "x1": float(row["XMin"]), "y1": float(row["YMin"]), "x2": float(row["XMax"]), "y2": float(row["YMax"]), "group_of": row["IsGroupOf"] == "1"})
    predictions = json.loads(prediction_file.read_text(encoding="utf-8"))
    items = []
    for pred in predictions:
        image_id = pred["image_id"]
        meta = manifest.get(image_id)
        if not meta:
            continue
        image_path = data_root / meta["local_image"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_height, image_width = image.shape[:2]
        gt_boxes = ground_truth.get(image_id, [])
        pred_boxes = [{"class_name": OPEN_CLASS_NAMES[int(label)], "confidence": float(score), "x1": float(box[0]) / image_width, "y1": float(box[1]) / image_height, "x2": float(box[2]) / image_width, "y2": float(box[3]) / image_height} for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]) if float(score) >= min_score and 0 <= int(label) < len(OPEN_CLASS_NAMES)]
        if class_name and not any(box["class_name"].casefold() == class_name.casefold() for box in gt_boxes):
            continue
        items.append({"image_id": image_id, "image_url": f"/data/open-images-300/{image_id}.jpg", "image_size": {"width": image_width, "height": image_height}, "class_names": sorted({box["class_name"] for box in gt_boxes}), "ground_truth": gt_boxes, "predictions": pred_boxes, "source": meta.get("OriginalLandingURL", ""), "license": meta.get("License", ""), "author": meta.get("Author", ""), "dataset": "Open Images V7 validation subset", "model": "YOLOv8s · COCO pretrained · img 320", "prediction_threshold": min_score})
    total = len(items)
    start = (page - 1) * page_size
    return {"items": items[start:start + page_size], "page": page, "page_size": page_size, "total": total, "classes": OPEN_CLASS_NAMES, "protocol": "Open Images V7 · 300-image disjoint labeled subset · YOLOv8s predictions from recorded img320 evaluation · confidence threshold is a display filter, not a new metric"}

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "coco_normalized_320",
        "title": "COCO val2017 · shared evaluator · 320 px",
        "dataset": "COCO val2017",
        "dataset_kind": "COCO-80",
        "protocol": "5,000 images; shared 101-point AP implementation; YOLOv8n, YOLOv8s, and SSDLite at input size 320; IoU 0.50:0.95; count threshold 0.25 / matching IoU 0.50.",
        "source": "runs/normalized_detector_comparison_fullval/metrics.csv",
        "kind": "multi_model_corruption",
    },
    {
        "id": "coco_intensity_320",
        "title": "COCO val2017 · intensity sweep & restoration · 320 px",
        "dataset": "COCO val2017",
        "dataset_kind": "COCO-80",
        "protocol": "5,000 images per recorded condition; 320 px is identified by the detector IDs. Clean baseline is read from the paired normalized metrics.csv. The intensity CSVs do not record evaluator thresholds or implementation, so those settings are left unspecified. YOLOv8n and SSD-MobileNet only. Three raw severity levels are recorded per family; restored rows appear only for combinations present in the recovery source, which does not provide restored TP.",
        "source": "runs/normalized_detector_comparison_fullval/raw_intensity_metrics.csv",
        "secondary_sources": [
            "runs/normalized_detector_comparison_fullval/metrics.csv",
            "runs/normalized_detector_comparison_fullval/recovery_intensity_metrics.csv",
        ],
        "kind": "intensity_sweep",
    },
    {
        "id": "coco_yolo_family_640",
        "title": "COCO val2017 · YOLOv8n/s comparison · 640 px",
        "dataset": "COCO val2017",
        "dataset_kind": "COCO-80",
        "protocol": "5,000 images; Ultralytics validation evaluator; input size 640; matched clean, raw, and recovery conditions for YOLOv8n and YOLOv8s. SSDLite is evaluated at its native 320 px input and is not included in this 640 px comparison.",
        "source": "runs/yolo_family_comparison_640_fullval/metrics.csv",
        "kind": "multi_model_corruption",
    },
    {
        "id": "coco_yolov8s_640_recovery",
        "title": "COCO val2017 · YOLOv8s recovery policy · 640 px",
        "dataset": "COCO val2017",
        "dataset_kind": "COCO-80",
        "protocol": "5,000 images; YOLO validation metrics; input size 640; same YOLOv8s checkpoint and GPU experiment setup.",
        "source": "runs/yolov8s_fullval_recovery_policy/metrics.csv",
        "kind": "single_model_recovery",
    },
    {
        "id": "coco_3x500_subsets",
        "title": "COCO val2017 · three 500-image subsets",
        "dataset": "COCO val2017 subsets",
        "dataset_kind": "COCO-80",
        "protocol": "Three image-disjoint 500-image subsets; reported mean and sample standard deviation. Seeded subset manifests are stored under datasets/coco/.",
        "source": "runs/image_subset_analysis/metrics_summary_by_image_subset.csv",
        "kind": "three_subset_aggregate",
    },
    {
        "id": "coco_yolov8_clean_640",
        "title": "COCO val2017 · clean model comparison · 640 px",
        "dataset": "COCO val2017",
        "dataset_kind": "COCO-80",
        "protocol": "5,000 images; clean only; YOLOv8n and YOLOv8s at input size 640; latency measured on the recorded experiment hardware.",
        "source": "runs/yolo_model_size_comparison/yolov8n_vs_yolov8s_fullval.csv",
        "kind": "clean_model_comparison",
    },
    {
        "id": "open_images_100",
        "title": "Open Images V7 · 100-image pilot",
        "dataset": "Open Images V7 validation subset",
        "dataset_kind": "Open Images mapped classes",
        "protocol": "100 images; 193 boxes; 8 mapped classes; Open Images positive/negative and group-of handling. Not a COCO-80 score.",
        "source": "runs/open_images_v7_eval/summary.csv",
        "kind": "external_labeled_subset",
    },
    {
        "id": "open_images_300",
        "title": "Open Images V7 · disjoint 300-image expansion",
        "dataset": "Open Images V7 validation subset",
        "dataset_kind": "Open Images mapped classes",
        "protocol": "300 images; 782 boxes; 7 classes with ground truth (Remote control absent); Open Images positive/negative and group-of handling. Not a COCO-80 score.",
        "source": "runs/open_images_v7_test_extra_eval/summary.csv",
        "kind": "external_labeled_subset",
    },
]

REFERENCE_ROWS = [
    {"model": "YOLOv8n", "map50": 0.658, "tp": 114, "fp": 50, "fn": 79},
    {"model": "YOLOv8s", "map50": 0.654, "tp": 119, "fp": 69, "fn": 74},
    {"model": "SSD-MobileNet", "map50": 0.619, "tp": 105, "fp": 62, "fn": 88},
]

CONDITION_NAMES = {
    "clean": ("Ảnh gốc", "Clean"),
    "gaussian": ("Nhiễu Gaussian", "Gaussian"),
    "salt_pepper": ("Nhiễu muối tiêu", "Salt-and-Pepper"),
    "fog": ("Sương mù", "Fog"),
    "low_light": ("Thiếu sáng", "Low light"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if result == result:  # filter NaN without importing math
                return result
    return None


def _int(row: dict[str, str], *names: str) -> int | None:
    value = _float(row, *names)
    return int(value) if value is not None else None


def _condition(row: dict[str, str]) -> tuple[str, str, float | None]:
    raw = row.get("condition", "clean").strip()
    stage = "raw"
    for suffix in ("_recovered", "_bilateral", "_clahe", "_nlm", "_median"):
        if raw.lower().endswith(suffix):
            raw = raw[:-len(suffix)]
            stage = "restored"
            break
    if raw == "clean":
        stage = "clean"
    family = _family_slug(row.get("family", "") or (raw.split("_", 1)[0] if raw != "clean" else "clean"))
    level = _float(row, "level")
    if family == "clean":
        label = "Ảnh gốc / Clean"
    else:
        vi, en = CONDITION_NAMES.get(family, (family, family))
        label = f"{vi} · {level:g}" if level is not None else vi
        label += " / " + (f"{en} · {level:g}" if level is not None else en)
    return raw, stage, level


def _normalise(row: dict[str, str], exp: dict[str, Any]) -> dict[str, Any]:
    condition, stage, level = _condition(row)
    family = _family_slug(row.get("family") or (condition.split("_", 1)[0] if condition != "clean" else "clean"))
    recovery_method = row.get("recovery_method", "").strip() or _recovery_method_from_condition(row.get("condition", ""))
    if family == "clean":
        condition_label = "Ảnh gốc / Clean"
    else:
        vi, en = CONDITION_NAMES.get(family, (family, family))
        suffix = f" · {level:g}" if level is not None else ""
        if stage == "restored" and recovery_method:
            suffix += f" · {recovery_method}"
        condition_label = f"{vi}{suffix} / {en}{suffix}"
    tp = _int(row, "tp_iou50", "TP", "TP_conf025_iou05")
    fp = _int(row, "fp_iou50", "FP", "FP_conf025_iou05")
    fn = _int(row, "fn_iou50", "FN", "FN_conf025_iou05")
    precision = _float(row, "precision_ultralytics", "precision", "precision_conf025_iou50")
    recall = _float(row, "recall_ultralytics", "recall", "recall_conf025_iou50")
    if precision is None and tp is not None and fp is not None and tp + fp:
        precision = tp / (tp + fp)
    if recall is None and tp is not None and fn is not None and tp + fn:
        recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    model = _canonical_model(row.get("model") or row.get("detector") or row.get("model_id") or "YOLOv8n")
    return {
        "experiment_id": exp["id"],
        "protocol_id": exp["id"],
        "dataset": exp["dataset"],
        "dataset_kind": exp["dataset_kind"],
        "protocol": exp["protocol"],
        "model": model,
        "model_version": "COCO-pretrained checkpoint; exact file identified by model ID in project root",
        "condition": condition,
        "condition_label": condition_label,
        "condition_label_bilingual": condition_label,
        "family": family,
        "severity": level,
        "stage": stage,
        "recovery_method": recovery_method or None,
        "mAP50": _float(row, "mAP50", "mAP50_open_images"),
        "mAP50_95": _float(row, "mAP50_95", "map50_95"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "latency_ms": _float(row, "latency_mean_ms", "inference_ms_per_image"),
        "fps": _float(row, "fps", "single_image_fps_mean"),
        "images": _int(row, "images", "evaluated_images"),
        "ground_truth_boxes": _int(row, "ground_truth_boxes"),
        "classes_scored": _int(row, "classes_scored"),
        "input_size": _int(row, "input_size", "imgsz"),
        "confidence_threshold": _float(row, "fp_conf_threshold", "confusion_conf"),
        "matching_iou": _float(row, "confusion_iou"),
        "metric_implementation": row.get("ap_implementation") or row.get("ap_method") or row.get("note") or None,
        "seed": _int(row, "seed"),
        "derived_fields": [key for key, original in (("precision", "precision_ultralytics"), ("recall", "recall_ultralytics"), ("f1", "")) if not row.get(original)] if exp["kind"] == "external_labeled_subset" else [],
        "source_file": row.get("_source_file") or exp["source"],
        "source_note": exp.get("protocol"),
    }


def _family_slug(value: str) -> str:
    aliases = {
        "gaussian": "gaussian", "salt-and-pepper": "salt_pepper", "salt_pepper": "salt_pepper",
        "fog": "fog", "low-light factor": "low_light", "low_light": "low_light", "clean": "clean",
    }
    return aliases.get(value.strip().lower().replace("†", ""), value.strip().lower())


def _recovery_method_from_condition(condition: str) -> str:
    for suffix, method in (("_bilateral", "bilateral"), ("_clahe", "CLAHE"), ("_nlm", "NLM"), ("_median", "median")):
        if condition.lower().endswith(suffix):
            return method
    return ""


def _canonical_model(value: str) -> str:
    if value.startswith("YOLOv8") and value.rsplit("-", 1)[-1] in {"320", "640"}:
        return value.rsplit("-", 1)[0]
    if value.startswith("SSDLite"):
        return "SSD-MobileNet"
    return value


def _intensity_rows(exp: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    primary_path = ROOT / exp["source"]
    if not primary_path.is_file():
        return records
    base_exp = next(item for item in EXPERIMENTS if item["id"] == "coco_normalized_320")
    base_rows = _read_csv(ROOT / base_exp["source"])
    for row in base_rows:
        if row.get("condition") == "clean" and row.get("detector") != "YOLOv8s-320":
            records.append(_normalise({**row, "_source_file": base_exp["source"]}, exp))
    for row in _read_csv(primary_path):
        normal = {**row, "family": _family_slug(row.get("family", "")), "input_size": "320", "_source_file": exp["source"]}
        records.append(_normalise(normal, exp))
    recovery_path = ROOT / exp["secondary_sources"][1]
    if recovery_path.is_file():
        for row in _read_csv(recovery_path):
            family = _family_slug(row.get("family", ""))
            records.append(_normalise({
                "family": family,
                "condition": row.get("recovered_condition", ""),
                "level": row.get("level", ""),
                "detector": row.get("detector", ""),
                "images": "5000",
                "input_size": "320",
                "mAP50": row.get("recovered_mAP50", ""),
                "mAP50_95": row.get("recovered_mAP50_95", ""),
                "fp_iou50": row.get("recovered_fp", ""),
                "fn_iou50": row.get("recovered_fn", ""),
                "_source_file": exp["secondary_sources"][1],
            }, exp))
    return records


def _subset_rows(row: dict[str, str], exp: dict[str, Any]) -> dict[str, Any]:
    result = _normalise(row, exp)
    for metric in ("mAP50", "mAP50_95", "precision", "recall", "tp", "fp", "fn"):
        source_key = {
            "mAP50": "mAP50_mean", "mAP50_95": "mAP50_95_mean",
            "precision": "precision_ultralytics_mean", "recall": "recall_ultralytics_mean",
            "tp": "tp_iou50_mean", "fp": "fp_iou50_mean", "fn": "fn_iou50_mean",
        }[metric]
        result[metric] = _float(row, source_key)
        std_key = source_key.removesuffix("_mean") + "_std"
        result[f"{metric}_std"] = _float(row, std_key)
    result["f1"] = None
    result["f1_std"] = None
    result["subset_count"] = _int(row, "n_image_subsets")
    result["images"] = 500
    result["model"] = "YOLOv8n"
    result["model_version"] = "COCO-pretrained"
    result["input_size"] = 640
    result["stage"] = "restored" if row.get("recovered", "False").lower() == "true" else "raw"
    if result["family"] == "clean":
        result["stage"] = "clean"
    result["relative_change_vs_clean_pct"] = _float(row, "relative_change_vs_clean_pct_mean")
    result["relative_change_vs_clean_pct_std"] = _float(row, "relative_change_vs_clean_pct_std")
    return result


def _reference_records(exp: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in REFERENCE_ROWS:
        item = _normalise({
            "model": row["model"], "condition": "clean", "mAP50": str(row["map50"]),
            "tp_iou50": str(row["tp"]), "fp_iou50": str(row["fp"]), "fn_iou50": str(row["fn"]),
        }, exp)
        item.update({
            "condition": "unconfirmed",
            "condition_label": "Không xác nhận / Unconfirmed",
            "condition_label_bilingual": "Không xác nhận / Unconfirmed",
            "protocol": "User-provided reference in the supplied specification; condition, image count, input size, and evaluator were not confirmed. Excluded from benchmarks and recommendations.",
            "images": None,
            "ground_truth_boxes": None,
            "input_size": None,
            "confidence_threshold": 0.25,
            "matching_iou": 0.5,
            "metric_implementation": "User-provided, unverified",
            "derived_fields": ["precision", "recall", "f1"],
            "reference_only": True,
        })
        result.append(item)
    return result


def get_experiments() -> list[dict[str, Any]]:
    experiments = []
    for exp in EXPERIMENTS:
        path = ROOT / exp["source"]
        source_rows = _read_csv(path) if path.is_file() else []
        if exp["kind"] == "intensity_sweep":
            measured_rows = _intensity_rows(exp)
        elif exp["kind"] == "three_subset_aggregate":
            measured_rows = [_subset_rows(row, exp) for row in source_rows]
        else:
            measured_rows = [_normalise(row, exp) for row in source_rows]
        source_paths = [path, *(ROOT / item for item in exp.get("secondary_sources", []))]
        is_available = path.is_file() and all(source.is_file() for source in source_paths[1:])
        experiments.append({
            **exp,
            "available": is_available,
            "source_modified": max((source.stat().st_mtime for source in source_paths if source.is_file()), default=None),
            "record_count": len(measured_rows),
            "model_count": len({row["model"] for row in measured_rows}),
            "map50_count": sum(row["mAP50"] is not None for row in measured_rows),
            "map50_95_count": sum(row["mAP50_95"] is not None for row in measured_rows),
            "models": sorted({row["model"] for row in measured_rows}),
            "conditions": sorted({row["condition"] for row in measured_rows}),
            "stages": sorted({row["stage"] for row in measured_rows}),
            "reference_only": False,
        })
    reference = {
        "id": "user_reference_unverified",
        "title": "User-provided reference · protocol unconfirmed",
        "dataset": "Unknown",
        "dataset_kind": "Unverified reference",
        "protocol": "Not eligible for comparisons or recommendations.",
        "source": "Object_Detection_Robustness_Lab_Codex_Spec.docx (table supplied by user)",
        "kind": "reference_only",
        "available": True,
        "source_modified": None,
        "reference_only": True,
    }
    reference_rows = _reference_records(reference)
    reference.update({
        "record_count": len(reference_rows),
        "model_count": 0,
        "map50_count": 0,
        "map50_95_count": 0,
        "models": [],
        "conditions": [],
        "stages": [],
    })
    experiments.append(reference)
    return experiments


def get_records(experiment_id: str | None = None) -> list[dict[str, Any]]:
    selected = [exp for exp in EXPERIMENTS if experiment_id is None or exp["id"] == experiment_id]
    records: list[dict[str, Any]] = []
    for exp in selected:
        path = ROOT / exp["source"]
        if not path.is_file():
            continue
        if exp["kind"] == "intensity_sweep":
            records.extend(_intensity_rows(exp))
            continue
        rows = _read_csv(path)
        if exp["kind"] == "three_subset_aggregate":
            records.extend(_subset_rows(row, exp) for row in rows)
        else:
            records.extend(_normalise(row, exp) for row in rows)
    if experiment_id in (None, "user_reference_unverified"):
        ref = next(item for item in get_experiments() if item["id"] == "user_reference_unverified")
        records.extend(_reference_records(ref))
    return records


def get_classwise_records(*, model: str | None = None, condition: str | None = None,
                          family: str | None = None, stage: str | None = None,
                          recovery_method: str | None = None) -> list[dict[str, Any]]:
    path = ROOT / "runs/normalized_detector_comparison_fullval/classwise/classwise_confusion_metrics.csv"
    if not path.is_file():
        return []
    result = []
    for row in _read_csv(path):
        raw_condition, parsed_stage, _ = _condition(row)
        canonical_model = _canonical_model(row.get("detector", ""))
        if model and canonical_model != model:
            continue
        row_family = _family_slug(row.get("family") or (raw_condition.split("_", 1)[0] if raw_condition != "clean" else "clean"))
        if condition and raw_condition != condition:
            continue
        if family and row_family != family:
            continue
        if stage and parsed_stage != stage:
            continue
        source_condition = row.get("condition", "")
        source_method = _recovery_method_from_condition(source_condition)
        if parsed_stage == "restored" and recovery_method:
            if source_method and source_method.lower() != recovery_method.lower():
                continue
            if not source_method and not source_condition.endswith("_recovered"):
                continue
        result.append({
            "model": canonical_model,
            "condition": raw_condition,
            "source_condition": source_condition,
            "family": row_family,
            "stage": parsed_stage,
            "class_id": _int(row, "class_id"),
            "class_name": row.get("class_name", ""),
            "tp": _int(row, "tp"),
            "fp": _int(row, "fp"),
            "fn": _int(row, "fn"),
            "support": _int(row, "support"),
            "predicted": _int(row, "predicted"),
            "precision": _float(row, "precision_at_conf025_iou05"),
            "recall": _float(row, "recall_at_conf025_iou05"),
            "source_file": "runs/normalized_detector_comparison_fullval/classwise/classwise_confusion_metrics.csv",
        })
    # A canonical condition can have multiple recovery methods in the source file.
    # Do not silently combine those variants when the caller has no method to match.
    if stage == "restored" and not recovery_method and len({row["source_condition"] for row in result}) > 1:
        return []
    return result


def filter_records(records: list[dict[str, Any]], *, model: str | None = None,
                   condition: str | None = None, family: str | None = None,
                   stage: str | None = None) -> list[dict[str, Any]]:
    out = records
    for key, value in (("model", model), ("condition", condition), ("family", family), ("stage", stage)):
        if value is not None:
            out = [row for row in out if str(row.get(key)) == value]
    return out


#!/usr/bin/env python
"""Generate controlled image corruptions and measure YOLO detection robustness.

Input data must use the standard Ultralytics detection layout: a dataset YAML,
an images/val directory, and matching labels/val YOLO text files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from preprocessing import recover_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    level: float
    recovered: bool = False


def conditions() -> list[Condition]:
    out = [Condition("clean", "clean", 0.0)]
    for family, levels in (
        ("gaussian", (0.01, 0.03, 0.06)),
        ("salt_pepper", (0.01, 0.03, 0.06)),
        ("fog", (0.15, 0.30, 0.50)),
        ("low_light", (0.60, 0.40, 0.25)),
    ):
        out.extend(Condition(f"{family}_{level:g}", family, level) for level in levels)
    return out


def load_dataset(data_yaml: Path) -> tuple[dict[str, Any], list[Path]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "val" not in data:
        raise ValueError(f"{data_yaml} must define a 'val' image directory")
    root_value = Path(data.get("path", data_yaml.parent))
    root = root_value if root_value.is_absolute() else (data_yaml.parent / root_value).resolve()
    val_entries = data["val"] if isinstance(data["val"], list) else [data["val"]]
    images: list[Path] = []
    for entry in val_entries:
        path = Path(entry)
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                item = Path(line.strip())
                if not line.strip():
                    continue
                images.append(item if item.is_absolute() else (path.parent / item).resolve())
        else:
            raise FileNotFoundError(f"Validation path does not exist: {path}")
    images = sorted(set(p.resolve() for p in images))
    if not images:
        raise ValueError("No validation images found")
    return data, images


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    image_index = next((i for i, part in enumerate(parts) if part.lower() == "images"), None)
    if image_index is None:
        raise ValueError(f"Cannot infer YOLO label path (expected an 'images' folder): {image}")
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def perturb(image: np.ndarray, condition: Condition, rng: np.random.Generator) -> np.ndarray:
    if condition.family == "clean":
        return image
    x = image.astype(np.float32)
    if condition.family == "gaussian":
        sigma = condition.level * 255.0
        x += rng.normal(0.0, sigma, x.shape).astype(np.float32)
    elif condition.family == "salt_pepper":
        amount = condition.level
        mask = rng.random(x.shape[:2])
        x[mask < amount / 2] = 0
        x[mask > 1 - amount / 2] = 255
    elif condition.family == "fog":
        # Uniform atmospheric veil; alpha controls fog density.
        alpha = condition.level
        x = x * (1.0 - alpha) + 235.0 * alpha
    elif condition.family == "low_light":
        # Scale brightness, then apply a mild darkening gamma.
        factor = condition.level
        normalized = np.clip(x / 255.0, 0, 1)
        x = 255.0 * np.power(normalized * factor, 1.15)
    else:
        raise ValueError(f"Unknown corruption: {condition.family}")
    return np.clip(x, 0, 255).astype(np.uint8)


def recover(image: np.ndarray, source_family: str, method: str = "auto") -> np.ndarray:
    if method == "none":
        return image
    return recover_image(image, source_family, method)


def make_dataset(
    images: list[Path], source_yaml: dict[str, Any], data_yaml_path: Path,
    out_dir: Path, condition: Condition, seed: int, preprocess: str,
) -> Path:
    image_root = out_dir / "images" / "val"
    label_root = out_dir / "labels" / "val"
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for index, src in enumerate(images):
        label = label_for_image(src)
        if not label.exists():
            raise FileNotFoundError(f"Missing YOLO label file for {src}: expected {label}")
        # Keep the validation images flat: Ultralytics' prediction directory loader
        # does not recurse consistently across versions. The index avoids collisions.
        rel = Path(f"{index:06d}_{src.name}")
        dst_image = image_root / rel
        dst_label = (label_root / rel).with_suffix(".txt")
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {src}")
        image = perturb(image, condition, rng)
        if condition.recovered:
            image = recover(image, condition.family, preprocess)
        if not cv2.imwrite(str(dst_image), image):
            raise OSError(f"Could not write generated image: {dst_image}")
        shutil.copy2(label, dst_label)
        records.append({"source": str(src), "generated": str(dst_image), "label": str(dst_label)})
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        # Ultralytics requires train and val keys even for validation-only runs.
        "train": "images/val",
        "val": "images/val",
        "names": source_yaml.get("names", {}),
    }
    if "nc" in source_yaml:
        dataset_yaml["nc"] = source_yaml["nc"]
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"condition": condition.__dict__, "images": records}, f, indent=2)
    return out_dir / "data.yaml"


def read_yolo_labels(path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, xc, yc, w, h = map(float, values[:5])
        x1, y1 = (xc - w / 2) * width, (yc - h / 2) * height
        x2, y2 = (xc + w / 2) * width, (yc + h / 2) * height
        result.append((int(cls), np.array([x1, y1, x2, y2], dtype=float)))
    return result


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def evaluate(model: Any, data_yaml: Path, dataset_dir: Path, condition: Condition,
             out_dir: Path, imgsz: int, batch: int, device: str | None,
             conf: float, fp_conf: float, iou_threshold: float, classes: int,
             class_names: list[str], reuse_confusion: bool = False) -> dict[str, Any]:
    val_args = dict(data=str(data_yaml), imgsz=imgsz, batch=batch, conf=conf,
                    iou=0.7, plots=True, project=str(out_dir / "ultralytics"),
                    name=condition.name, exist_ok=True, verbose=False)
    if device:
        val_args["device"] = device
    metrics = model.val(**val_args)
    box_metrics = getattr(metrics, "box", None)
    map50 = float(getattr(box_metrics, "map50", float("nan")))
    map5095 = float(getattr(box_metrics, "map", float("nan")))
    precision = float(getattr(box_metrics, "mp", float("nan")))
    recall = float(getattr(box_metrics, "mr", float("nan")))

    matrix_path = out_dir / f"confusion_{condition.name}.csv"
    matrix_image_path = out_dir / f"confusion_{condition.name}.png"
    predictions = None
    if reuse_confusion and matrix_path.exists():
        matrix = np.loadtxt(matrix_path, delimiter=",", dtype=np.int64)
    else:
        predict_args = dict(source=str(dataset_dir / "images" / "val"), imgsz=imgsz,
                            batch=batch, conf=min(conf, fp_conf), iou=0.7,
                            stream=False, verbose=False)
        if device:
            predict_args["device"] = device
        predictions = model.predict(**predict_args)
        matrix = np.zeros((classes + 1, classes + 1), dtype=np.int64)
        for pred in predictions:
            pred_path = Path(pred.path)
            rel = pred_path.relative_to(dataset_dir / "images" / "val")
            gt_path = (dataset_dir / "labels" / "val" / rel).with_suffix(".txt")
            image = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            gt = read_yolo_labels(gt_path, width, height)
            boxes = pred.boxes
            pboxes = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) else np.empty((0, 4))
            pclasses = boxes.cls.cpu().numpy().astype(int) if boxes is not None and len(boxes) else np.empty((0,), dtype=int)
            scores = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else np.empty((0,))
            keep = scores >= fp_conf
            pboxes, pclasses, scores = pboxes[keep], pclasses[keep], scores[keep]
            candidates = []
            for gi, (gcls, gbox) in enumerate(gt):
                for pi, (pcls, pbox) in enumerate(zip(pclasses, pboxes)):
                    overlap = box_iou(gbox, pbox)
                    if overlap >= iou_threshold:
                        candidates.append((overlap, float(scores[pi]), gi, pi))
            candidates.sort(reverse=True)
            matched_gt, matched_pred = set(), set()
            for _, _, gi, pi in candidates:
                if gi in matched_gt or pi in matched_pred:
                    continue
                matched_gt.add(gi); matched_pred.add(pi)
                gcls, _ = gt[gi]; pcls = int(pclasses[pi])
                if 0 <= gcls < classes and 0 <= pcls < classes:
                    matrix[gcls, pcls] += 1
            for gi, (gcls, _) in enumerate(gt):
                if gi not in matched_gt and 0 <= gcls < classes:
                    matrix[gcls, classes] += 1
            for pi, pcls in enumerate(pclasses):
                if pi not in matched_pred and 0 <= pcls < classes:
                    matrix[classes, pcls] += 1
        np.savetxt(matrix_path, matrix, delimiter=",", fmt="%d")
    tp = int(np.trace(matrix[:classes, :classes]))
    misclassified = int(matrix[:classes, :classes].sum()) - tp
    fp = int(matrix[classes, :classes].sum()) + misclassified
    fn = int(matrix[:classes, classes].sum()) + misclassified
    if not matrix_image_path.exists():
        labels = class_names + ["background"]
        fig, ax = plt.subplots(figsize=(max(6, classes * 0.55), max(5, classes * 0.5)))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(classes + 1), labels=labels, rotation=90)
        ax.set_yticks(range(classes + 1), labels=labels)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class")
        ax.set_title(f"{condition.name}: confusion matrix (IoU≥{iou_threshold:g}, conf≥{fp_conf:g})")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(matrix_image_path, dpi=150); plt.close(fig)
    denom = tp + fp
    return {
        "condition": condition.name, "family": condition.family, "level": condition.level,
        "recovered": condition.recovered, "mAP50": map50, "mAP50_95": map5095,
        "precision_ultralytics": precision, "recall_ultralytics": recall,
        "tp_iou50": tp, "fp_iou50": fp, "fn_iou50": fn,
        "precision_iou50": tp / denom if denom else 0.0,
        "fp_conf_threshold": fp_conf,
        "images": len(predictions) if predictions is not None else len(list((dataset_dir / "images" / "val").glob("*"))),
    }


def save_report(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = list(rows[0])
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    families = sorted({row["family"] for row in rows if row["family"] != "clean"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, family in zip(axes.flat, families):
        for recovered, label, marker in ((False, "Raw", "o"), (True, "Recovered", "s")):
            subset = sorted((r for r in rows if r["family"] == family and r["recovered"] == recovered), key=lambda r: r["level"])
            if subset:
                ax.plot([r["level"] for r in subset], [r["mAP50"] for r in subset], marker=marker, label=label)
        ax.set_title(family.replace("_", " ").title())
        ax.set_xlabel("Intensity (see README for definition)")
        ax.set_ylabel("mAP@0.5")
        ax.grid(alpha=0.25); ax.legend()
    fig.suptitle("Object detection robustness under synthetic distortions")
    fig.savefig(out_dir / "map_vs_intensity.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, family in zip(axes.flat, families):
        for recovered, label, marker in ((False, "Raw", "o"), (True, "Recovered", "s")):
            subset = sorted((r for r in rows if r["family"] == family and r["recovered"] == recovered), key=lambda r: r["level"])
            if subset:
                ax.plot([r["level"] for r in subset], [r["fp_iou50"] for r in subset], marker=marker, label=label)
        ax.set_title(family.replace("_", " ").title())
        ax.set_xlabel("Intensity (see README for definition)")
        ax.set_ylabel("False positives / validation set")
        ax.grid(alpha=0.25); ax.legend()
    fig.suptitle("False positive counts by distortion intensity")
    fig.savefig(out_dir / "false_positives_vs_intensity.png", dpi=180)
    plt.close(fig)


def write_partial_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_partial_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    integer_fields = {"tp_iou50", "fp_iou50", "fn_iou50", "images"}
    float_fields = {"level", "mAP50", "mAP50_95", "precision_ultralytics",
                    "recall_ultralytics", "precision_iou50", "fp_conf_threshold"}
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for key in integer_fields:
                row[key] = int(row[key])
            for key in float_fields:
                row[key] = float(row[key])
            row["recovered"] = row["recovered"].lower() == "true"
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--model", required=True, help="YOLO checkpoint (.pt), e.g. yolov8n.pt")
    parser.add_argument("--out", type=Path, default=Path("robustness_results"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu")
    parser.add_argument("--conf", type=float, default=0.001, help="Low confidence for mAP and FP analysis")
    parser.add_argument("--fp-conf", type=float, default=0.25, help="Confidence threshold for FP/FN confusion matrices")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess", choices=("auto", "median", "nlmeans", "clahe", "bilateral"), default="auto")
    parser.add_argument("--recovery-tag", help="Optional suffix for recovered condition names, e.g. bilateral")
    parser.add_argument("--no-recovery", action="store_true", help="Skip the recovered-image comparison")
    parser.add_argument("--families", nargs="+",
                        choices=("clean", "gaussian", "salt_pepper", "fog", "low_light"),
                        help="Run only selected condition families")
    parser.add_argument("--levels", nargs="+", type=float,
                        help="Run only selected distortion levels within the chosen families")
    parser.add_argument("--recovered-only", action="store_true",
                        help="Evaluate only recovered variants (use with --families)")
    parser.add_argument("--only-clean", action="store_true", help="Run only the clean baseline condition")
    parser.add_argument("--resume", action="store_true", help="Resume from partial metrics and reuse saved confusion matrices")
    args = parser.parse_args()

    if args.recovered_only and args.no_recovery:
        parser.error("--recovered-only cannot be combined with --no-recovery")
    if args.recovery_tag and not args.recovered_only:
        parser.error("--recovery-tag requires --recovered-only")
    if args.recovery_tag and not args.recovery_tag.replace("_", "").isalnum():
        parser.error("--recovery-tag may contain only letters, numbers, and underscores")
    if args.recovered_only and (not args.families or "clean" in args.families):
        parser.error("--recovered-only requires --families without clean")
    if args.only_clean and args.families:
        parser.error("--only-clean cannot be combined with --families")
    if args.levels is not None and not args.families:
        parser.error("--levels requires --families")
    if args.levels is not None and args.only_clean:
        parser.error("--levels cannot be combined with --only-clean")

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    source_yaml = args.data.resolve()
    data, images = load_dataset(source_yaml)
    classes = len(data["names"]) if isinstance(data.get("names"), (dict, list)) else int(data.get("nc", 0))
    if classes <= 0:
        raise ValueError("Dataset YAML must define class names ('names') or 'nc'")
    raw_names = data.get("names", [str(i) for i in range(classes)])
    class_names = ([str(raw_names.get(i, raw_names.get(str(i), i))) for i in range(classes)]
                   if isinstance(raw_names, dict) else [str(x) for x in raw_names])
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc
    model = YOLO(args.model)
    specs = [Condition("clean", "clean", 0.0)] if args.only_clean else conditions()
    if args.families:
        specs = [spec for spec in specs if spec.family in args.families]
    if args.levels is not None:
        selected_levels = {round(level, 6) for level in args.levels}
        specs = [spec for spec in specs if round(spec.level, 6) in selected_levels]
    partial_path = args.out / "metrics.partial.csv"
    rows = read_partial_rows(partial_path) if args.resume else []
    completed = {row["condition"] for row in rows}
    for original in specs:
        variants = [] if args.recovered_only else [original]
        if original.family != "clean" and not args.no_recovery:
            suffix = args.recovery_tag or "recovered"
            variants.append(Condition(original.name + "_" + suffix, original.family,
                                      original.level, recovered=True))
        for condition in variants:
            if condition.name in completed:
                print(f"Skipping completed condition {condition.name}", flush=True)
                continue
            dataset_dir = args.out / "datasets" / condition.name
            image_dir = dataset_dir / "images" / "val"
            if args.resume and len(list(image_dir.glob("*"))) == len(images) and (dataset_dir / "data.yaml").exists():
                dataset_yaml = dataset_dir / "data.yaml"
            else:
                dataset_yaml = make_dataset(images, data, source_yaml, dataset_dir,
                                            condition, args.seed, args.preprocess)
            print(f"Evaluating {condition.name} ({len(images)} images)", flush=True)
            row = evaluate(model, dataset_yaml, dataset_dir, condition, args.out,
                           args.imgsz, args.batch, args.device, args.conf, args.fp_conf,
                           args.match_iou, classes, class_names, reuse_confusion=args.resume)
            rows.append(row)
            completed.add(condition.name)
            write_partial_rows(rows, partial_path)
    save_report(rows, args.out)
    print(f"Saved {len(rows)} result rows to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()

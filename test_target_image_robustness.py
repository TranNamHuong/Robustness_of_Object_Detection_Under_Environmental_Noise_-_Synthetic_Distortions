"""Qualitative clean/noise/recovery comparison for unlabeled target images."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from preprocessing import recover_image
from robustness_eval import Condition, perturb


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCENARIOS = [
    ("clean", None, 0.0, "none"),
    ("gaussian_0.06_raw", "gaussian", 0.06, "none"),
    ("gaussian_0.06_bilateral", "gaussian", 0.06, "bilateral"),
    ("salt_pepper_0.06_raw", "salt_pepper", 0.06, "none"),
    ("salt_pepper_0.06_median", "salt_pepper", 0.06, "median"),
    ("fog_0.50_raw", "fog", 0.50, "none"),
    ("fog_0.50_clahe", "fog", 0.50, "clahe"),
    ("low_light_0.25_raw", "low_light", 0.25, "none"),
    ("low_light_0.25_clahe", "low_light", 0.25, "clahe"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "target_eval/images/val")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/target_image_robustness")
    parser.add_argument("--model", type=Path, default=ROOT / "yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"No supported images found in: {source}")
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(args.model.resolve()))
    rows = []
    previews = out / "previews"
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        relative = image_path.relative_to(source)
        condition_images: dict[str, np.ndarray] = {"clean": image}
        # Use the same seeded corrupted image for raw and recovered comparisons.
        for family, level in (("gaussian", 0.06), ("salt_pepper", 0.06), ("fog", 0.50), ("low_light", 0.25)):
            corrupted = perturb(image, Condition(f"{family}_{level:g}", family, level), np.random.default_rng(args.seed))
            condition_images[f"{family}_{level:.2f}_raw"] = corrupted
            recovery = {"gaussian": "bilateral", "salt_pepper": "median", "fog": "clahe", "low_light": "clahe"}[family]
            condition_images[f"{family}_{level:.2f}_{recovery}"] = recover_image(corrupted, family, recovery)
        preview_dir = previews / relative.parent / image_path.stem
        preview_dir.mkdir(parents=True, exist_ok=True)
        for scenario, family, level, recovery in SCENARIOS:
            if scenario == "clean":
                input_image = image
            elif scenario.endswith("_raw"):
                input_image = condition_images[scenario]
            else:
                input_image = condition_images[scenario]
            result = model.predict(source=input_image, imgsz=args.imgsz, conf=args.conf,
                                   iou=0.7, device=device, verbose=False)[0]
            ids = result.boxes.cls.int().cpu().tolist() if result.boxes is not None else []
            scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
            counts = Counter(str(model.names[class_id]) for class_id in ids)
            rows.append({
                "image": str(relative), "condition": scenario,
                "detections": len(ids), "mean_confidence": float(np.mean(scores)) if scores else 0.0,
                "class_counts": json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
                "device": device, "conf": args.conf, "imgsz": args.imgsz,
                "warning": "Unlabeled qualitative comparison; counts/confidence are not accuracy or mAP.",
            })
            preview_path = preview_dir / f"{scenario}.jpg"
            if not cv2.imwrite(str(preview_path), result.plot()):
                raise OSError(f"Could not save preview: {preview_path}")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    print(f"Processed {len(images)} images across {len(SCENARIOS)} scenarios on {device}.")
    print(f"CSV: {out / 'metrics.csv'}")
    print(f"Previews: {previews}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Evaluate a YOLO checkpoint on the existing full COCO val corruption sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
CONDITIONS = [
    "clean",
    "gaussian_0.01", "gaussian_0.03", "gaussian_0.06",
    "salt_pepper_0.01", "salt_pepper_0.03", "salt_pepper_0.06",
    "fog_0.15", "fog_0.3", "fog_0.5",
    "low_light_0.6", "low_light_0.4", "low_light_0.25",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "runs/full_val2017_raw_seed42/datasets")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model_path = args.model.resolve()
    data_root = args.data_root.resolve()
    out_dir = args.out.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    for condition in CONDITIONS:
        if not (data_root / condition / "data.yaml").is_file():
            raise FileNotFoundError(f"Missing full-val condition: {data_root / condition}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    rows = []
    csv_path = out_dir / "metrics.csv"
    for condition in CONDITIONS:
        data_yaml = data_root / condition / "data.yaml"
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        n_images = len(list((Path(data["path"]) / data["val"]).glob("*")))
        print(f"Evaluating {condition} ({n_images} images)", flush=True)
        result = model.val(
            data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device,
            conf=0.001, iou=0.7, plots=False, verbose=False,
            project=str(out_dir / "ultralytics"), name=condition, exist_ok=True,
        )
        rows.append({
            "condition": condition,
            "images": n_images,
            "mAP50": float(result.box.map50),
            "mAP50_95": float(result.box.map),
            "precision": float(result.box.mp),
            "recall": float(result.box.mr),
        })
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{condition}: mAP50={rows[-1]['mAP50']:.4f}, "
              f"mAP50-95={rows[-1]['mAP50_95']:.4f}", flush=True)
    print(f"Saved {len(rows)} conditions to {csv_path}", flush=True)


if __name__ == "__main__":
    main()

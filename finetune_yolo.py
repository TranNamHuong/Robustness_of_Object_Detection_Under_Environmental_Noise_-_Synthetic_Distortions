#!/usr/bin/env python
"""Fine-tune a COCO-pretrained Ultralytics detector on the prepared COCO-80 subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="COCO-pretrained YOLO weights, e.g. yolov8n.pt")
    parser.add_argument("--name", required=True, help="Run name, e.g. yolov8n_coco80_1k")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/coco/train_subset_1000/data.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience in epochs")
    args = parser.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if len(data.get("names", {})) != 80:
        raise ValueError("Fine-tuning requires the same 80 COCO classes as the baseline")
    weights = Path(args.model)
    if not weights.is_absolute():
        weights = (ROOT / weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {weights}")
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0:
        parser.error("epochs, batch and imgsz must be positive")

    model = YOLO(str(weights))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs/finetune_coco80"),
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        patience=args.patience,
        seed=42,
        deterministic=True,
        amp=True,
        cache=False,
        close_mosaic=3,
        plots=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

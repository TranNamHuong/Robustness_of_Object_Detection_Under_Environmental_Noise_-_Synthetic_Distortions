"""Create YOLO pseudo-label drafts and annotated previews for human review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory of images to pre-annotate")
    parser.add_argument("--out", type=Path, required=True, help="Separate output directory for draft labels/previews")
    parser.add_argument("--model", type=Path, default=ROOT / "yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="0 for CUDA GPU or cpu; default auto-selects")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum image count for a quick preview")
    args = parser.parse_args()

    source = args.source.resolve()
    out = args.out.resolve()
    model_path = args.model.resolve()
    if not source.is_dir():
        parser.error(f"Image directory does not exist: {source}")
    if not model_path.is_file():
        parser.error(f"Model checkpoint does not exist: {model_path}")
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"No supported images found in: {source}")
    if args.limit > 0:
        images = images[:args.limit]

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(model_path))
    labels_root = out / "draft_labels"
    previews_root = out / "previews"
    metadata = {
        "source": str(source),
        "checkpoint": str(model_path),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": device,
        "warning": "Pseudo-labels from model predictions. Review/correct every box and class before using as ground truth.",
        "classes": model.names,
        "images": [],
    }
    for image_path in images:
        relative = image_path.relative_to(source)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        result = model.predict(source=image, imgsz=args.imgsz, conf=args.conf,
                               device=device, verbose=False)[0]
        label_path = (labels_root / relative).with_suffix(".txt")
        preview_path = previews_root / relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        detections = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xywhn.detach().cpu().tolist()
            class_ids = result.boxes.cls.detach().cpu().tolist()
            confidences = result.boxes.conf.detach().cpu().tolist()
            for class_id, box, confidence in zip(class_ids, boxes, confidences):
                class_id = int(class_id)
                values = " ".join(f"{float(value):.6f}" for value in box)
                rows.append(f"{class_id} {values}")
                detections.append({
                    "class_id": class_id,
                    "class_name": str(model.names[class_id]),
                    "confidence": float(confidence),
                    "xywh_normalized": [float(value) for value in box],
                })
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        if not cv2.imwrite(str(preview_path), result.plot()):
            raise OSError(f"Could not save preview: {preview_path}")
        metadata["images"].append({
            "image": str(relative),
            "draft_label": str(label_path.relative_to(out)),
            "preview": str(preview_path.relative_to(out)),
            "detections": detections,
        })

    out.mkdir(parents=True, exist_ok=True)
    (out / "draft_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created drafts for {len(images)} images in {out}")
    print("Review every preview/box/class and keep these labels separate from evaluation ground truth until verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

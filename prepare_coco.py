#!/usr/bin/env python
"""Create a reproducible COCO val subset in Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="COCO val2017 image folder")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="instances_val2017.json")
    parser.add_argument("--out", type=Path, default=Path("data/coco_subset"))
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="Manifest(s) whose image IDs must be excluded from sampling")
    args = parser.parse_args()

    images_dir = args.images.resolve()
    annotations_file = args.annotations.resolve()
    out_dir = args.out.resolve()
    if args.num_images <= 0:
        parser.error("--num-images must be positive")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO image folder not found: {images_dir}")
    if not annotations_file.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_file}")

    coco: dict[str, Any] = json.loads(annotations_file.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
    excluded_ids: set[int] = set()
    for manifest_path in args.exclude_manifest:
        manifest = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
        excluded_ids.update(int(row["image_id"]) for row in manifest.get("images", []))
    candidates = [image for image in images if int(image["id"]) not in excluded_ids]
    if args.num_images > len(candidates):
        parser.error(f"Requested {args.num_images} images, but only {len(candidates)} remain after exclusions")
    if not categories:
        raise ValueError("COCO annotation JSON has no categories")

    category_to_yolo = {category["id"]: index for index, category in enumerate(categories)}
    category_names = {category["id"]: category["name"] for category in categories}
    annotation_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        annotation_by_image.setdefault(annotation["image_id"], []).append(annotation)

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(candidates, args.num_images), key=lambda item: item["id"])
    image_out = out_dir / "images" / "val"
    label_out = out_dir / "labels" / "val"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    records = []

    for image in selected:
        image_id = image["id"]
        filename = Path(image["file_name"]).name
        source = images_dir / filename
        destination = image_out / filename
        if not source.is_file():
            raise FileNotFoundError(f"COCO image missing: {source}")
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        label_path = (label_out / Path(filename)).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        present_classes = set()
        width, height = float(image["width"]), float(image["height"])
        for annotation in annotation_by_image.get(image_id, []):
            category_id = annotation["category_id"]
            if category_id not in category_to_yolo:
                continue
            x, y, box_width, box_height = map(float, annotation["bbox"])
            # Clip malformed/out-of-frame boxes to image bounds.
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(width, x + box_width), min(height, y + box_height)
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            lines.append(f"{category_to_yolo[category_id]} {xc:.6f} {yc:.6f} "
                         f"{box_width / width:.6f} {box_height / height:.6f}")
            category_name = str(category_names[category_id])
            instance_counts[category_name] += 1
            present_classes.add(category_name)
        class_counts.update(present_classes)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        records.append({"image_id": image_id, "file_name": filename,
                        "annotations": len(lines), "classes": sorted(present_classes)})

    dataset_yaml = {
        "path": str(out_dir),
        # Required by Ultralytics' dataset validator; this project only calls model.val().
        "train": "images/val",
        "val": "images/val",
        "names": {index: category["name"] for index, category in enumerate(categories)},
    }
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "source": "COCO val2017",
        "selection": "uniform random sample from COCO val2017 images after manifest exclusions",
        "seed": args.seed,
        "excluded_manifests": [str(path.resolve()) for path in args.exclude_manifest],
        "excluded_images": len(excluded_ids),
        "requested_images": args.num_images,
        "selected_images": len(records),
        "category_id_to_yolo_id": {str(key): value for key, value in category_to_yolo.items()},
        "images": records,
        "images_per_class": dict(sorted(class_counts.items())),
        "instances_per_class": dict(sorted(instance_counts.items())),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} images at {out_dir}")
    print(f"Dataset YAML: {out_dir / 'data.yaml'}")
if __name__ == "__main__":
    main()

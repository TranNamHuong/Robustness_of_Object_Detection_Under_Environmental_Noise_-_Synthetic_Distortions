#!/usr/bin/env python
"""Prepare a compact, fully annotated COCO-80 train subset for fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


def download_image(image: dict, image_dir: Path, retries: int = 3) -> tuple[str, str | None]:
    filename = Path(image["file_name"]).name
    destination = image_dir / filename
    if destination.is_file() and destination.stat().st_size > 0:
        return filename, None

    # COCO's official image endpoint currently serves the per-image URL over HTTP.
    url = f"http://images.cocodataset.org/train2017/{filename}"
    request = Request(url, headers={"User-Agent": "COCO-80-subset-preparer/1.0"})
    last_error = "unknown download error"
    for attempt in range(retries):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if temporary.stat().st_size == 0:
                raise OSError("downloaded an empty file")
            temporary.replace(destination)
            return filename, None
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = str(error)
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    return filename, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=Path("datasets/coco/annotations/instances_train2017.json"))
    parser.add_argument("--out", type=Path, default=Path("datasets/coco/train_subset_1000"))
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-images", type=Path,
                        default=Path("datasets/coco/full_val2017/images/val"))
    args = parser.parse_args()

    annotation_path = args.annotations.resolve()
    output_dir = args.out.resolve()
    eval_images = args.eval_images.resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO train annotations not found: {annotation_path}")
    if not eval_images.is_dir():
        raise FileNotFoundError(f"Existing COCO validation images not found: {eval_images}")
    if args.num_images <= 0 or args.workers <= 0:
        parser.error("--num-images and --workers must be positive")

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = coco["images"]
    categories = sorted(coco["categories"], key=lambda item: item["id"])
    category_to_yolo = {category["id"]: index for index, category in enumerate(categories)}
    category_names = {category["id"]: category["name"] for category in categories}
    image_by_id = {image["id"]: image for image in images}
    annotation_by_image: dict[int, list[dict]] = defaultdict(list)
    image_ids_by_category: dict[int, set[int]] = defaultdict(set)
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        image_id = annotation["image_id"]
        annotation_by_image[image_id].append(annotation)
        image_ids_by_category[annotation["category_id"]].add(image_id)

    if args.num_images > len(images):
        parser.error(f"Requested {args.num_images} images but COCO train has only {len(images)}")

    # Round-robin over rare-to-common categories to improve coverage while keeping
    # every selected image's complete COCO annotation set.
    rng = random.Random(args.seed)
    class_queues: list[list[int]] = []
    for category in categories:
        queue = list(image_ids_by_category.get(category["id"], set()))
        rng.shuffle(queue)
        class_queues.append(queue)
    class_queues.sort(key=len)
    selected_ids: list[int] = []
    selected_set: set[int] = set()
    cursors = [0] * len(class_queues)
    while len(selected_ids) < args.num_images:
        added_this_round = 0
        for queue_index, queue in enumerate(class_queues):
            while cursors[queue_index] < len(queue):
                image_id = queue[cursors[queue_index]]
                cursors[queue_index] += 1
                if image_id not in selected_set:
                    selected_set.add(image_id)
                    selected_ids.append(image_id)
                    added_this_round += 1
                    break
            if len(selected_ids) >= args.num_images:
                break
        if added_this_round == 0:
            break
    if len(selected_ids) < args.num_images:
        remaining = [image["id"] for image in images if image["id"] not in selected_set]
        rng.shuffle(remaining)
        selected_ids.extend(remaining[:args.num_images - len(selected_ids)])
    selected = [image_by_id[image_id] for image_id in selected_ids]

    image_dir = output_dir / "images" / "train"
    label_dir = output_dir / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_image, image, image_dir) for image in selected]
        for index, future in enumerate(as_completed(futures), start=1):
            filename, error = future.result()
            if error:
                errors.append((filename, error))
            if index % 100 == 0 or index == len(futures):
                print(f"Downloaded {index}/{len(futures)} images; failures: {len(errors)}")
    if errors:
        details = "\n".join(f"{name}: {error}" for name, error in errors[:20])
        raise RuntimeError(f"{len(errors)} image downloads failed:\n{details}")

    instance_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    manifest_images = []
    for image in selected:
        image_id = image["id"]
        filename = Path(image["file_name"]).name
        width, height = float(image["width"]), float(image["height"])
        label_lines = []
        present_classes = set()
        for annotation in annotation_by_image.get(image_id, []):
            category_id = annotation["category_id"]
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(width, x + box_width), min(height, y + box_height)
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            label_lines.append(
                f"{category_to_yolo[category_id]} {xc:.6f} {yc:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}"
            )
            name = category_names[category_id]
            instance_counts[name] += 1
            present_classes.add(name)
        image_counts.update(present_classes)
        (label_dir / Path(filename).with_suffix(".txt").name).write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        manifest_images.append({
            "image_id": image_id,
            "file_name": filename,
            "annotations": len(label_lines),
            "classes": sorted(present_classes),
        })

    dataset_yaml = {
        "path": str(output_dir).replace("\\", "/"),
        "train": "images/train",
        "val": str(eval_images).replace("\\", "/"),
        "names": {index: category["name"] for index, category in enumerate(categories)},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "source": "COCO Train2017",
        "source_annotations": str(annotation_path),
        "selection": "seeded rare-category-first round-robin; all non-crowd COCO boxes retained",
        "seed": args.seed,
        "requested_images": args.num_images,
        "selected_images": len(manifest_images),
        "class_count": len(categories),
        "category_id_to_yolo_id": {str(key): value for key, value in category_to_yolo.items()},
        "images": manifest_images,
        "images_per_class": dict(sorted(image_counts.items())),
        "instances_per_class": dict(sorted(instance_counts.items())),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest_images)} images across {len(categories)} classes at {output_dir}")
    print(f"YAML: {output_dir / 'data.yaml'}")
    print(f"Classes represented: {len(image_counts)}/{len(categories)}")


if __name__ == "__main__":
    main()

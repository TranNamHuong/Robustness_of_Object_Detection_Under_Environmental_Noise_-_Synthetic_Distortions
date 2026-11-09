"""Prepare a small, reproducible Open Images V7 validation subset.

This keeps the original Open Images boxes and class names. It intentionally
does not convert them to YOLO labels because Open Images and COCO use
different annotation/evaluation semantics.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


PROJECT_CLASSES = {
    "Bed": "bed",
    "Chair": "chair",
    "Clock": "clock",
    "Couch": "couch",
    "Mobile phone": "cell phone",
    "Person": "person",
    "Remote control": "remote",
    "Teddy bear": "teddy bear",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("datasets/open_images_v7")
    )
    parser.add_argument(
        "--source-root", type=Path,
        help="Directory containing the official Open Images source CSVs (defaults to ROOT/source).",
    )
    parser.add_argument(
        "--exclude-ids-file", type=Path,
        help="Optional image_ids.txt whose IDs must not appear in this subset.",
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample", action="store_true", help="Choose a new subset instead of reusing image_ids.txt")
    args = parser.parse_args()

    source = args.source_root or (args.root / "source")
    class_file = source / "oidv7-class-descriptions-boxable.csv"
    boxes_file = source / "validation-annotations-bbox.csv"
    if not class_file.exists() or not boxes_file.exists():
        raise SystemExit(
            "Download the official Open Images class list and validation "
            "box CSV into datasets/open_images_v7/source first."
        )

    id_to_name: dict[str, str] = {}
    with class_file.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                id_to_name[row[0]] = row[1]
    name_to_id = {name: mid for mid, name in id_to_name.items()}
    missing = sorted(set(PROJECT_CLASSES) - name_to_id.keys())
    if missing:
        raise SystemExit(f"Open Images is missing expected classes: {missing}")
    selected_class_ids = {name_to_id[name] for name in PROJECT_CLASSES}

    rows: list[dict[str, str]] = []
    image_ids_by_class: dict[str, set[str]] = defaultdict(set)
    with boxes_file.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] in selected_class_ids:
                rows.append(row)
                image_ids_by_class[row["LabelName"]].add(row["ImageID"])

    rng = random.Random(args.seed)
    shuffled_by_class: dict[str, list[str]] = {}
    # Sort before shuffling so the seed is reproducible across Python processes.
    for class_id in sorted(selected_class_ids):
        ids = sorted(image_ids_by_class[class_id])
        rng.shuffle(ids)
        shuffled_by_class[class_id] = ids

    excluded: set[str] = set()
    if args.exclude_ids_file:
        excluded = {
            line.strip().split("/", 1)[-1]
            for line in args.exclude_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for class_id in shuffled_by_class:
            shuffled_by_class[class_id] = [
                image_id for image_id in shuffled_by_class[class_id]
                if image_id not in excluded
            ]

    image_dir = args.root / "images" / "val"
    existing_image_ids = {path.stem for path in image_dir.glob("*.jpg")}
    saved_ids_file = args.root / "image_ids.txt"
    if saved_ids_file.exists() and not args.resample:
        chosen = {
            line.strip().split("/", 1)[-1]
            for line in saved_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    elif existing_image_ids and not args.resample:
        # Keep annotations synchronized with images already downloaded.
        chosen = existing_image_ids
    else:
        chosen: set[str] = set()
        cursors = {class_id: 0 for class_id in shuffled_by_class}
        class_order = list(shuffled_by_class)
        rng.shuffle(class_order)
        while len(chosen) < args.max_images:
            added = False
            for class_id in class_order:
                ids = shuffled_by_class[class_id]
                while cursors[class_id] < len(ids):
                    image_id = ids[cursors[class_id]]
                    cursors[class_id] += 1
                    if image_id not in chosen:
                        chosen.add(image_id)
                        added = True
                        break
                if len(chosen) >= args.max_images:
                    break
            if not added:
                break

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "image_ids.txt").write_text(
        "".join(f"validation/{image_id}\n" for image_id in sorted(chosen)),
        encoding="utf-8",
    )
    selected_rows = [row for row in rows if row["ImageID"] in chosen]
    with (args.root / "selected_annotations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=selected_rows[0].keys())
        writer.writeheader()
        writer.writerows(selected_rows)

    with (args.root / "class_mapping.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["open_images_class", "project_coco_class"])
        writer.writerows(PROJECT_CLASSES.items())

    metadata_file = source / "validation-images-with-rotation.csv"
    if metadata_file.exists():
        metadata: dict[str, dict[str, str]] = {}
        with metadata_file.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["ImageID"] in chosen:
                    metadata[row["ImageID"]] = row
        with (args.root / "subset_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            fields = [
                "ImageID",
                "local_image",
                "mapped_classes",
                "License",
                "Author",
                "OriginalLandingURL",
                "OriginalURL",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for image_id in sorted(chosen):
                source_row = metadata.get(image_id, {})
                present_names = sorted(
                    {
                        id_to_name[row["LabelName"]]
                        for row in selected_rows
                        if row["ImageID"] == image_id
                    }
                )
                writer.writerow(
                    {
                        "ImageID": image_id,
                        "local_image": f"images/val/{image_id}.jpg",
                        "mapped_classes": ";".join(present_names),
                        **{
                            field: source_row.get(field, "")
                            for field in fields[3:]
                        },
                    }
                )

    counts = defaultdict(int)
    for row in selected_rows:
        counts[id_to_name[row["LabelName"]]] += 1
    print(f"Selected {len(chosen)} of at most {args.max_images} images; excluded {len(excluded)} IDs")
    print(f"Retained {len(selected_rows)} original boxes from 8 mapped classes")
    for name in sorted(PROJECT_CLASSES):
        print(f"{name}: {counts[name]} boxes")


if __name__ == "__main__":
    main()

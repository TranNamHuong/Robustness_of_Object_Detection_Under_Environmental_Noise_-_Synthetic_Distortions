"""Validate YOLO detection dataset paths, labels, class IDs and split leakage."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def yaml_root(data: dict[str, Any], yaml_path: Path) -> Path:
    root = Path(data.get("path", yaml_path.parent))
    return root.resolve() if root.is_absolute() else (yaml_path.parent / root).resolve()


def split_dirs(value: Any, root: Path) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        path = Path(item)
        result.append((path if path.is_absolute() else root / path).resolve())
    return result


def infer_label_path(image: Path) -> Path:
    parts = list(image.parts)
    image_indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def class_count(names: Any) -> int:
    if isinstance(names, dict):
        ids = sorted(int(key) for key in names)
        if ids != list(range(len(ids))):
            raise ValueError(f"Class IDs in names must be contiguous from 0; got {ids[:10]}")
        return len(ids)
    if isinstance(names, list):
        return len(names)
    raise ValueError("data.yaml must define names as a list or ID-to-name mapping")


def inspect_split(name: str, dirs: list[Path], nclasses: int) -> tuple[set[Path], list[str], Counter[int], int]:
    errors: list[str] = []
    images: set[Path] = set()
    class_instances: Counter[int] = Counter()
    empty_labels = 0
    for directory in dirs:
        if not directory.is_dir():
            errors.append(f"{name}: image directory does not exist: {directory}")
            continue
        for image in sorted(p.resolve() for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
            images.add(image)
            label = infer_label_path(image)
            if not label.is_file():
                errors.append(f"{name}: missing label for {image}")
                continue
            try:
                rows = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            except UnicodeError:
                errors.append(f"{name}: label is not UTF-8 text: {label}")
                continue
            if not rows:
                empty_labels += 1
                continue
            for line_num, row in enumerate(rows, start=1):
                fields = row.split()
                if len(fields) != 5:
                    errors.append(f"{name}: {label}:{line_num}: expected 5 YOLO fields, got {len(fields)}")
                    continue
                try:
                    class_float = float(fields[0])
                    box = [float(v) for v in fields[1:]]
                except ValueError:
                    errors.append(f"{name}: {label}:{line_num}: non-numeric value")
                    continue
                class_id = int(class_float)
                if class_float != class_id or not 0 <= class_id < nclasses:
                    errors.append(f"{name}: {label}:{line_num}: class ID {fields[0]} outside 0..{nclasses - 1}")
                    continue
                xc, yc, width, height = box
                if not all(0.0 <= v <= 1.0 for v in box):
                    errors.append(f"{name}: {label}:{line_num}: bbox values must be normalized to [0,1]")
                    continue
                eps = 1e-5  # tolerate decimal rounding in normalized COCO coordinates
                if width <= 0.0 or height <= 0.0 or xc - width / 2 < -eps or yc - height / 2 < -eps or xc + width / 2 > 1 + eps or yc + height / 2 > 1 + eps:
                    errors.append(f"{name}: {label}:{line_num}: invalid or out-of-image bbox")
                    continue
                class_instances[class_id] += 1
    if not images:
        errors.append(f"{name}: no supported image files found in the configured image directory")
    return images, errors, class_instances, empty_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics data.yaml")
    parser.add_argument("--require-coco80", action="store_true", help="Require COCO's contiguous 80 class IDs")
    parser.add_argument("--show-class-support", action="store_true", help="Print instance counts for each present class")
    args = parser.parse_args()
    data_path = args.data.resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    nclasses = class_count(data.get("names"))
    if args.require_coco80 and nclasses != 80:
        raise SystemExit(f"Expected 80 COCO classes, found {nclasses}")
    root = yaml_root(data, data_path)
    split_names = [name for name in ("train", "val", "test") if data.get(name) is not None]
    split_paths = {name: split_dirs(data[name], root) for name in split_names}
    seen: dict[Path, str] = {}
    train_val_alias = "train" in split_paths and "val" in split_paths and set(split_paths["train"]) == set(split_paths["val"])
    if train_val_alias:
        print("Note: train and val point to the same image directory (validation-only alias).")
    total_errors = 0
    print(f"Dataset: {data_path}")
    print(f"Root: {root}")
    print(f"Classes: {nclasses}")
    for split_name in split_names:
        dirs = split_paths[split_name]
        images, errors, counts, empty = inspect_split(split_name, dirs, nclasses)
        skip_alias_overlap = train_val_alias and split_name == "val"
        leaked = [] if skip_alias_overlap else [image for image in images if image in seen]
        if leaked:
            errors.extend(f"image leakage: {image} appears in {seen[image]} and {split_name}" for image in leaked[:20])
        for image in images:
            seen[image] = split_name
        total_errors += len(errors)
        instances = sum(counts.values())
        present = len(counts)
        print(f"{split_name}: {len(images)} images, {instances} instances, {empty} empty/background labels, {present}/{nclasses} classes present")
        if counts and args.show_class_support:
            print("  class support: " + ", ".join(f"{class_id}={count}" for class_id, count in sorted(counts.items())))
        for error in errors[:20]:
            print(f"  ERROR: {error}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more errors")
    if total_errors:
        print(f"FAILED: {total_errors} issue(s) found.")
        return 1
    print("PASSED: labels are valid and no unintended split overlap was detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build paired clean/corrupted COCO-80 training data without new downloads."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONDITIONS = [(family, level) for family, levels in (
    ("gaussian", (0.01, 0.03, 0.06)),
    ("salt_pepper", (0.01, 0.03, 0.06)),
    ("fog", (0.15, 0.30, 0.50)),
    ("low_light", (0.60, 0.40, 0.25)),
) for level in levels]


def perturb(image: np.ndarray, family: str, level: float,
            rng: np.random.Generator) -> np.ndarray:
    x = image.astype(np.float32)
    if family == "gaussian":
        x += rng.normal(0.0, level * 255.0, x.shape).astype(np.float32)
    elif family == "salt_pepper":
        mask = rng.random(x.shape[:2])
        x[mask < level / 2] = 0
        x[mask > 1 - level / 2] = 255
    elif family == "fog":
        x = x * (1.0 - level) + 235.0 * level
    elif family == "low_light":
        x = 255.0 * np.power(np.clip(x / 255.0, 0, 1) * level, 1.15)
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    return np.clip(x, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "datasets/coco/train_subset_5000/data.yaml")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "datasets/coco/train_subset_5000_noise_aware")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_yaml = args.source.resolve()
    out_dir = args.out.resolve()
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Source YAML not found: {source_yaml}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    source_data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    names = source_data.get("names", {})
    if len(names) != 80:
        raise ValueError(f"Expected exactly 80 COCO classes; found {len(names)}")
    source_root = Path(source_data.get("path", source_yaml.parent))
    if not source_root.is_absolute():
        source_root = (source_yaml.parent / source_root).resolve()
    train_value = source_data["train"]
    train_entries = train_value if isinstance(train_value, list) else [train_value]
    source_images: list[Path] = []
    for entry in train_entries:
        path = Path(entry)
        path = path if path.is_absolute() else source_root / path
        if path.is_dir():
            source_images.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                p = Path(line.strip())
                if line.strip():
                    source_images.append(p if p.is_absolute() else path.parent / p)
        else:
            raise FileNotFoundError(f"Unsupported/missing train path: {path}")
    source_images = sorted(set(p.resolve() for p in source_images))
    if not source_images:
        raise ValueError("No training images found")

    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    # Balanced assignment: each family and each defined severity gets near-equal use.
    specs = [CONDITIONS[i % len(CONDITIONS)] for i in range(len(source_images))]
    py_rng.shuffle(specs)

    image_out = out_dir / "images" / "train"
    label_out = out_dir / "labels" / "train"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    manifest = []
    family_counts: dict[str, int] = {}
    for index, (src, (family, level)) in enumerate(zip(source_images, specs)):
        src_label = src.parents[2] / "labels" / "train" / src.name
        src_label = src_label.with_suffix(".txt")
        if not src_label.is_file():
            raise FileNotFoundError(f"Missing label: {src_label}")
        label_text = src_label.read_text(encoding="utf-8")
        for line in label_text.splitlines():
            fields = line.split()
            if fields and (int(fields[0]) < 0 or int(fields[0]) >= 80):
                raise ValueError(f"Invalid COCO class ID in {src_label}: {fields[0]}")

        clean_name = f"clean_{index:06d}_{src.name}"
        noisy_name = f"aug_{index:06d}_{family}_{level:g}_{src.name}"
        shutil.copy2(src, image_out / clean_name)
        (label_out / Path(clean_name).with_suffix(".txt").name).write_text(
            label_text, encoding="utf-8")

        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {src}")
        corrupted = perturb(image, family, level, rng)
        if not cv2.imwrite(str(image_out / noisy_name), corrupted):
            raise RuntimeError(f"Could not write corrupted image: {image_out / noisy_name}")
        (label_out / Path(noisy_name).with_suffix(".txt").name).write_text(
            label_text, encoding="utf-8")
        family_counts[family] = family_counts.get(family, 0) + 1
        manifest.append({"source": src.name, "clean": clean_name, "augmented": noisy_name,
                         "family": family, "level": level})
        if (index + 1) % 500 == 0 or index + 1 == len(source_images):
            print(f"Prepared {index + 1}/{len(source_images)} source images", flush=True)

    val_value = source_data["val"]
    val_entries = val_value if isinstance(val_value, list) else [val_value]
    resolved_val = [str((Path(v) if Path(v).is_absolute() else source_root / Path(v)).resolve())
                    for v in val_entries]
    output_data = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images/train",
        "val": resolved_val if isinstance(val_value, list) else resolved_val[0],
        "names": names,
    }
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(output_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps({
        "source_yaml": str(source_yaml), "seed": args.seed,
        "source_images": len(source_images),
        "clean_images": len(source_images), "corrupted_images": len(source_images),
        "corruption_family_counts": family_counts,
        "condition_levels": {name: [level for fam, level in CONDITIONS if fam == name]
                             for name in sorted({fam for fam, _ in CONDITIONS})},
        "corruptions_match": "robustness_eval.py perturb() definitions",
        "classes": names, "images": manifest,
    }, indent=2), encoding="utf-8")
    print(f"Prepared {len(source_images) * 2} training images; family counts: {family_counts}")
    print(f"Dataset YAML: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

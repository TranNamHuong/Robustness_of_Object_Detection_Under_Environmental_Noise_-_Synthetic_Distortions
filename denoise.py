#!/usr/bin/env python
"""Apply the project's automatic or selected recovery method to an image/folder."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from preprocessing import NOISE_FAMILIES, RECOVERY_METHODS, recover_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Validation-tuned choices from the full COCO val2017 comparison of YOLOv8n
# and SSDLite. Levels are restricted to those measured in this project.
LEVEL_AWARE_AUTO = {
    "gaussian": {0.01: "none", 0.03: "none", 0.06: "bilateral"},
    "salt_pepper": {0.01: "median", 0.03: "median", 0.06: "median"},
    "fog": {0.15: "none", 0.30: "clahe", 0.50: "clahe"},
    # Low-light level is a brightness multiplier: smaller means darker.
    "low_light": {0.25: "clahe", 0.40: "clahe", 0.60: "none"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input image or folder of images")
    parser.add_argument("--output", type=Path, required=True, help="Output image path or folder")
    parser.add_argument("--family", choices=NOISE_FAMILIES, required=True,
                        help="Known noise/environment family for the automatic recovery policy")
    parser.add_argument("--method", choices=RECOVERY_METHODS, default="auto")
    parser.add_argument("--level", type=float,
                        help="Known benchmark level; with --method auto, use the validated level-aware policy")
    parser.add_argument("--overwrite", action="store_true", help="Replace output files that already exist")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if not source.exists():
        parser.error(f"input path does not exist: {source}")

    selected_method = args.method
    if args.method == "auto" and args.level is not None:
        levels = LEVEL_AWARE_AUTO[args.family]
        match = next((known for known in levels if abs(known - args.level) < 1e-8), None)
        if match is None:
            supported = ", ".join(f"{level:g}" for level in levels)
            parser.error(f"level {args.level:g} was not benchmarked for {args.family}; supported: {supported}")
        selected_method = levels[match]

    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            parser.error(f"unsupported image extension: {source.suffix}")
        target = destination / source.name if destination.exists() and destination.is_dir() else destination
        pairs = [(source, target)]
    elif source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            parser.error(f"no supported images found under: {source}")
        if destination == source or source in destination.parents:
            parser.error("output folder must be outside the input folder")
        pairs = [(p, destination / p.relative_to(source)) for p in files]
    else:
        parser.error(f"input must be an image file or directory: {source}")

    for src, dst in pairs:
        if dst.exists() and not args.overwrite:
            raise FileExistsError(f"output exists (pass --overwrite to replace it): {dst}")
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"could not decode image: {src}")
        recovered = (image if selected_method == "none" else
                     recover_image(image, args.family, selected_method))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), recovered):
            raise OSError(f"could not write image: {dst}")
    print(f"Processed {len(pairs)} image(s) with method={selected_method}, family={args.family}; output={destination}")


if __name__ == "__main__":
    main()

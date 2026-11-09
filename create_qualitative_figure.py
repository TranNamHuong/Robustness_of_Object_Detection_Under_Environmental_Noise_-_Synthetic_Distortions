#!/usr/bin/env python
"""Create a same-image visual comparison for COCO corruption/recovery examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/coco/full_val2017"
RAW = ROOT / "runs/full_val2017_raw_seed42/datasets"
RECOVERED = ROOT / "runs/full_val2017_recovered_seed42/datasets"
BILATERAL = ROOT / "runs/full_val2017_gaussian_bilateral_seed42/datasets"

ROWS = [
    ("Gaussian 0.06", RAW / "gaussian_0.06", BILATERAL / "gaussian_0.06_bilateral"),
    ("Salt-and-Pepper 0.06", RAW / "salt_pepper_0.06", RECOVERED / "salt_pepper_0.06_recovered"),
    ("Fog 0.50", RAW / "fog_0.5", RECOVERED / "fog_0.5_recovered"),
    ("Low-light 0.25", RAW / "low_light_0.25", RECOVERED / "low_light_0.25_recovered"),
]


def read_names() -> list[str]:
    config = yaml.safe_load((DATA / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        return [str(names.get(i, names.get(str(i), i))) for i in range(len(names))]
    return [str(name) for name in names]


def read_boxes(label_path: Path, width: int, height: int):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        class_id, xc, yc, bw, bh = map(float, values[:5])
        boxes.append((int(class_id), round((xc-bw/2)*width), round((yc-bh/2)*height),
                      round((xc+bw/2)*width), round((yc+bh/2)*height)))
    return boxes


def choose_image() -> str:
    clean_labels = DATA / "labels/val"
    images = sorted((DATA / "images/val").glob("*.jpg"))
    candidates = []
    for image_path in images:
        label_path = (clean_labels / image_path.name).with_suffix(".txt")
        if not label_path.exists():
            continue
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if 3 <= len(lines) <= 8:
            candidates.append((len(lines), image_path.name))
    if not candidates:
        raise RuntimeError("No COCO image with 3–8 labeled objects was found")
    # Prefer a representative scene with several boxes while keeping the panel legible.
    candidates.sort(key=lambda item: (abs(item[0] - 5), item[1]))
    return candidates[0][1]


def render_panel(image_path: Path, label_path: Path, names: list[str]):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    for class_id, x1, y1, x2, y2 in read_boxes(label_path, width, height):
        cv2.rectangle(image, (x1, y1), (x2, y2), (20, 210, 90), 2)
        if 0 <= class_id < len(names):
            cv2.putText(image, names[class_id], (x1, max(16, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 210, 90), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def condition_image(dataset_root: Path, filename: str) -> Path:
    image_dir = dataset_root / "images/val"
    exact = image_dir / filename
    if exact.is_file():
        return exact
    matches = sorted(image_dir.glob(f"*_{filename}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not uniquely match {filename} in {image_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="COCO filename; default chooses a scene with about five objects")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs/qualitative_coco_examples/qualitative_recovery_examples.png")
    args = parser.parse_args()

    names = read_names()
    filename = args.image or choose_image()
    clean_image = DATA / "images/val" / filename
    clean_label = DATA / "labels/val" / Path(filename).with_suffix(".txt")
    if not clean_image.is_file() or not clean_label.is_file():
        raise FileNotFoundError(f"COCO image/label pair not found for {filename}")

    fig, axes = plt.subplots(len(ROWS), 3, figsize=(15, 17))
    column_titles = ["Clean", "Corrupted", "Recovered"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=14, weight="bold")
    for row_idx, (condition, raw_root, recovered_root) in enumerate(ROWS):
        paths = [clean_image, condition_image(raw_root, filename),
                 condition_image(recovered_root, filename)]
        for col, path in enumerate(paths):
            if not path.is_file():
                raise FileNotFoundError(f"Missing image for qualitative panel: {path}")
            # GT boxes are overlaid identically to make object visibility comparable.
            label = clean_label
            axes[row_idx, col].imshow(render_panel(path, label, names))
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].set_ylabel("")

    fig.suptitle(f"COCO val2017 qualitative recovery examples — {filename}\nGreen boxes: unchanged ground truth",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=(0.07, 0, 1, 0.97))
    for row_idx, (condition, _, _) in enumerate(ROWS):
        position = axes[row_idx, 0].get_position()
        fig.text(0.025, position.y0 + position.height / 2, condition,
                 fontsize=11, weight="bold", rotation=90, ha="center", va="center")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    metadata = {"image": filename, "conditions": [
        {"family": family, "raw": str(raw), "recovered": str(recovered)}
        for family, raw, recovered in ROWS],
        "ground_truth_box_color": "green"}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved figure: {args.out}")
    print(f"Selected COCO image: {filename}")


if __name__ == "__main__":
    main()

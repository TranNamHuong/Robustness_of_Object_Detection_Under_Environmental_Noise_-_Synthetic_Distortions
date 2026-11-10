"""Download the pretrained checkpoints required by the local detection demo.

This restores inference assets only. It does not train a model or run a benchmark.
Run from the project root after installing requirements.txt.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    for filename in ("yolov8n.pt", "yolov8s.pt"):
        destination = PROJECT_ROOT / filename
        if destination.is_file():
            print(f"Already present: {destination}")
            continue
        print(f"Downloading COCO-pretrained {filename}...")
        YOLO(str(destination))
        if not destination.is_file():
            raise RuntimeError(f"Ultralytics did not create the expected checkpoint: {destination}")
        print(f"Saved: {destination}")

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    cache_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if cache_file.is_file():
        print(f"SSD-MobileNet weights already present: {cache_file}")
    else:
        print("Downloading the Torchvision COCO-pretrained SSD-MobileNet weights...")
        ssdlite320_mobilenet_v3_large(weights=weights).eval()
        if not cache_file.is_file():
            raise RuntimeError(f"Torchvision did not create the expected checkpoint: {cache_file}")
        print(f"Saved: {cache_file}")

    print("Pretrained model weights are ready. No training or benchmark was run.")


if __name__ == "__main__":
    main()

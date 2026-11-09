"""Inference helpers shared by the desktop app and evaluation scripts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import batched_nms


@dataclass
class DetectionBoxes:
    xyxy: torch.Tensor
    cls: torch.Tensor
    conf: torch.Tensor

    def __len__(self) -> int:
        return int(self.cls.numel())


@dataclass
class DetectionResult:
    boxes: DetectionBoxes


def predict_image(model, image: np.ndarray, imgsz: int, confidence: float,
                  device: str, tiled: bool = False):
    """Run full-frame or overlapping 640px tiles, then merge duplicate boxes."""
    if not tiled:
        return model.predict(source=image, imgsz=imgsz, conf=confidence,
                             iou=0.7, max_det=300, device=device, verbose=False)[0]

    height, width = image.shape[:2]
    tile_size = 640
    stride = int(tile_size * 0.75)

    def positions(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, stride))
        final = length - tile_size
        if starts[-1] != final:
            starts.append(final)
        return starts

    all_boxes, all_scores, all_classes = [], [], []
    for top in positions(height):
        for left in positions(width):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            crop = image[top:bottom, left:right]
            result = model.predict(source=crop, imgsz=imgsz, conf=confidence,
                                   iou=0.7, max_det=300, device=device, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.detach().float().cpu().clone()
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            all_boxes.append(boxes)
            all_scores.append(result.boxes.conf.detach().float().cpu())
            all_classes.append(result.boxes.cls.detach().long().cpu())

    if not all_boxes:
        return DetectionResult(DetectionBoxes(
            xyxy=torch.empty((0, 4)), cls=torch.empty((0,), dtype=torch.long),
            conf=torch.empty((0,))))
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    classes = torch.cat(all_classes)
    keep = batched_nms(boxes, scores, classes, iou_threshold=0.55)[:300]
    return DetectionResult(DetectionBoxes(boxes[keep], classes[keep], scores[keep]))

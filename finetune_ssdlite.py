#!/usr/bin/env python
"""Fine-tune torchvision SSDLite-MobileNetV3 on the same COCO-80 subset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)


ROOT = Path(__file__).resolve().parent


class CocoYoloTrainDataset(Dataset):
    def __init__(self, root: Path, names: dict[int, str], category_ids: dict[str, int]):
        self.images = sorted((root / "images/train").glob("*"))
        self.labels = root / "labels/train"
        self.name_by_id = names
        self.category_ids = category_ids
        if not self.images:
            raise FileNotFoundError(f"No training images under {root / 'images/train'}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        image = read_image(str(path), mode=ImageReadMode.RGB).float().div_(255.0)
        _, height, width = image.shape
        boxes, labels = [], []
        label_path = self.labels / path.with_suffix(".txt").name
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            class_id, xc, yc, bw, bh = map(float, fields[:5])
            name = self.name_by_id[int(class_id)]
            # Torchvision's pretrained detector uses COCO category IDs with 0 reserved
            # for background; map by category name to preserve its pretrained head.
            labels.append(self.category_ids[name])
            x1, y1 = (xc - bw / 2) * width, (yc - bh / 2) * height
            x2, y2 = (xc + bw / 2) * width, (yc + bh / 2) * height
            boxes.append([max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/coco/train_subset_1000/data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data_path = args.data.resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    raw_names = data.get("names", {})
    names = {int(index): str(name) for index, name in raw_names.items()} if isinstance(raw_names, dict) else {
        index: str(name) for index, name in enumerate(raw_names)
    }
    if len(names) != 80:
        raise ValueError("Fine-tuning requires the same 80 COCO classes as the baseline")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    category_ids = {name: index for index, name in enumerate(weights.meta["categories"]) if index > 0}
    missing = sorted(set(names.values()) - set(category_ids))
    if missing:
        raise ValueError(f"Pretrained SSDLite COCO classes missing: {missing}")

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    torch.hub.set_dir(str(ROOT / "runs/torch_hub"))
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device)
    dataset = CocoYoloTrainDataset(data_path.parent, names, category_ids)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_dir = ROOT / "runs/finetune_coco80/ssdlite320_mobilenet_v3_large"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training {len(dataset)} images, 80 classes, device={device}, output={run_dir}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()}
                       for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite SSD loss at epoch {epoch}: {losses}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            steps += 1
        mean_loss = total_loss / max(steps, 1)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "mean_train_loss": mean_loss,
            "class_names": names,
            "category_ids": category_ids,
            "seed": args.seed,
        }
        torch.save(checkpoint, run_dir / "last.pth")
        print(f"epoch {epoch}/{args.epochs}: mean_train_loss={mean_loss:.5f}", flush=True)
    print(f"Saved checkpoint: {run_dir / 'last.pth'}", flush=True)


if __name__ == "__main__":
    main()

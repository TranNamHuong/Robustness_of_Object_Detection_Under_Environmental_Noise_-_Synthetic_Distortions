#!/usr/bin/env python
"""Evaluate torchvision SSDLite320-MobileNetV3 on selected full COCO val conditions."""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import yaml

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "runs/full_val2017_raw_seed42/datasets"
REC_ROOTS = [ROOT / "runs/full_val2017_recovered_seed42/datasets",
             ROOT / "runs/full_val2017_gaussian_recovered_seed42/datasets",
             ROOT / "runs/full_val2017_gaussian_bilateral_seed42/datasets"]
DEFAULT_CONDITIONS = ["clean", "gaussian_0.06", "gaussian_0.06_recovered",
                      "salt_pepper_0.06", "salt_pepper_0.06_recovered",
                      "fog_0.5", "fog_0.5_recovered",
                      "low_light_0.25", "low_light_0.25_recovered"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2]-a[:, 0], 0) * np.maximum(a[:, 3]-a[:, 1], 0)
    area_b = np.maximum(b[:, 2]-b[:, 0], 0) * np.maximum(b[:, 3]-b[:, 1], 0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class CocoYoloDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.root = root
        self.image_dir = root / "images/val"
        self.label_dir = root / "labels/val"
        self.images = sorted(p for p in self.image_dir.iterdir() if p.is_file())
        self.transform = transform
        if len(self.images) != 5000:
            raise ValueError(f"Expected 5000 COCO val images in {self.image_dir}, found {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        _, height, width = image.shape
        label_path = (self.label_dir / image_path.name).with_suffix(".txt")
        classes, boxes = [], []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            vals = line.split()
            if len(vals) < 5:
                continue
            cls, xc, yc, bw, bh = map(float, vals[:5])
            classes.append(int(cls))
            boxes.append([(xc-bw/2)*width, (yc-bh/2)*height,
                          (xc+bw/2)*width, (yc+bh/2)*height])
        return self.transform(image), (np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                                      np.asarray(classes, dtype=np.int64)), image_path.name


def collate(batch):
    images, targets, filenames = zip(*batch)
    return list(images), list(targets), filenames


def ap_for_class(image_predictions, image_targets, cls, thresholds):
    gt_by_image = {}
    total_gt = 0
    detections = []
    for image_id, (boxes, labels) in enumerate(image_targets):
        gt = boxes[labels == cls]
        gt_by_image[image_id] = gt
        total_gt += len(gt)
        pred = image_predictions[image_id]
        keep = pred["labels"] == cls
        for box, score in zip(pred["boxes"][keep], pred["scores"][keep]):
            detections.append((float(score), image_id, box))
    if total_gt == 0:
        return [float("nan")] * len(thresholds)
    detections.sort(key=lambda item: item[0], reverse=True)
    # Compute per-detection overlaps once; reuse them across the ten IoU thresholds.
    cached_overlaps = []
    for _, image_id, box in detections:
        gt = gt_by_image[image_id]
        if len(gt):
            overlaps = iou_matrix(box.reshape(1, 4), gt)[0]
            order = np.argsort(overlaps)[::-1]
            cached_overlaps.append((image_id, [(int(j), float(overlaps[j])) for j in order]))
        else:
            cached_overlaps.append((image_id, []))
    aps = []
    recall_points = np.linspace(0.0, 1.0, 101)
    for threshold in thresholds:
        matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_by_image.items()}
        tp = np.zeros(len(detections), dtype=np.float64)
        fp = np.zeros(len(detections), dtype=np.float64)
        for i, (image_id, overlaps) in enumerate(cached_overlaps):
            match = next((j for j, overlap in overlaps
                          if overlap >= threshold and not matched[image_id][j]), None)
            if match is None:
                fp[i] = 1
            else:
                tp[i] = 1
                matched[image_id][match] = True
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
        recall = cum_tp / total_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        if len(precision):
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            sampled = [precision[recall >= r].max() if np.any(recall >= r) else 0.0 for r in recall_points]
            aps.append(float(np.mean(sampled)))
        else:
            aps.append(0.0)
    return aps


def confusion_counts(image_predictions, image_targets, nclasses, conf=0.25, iou_thr=0.5):
    matrix = np.zeros((nclasses + 1, nclasses + 1), dtype=np.int64)
    for pred, (gt_boxes, gt_classes) in zip(image_predictions, image_targets):
        keep = pred["scores"] >= conf
        boxes = pred["boxes"][keep]
        classes = pred["labels"][keep]
        scores = pred["scores"][keep]
        overlaps = iou_matrix(gt_boxes, boxes)
        candidates = [(overlaps[g, p], scores[p], g, p)
                      for g in range(len(gt_boxes)) for p in range(len(boxes))
                      if overlaps[g, p] >= iou_thr]
        candidates.sort(reverse=True)
        matched_g, matched_p = set(), set()
        for _, _, g, p in candidates:
            if g in matched_g or p in matched_p:
                continue
            matrix[gt_classes[g], classes[p]] += 1
            matched_g.add(g); matched_p.add(p)
        for g, cls in enumerate(gt_classes):
            if g not in matched_g:
                matrix[cls, nclasses] += 1
        for p, cls in enumerate(classes):
            if p not in matched_p:
                matrix[nclasses, cls] += 1
    tp = int(np.trace(matrix[:nclasses, :nclasses]))
    misclassified = int(matrix[:nclasses, :nclasses].sum()) - tp
    fp = int(matrix[nclasses, :nclasses].sum()) + misclassified
    fn = int(matrix[:nclasses, nclasses].sum()) + misclassified
    return matrix, tp, fp, fn


def locate_condition(name: str) -> Path:
    if "_seed" in name and name.rsplit("_seed", 1)[1].isdigit():
        base, seed = name.rsplit("_seed", 1)
        candidate = ROOT / f"runs/full_val2017_seed{seed}_selected/datasets" / base
        if (candidate / "data.yaml").exists():
            return candidate
    for root in [RAW_ROOT, *REC_ROOTS]:
        candidate = root / name
        if (candidate / "data.yaml").exists():
            return candidate
    raise FileNotFoundError(f"No prepared dataset found for {name}")


def evaluate_condition(model, transform, device, data_names, output, condition, batch_size):
    dataset_dir = locate_condition(condition)
    ds = CocoYoloDataset(dataset_dir, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=device.type == "cuda", collate_fn=collate)
    image_predictions, image_targets = [], []
    started = time.time()
    with torch.inference_mode():
        for batch_no, (images, targets, filenames) in enumerate(loader, start=1):
            outputs = model([image.to(device, non_blocking=True) for image in images])
            for result, (gt_boxes, gt_classes) in zip(outputs, targets):
                labels = result["labels"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()
                boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
                mapped = np.asarray([output_to_yolo.get(int(label), -1) for label in labels], dtype=np.int64)
                keep = mapped >= 0
                image_predictions.append({"boxes": boxes[keep], "labels": mapped[keep], "scores": scores[keep]})
                image_targets.append((gt_boxes, gt_classes))
            if batch_no % 50 == 0:
                print(f"{condition}: {min(batch_no*batch_size, len(ds))}/{len(ds)} images", flush=True)
    thresholds = [0.50 + 0.05*i for i in range(10)]
    aps = np.asarray([ap_for_class(image_predictions, image_targets, cls, thresholds)
                      for cls in range(len(data_names))], dtype=np.float64)
    map50 = float(np.nanmean(aps[:, 0]))
    map5095 = float(np.nanmean(aps))
    matrix, tp, fp, fn = confusion_counts(image_predictions, image_targets, len(data_names))
    np.savetxt(output / f"confusion_{condition}.csv", matrix, fmt="%d", delimiter=",")
    inference_sec = time.time() - started
    row = {"model": "SSDLite320-MobileNetV3-Large-COCO_V1", "condition": condition,
           "images": len(ds), "mAP50": map50, "mAP50_95": map5095,
           "tp_iou50": tp, "fp_iou50": fp, "fn_iou50": fn,
           "confusion_conf": 0.25, "confusion_iou": 0.5,
           "elapsed_sec": round(inference_sec, 2)}
    print(f"{condition}: mAP50={map50:.4f}, mAP50-95={map5095:.4f}, FP={fp}, FN={fn}, elapsed={inference_sec:.1f}s", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional fine-tuned SSDLite checkpoint (.pth)")
    parser.add_argument("--out", type=Path,
                        help="Output directory; defaults to the baseline or fine-tune run folder")
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    data = yaml.safe_load((ROOT / "datasets/coco/full_val2017/data.yaml").read_text(encoding="utf-8"))
    raw_names = data["names"]
    names = [raw_names.get(i, raw_names.get(str(i), str(i))) for i in range(len(raw_names))] if isinstance(raw_names, dict) else raw_names
    name_to_yolo = {name: i for i, name in enumerate(names)}
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    torch.hub.set_dir(str(ROOT / "runs/torch_hub"))
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
    categories = weights.meta["categories"]
    global output_to_yolo
    output_to_yolo = {i: name_to_yolo[name] for i, name in enumerate(categories)
                      if name in name_to_yolo}
    print(f"Loaded {weights}; CUDA={device.type == 'cuda'}; mapped classes={len(output_to_yolo)}", flush=True)
    default_out = (ROOT / "runs/ssdlite_finetune_coco80"
                   if args.checkpoint else ROOT / "runs/ssdlite_fullval_seed42")
    out = (args.out.resolve() if args.out else default_out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.csv"
    existing = {}
    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8-sig") as f:
            existing = {row["condition"]: row for row in csv.DictReader(f)}
    for condition in args.conditions:
        if condition in existing:
            print(f"Skipping existing {condition}", flush=True)
            continue
        row = evaluate_condition(model, weights.transforms(), device, names, out, condition, args.batch)
        existing[condition] = row
        with metrics_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader(); writer.writerows(existing.values())
    print(f"Saved {len(existing)} SSD results to {metrics_path}")


if __name__ == "__main__":
    main()

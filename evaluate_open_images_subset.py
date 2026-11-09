"""Evaluate COCO-pretrained YOLOv8 and SSDLite on a small Open Images subset.

The evaluator uses the Open Images image-level positive/negative/unannotated
states, ignores unannotated-class predictions, and supports group-of boxes.
Only the eight explicitly mapped classes in the subset README are scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/open_images_v7"
OUT = ROOT / "runs/open_images_v7_eval"
SOURCE = DATA / "source"
OPEN_TO_COCO = {
    "/m/01g317": ("Person", 0),
    "/m/01mzpv": ("Chair", 56),
    "/m/02crq1": ("Couch", 57),
    "/m/03ssj5": ("Bed", 59),
    "/m/0qjjc": ("Remote control", 65),
    "/m/050k8": ("Mobile phone", 67),
    "/m/01x3z": ("Clock", 74),
    "/m/0kmg4": ("Teddy bear", 77),
}
CLASSES = [
    ("Person", "person", 0),
    ("Chair", "chair", 56),
    ("Couch", "couch", 57),
    ("Bed", "bed", 59),
    ("Remote control", "remote", 65),
    ("Mobile phone", "cell phone", 67),
    ("Clock", "clock", 74),
    ("Teddy bear", "teddy bear", 77),
]
LOCAL_TO_COCO = {i: item[2] for i, item in enumerate(CLASSES)}
COCO_TO_LOCAL = {item[2]: i for i, item in enumerate(CLASSES)}


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    lt = np.maximum(boxes[:, :2], box[:2])
    rb = np.minimum(boxes[:, 2:], box[2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def intersection_over_detection(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    lt = np.maximum(boxes[:, :2], box[:2])
    rb = np.minimum(boxes[:, 2:], box[2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / max(area, 1e-9)


def load_ground_truth() -> tuple[list[str], dict[str, dict[str, Any]]]:
    image_ids = [
        line.strip().split("/", 1)[-1]
        for line in (DATA / "image_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    id_set = set(image_ids)
    status: dict[tuple[str, int], str] = {}
    rank: dict[tuple[str, int], int] = {}
    labels_file = SOURCE / "validation-annotations-human-imagelabels-boxable.csv"
    with labels_file.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image_id = row["ImageID"]
            mapped = OPEN_TO_COCO.get(row["LabelName"])
            if image_id not in id_set or mapped is None:
                continue
            source = row["Source"]
            source_rank = {"verification": 2, "crowdsource-verification": 1}.get(source, 0)
            if source_rank == 0:
                continue
            key = (image_id, COCO_TO_LOCAL[mapped[1]])
            if source_rank >= rank.get(key, 0):
                status[key] = "positive" if row["Confidence"] == "1" else "negative"
                rank[key] = source_rank

    annotations: dict[str, dict[str, Any]] = {
        image_id: {"boxes": [], "status": {}} for image_id in image_ids
    }
    for (image_id, cls), value in status.items():
        annotations[image_id]["status"][cls] = value

    boxes_file = DATA / "selected_annotations.csv"
    with boxes_file.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image_id = row["ImageID"]
            mapped = OPEN_TO_COCO.get(row["LabelName"])
            if image_id not in annotations or mapped is None:
                continue
            local_cls = COCO_TO_LOCAL[mapped[1]]
            # The normalized coordinates are converted to pixels after reading
            # each image. Keep Open Images group-of semantics as an attribute.
            annotations[image_id]["boxes"].append(
                {
                    "class": local_cls,
                    "xyxy_norm": [
                        float(row["XMin"]),
                        float(row["YMin"]),
                        float(row["XMax"]),
                        float(row["YMax"]),
                    ],
                    "group_of": row["IsGroupOf"] == "1",
                }
            )

    # A box is evidence of a positive class even if the separate image-level
    # verification file has no entry for that image/class.
    for image_id, record in annotations.items():
        for box in record["boxes"]:
            record["status"].setdefault(box["class"], "positive")

    return image_ids, annotations


class ImageDataset(Dataset):
    def __init__(self, image_ids: list[str], transform=None):
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        path = DATA / "images/val" / f"{image_id}.jpg"
        image = read_image(str(path), mode=ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id


def collate(batch):
    images, ids = zip(*batch)
    return list(images), list(ids)


def run_yolo(model_path: str, image_ids: list[str], imgsz: int, device: str,
             conf: float, batch: int, refresh: bool = False) -> tuple[list[dict[str, Any]], float | None]:
    model_name = Path(model_path).stem
    cache = OUT / f"predictions_{model_name}_img{imgsz}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8")), None
    model = YOLO(model_path)
    paths = [str(DATA / "images/val" / f"{image_id}.jpg") for image_id in image_ids]
    start = time.perf_counter()
    results = model.predict(
        source=paths,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        max_det=300,
        verbose=False,
        stream=False,
    )
    elapsed = time.perf_counter() - start
    predictions = []
    for image_id, result in zip(image_ids, results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            predictions.append({"image_id": image_id, "boxes": [], "labels": [], "scores": []})
            continue
        raw_classes = boxes.cls.cpu().numpy().astype(int)
        keep = np.asarray([int(cls) in COCO_TO_LOCAL for cls in raw_classes], dtype=bool)
        predictions.append(
            {
                "image_id": image_id,
                "boxes": boxes.xyxy.cpu().numpy()[keep].astype(float).tolist(),
                "labels": [COCO_TO_LOCAL[int(cls)] for cls in raw_classes[keep]],
                "scores": boxes.conf.cpu().numpy()[keep].astype(float).tolist(),
            }
        )
    cache.write_text(json.dumps(predictions), encoding="utf-8")
    return predictions, elapsed


def run_ssdlite(image_ids: list[str], device_name: str, conf: float,
                batch: int, refresh: bool = False) -> tuple[list[dict[str, Any]], float | None, str]:
    cache = OUT / "predictions_ssdlite320_img320.json"
    if cache.exists() and not refresh:
        device = "cuda:0" if device_name == "0" and torch.cuda.is_available() else device_name
        return json.loads(cache.read_text(encoding="utf-8")), None, str(device)
    device = torch.device("cuda:0" if device_name == "0" and torch.cuda.is_available() else device_name)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    torch.hub.set_dir(str(ROOT / "runs/torch_hub"))
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()
    dataset = ImageDataset(image_ids, weights.transforms())
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)
    coco_names = weights.meta["categories"]
    # Torchvision COCO labels are category IDs in the weights metadata order;
    # class-name lookup avoids assuming their integer IDs match Ultralytics.
    name_to_local = {coco_name: i for i, (_, coco_name, _) in enumerate(CLASSES)}
    model_to_local = {i: name_to_local[name] for i, name in enumerate(coco_names) if name in name_to_local}
    predictions = []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, ids in loader:
            outputs = model([image.to(device) for image in images])
            for image_id, output in zip(ids, outputs):
                labels = output["labels"].detach().cpu().numpy().astype(int)
                scores = output["scores"].detach().cpu().numpy()
                boxes = output["boxes"].detach().cpu().numpy().astype(np.float32)
                keep = np.asarray(
                    [score >= conf and int(label) in model_to_local for label, score in zip(labels, scores)],
                    dtype=bool,
                )
                predictions.append(
                    {
                        "image_id": image_id,
                        "boxes": boxes[keep].astype(float).tolist(),
                        "labels": [model_to_local[int(label)] for label in labels[keep]],
                        "scores": scores[keep].astype(float).tolist(),
                    }
                )
    elapsed = time.perf_counter() - start
    cache.write_text(json.dumps(predictions), encoding="utf-8")
    return predictions, elapsed, str(device)


def pixel_ground_truth(image_ids: list[str], annotations: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for image_id in image_ids:
        image = cv2.imread(str(DATA / "images/val" / f"{image_id}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image {image_id}")
        height, width = image.shape[:2]
        boxes = []
        for item in annotations[image_id]["boxes"]:
            x1, y1, x2, y2 = item["xyxy_norm"]
            boxes.append(
                {
                    "class": item["class"],
                    "box": np.asarray([x1 * width, y1 * height, x2 * width, y2 * height], dtype=np.float32),
                    "group_of": item["group_of"],
                }
            )
        result[image_id] = {"boxes": boxes, "status": annotations[image_id]["status"]}
    return result


def score_class(predictions: list[dict[str, Any]], ground_truth: dict[str, dict[str, Any]],
                cls: int, iou_threshold: float = 0.5) -> tuple[float, int, int, int, int]:
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    total_gt = 0
    positive_images = negative_images = unannotated_images = 0
    detections = []
    for image_id, record in ground_truth.items():
        status = record["status"].get(cls)
        if status == "positive":
            positive_images += 1
            gt = [x for x in record["boxes"] if x["class"] == cls]
            total_gt += len(gt)
        elif status == "negative":
            negative_images += 1
            gt = []
        else:
            unannotated_images += 1
            gt = []
        gt_by_image[image_id] = gt

    for record in predictions:
        image_id = record["image_id"]
        if ground_truth[image_id]["status"].get(cls) is None:
            continue  # Open Images: ignore detections for unannotated classes
        for box, label, score in zip(record["boxes"], record["labels"], record["scores"]):
            if int(label) == cls:
                detections.append((float(score), image_id, np.asarray(box, dtype=np.float32)))
    detections.sort(key=lambda x: x[0], reverse=True)
    if total_gt == 0:
        return float("nan"), 0, positive_images, negative_images, unannotated_images

    matched = {
        image_id: np.zeros(sum(not x["group_of"] for x in boxes), dtype=bool)
        for image_id, boxes in gt_by_image.items()
    }
    group_matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    tp, fp = [], []
    for _score, image_id, pred_box in detections:
        status = ground_truth[image_id]["status"].get(cls)
        if status == "negative":
            tp.append(0.0); fp.append(1.0)
            continue
        gt = gt_by_image[image_id]
        normal = [x for x in gt if not x["group_of"]]
        group = [x for x in gt if x["group_of"]]
        overlaps = iou_one(pred_box, np.asarray([x["box"] for x in normal], dtype=np.float32).reshape(-1, 4))
        choices = np.argsort(overlaps)[::-1]
        match = next((int(i) for i in choices if overlaps[i] > iou_threshold and not matched[image_id][i]), None)
        if match is not None:
            matched[image_id][match] = True
            tp.append(1.0); fp.append(0.0)
            continue
        group_boxes = np.asarray([x["box"] for x in group], dtype=np.float32).reshape(-1, 4)
        group_overlap = intersection_over_detection(pred_box, group_boxes)
        group_match = next(
            (int(i) for i in np.argsort(group_overlap)[::-1]
             if group_overlap[i] > 0.5 and i not in group_matched[image_id]),
            None,
        )
        if group_match is not None:
            group_matched[image_id].add(group_match)
            tp.append(1.0); fp.append(0.0)
        elif any(group_overlap > 0.5):
            # A group-of region is one GT instance for AP; subsequent detections
            # inside that same region do not create extra TP or FP.
            continue
        else:
            tp.append(0.0); fp.append(1.0)

    if not tp:
        return 0.0, total_gt, positive_images, negative_images, unannotated_images
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))
    return ap, total_gt, positive_images, negative_images, unannotated_images


def confusion_counts(predictions: list[dict[str, Any]], ground_truth: dict[str, dict[str, Any]],
                     conf: float = 0.25, iou_threshold: float = 0.5) -> tuple[np.ndarray, int, int, int]:
    n = len(CLASSES)
    matrix = np.zeros((n + 1, n + 1), dtype=np.int64)
    for prediction in predictions:
        image_id = prediction["image_id"]
        gt_all = ground_truth[image_id]["boxes"]
        status = ground_truth[image_id]["status"]
        pred_items = [
            (np.asarray(box, dtype=np.float32), int(cls), float(score))
            for box, cls, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"])
            if float(score) >= conf and status.get(int(cls)) is not None
        ]
        regular = [x for x in gt_all if not x["group_of"]]
        group = [x for x in gt_all if x["group_of"]]
        pairs = []
        for gi, gt in enumerate(regular):
            for pi, (pbox, _pcls, score) in enumerate(pred_items):
                overlap = float(iou_one(pbox, np.asarray([gt["box"]]))[0])
                if overlap > iou_threshold:
                    pairs.append((overlap, score, gi, pi))
        pairs.sort(reverse=True)
        matched_gt, matched_pred = set(), set()
        for _ov, _score, gi, pi in pairs:
            if gi in matched_gt or pi in matched_pred:
                continue
            matrix[regular[gi]["class"], pred_items[pi][1]] += 1
            matched_gt.add(gi); matched_pred.add(pi)

        ignored_pred: set[int] = set()
        for gi, gt in enumerate(group):
            candidates = []
            for pi, (pbox, _pcls, score) in enumerate(pred_items):
                if pi in matched_pred:
                    continue
                coverage = float(intersection_over_detection(pbox, np.asarray([gt["box"]]))[0])
                if coverage > 0.5:
                    candidates.append((coverage, score, pi))
            if candidates:
                _coverage, _score, pi = max(candidates)
                matrix[gt["class"], pred_items[pi][1]] += 1
                matched_pred.add(pi)
                # One group box contributes one positive instance.
                matched_gt.add(len(regular) + gi)
                for _cov, _score, duplicate_pi in candidates:
                    if duplicate_pi != pi:
                        ignored_pred.add(duplicate_pi)

        for gi, gt in enumerate(regular):
            if gi not in matched_gt:
                matrix[gt["class"], n] += 1
        for gi, gt in enumerate(group):
            if len(regular) + gi not in matched_gt:
                matrix[gt["class"], n] += 1
        for pi, (_box, cls, _score) in enumerate(pred_items):
            if pi not in matched_pred and pi not in ignored_pred:
                matrix[n, cls] += 1

    tp = int(np.trace(matrix[:n, :n]))
    misclassified = int(matrix[:n, :n].sum()) - tp
    fp = int(matrix[n, :n].sum()) + misclassified
    fn = int(matrix[:n, n].sum()) + misclassified
    return matrix, tp, fp, fn


def score_model(model_name: str, predictions: list[dict[str, Any]], elapsed: float,
                ground_truth: dict[str, dict[str, Any]], imgsz: int, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_class = []
    for local_cls, (oi_name, coco_name, coco_id) in enumerate(CLASSES):
        ap, num_gt, pos, neg, unannotated = score_class(predictions, ground_truth, local_cls)
        per_class.append(
            {
                "model": model_name,
                "open_images_class": oi_name,
                "project_class": coco_name,
                "coco_class_id": coco_id,
                "ground_truth_boxes": num_gt,
                "positive_images": pos,
                "negative_images": neg,
                "unannotated_images_ignored": unannotated,
                "AP50": ap,
            }
        )
    valid_ap = [row["AP50"] for row in per_class if np.isfinite(row["AP50"])]
    matrix, tp, fp, fn = confusion_counts(predictions, ground_truth)
    model_key = model_name.lower().replace(" ", "_").replace("/", "_")
    np.savetxt(OUT / f"confusion_{model_key}.csv", matrix, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="Blues")
    labels = [item[1] for item in CLASSES] + ["background"]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=75, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class")
    ax.set_title(f"{model_name}: Open Images subset confusion (IoU>0.5, conf≥0.25)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(OUT / f"confusion_{model_key}.png", dpi=160); plt.close(fig)
    row = {
        "model": model_name,
        "images": len(predictions),
        "ground_truth_boxes": int(sum(x["ground_truth_boxes"] for x in per_class)),
        "classes_scored": len(valid_ap),
        "imgsz": imgsz,
        "device": device,
        "mAP50_open_images": float(np.mean(valid_ap)) if valid_ap else float("nan"),
        "TP_conf025_iou05": tp,
        "FP_conf025_iou05": fp,
        "FN_conf025_iou05": fn,
        "inference_seconds": round(elapsed, 2) if elapsed is not None else "",
        "note": "Open Images protocol subset; unannotated-class predictions ignored; group-of supported; metrics average classes with ground truth",
    }
    return row, per_class


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    global DATA, OUT, SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--source-root", type=Path, help="Official Open Images source CSV directory; defaults to DATA/source.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "ssdlite"])
    parser.add_argument("--refresh", action="store_true", help="Rerun inference instead of reusing prediction caches")
    args = parser.parse_args()
    DATA, OUT = args.data_root.resolve(), args.out.resolve()
    SOURCE = (args.source_root or (DATA / "source")).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    image_ids, annotations = load_ground_truth()
    ground_truth = pixel_ground_truth(image_ids, annotations)
    summary, per_class_all = [], []

    for model_path in args.models:
        if model_path.lower() == "ssdlite":
            predictions, elapsed, device = run_ssdlite(image_ids, args.device, args.conf, args.batch, args.refresh)
            model_name = "SSDLite320-MobileNetV3-Large"
            imgsz = 320
        else:
            predictions, elapsed = run_yolo(
                model_path, image_ids, args.imgsz, args.device, args.conf, args.batch, args.refresh
            )
            model_name = Path(model_path).stem
            device = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
            imgsz = args.imgsz
        row, per_class = score_model(model_name, predictions, elapsed, ground_truth, imgsz, device)
        summary.append(row); per_class_all.extend(per_class)
        print(
            f"{model_name}: mAP50={row['mAP50_open_images']:.4f} "
            f"({row['mAP50_open_images']*100:.2f}%), TP/FP/FN="
            f"{row['TP_conf025_iou05']}/{row['FP_conf025_iou05']}/{row['FN_conf025_iou05']}",
            flush=True,
        )

    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "per_class_ap50.csv", per_class_all)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar([row["model"] for row in summary], [row["mAP50_open_images"] * 100 for row in summary], color=["#3973ac", "#ed9b40", "#5a9e6f"])
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Open Images V7 subset: model comparison (imgsz=320)")
    ax.tick_params(axis="x", rotation=15)
    for index, row in enumerate(summary):
        ax.text(index, row["mAP50_open_images"] * 100 + 1.2, f"{row['mAP50_open_images']*100:.1f}%", ha="center")
    fig.tight_layout(); fig.savefig(OUT / "map50_model_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    class_names = [item[1] for item in CLASSES]
    x = np.arange(len(class_names)); width = 0.25
    for index, model_name in enumerate(row["model"] for row in summary):
        values = [next(r["AP50"] for r in per_class_all if r["model"] == model_name and r["project_class"] == class_name) * 100 for class_name in class_names]
        ax.bar(x + (index - 1) * width, values, width, label=model_name)
    ax.set_xticks(x, class_names, rotation=30, ha="right")
    ax.set_ylabel("AP@0.5 (%)"); ax.set_ylim(0, 100)
    ax.set_title("Per-class AP@0.5 on Open Images V7 subset")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "per_class_ap50.png", dpi=160); plt.close(fig)
    print(f"Saved Open Images evaluation to {OUT}")


if __name__ == "__main__":
    main()

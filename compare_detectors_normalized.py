#!/usr/bin/env python
"""Compare YOLOv8n, YOLOv8s, and SSDLite with 320px inputs and shared AP."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
YOLO = ROOT / "runs/yolo_custom_fullval_seed42/metrics.csv"
YOLO_S = ROOT / "runs/yolo_custom_fullval_yolov8s_img320_seed42/metrics.csv"
SSD = ROOT / "runs/ssdlite_fullval_seed42/metrics.csv"
OUT = ROOT / "runs/normalized_detector_comparison_fullval"
YOLO_N_640_RAW = ROOT / "runs/full_val2017_raw_seed42/metrics.csv"
YOLO_N_640_RECOVERED = ROOT / "runs/full_val2017_recovered_seed42/metrics.csv"
YOLO_N_640_GAUSSIAN_BILATERAL = ROOT / "runs/full_val2017_gaussian_bilateral_seed42/metrics.csv"
YOLO_S_640 = ROOT / "runs/yolov8s_fullval_recovery_policy/metrics.csv"
OUT_640 = ROOT / "runs/yolo_family_comparison_640_fullval"
CONDITIONS = ["clean", "gaussian_0.06", "gaussian_0.06_recovered",
              "salt_pepper_0.06", "salt_pepper_0.06_recovered", "fog_0.5",
              "fog_0.5_recovered", "low_light_0.25", "low_light_0.25_recovered"]
RECOVERY_METHODS = {
    "gaussian_0.06_recovered": "NLM",
    "salt_pepper_0.06_recovered": "median",
    "fog_0.5_recovered": "CLAHE",
    "low_light_0.25_recovered": "CLAHE",
}

def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}


def write_640_family_comparison():
    """Pair existing 640px Ultralytics evaluations under matched conditions."""
    n_raw = read(YOLO_N_640_RAW)
    n_recovered = read(YOLO_N_640_RECOVERED)
    n_gaussian_bilateral = read(YOLO_N_640_GAUSSIAN_BILATERAL)
    s_rows = read(YOLO_S_640)
    conditions = [
        ("clean", "clean", "clean", "none"),
        ("gaussian_0.06", "gaussian_0.06", "gaussian_0.06", "none"),
        ("gaussian_0.06_recovered", "gaussian_0.06_bilateral", "gaussian_0.06_bilateral", "bilateral"),
        ("salt_pepper_0.06", "salt_pepper_0.06", "salt_pepper_0.06", "none"),
        ("salt_pepper_0.06_recovered", "salt_pepper_0.06_recovered", "salt_pepper_0.06_recovered", "median"),
        ("fog_0.5", "fog_0.5", "fog_0.50", "none"),
        ("fog_0.5_recovered", "fog_0.5_recovered", "fog_0.50_CLAHE", "CLAHE"),
        ("low_light_0.25", "low_light_0.25", "low_light_0.25", "none"),
        ("low_light_0.25_recovered", "low_light_0.25_recovered", "low_light_0.25_CLAHE", "CLAHE"),
    ]

    def n_source(source_condition):
        if source_condition == "gaussian_0.06_bilateral":
            return n_gaussian_bilateral[source_condition]
        if source_condition == "clean" or not source_condition.endswith("_recovered"):
            return n_raw[source_condition]
        return n_recovered[source_condition]

    OUT_640.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, n_key, s_key, recovery_method in conditions:
        family = "clean" if condition == "clean" else (
            "salt_pepper" if condition.startswith("salt_pepper") else
            "low_light" if condition.startswith("low_light") else
            condition.split("_", 1)[0]
        )
        level = 0 if condition == "clean" else float(condition.removesuffix("_recovered").rsplit("_", 1)[1])
        for model_name, source in (("YOLOv8n-640", n_source(n_key)), ("YOLOv8s-640", s_rows[s_key])):
            rows.append({
                "condition": condition, "family": family, "level": level,
                "recovered": condition.endswith("_recovered"),
                "recovery_method": recovery_method if condition.endswith("_recovered") else "none",
                "model": model_name, "images": int(source["images"]), "imgsz": 640,
                "mAP50": float(source["mAP50"]), "mAP50_95": float(source["mAP50_95"]),
                "precision_ultralytics": float(source["precision_ultralytics"]),
                "recall_ultralytics": float(source["recall_ultralytics"]),
                "tp_iou50": source.get("tp_iou50", ""),
                "fp_iou50": source.get("fp_iou50", ""),
                "fn_iou50": source.get("fn_iou50", ""),
                "ap_method": "Ultralytics validation; matched 5,000-image conditions",
            })

    with (OUT_640 / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    yolo, yolo_s, ssd = read(YOLO), read(YOLO_S), read(SSD)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        for detector, source in (("YOLOv8n-320", yolo[condition]),
                                 ("YOLOv8s-320", yolo_s[condition]),
                                 ("SSDLite320-MobileNetV3-Large", ssd[condition])):
            rows.append({"condition": condition, "detector": detector,
                         "recovery_method": RECOVERY_METHODS.get(condition, "none"),
                         "images": int(source["images"]), "mAP50": float(source["mAP50"]),
                         "mAP50_95": float(source["mAP50_95"]), "tp_iou50": int(source["tp_iou50"]),
                         "fp_iou50": int(source["fp_iou50"]), "fn_iou50": int(source["fn_iou50"]),
                         "confusion_conf": 0.25, "confusion_iou": 0.5,
                         "input_size": 320, "ap_method": "shared 101-point interpolation, IoU 0.50:0.95"})
    bases = {r["detector"]: r["mAP50"] for r in rows if r["condition"] == "clean"}
    for row in rows:
        base = bases[row["detector"]]
        row["delta_mAP50_vs_clean"] = row["mAP50"] - base
        row["relative_mAP50_change_vs_clean_pct"] = 100 * (row["mAP50"] - base) / base
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_640_family_comparison()

    labels = ["Clean", "Gaussian raw", "Gaussian rec.", "S&P raw", "S&P rec.",
              "Fog raw", "Fog rec.", "Low-light raw", "Low-light rec."]
    detectors = ("YOLOv8n-320", "YOLOv8s-320", "SSDLite320-MobileNetV3-Large")
    x, width = np.arange(len(CONDITIONS)), 0.25
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for i, detector in enumerate(detectors):
        vals = [next(r["mAP50"] for r in rows if r["condition"] == c and r["detector"] == detector) for c in CONDITIONS]
        ax.bar(x + (i - 1) * width, vals, width, label=detector)
    ax.set_ylabel("mAP@0.5 (shared evaluator)")
    ax.set_title("COCO val2017 full split: normalized detector comparison")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "map50_normalized_detector_comparison.png", dpi=180)
    plt.close(fig)

    pairs = [("Gaussian", "gaussian_0.06", "gaussian_0.06_recovered"),
             ("Salt-Pepper", "salt_pepper_0.06", "salt_pepper_0.06_recovered"),
             ("Fog", "fog_0.5", "fog_0.5_recovered"),
             ("Low-light", "low_light_0.25", "low_light_0.25_recovered")]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pairs))
    for i, detector in enumerate(detectors):
        deltas = []
        for _, raw, rec in pairs:
            before = next(r["mAP50"] for r in rows if r["condition"] == raw and r["detector"] == detector)
            after = next(r["mAP50"] for r in rows if r["condition"] == rec and r["detector"] == detector)
            deltas.append(after - before)
        ax.bar(x + (i - 1) * width, deltas, width, label=detector)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ mAP@0.5 (recovered − raw)")
    ax.set_title("Recovery effect with shared evaluator")
    ax.set_xticks(x, [p[0] for p in pairs]); ax.grid(axis="y", alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "recovery_delta_normalized_detector_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote 320px normalized results/charts to {OUT}")
    print(f"Wrote matched 640px YOLO comparison to {OUT_640}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Join full-val YOLOv8n and torchvision SSDLite robustness metrics."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
YOLO = ROOT / "runs/full_val2017_comparison_seed42/raw_vs_recovered.csv"
SSD = ROOT / "runs/ssdlite_fullval_seed42/metrics.csv"
OUT = ROOT / "runs/detector_comparison_fullval"
CONDITIONS = [
    "clean", "gaussian_0.06", "gaussian_0.06_recovered",
    "salt_pepper_0.06", "salt_pepper_0.06_recovered",
    "fog_0.5", "fog_0.5_recovered",
    "low_light_0.25", "low_light_0.25_recovered",
]


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["condition"]: row for row in csv.DictReader(f)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    yolo, ssd = read_csv(YOLO), read_csv(SSD)
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        recovered = condition.endswith("_recovered")
        yolo_condition = condition.removesuffix("_recovered")
        if yolo_condition not in yolo or condition not in ssd:
            raise KeyError(f"Missing condition {condition}: YOLO base={yolo_condition in yolo}, SSD={condition in ssd}")
        yolo_suffix = "_recovered" if recovered else ""
        sources = (
            ("YOLOv8n", yolo[yolo_condition],
             (f"mAP50{yolo_suffix}", f"mAP50_95{yolo_suffix}", f"tp_iou50{yolo_suffix}",
              f"fp_iou50{yolo_suffix}", f"fn_iou50{yolo_suffix}")),
            ("SSDLite320-MobileNetV3-Large", ssd[condition],
             ("mAP50", "mAP50_95", "tp_iou50", "fp_iou50", "fn_iou50")),
        )
        for detector, source, keys in sources:
            if not source[keys[0]]:
                raise ValueError(f"YOLO recovered metrics are not present for {condition}")
            rows.append({"condition": condition, "detector": detector,
                         "images": int(source["images"]),
                         "mAP50": float(source[keys[0]]),
                         "mAP50_95": float(source[keys[1]]),
                         "tp_iou50": int(float(source[keys[2]])),
                         "fp_iou50": int(float(source[keys[3]])),
                         "fn_iou50": int(float(source[keys[4]])),
                         "confusion_conf": 0.25, "confusion_iou": 0.5})

    baselines = {row["detector"]: row["mAP50"] for row in rows if row["condition"] == "clean"}
    for row in rows:
        detector = row["detector"]
        base = float(baselines[detector])
        row["delta_mAP50_vs_clean"] = float(row["mAP50"]) - base
        row["relative_mAP50_change_vs_clean_pct"] = 100 * (float(row["mAP50"]) - base) / base

    fields = list(rows[0])
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Grouped mAP bars: clean, raw corruption and corresponding preprocessing.
    labels = ["Clean", "Gaussian\nraw", "Gaussian\nrecovered", "S&P\nraw", "S&P\nrecovered",
              "Fog\nraw", "Fog\nrecovered", "Low light\nraw", "Low light\nrecovered"]
    x = np.arange(len(CONDITIONS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for i, detector in enumerate(("YOLOv8n", "SSDLite320-MobileNetV3-Large")):
        values = [float(next(r["mAP50"] for r in rows if r["condition"] == cond and r["detector"] == detector))
                  for cond in CONDITIONS]
        ax.bar(x + (i - 0.5) * width, values, width, label=detector)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Full COCO val2017 (5,000 images): detector robustness at selected severe levels")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "map50_detector_comparison.png", dpi=180)
    plt.close(fig)

    # Recovery deltas, shown separately so baseline differences are not hidden.
    pairs = [("Gaussian", "gaussian_0.06", "gaussian_0.06_recovered"),
             ("Salt-Pepper", "salt_pepper_0.06", "salt_pepper_0.06_recovered"),
             ("Fog", "fog_0.5", "fog_0.5_recovered"),
             ("Low-light", "low_light_0.25", "low_light_0.25_recovered")]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pairs))
    for i, detector in enumerate(("YOLOv8n", "SSDLite320-MobileNetV3-Large")):
        deltas = []
        for _, raw, recovered in pairs:
            before = next(float(r["mAP50"]) for r in rows if r["condition"] == raw and r["detector"] == detector)
            after = next(float(r["mAP50"]) for r in rows if r["condition"] == recovered and r["detector"] == detector)
            deltas.append(after - before)
        ax.bar(x + (i - 0.5) * width, deltas, width, label=detector)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ mAP@0.5 (recovered − raw)")
    ax.set_title("Preprocessing recovery effect on full COCO val2017")
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "recovery_delta_detector_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {OUT / 'metrics.csv'} and comparison charts")


if __name__ == "__main__":
    main()

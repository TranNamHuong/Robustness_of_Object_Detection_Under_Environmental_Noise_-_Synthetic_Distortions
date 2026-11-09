#!/usr/bin/env python
"""Compare per-class confusion-count changes for both 320px COCO detectors."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
RUNS = {
    "YOLOv8n-320": ROOT / "runs/yolo_custom_fullval_seed42",
    "SSDLite-320": ROOT / "runs/ssdlite_fullval_seed42",
}
OUT = ROOT / "runs/normalized_detector_comparison_fullval/classwise"
OUT.mkdir(parents=True, exist_ok=True)

data = yaml.safe_load((ROOT / "datasets/coco/full_val2017/data.yaml").read_text(encoding="utf-8"))
raw_names = data["names"]
names = ([raw_names.get(i, raw_names.get(str(i), str(i))) for i in range(len(raw_names))]
         if isinstance(raw_names, dict) else raw_names)
n = len(names)
records = []

for detector, folder in RUNS.items():
    for path in folder.glob("confusion_*.csv"):
        condition = path.stem.removeprefix("confusion_")
        # Seed repeat files are retained separately and must not masquerade as
        # new distortion conditions in the primary per-class table.
        if condition.endswith(("_seed43", "_seed44")):
            continue
        matrix = pd.read_csv(path, header=None).to_numpy()
        if matrix.shape != (n + 1, n + 1):
            continue
        for cls, name in enumerate(names):
            tp = int(matrix[cls, cls])
            # Every row entry is a ground-truth instance, including misses in
            # the final background column.
            support = int(matrix[cls, :].sum())
            predicted = int(matrix[:, cls].sum())
            fn = int(matrix[cls, :].sum() - tp)
            fp = int(matrix[:, cls].sum() - tp)
            records.append({"detector": detector, "condition": condition,
                            "class_id": cls, "class_name": name, "tp": tp,
                            "fp": fp, "fn": fn, "support": support,
                            "predicted": predicted,
                            "precision_at_conf025_iou05": tp / predicted if predicted else 0.0,
                            "recall_at_conf025_iou05": tp / support if support else 0.0})

frame = pd.DataFrame(records)
frame.to_csv(OUT / "classwise_confusion_metrics.csv", index=False)

clean = (frame[frame.condition == "clean"]
         [["detector", "class_id", "class_name", "fp", "fn",
           "precision_at_conf025_iou05", "recall_at_conf025_iou05"]]
         .rename(columns={"fp": "fp_clean", "fn": "fn_clean",
                          "precision_at_conf025_iou05": "precision_clean",
                          "recall_at_conf025_iou05": "recall_clean"}))
deltas = frame.merge(clean, on=["detector", "class_id", "class_name"], how="left")
deltas["fp_delta_vs_clean"] = deltas.fp - deltas.fp_clean
deltas["fn_delta_vs_clean"] = deltas.fn - deltas.fn_clean
deltas["precision_delta_vs_clean_pp"] = 100 * (deltas.precision_at_conf025_iou05 - deltas.precision_clean)
deltas["recall_delta_vs_clean_pp"] = 100 * (deltas.recall_at_conf025_iou05 - deltas.recall_clean)
deltas.to_csv(OUT / "classwise_deltas_vs_clean.csv", index=False)

raw_condition = "salt_pepper_0.06"
recovered_condition = "salt_pepper_0.06_recovered"
fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
for row, detector in enumerate(RUNS):
    subset = deltas[deltas.detector == detector].set_index(["condition", "class_id"])
    rows = []
    for cls, name in enumerate(names):
        try:
            raw = subset.loc[(raw_condition, cls)]
            recovered = subset.loc[(recovered_condition, cls)]
        except KeyError:
            continue
        rows.append({"class_name": name,
                     "fp_raw_delta": raw.fp_delta_vs_clean,
                     "fp_recovered_delta": recovered.fp_delta_vs_clean,
                     "fn_raw_delta": raw.fn_delta_vs_clean,
                     "fn_recovered_delta": recovered.fn_delta_vs_clean})
    plot = pd.DataFrame(rows)
    plot["rank"] = plot.fp_raw_delta.clip(lower=0) + plot.fn_raw_delta.clip(lower=0)
    top = plot.nlargest(12, "rank").sort_values("rank")
    for ax, raw_col, rec_col, title, xlabel in [
        (axes[row, 0], "fp_raw_delta", "fp_recovered_delta", "False positive change", "Δ FP vs clean"),
        (axes[row, 1], "fn_raw_delta", "fn_recovered_delta", "Missed object change", "Δ FN vs clean"),
    ]:
        y = range(len(top))
        ax.barh([v - 0.18 for v in y], top[raw_col], height=0.34, label="Raw S&P 0.06")
        ax.barh([v + 0.18 for v in y], top[rec_col], height=0.34, label="Median recovered")
        ax.set_yticks(list(y), labels=top.class_name)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{detector}: {title}")
        ax.grid(axis="x", alpha=0.25)
        ax.legend()

fig.suptitle("Full COCO val2017: per-class errors under severe Salt-and-Pepper (320px)")
fig.savefig(OUT / "salt_pepper_0.06_class_errors_both_detectors.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
for ax, detector in zip(axes, RUNS):
    subset = deltas[deltas.detector == detector]
    raw = subset[subset.condition == raw_condition].set_index("class_id")
    recovered = subset[subset.condition == recovered_condition].set_index("class_id")
    plot = pd.DataFrame({"class_name": raw.class_name,
                         "raw_recall_delta_pp": raw.recall_delta_vs_clean_pp,
                         "recovered_recall_delta_pp": recovered.recall_delta_vs_clean_pp})
    top = plot.nsmallest(12, "raw_recall_delta_pp").sort_values("raw_recall_delta_pp")
    y = range(len(top))
    ax.barh([v - 0.18 for v in y], top.raw_recall_delta_pp, height=0.34, label="Raw S&P 0.06")
    ax.barh([v + 0.18 for v in y], top.recovered_recall_delta_pp, height=0.34,
            label="Median recovered")
    ax.set_yticks(list(y), labels=top.class_name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ recall vs clean (percentage points)")
    ax.set_title(detector)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
fig.suptitle("Full COCO val2017: class recall loss under severe Salt-and-Pepper")
fig.savefig(OUT / "salt_pepper_0.06_class_recall_both_detectors.png", dpi=180)
plt.close(fig)

print(f"Wrote {len(frame)} class-condition rows for {frame.detector.nunique()} detectors and "
      f"{frame.condition.nunique()} conditions to {OUT}")
for detector in RUNS:
    part = deltas[(deltas.detector == detector) & (deltas.condition == raw_condition)]
    print(f"\n{detector} — largest recall drops under raw S&P 0.06 (percentage points):")
    print(part.nsmallest(8, "recall_delta_vs_clean_pp")
          [["class_name", "recall_clean", "recall_at_conf025_iou05",
            "recall_delta_vs_clean_pp", "support"]]
          .to_string(index=False))
    print(f"\n{detector} — largest FP increases under raw S&P 0.06:")
    print(part.nlargest(8, "fp_delta_vs_clean")
          [["class_name", "fp_delta_vs_clean", "fn_delta_vs_clean", "support"]]
          .to_string(index=False))
    print(f"{detector} — largest FN increases under raw S&P 0.06:")
    print(part.nlargest(8, "fn_delta_vs_clean")
          [["class_name", "fp_delta_vs_clean", "fn_delta_vs_clean", "support"]]
          .to_string(index=False))

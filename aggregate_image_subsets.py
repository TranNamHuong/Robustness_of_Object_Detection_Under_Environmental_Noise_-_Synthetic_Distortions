#!/usr/bin/env python
"""Aggregate severe-condition YOLO results across disjoint COCO image subsets."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SETS = [
    ("image_set_1", ROOT / "datasets/coco/subset_500/manifest.json",
     ROOT / "runs/noise_sweep_seed42/metrics.csv"),
    ("image_set_2", ROOT / "datasets/coco/subset_500_image_set2/manifest.json",
     ROOT / "runs/image_subset_set2/metrics.csv"),
    ("image_set_3", ROOT / "datasets/coco/subset_500_image_set3/manifest.json",
     ROOT / "runs/image_subset_set3/metrics.csv"),
]
CONDITIONS = ["clean", "gaussian_0.06", "gaussian_0.06_recovered",
              "salt_pepper_0.06", "salt_pepper_0.06_recovered", "fog_0.5",
              "fog_0.5_recovered", "low_light_0.25", "low_light_0.25_recovered"]
METRICS = ["mAP50", "mAP50_95", "precision_ultralytics", "recall_ultralytics",
           "tp_iou50", "fp_iou50", "fn_iou50"]
OUT = ROOT / "runs/image_subset_analysis"
OUT.mkdir(parents=True, exist_ok=True)

frames = []
seen_image_ids: dict[str, set[int]] = {}
for subset, manifest_path, metrics_path in SETS:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = {int(row["image_id"]) for row in manifest["images"]}
    if len(ids) != int(manifest["selected_images"]):
        raise ValueError(f"Duplicate image IDs within {subset}")
    for earlier, previous_ids in seen_image_ids.items():
        overlap = ids & previous_ids
        if overlap:
            raise ValueError(f"{subset} overlaps {earlier} by {len(overlap)} images")
    seen_image_ids[subset] = ids

    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["condition"].isin(CONDITIONS)].copy()
    missing = sorted(set(CONDITIONS) - set(metrics["condition"]))
    if missing:
        raise FileNotFoundError(f"{subset} is missing conditions: {missing}")
    metrics.insert(0, "image_subset", subset)
    metrics["instances"] = sum(int(row["annotations"]) for row in manifest["images"])
    frames.append(metrics)

by_subset = pd.concat(frames, ignore_index=True)
by_subset.to_csv(OUT / "metrics_by_image_subset.csv", index=False)

clean = by_subset[by_subset.condition == "clean"].set_index("image_subset")["mAP50"]
by_subset["delta_mAP50_vs_clean"] = by_subset.apply(
    lambda row: row["mAP50"] - clean[row["image_subset"]], axis=1)
by_subset["relative_change_vs_clean_pct"] = by_subset.apply(
    lambda row: 100 * row["delta_mAP50_vs_clean"] / clean[row["image_subset"]], axis=1)

summary = (by_subset.groupby(["condition", "family", "level", "recovered"], sort=False)
           [METRICS + ["delta_mAP50_vs_clean", "relative_change_vs_clean_pct"]]
           .agg(["mean", "std"]).reset_index())
summary.columns = ["_".join(str(part) for part in col if part != "")
                   if isinstance(col, tuple) else col for col in summary.columns]
summary = summary.rename(columns={"condition_": "condition", "family_": "family",
                                  "level_": "level", "recovered_": "recovered"})
summary.insert(0, "n_image_subsets", 3)
summary.to_csv(OUT / "metrics_summary_by_image_subset.csv", index=False)

labels = ["Clean", "Gaussian raw", "Gaussian NLM", "S&P raw", "S&P median",
          "Fog raw", "Fog CLAHE", "Low-light raw", "Low-light CLAHE"]
fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
x = np.arange(len(CONDITIONS))
means = summary["mAP50_mean"].to_numpy()
stds = summary["mAP50_std"].fillna(0).to_numpy()
ax.errorbar(x, means, yerr=stds, fmt="o", capsize=4, linewidth=1.4,
            markersize=6, label="Mean ± sample SD across image subsets")
offsets = [-0.11, 0.0, 0.11]
for i, subset in enumerate(by_subset.image_subset.unique()):
    part = by_subset[by_subset.image_subset == subset].set_index("condition").loc[CONDITIONS]
    ax.scatter(x + offsets[i], part["mAP50"], s=22, alpha=0.55, label=subset)
ax.set_xticks(x, labels, rotation=25, ha="right")
ax.set_ylabel("YOLOv8n mAP@0.5 (imgsz 640)")
ax.set_title("Severe distortion conditions across three disjoint COCO val subsets")
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=2)
fig.savefig(OUT / "mAP50_by_image_subset.png", dpi=180)
plt.close(fig)

print(f"Verified {len(seen_image_ids)} pairwise-disjoint subsets; wrote "
      f"{len(by_subset)} subset-condition rows to {OUT}")
print(summary[["condition", "mAP50_mean", "mAP50_std",
               "relative_change_vs_clean_pct_mean", "relative_change_vs_clean_pct_std"]]
      .to_string(index=False))

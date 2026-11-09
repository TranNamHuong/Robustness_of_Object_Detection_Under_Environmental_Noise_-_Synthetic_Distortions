#!/usr/bin/env python
"""Summarize per-class detection recall across three COCO image subsets."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SUBSETS = {
    "subset1": ROOT / "runs/noise_sweep_seed42",
    "subset2": ROOT / "runs/image_subset_set2",
    "subset3": ROOT / "runs/image_subset_set3",
}
CONDITIONS = {
    "clean": "Clean",
    "salt_pepper_0.06": "S&P 0.06 raw",
    "salt_pepper_0.06_recovered": "S&P 0.06 median",
}


def class_names():
    data = yaml.safe_load((ROOT / "datasets/coco/full_val2017/data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names.get(i, names.get(str(i), str(i))) for i in range(len(names))]
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/classwise_subset_analysis")
    args = parser.parse_args()
    names = class_names()
    records = []

    for subset, root in SUBSETS.items():
        for condition, label in CONDITIONS.items():
            path = root / f"confusion_{condition}.csv"
            matrix = pd.read_csv(path, header=None).to_numpy()
            if matrix.shape != (len(names) + 1, len(names) + 1):
                raise ValueError(f"Unexpected confusion matrix shape at {path}: {matrix.shape}")
            for class_id, name in enumerate(names):
                tp = int(matrix[class_id, class_id])
                support = int(matrix[class_id, :].sum())
                fn = support - tp
                predicted = int(matrix[:, class_id].sum())
                fp = predicted - tp
                records.append({
                    "subset": subset, "condition": label, "class_id": class_id,
                    "class_name": name, "support": support, "tp": tp, "fn": fn,
                    "fp": fp, "recall": tp / support if support else np.nan,
                    "precision": tp / predicted if predicted else np.nan,
                })

    detail = pd.DataFrame(records)
    wide = detail.pivot(index=["subset", "class_id", "class_name"],
                        columns="condition", values=["support", "tp", "fn", "fp", "recall"])
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    for condition in ("S&P 0.06 raw", "S&P 0.06 median"):
        wide[f"recall_delta_pp_{condition}"] = 100 * (
            wide[f"recall_{condition}"] - wide["recall_Clean"])
        wide[f"fn_delta_{condition}"] = wide[f"fn_{condition}"] - wide["fn_Clean"]

    summary_rows = []
    for class_id, group in wide.groupby(["class_id", "class_name"], sort=True):
        row = {"class_id": class_id[0], "class_name": class_id[1],
               "support_total": int(group["support_Clean"].sum()),
               "n_subsets_with_support": int(group["recall_Clean"].notna().sum())}
        for condition in ("S&P 0.06 raw", "S&P 0.06 median"):
            deltas = group[f"recall_delta_pp_{condition}"].dropna().to_numpy(dtype=float)
            row[f"recall_delta_pp_mean_{condition}"] = float(np.mean(deltas)) if len(deltas) else np.nan
            row[f"recall_delta_pp_sd_{condition}"] = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
            fn_deltas = group[f"fn_delta_{condition}"].to_numpy(dtype=float)
            row[f"fn_delta_mean_{condition}"] = float(np.mean(fn_deltas))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    detail_path = args.out / "classwise_errors_by_subset.csv"
    summary_path = args.out / "classwise_errors_summary.csv"
    detail.to_csv(detail_path, index=False)
    wide.to_csv(args.out / "classwise_errors_deltas_by_subset.csv", index=False)
    summary.to_csv(summary_path, index=False)

    # Rank only classes with at least 30 clean ground-truth instances pooled
    # across the three subsets; show how recall changes for raw and recovered.
    ranked = summary[summary.support_total >= 30].nsmallest(
        15, "recall_delta_pp_mean_S&P 0.06 raw").sort_values(
            "recall_delta_pp_mean_S&P 0.06 raw")
    fig, ax = plt.subplots(figsize=(11, 7.5), constrained_layout=True)
    positions = np.arange(len(ranked))
    raw = ranked["recall_delta_pp_mean_S&P 0.06 raw"].to_numpy()
    raw_sd = ranked["recall_delta_pp_sd_S&P 0.06 raw"].to_numpy()
    med = ranked["recall_delta_pp_mean_S&P 0.06 median"].to_numpy()
    med_sd = ranked["recall_delta_pp_sd_S&P 0.06 median"].to_numpy()
    ax.barh(positions - 0.18, raw, height=0.34, xerr=raw_sd, capsize=2, label="Raw S&P 0.06")
    ax.barh(positions + 0.18, med, height=0.34, xerr=med_sd, capsize=2, label="Median recovered")
    ax.set_yticks(positions, ranked["class_name"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Recall change vs clean (percentage points; mean ± SD across subsets)")
    ax.set_title("Classes most affected by Salt-and-Pepper across COCO subsets")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    chart_path = args.out / "classwise_salt_pepper_recall_change.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    print(f"Saved per-subset class counts: {detail_path}")
    print(f"Saved per-class summary: {summary_path}")
    print(f"Saved chart: {chart_path}")
    print("Largest mean recall losses (minimum 30 clean instances pooled):")
    print(ranked[["class_name", "support_total",
                  "recall_delta_pp_mean_S&P 0.06 raw",
                  "recall_delta_pp_mean_S&P 0.06 median"]].to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compare per-class recall changes for all severe COCO distortions."""
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
    "clean": ("Clean", None, None),
    "gaussian_0.06": ("Gaussian 0.06", "Gaussian", "raw"),
    "gaussian_0.06_recovered": ("Gaussian 0.06", "Gaussian", "recovered"),
    "salt_pepper_0.06": ("S&P 0.06", "S&P", "raw"),
    "salt_pepper_0.06_recovered": ("S&P 0.06", "S&P", "recovered"),
    "fog_0.5": ("Fog 0.50", "Fog", "raw"),
    "fog_0.5_recovered": ("Fog 0.50", "Fog", "recovered"),
    "low_light_0.25": ("Low-light 0.25", "Low-light", "raw"),
    "low_light_0.25_recovered": ("Low-light 0.25", "Low-light", "recovered"),
}
FAMILIES = ["Gaussian", "S&P", "Fog", "Low-light"]


def load_class_names():
    data = yaml.safe_load((ROOT / "datasets/coco/full_val2017/data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names.get(i, names.get(str(i), str(i))) for i in range(len(names))]
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-support", type=int, default=30,
                        help="Minimum pooled clean GT support to include a class in plots")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/classwise_all_conditions_subsets")
    args = parser.parse_args()
    names = load_class_names()
    records = []

    for subset, root in SUBSETS.items():
        matrices = {}
        for condition_key in CONDITIONS:
            path = root / f"confusion_{condition_key}.csv"
            matrix = pd.read_csv(path, header=None).to_numpy()
            if matrix.shape != (len(names) + 1, len(names) + 1):
                raise ValueError(f"Unexpected confusion matrix shape at {path}: {matrix.shape}")
            matrices[condition_key] = matrix
        clean = matrices["clean"]
        for condition_key, (label, family, status) in CONDITIONS.items():
            matrix = matrices[condition_key]
            for class_id, class_name in enumerate(names):
                tp = int(matrix[class_id, class_id])
                support = int(matrix[class_id, :].sum())
                predicted = int(matrix[:, class_id].sum())
                recall = tp / support if support else np.nan
                precision = tp / predicted if predicted else np.nan
                clean_tp = int(clean[class_id, class_id])
                clean_support = int(clean[class_id, :].sum())
                clean_recall = clean_tp / clean_support if clean_support else np.nan
                records.append({
                    "subset": subset, "condition": label, "family": family or "clean",
                    "status": status or "clean", "class_id": class_id,
                    "class_name": class_name, "support": support, "tp": tp,
                    "fp": predicted - tp, "fn": support - tp,
                    "recall": recall, "precision": precision,
                    "clean_support": clean_support, "clean_recall": clean_recall,
                    "recall_delta_pp": 100 * (recall - clean_recall)
                    if np.isfinite(recall) and np.isfinite(clean_recall) else np.nan,
                })

    detail = pd.DataFrame(records)
    evaluated = detail[detail.status != "clean"].copy()
    summary = (evaluated.groupby(["family", "condition", "status", "class_id", "class_name"],
                                as_index=False)
               .agg(support_total=("clean_support", "sum"),
                    support_condition_total=("support", "sum"),
                    n_subsets_with_support=("recall_delta_pp", "count"),
                    recall_delta_pp_mean=("recall_delta_pp", "mean"),
                    recall_delta_pp_sd=("recall_delta_pp", "std"),
                    fn_mean=("fn", "mean"), fn_sd=("fn", "std")))
    summary["recall_delta_pp_sd"] = summary["recall_delta_pp_sd"].fillna(0.0)
    summary["fn_sd"] = summary["fn_sd"].fillna(0.0)
    effect = summary.pivot(index=["family", "class_id", "class_name", "support_total"],
                           columns="status", values="recall_delta_pp_mean").reset_index()
    effect.columns.name = None
    effect = effect.rename(columns={"raw": "raw_recall_delta_pp",
                                   "recovered": "recovered_recall_delta_pp"})
    effect["recovery_gain_pp"] = effect["recovered_recall_delta_pp"] - effect["raw_recall_delta_pp"]

    args.out.mkdir(parents=True, exist_ok=True)
    detail_path = args.out / "classwise_all_conditions_by_subset.csv"
    summary_path = args.out / "classwise_all_conditions_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    effect.to_csv(args.out / "classwise_recovery_effect_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    for ax, family in zip(axes.flat, FAMILIES):
        family_rows = summary[(summary.family == family) &
                              (summary.support_total >= args.min_support)]
        raw = family_rows[family_rows.status == "raw"].nsmallest(8, "recall_delta_pp_mean")
        if raw.empty:
            ax.set_visible(False)
            continue
        class_order = raw.sort_values("recall_delta_pp_mean")["class_name"].tolist()
        selected = family_rows[family_rows.class_name.isin(class_order)]
        raw_plot = (selected[selected.status == "raw"]
                    .set_index("class_name").reindex(class_order))
        recovered_plot = (selected[selected.status == "recovered"]
                          .set_index("class_name").reindex(class_order))
        y = np.arange(len(class_order))
        ax.barh(y - 0.18, raw_plot.recall_delta_pp_mean, height=0.34,
                xerr=raw_plot.recall_delta_pp_sd, capsize=2, label="Raw")
        ax.barh(y + 0.18, recovered_plot.recall_delta_pp_mean, height=0.34,
                xerr=recovered_plot.recall_delta_pp_sd, capsize=2, label="Recovered")
        ax.set_yticks(y, class_order)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(family)
        ax.set_xlabel("Recall change vs clean (percentage points)")
        ax.grid(axis="x", alpha=0.25)
        ax.legend()
    fig.suptitle("Per-class recall under severe distortions across three COCO subsets")
    chart_path = args.out / "classwise_recall_all_families.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    print(f"Saved per-subset class metrics: {detail_path}")
    print(f"Saved mean/SD summary: {summary_path}")
    print(f"Saved chart: {chart_path}")
    for family in FAMILIES:
        rows = summary[(summary.family == family) & (summary.status == "raw") &
                       (summary.support_total >= args.min_support)]
        print(f"\n{family}: greatest recall losses (raw)")
        print(rows.nsmallest(5, "recall_delta_pp_mean")[[
            "class_name", "support_total", "recall_delta_pp_mean", "recall_delta_pp_sd"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()

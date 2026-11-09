#!/usr/bin/env python
"""Compare baseline and noise-aware YOLOv8n on the same full-val sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "runs/full_val2017_raw_seed42/metrics.csv"
TRAINED = ROOT / "runs/noiseaware_yolov8n_fullval_sweep_seed42/metrics.csv"
OUT = ROOT / "runs/noiseaware_yolov8n_fullval_sweep_seed42"
SERIES = {
    "Gaussian": ("gaussian", [0.01, 0.03, 0.06]),
    "Salt-and-Pepper": ("salt_pepper", [0.01, 0.03, 0.06]),
    "Fog": ("fog", [0.15, 0.30, 0.50]),
    "Low light (smaller factor = darker)": ("low_light", [0.60, 0.40, 0.25]),
}


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row["condition"]: row for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--trained", type=Path, default=TRAINED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    baseline, trained = read_csv(args.baseline), read_csv(args.trained)
    conditions = ["clean"] + [f"{family}_{level:g}" for family, levels in SERIES.values()
                               for level in levels]
    clean_base = float(baseline["clean"]["mAP50"])
    clean_trained = float(trained["clean"]["mAP50"])
    rows = []
    for condition in conditions:
        if condition not in baseline or condition not in trained:
            raise KeyError(f"Condition missing from one result file: {condition}")
        b, t = float(baseline[condition]["mAP50"]), float(trained[condition]["mAP50"])
        b95 = float(baseline[condition]["mAP50_95"])
        t95 = float(trained[condition]["mAP50_95"])
        rows.append({"condition": condition, "baseline_mAP50": b,
                     "noiseaware_mAP50": t, "delta_mAP50_pp": (t - b) * 100,
                     "baseline_drop_from_clean_pp": (b - clean_base) * 100,
                     "noiseaware_drop_from_clean_pp": (t - clean_trained) * 100,
                     "relative_drop_change_pp": ((t - clean_trained) - (b - clean_base)) * 100,
                     "baseline_mAP50_95": b95, "noiseaware_mAP50_95": t95,
                     "delta_mAP50_95_pp": (t95 - b95) * 100})
    comparison_path = args.out / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (title, (family, levels)) in zip(axes.flat, SERIES.items()):
        xs = list(range(len(levels)))
        base_values = [float(baseline[f"{family}_{level:g}"]["mAP50"]) * 100 for level in levels]
        new_values = [float(trained[f"{family}_{level:g}"]["mAP50"]) * 100 for level in levels]
        axis.plot(xs, base_values, marker="o", label="YOLOv8n baseline")
        axis.plot(xs, new_values, marker="o", label="Noise-aware fine-tune")
        axis.set_xticks(xs, [f"{level:g}" for level in levels])
        axis.set_xlabel("Severity" if family != "low_light" else "Brightness factor")
        axis.set_ylabel("mAP@0.5 (%)")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes.flat[0].legend()
    fig.suptitle("YOLOv8n: baseline vs noise-aware training — full COCO val2017")
    fig.savefig(args.out / "mAP50_corruption_curves.png", dpi=180)
    plt.close(fig)
    avg_noisy_delta = sum(row["delta_mAP50_pp"] for row in rows[1:]) / (len(rows) - 1)
    avg_relative_drop = sum(row["relative_drop_change_pp"] for row in rows[1:]) / (len(rows) - 1)
    avg_base_noisy = sum(row["baseline_mAP50"] for row in rows[1:]) / (len(rows) - 1)
    avg_trained_noisy = sum(row["noiseaware_mAP50"] for row in rows[1:]) / (len(rows) - 1)
    print(f"Saved {comparison_path}")
    print(f"Saved {args.out / 'mAP50_corruption_curves.png'}")
    print(f"Mean delta over 12 noisy conditions: {avg_noisy_delta:+.2f} pp")
    print(f"Mean noisy mAP50: {avg_base_noisy*100:.2f}% -> {avg_trained_noisy*100:.2f}%")
    print(f"Mean reduction in clean-to-noisy drop: {avg_relative_drop:+.2f} pp "
          "(positive means less degradation from its own clean score)")
    for row in rows:
        print(f"{row['condition']}: {row['baseline_mAP50']*100:.2f}% -> "
              f"{row['noiseaware_mAP50']*100:.2f}% ({row['delta_mAP50_pp']:+.2f} pp)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Summarize SSDLite per-class confusion changes versus its clean baseline."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/ssdlite_fullval_seed42"
OUT = RUN / "classwise"
OUT.mkdir(parents=True, exist_ok=True)
data = yaml.safe_load((ROOT / "datasets/coco/full_val2017/data.yaml").read_text(encoding="utf-8"))
raw_names = data["names"]
names = [raw_names.get(i, raw_names.get(str(i), str(i))) for i in range(len(raw_names))] if isinstance(raw_names, dict) else raw_names
n = len(names)
records = []
for path in RUN.glob("confusion_*.csv"):
    matrix = pd.read_csv(path, header=None).to_numpy()
    if matrix.shape != (n + 1, n + 1):
        continue
    condition = path.stem.removeprefix("confusion_")
    for cls, name in enumerate(names):
        tp = int(matrix[cls, cls])
        # Every row entry is a ground-truth instance, including misses in the
        # final background column.
        support = int(matrix[cls, :].sum())
        predicted = int(matrix[:, cls].sum())
        fn = int(matrix[cls, :].sum() - tp)
        fp = int(matrix[:, cls].sum() - tp)
        records.append({"condition": condition, "class_id": cls, "class_name": name,
                        "tp": tp, "fp": fp, "fn": fn, "support": support,
                        "precision_at_conf025_iou05": tp / predicted if predicted else 0.0,
                        "recall_at_conf025_iou05": tp / support if support else 0.0})
frame = pd.DataFrame(records)
frame.to_csv(OUT / "classwise_confusion_metrics.csv", index=False)
base = frame[frame.condition == "clean"]["class_id class_name fp fn".split()].rename(
    columns={"fp": "fp_clean", "fn": "fn_clean"})
deltas = frame.merge(base, on=["class_id", "class_name"], how="left")
deltas["fp_delta_vs_clean"] = deltas.fp - deltas.fp_clean
deltas["fn_delta_vs_clean"] = deltas.fn - deltas.fn_clean
deltas.to_csv(OUT / "classwise_deltas_vs_clean.csv", index=False)

raw = frame[frame.condition == "salt_pepper_0.06"].set_index("class_id")
rec = frame[frame.condition == "salt_pepper_0.06_recovered"].set_index("class_id")
clean = frame[frame.condition == "clean"].set_index("class_id")
plot = pd.DataFrame({"class_name": raw.class_name,
                     "fp_raw": raw.fp - clean.fp, "fp_recovered": rec.fp - clean.fp,
                     "fn_raw": raw.fn - clean.fn, "fn_recovered": rec.fn - clean.fn})
plot["rank"] = plot.fp_raw.clip(lower=0) + plot.fn_raw.clip(lower=0)
top = plot.nlargest(15, "rank").sort_values("rank")
fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
for ax, a, b, title, xlabel in [
    (axes[0], "fp_raw", "fp_recovered", "FP change by class", "Δ FP vs clean"),
    (axes[1], "fn_raw", "fn_recovered", "FN change by class", "Δ FN vs clean")]:
    y = range(len(top))
    ax.barh([v - 0.18 for v in y], top[a], height=0.34, label="Raw S&P 0.06")
    ax.barh([v + 0.18 for v in y], top[b], height=0.34, label="Median recovered")
    ax.set_yticks(list(y), labels=top.class_name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
fig.suptitle("SSDLite on full COCO val2017: class errors under severe Salt-and-Pepper")
fig.savefig(OUT / "salt_pepper_0.06_class_errors.png", dpi=180)
plt.close(fig)

print(f"Wrote per-class results for {frame.condition.nunique()} conditions to {OUT}")
print("Largest positive FP deltas under raw S&P 0.06:")
print(deltas[deltas.condition == "salt_pepper_0.06"].nlargest(10, "fp_delta_vs_clean")[["class_name", "fp_delta_vs_clean", "fn_delta_vs_clean", "support"]].to_string(index=False))
print("Largest positive FN deltas under raw S&P 0.06:")
print(deltas[deltas.condition == "salt_pepper_0.06"].nlargest(10, "fn_delta_vs_clean")[["class_name", "fp_delta_vs_clean", "fn_delta_vs_clean", "support"]].to_string(index=False))

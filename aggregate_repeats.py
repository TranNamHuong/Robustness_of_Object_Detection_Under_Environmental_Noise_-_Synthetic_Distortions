from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
SOURCES = [(42, RUNS / "noise_sweep_seed42" / "metrics.csv"),
           (43, RUNS / "repeat_seed43" / "metrics.csv"),
           (44, RUNS / "repeat_seed44" / "metrics.csv")]
OUT = RUNS / "repeat_analysis_seed42_43_44"
OUT.mkdir(parents=True, exist_ok=True)

frames = []
for seed, path in SOURCES:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame.insert(0, "seed", seed)
    frames.append(frame)
all_rows = pd.concat(frames, ignore_index=True)
all_rows.to_csv(OUT / "metrics_by_seed.csv", index=False)

metrics = ["mAP50", "mAP50_95", "precision_ultralytics", "recall_ultralytics",
           "tp_iou50", "fp_iou50", "fn_iou50"]
groups = all_rows.groupby(["condition", "family", "level", "recovered"], sort=False)
summary = groups[metrics].agg(["mean", "std"]).reset_index()
summary.columns = ["_".join(str(x) for x in col if x != "") if isinstance(col, tuple) else col
                   for col in summary.columns]
summary = summary.rename(columns={"condition_": "condition", "family_": "family",
                                  "level_": "level", "recovered_": "recovered"})
summary.insert(0, "n_seeds", groups.size().values)
summary.to_csv(OUT / "metrics_summary.csv", index=False)

families = ["gaussian", "salt_pepper", "fog", "low_light"]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
for ax, family in zip(axes, families):
    subset = summary[summary["family"] == family]
    for recovered, label, marker in [(False, "Raw", "o"), (True, "Recovered", "s")]:
        part = subset[subset["recovered"] == recovered].sort_values("level")
        ax.errorbar(part["level"], part["mAP50_mean"], yerr=part["mAP50_std"].fillna(0),
                    marker=marker, capsize=3, label=label)
    ax.set_title(family.replace("_", " ").title())
    ax.set_xlabel("Distortion level")
    ax.set_ylabel("mAP@0.5")
    ax.grid(True, alpha=0.25)
    ax.legend()
fig.suptitle("YOLOv8n robustness across three noise seeds (mean ± sample SD)")
fig.savefig(OUT / "map_vs_intensity_mean_std.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
for ax, family in zip(axes, families):
    subset = summary[summary["family"] == family]
    for recovered, label, marker in [(False, "Raw", "o"), (True, "Recovered", "s")]:
        part = subset[subset["recovered"] == recovered].sort_values("level")
        ax.errorbar(part["level"], part["fp_iou50_mean"], yerr=part["fp_iou50_std"].fillna(0),
                    marker=marker, capsize=3, label=label)
    ax.set_title(family.replace("_", " ").title())
    ax.set_xlabel("Distortion level")
    ax.set_ylabel("False positives")
    ax.grid(True, alpha=0.25)
    ax.legend()
fig.suptitle("False positives at confidence 0.25, IoU 0.5 (mean ± sample SD)")
fig.savefig(OUT / "false_positives_mean_std.png", dpi=180)
plt.close(fig)

print(f"Wrote {len(all_rows)} per-seed rows and {len(summary)} summary rows to {OUT}")
print(summary[["condition", "n_seeds", "mAP50_mean", "mAP50_std", "fp_iou50_mean", "fp_iou50_std"]].to_string(index=False))

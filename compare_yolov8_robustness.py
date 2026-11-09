"""Join full COCO val robustness results for YOLOv8n and YOLOv8s."""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "yolov8_model_robustness_comparison"
OUT.mkdir(parents=True, exist_ok=True)

n_raw = pd.read_csv(ROOT / "runs/full_val2017_raw_seed42/metrics.csv")
n_recovered = pd.read_csv(ROOT / "runs/full_val2017_recovered_seed42/metrics.csv")
n_gaussian_recovered = pd.read_csv(
    ROOT / "runs/full_val2017_gaussian_bilateral_seed42/metrics.csv"
)
s_results = pd.read_csv(ROOT / "runs/yolov8s_fullval_recovery_policy/metrics.csv")

conditions = [
    ("clean", "Clean", "clean", "clean"),
    ("gaussian_0.06", "Gaussian 0.06", "gaussian_0.06", "gaussian_0.06"),
    ("salt_pepper_0.06", "Salt-and-pepper 0.06", "salt_pepper_0.06", "salt_pepper_0.06"),
    ("fog_0.50", "Fog 0.50", "fog_0.5", "fog_0.50"),
    ("low_light_0.25", "Low-light 0.25", "low_light_0.25", "low_light_0.25"),
]

def value(frame, condition, metric="mAP50"):
    rows = frame.loc[frame["condition"] == condition, metric]
    if rows.empty:
        raise ValueError(f"Missing condition {condition!r}")
    return float(rows.iloc[0])


rows = []
for key, label, n_raw_condition, s_raw_condition in conditions:
    n_clean = value(n_raw, "clean")
    s_clean = value(s_results, "clean")
    n_recovery_frame = n_gaussian_recovered if key.startswith("gaussian") else n_recovered
    n_rec_condition = {
        "gaussian_0.06": "gaussian_0.06_bilateral",
        "salt_pepper_0.06": "salt_pepper_0.06_recovered",
        "fog_0.50": "fog_0.5_recovered",
        "low_light_0.25": "low_light_0.25_recovered",
    }.get(key)
    n_raw_map = n_clean if key == "clean" else value(n_raw, n_raw_condition)
    s_raw_map = s_clean if key == "clean" else value(s_results, s_raw_condition)
    if key == "clean":
        n_rec_map, s_rec_map = n_clean, s_clean
    else:
        n_rec_map = value(n_recovery_frame, n_rec_condition)
        s_rec_condition = {
            "gaussian_0.06": "gaussian_0.06_bilateral",
            "salt_pepper_0.06": "salt_pepper_0.06_recovered",
            "fog_0.50": "fog_0.50_CLAHE",
            "low_light_0.25": "low_light_0.25_CLAHE",
        }[key]
        s_rec_map = value(s_results, s_rec_condition)
    rows.extend([
        {"condition": key, "label": label, "model": "YOLOv8n", "stage": "raw", "mAP50": n_raw_map, "mAP50_95": value(n_raw, "clean" if key == "clean" else n_raw_condition, "mAP50_95")},
        {"condition": key, "label": label, "model": "YOLOv8s", "stage": "raw", "mAP50": s_raw_map, "mAP50_95": value(s_results, "clean" if key == "clean" else s_raw_condition, "mAP50_95")},
        {"condition": key, "label": label, "model": "YOLOv8n", "stage": "recovered", "mAP50": n_rec_map, "mAP50_95": value(n_raw, "clean", "mAP50_95") if key == "clean" else value(n_recovery_frame, n_rec_condition, "mAP50_95")},
        {"condition": key, "label": label, "model": "YOLOv8s", "stage": "recovered", "mAP50": s_rec_map, "mAP50_95": value(s_results, "clean" if key == "clean" else s_rec_condition, "mAP50_95")},
    ])

table = pd.DataFrame(rows)
table["delta_vs_model_clean_pp"] = table.apply(
    lambda r: 100 * (r.mAP50 - (n_clean if r.model == "YOLOv8n" else s_clean)), axis=1
)
table.to_csv(OUT / "yolov8n_vs_yolov8s_robustness.csv", index=False, float_format="%.6f")

plot_data = table[table["stage"].isin(["raw", "recovered"])].copy()
labels = [item[1] for item in conditions]
models = ["YOLOv8n", "YOLOv8s"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
colors = {"clean": "#425466", "raw": "#e07a5f", "recovered": "#43aa8b"}
for ax, model in zip(axes, models):
    subset = plot_data[plot_data.model == model]
    x = range(len(labels))
    width = 0.24
    for j, stage in enumerate(("raw", "recovered")):
        vals = []
        for key, _, _, _ in conditions:
            match = subset[(subset.condition == key) & (subset.stage == stage)]
            vals.append(float(match.iloc[0].mAP50))
        ax.bar([i + (j - 0.5) * width for i in x], vals, width, label=stage.title(), color=colors[stage])
    ax.set_title(model)
    ax.set_xticks(list(x), labels, rotation=24, ha="right")
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
axes[0].set_ylabel("mAP@0.5")
fig.suptitle("YOLOv8 robustness and recovery on COCO val2017 (5,000 images)")
fig.tight_layout()
fig.savefig(OUT / "yolov8n_vs_yolov8s_robustness.png", dpi=180, bbox_inches="tight")
print(OUT / "yolov8n_vs_yolov8s_robustness.csv")
print(OUT / "yolov8n_vs_yolov8s_robustness.png")

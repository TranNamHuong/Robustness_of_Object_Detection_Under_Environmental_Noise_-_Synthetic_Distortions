from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parent
RUN_DIRS = [ROOT / "runs/full_val2017_raw_seed42",
            ROOT / "runs/full_val2017_recovered_seed42",
            ROOT / "runs/full_val2017_gaussian_recovered_seed42"]
OUT = ROOT / "runs/full_val2017_classwise"
OUT.mkdir(parents=True, exist_ok=True)
data = yaml.safe_load((ROOT / "datasets/coco/full_val2017/data.yaml").read_text(encoding="utf-8"))
names = data["names"]
if isinstance(names, dict):
    class_names = [names.get(i, names.get(str(i), str(i))) for i in range(len(names))]
else:
    class_names = names
n = len(class_names)
records = []
for folder in RUN_DIRS:
    for path in folder.glob("confusion_*.csv"):
        matrix = pd.read_csv(path, header=None).to_numpy()
        if matrix.shape != (n + 1, n + 1):
            continue
        condition = path.stem.removeprefix("confusion_")
        for cls, name in enumerate(class_names):
            tp = int(matrix[cls, cls])
            # Every row entry is a ground-truth instance, including misses in
            # the final background column.
            support = int(matrix[cls, :].sum())
            predicted = int(matrix[:, cls].sum())
            fn = int(matrix[cls, :].sum() - tp)
            fp = int(matrix[:, cls].sum() - tp)
            records.append({"condition": condition, "class_id": cls, "class_name": name,
                            "tp": tp, "fp": fp, "fn": fn, "support": support,
                            "predicted": predicted,
                            "precision_at_conf025_iou05": tp / predicted if predicted else 0.0,
                            "recall_at_conf025_iou05": tp / support if support else 0.0})
frame = pd.DataFrame(records)
frame.to_csv(OUT / "classwise_confusion_metrics.csv", index=False)

clean = frame[frame.condition == "clean"][["class_id", "class_name", "fp", "fn"]].rename(
    columns={"fp": "fp_clean", "fn": "fn_clean"})
deltas = frame.merge(clean, on=["class_id", "class_name"], how="left")
deltas["fp_delta_vs_clean"] = deltas.fp - deltas.fp_clean
deltas["fn_delta_vs_clean"] = deltas.fn - deltas.fn_clean
deltas.to_csv(OUT / "classwise_deltas_vs_clean.csv", index=False)

raw_name = "salt_pepper_0.06"
rec_name = "salt_pepper_0.06_recovered"
raw = frame[frame.condition == raw_name].set_index("class_id")
rec = frame[frame.condition == rec_name].set_index("class_id")
base = frame[frame.condition == "clean"].set_index("class_id")
plot_data = pd.DataFrame({"class_name": raw.class_name,
                          "fp_raw_delta": raw.fp - base.fp,
                          "fp_recovered_delta": rec.fp - base.fp,
                          "fn_raw_delta": raw.fn - base.fn,
                          "fn_recovered_delta": rec.fn - base.fn})
plot_data["rank"] = plot_data.fp_raw_delta.clip(lower=0) + plot_data.fn_raw_delta.clip(lower=0)
top = plot_data.nlargest(15, "rank").sort_values("rank")
fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
for ax, raw_col, rec_col, title, xlabel in [
    (axes[0], "fp_raw_delta", "fp_recovered_delta", "FP change by class", "Δ FP vs clean"),
    (axes[1], "fn_raw_delta", "fn_recovered_delta", "FN change by class", "Δ FN vs clean"),
]:
    y = range(len(top))
    ax.barh([v - 0.18 for v in y], top[raw_col], height=0.34, label="Raw S&P 0.06")
    ax.barh([v + 0.18 for v in y], top[rec_col], height=0.34, label="Median recovered")
    ax.set_yticks(list(y), labels=top.class_name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
fig.suptitle("Full COCO val2017: class errors under severe Salt-and-Pepper")
fig.savefig(OUT / "salt_pepper_0.06_class_errors.png", dpi=180)
plt.close(fig)

print(f"Wrote {len(frame)} class-condition rows across {frame.condition.nunique()} conditions")
print("Top increased FP classes (raw S&P 0.06):")
print(deltas[deltas.condition == raw_name].nlargest(12, "fp_delta_vs_clean")[["class_name", "fp_delta_vs_clean", "fn_delta_vs_clean"]].to_string(index=False))
print("Top increased FN classes (raw S&P 0.06):")
print(deltas[deltas.condition == raw_name].nlargest(12, "fn_delta_vs_clean")[["class_name", "fp_delta_vs_clean", "fn_delta_vs_clean"]].to_string(index=False))

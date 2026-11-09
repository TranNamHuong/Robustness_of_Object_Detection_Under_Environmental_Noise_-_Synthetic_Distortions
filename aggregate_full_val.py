from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
raw = pd.read_csv(RUNS / "full_val2017_raw_seed42" / "metrics.csv")
recovered_sources = [RUNS / "full_val2017_recovered_seed42" / "metrics.csv",
                     RUNS / "full_val2017_gaussian_recovered_seed42" / "metrics.csv"]
recovered = pd.concat([pd.read_csv(path) for path in recovered_sources if path.exists()],
                      ignore_index=True)
recovered["condition"] = recovered["condition"].str.removesuffix("_recovered")
rec_cols = ["condition", "mAP50", "mAP50_95", "tp_iou50", "fp_iou50", "fn_iou50"]
recovered = recovered[rec_cols].rename(columns={c: f"{c}_recovered" for c in rec_cols if c != "condition"})
merged = raw.merge(recovered, on="condition", how="left", validate="one_to_one")
clean = raw.loc[raw.condition == "clean", "mAP50"].iloc[0]
merged["delta_mAP50_vs_clean"] = merged["mAP50"] - clean
merged["relative_mAP50_change_vs_clean_pct"] = (merged["mAP50"] / clean - 1) * 100
merged["recovery_delta_mAP50"] = merged["mAP50_recovered"] - merged["mAP50"]
merged["recovery_delta_fp"] = merged["fp_iou50_recovered"] - merged["fp_iou50"]
merged["recovery_delta_fn"] = merged["fn_iou50_recovered"] - merged["fn_iou50"]
out = RUNS / "full_val2017_comparison_seed42"
out.mkdir(parents=True, exist_ok=True)
merged.to_csv(out / "raw_vs_recovered.csv", index=False)

families = ["gaussian", "salt_pepper", "fog", "low_light"]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
for ax, family in zip(axes, families):
    part = merged[merged.family == family].sort_values("level")
    ax.plot(part.level, part.mAP50, marker="o", label="Raw")
    has_recovery = part.mAP50_recovered.notna()
    if has_recovery.any():
        recovered_rows = part[has_recovery]
        ax.plot(recovered_rows.level, recovered_rows.mAP50_recovered, marker="s", label="Recovered")
    ax.axhline(clean, color="gray", linestyle="--", linewidth=1, label="Clean mAP" if family == families[0] else None)
    ax.set_title(family.replace("_", " ").title())
    ax.set_xlabel("Distortion level")
    ax.set_ylabel("mAP@0.5")
    ax.grid(True, alpha=0.25)
    ax.legend()
fig.suptitle("Full COCO val2017: raw vs recovered, YOLOv8n")
fig.savefig(out / "map_raw_vs_recovered.png", dpi=180)
plt.close(fig)

print(f"Wrote {len(merged)} comparisons to {out}")
print(merged[["condition", "mAP50", "mAP50_recovered", "recovery_delta_mAP50",
              "fp_iou50", "fp_iou50_recovered", "fn_iou50", "fn_iou50_recovered"]].to_string(index=False))

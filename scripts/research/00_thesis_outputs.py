"""Generate all thesis (Chapters 4 & 5) figures and tables from real project data.

Outputs -> research/figures/*.png and research/tables/*.csv|json|md
Run:  python scripts/research/00_thesis_outputs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "processed" / "wers2011_operational_risk_merged.csv"
RES = BASE / "results"
FIG = BASE / "research" / "figures"
TAB = BASE / "research" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

df = pd.read_csv(DATA)
TARGET = "operational_risk_category"
saved = []


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    saved.append(str(p.relative_to(BASE)))
    print(f"  [figure] {p.name}")


# ---------------------------------------------------------------- Ch4 figures
print("== Chapter 4 figures ==")

# Figure 4.1 – risk score histogram with thresholds
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["operational_risk_score"], bins=50, color="#3498db",
        edgecolor="black", alpha=0.8)
ax.axvline(60, color="red", ls="--", lw=2, label="High threshold (60)")
ax.axvline(35, color="orange", ls="--", lw=2, label="Medium threshold (35)")
ax.set(xlabel="Operational risk score (0-100)", ylabel="Frequency",
       title="Distribution of composite operational risk scores (n = 6,635)")
ax.legend()
save(fig, "fig_4_1_risk_score_distribution.png")

# Figure 4.2 – class distribution bar + pie
counts = df[TARGET].value_counts().reindex(["Low", "Medium", "High"])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
bars = axes[0].bar(counts.index, counts.values, color=colors,
                   edgecolor="black", width=0.55)
for b, v in zip(bars, counts.values):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 60,
                 f"{v:,}\n({v / len(df) * 100:.1f}%)",
                 ha="center", fontsize=9, fontweight="bold")
axes[0].set(ylabel="Observations", title="Class counts")
axes[0].set_ylim(0, counts.max() * 1.25)
axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90, explode=(0, 0, 0.12))
axes[1].set_title("Class proportions")
fig.suptitle("Target class distribution – operational_risk_category", y=1.02)
save(fig, "fig_4_2_class_distribution.png")

# Figure 4.3 – correlation heatmap (top 15 numeric predictors by variance)
num = df.select_dtypes(include=[np.number]).drop(
    columns=[c for c in ["serno", "weight"] if c in df.columns], errors="ignore")
top = num.var().sort_values(ascending=False).head(15).index
corr = num[top].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(top)), [c[:22] for c in top], rotation=45, ha="right")
ax.set_yticks(range(len(top)), [c[:22] for c in top])
fig.colorbar(im, shrink=0.8, label="Pearson r")
ax.set(title="Figure 4.3 – Correlation matrix (top 15 predictors)")
ax.grid(False)
save(fig, "fig_4_3_correlation_heatmap.png")

# Figure 4.4 – feature importance (Random Forest, top 20)
fi = pd.read_csv(RES / "tables" / "feature_importance.csv")
fi = fi.sort_values("importance", ascending=False).head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(fi["feature"], fi["importance"], color="#2980b9",
        edgecolor="black", alpha=0.85)
ax.set(xlabel="Mean decrease in impurity",
       title="Top 20 predictive features (Random Forest)")
save(fig, "fig_4_4_feature_importance.png")


# ---------------------------------------------------------------- Ch5 figures
print("== Chapter 5 figures ==")

mc = pd.read_csv(RES / "tables" / "model_comparison.csv")
ev = pd.read_csv(RES / "tables" / "evaluation_summary.csv")
NICE = {"logreg": "Logistic\nRegression", "random_forest": "Random\nForest",
        "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP"}

# Figure 5.1 – model comparison: accuracy vs balanced accuracy
core = mc[mc["model"].isin(NICE)].copy()
core["label"] = core["model"].map(NICE)
x = np.arange(len(core))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w / 2, core["test_accuracy"], w, label="Accuracy",
            color="#3498db", edgecolor="black")
b2 = ax.bar(x + w / 2, core["test_balanced_accuracy"], w,
            label="Balanced accuracy", color="#e67e22", edgecolor="black")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", fontsize=8)
ax.set_xticks(x, core["label"])
ax.set_ylim(0, 1.0)
ax.set(ylabel="Score", title="Figure 5.1 – Test accuracy vs balanced accuracy")
ax.legend(loc="lower right")
save(fig, "fig_5_1_model_comparison.png")

# Figure 5.2 – macro-F1 + ROC-AUC comparison
auc_vals = core["roc_auc_ovr"] if "roc_auc_ovr" in core.columns else 0.0
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w / 2, core["test_f1_macro"], w, label="Macro-F1",
            color="#9b59b6", edgecolor="black")
b2 = ax.bar(x + w / 2, auc_vals, w, label="ROC-AUC (OvR)",
            color="#1abc9c", edgecolor="black")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", fontsize=8)
ax.set_xticks(x, core["label"])
ax.set_ylim(0, 1.0)
ax.set(ylabel="Score", title="Figure 5.2 – Macro-F1 and ROC-AUC by model")
ax.legend(loc="lower right")
save(fig, "fig_5_2_f1_roc_comparison.png")

# Figure 5.3 – confusion matrices (5 panels)
fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
for ax, (m, nice) in zip(axes, NICE.items()):
    cm = pd.read_csv(RES / "tables" / f"confusion_matrix_{m}.csv", index_col=0)
    vals = cm.values.astype(float)
    ax.imshow(vals, cmap="Blues")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{int(vals[i, j])}", ha="center", va="center",
                    color="white" if vals[i, j] > vals.max() / 2 else "black",
                    fontweight="bold")
    ax.set_xticks(range(3), cm.columns, rotation=45)
    ax.set_yticks(range(3), cm.index)
    ax.set(xlabel="Predicted", ylabel="Actual", title=nice)
    ax.grid(False)
fig.suptitle("Figure 5.3 – Confusion matrices (test set, n = 1,327)", y=1.04)
save(fig, "fig_5_3_confusion_matrices.png")

# Figure 5.4 – imbalance-handling strategies
strat = mc[~mc["model"].isin(NICE)].copy()
strat = strat.sort_values("test_balanced_accuracy", ascending=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
colors_s = ["#e74c3c" if v < 0.5 else "#27ae60"
            for v in strat["test_balanced_accuracy"]]
ax.barh(strat["model"], strat["test_balanced_accuracy"], color=colors_s,
        edgecolor="black", alpha=0.85)
for i, (_, r) in enumerate(strat.iterrows()):
    ax.text(r["test_balanced_accuracy"] + 0.008, i,
            f"{r['test_balanced_accuracy']:.3f}", va="center", fontsize=9)
ax.axvline(0.5, color="gray", ls=":", lw=1.5, label="Chance (0.5)")
ax.set(xlim=(0, 0.75), xlabel="Balanced accuracy (test)",
       title="Figure 5.4 – Class-imbalance handling strategies")
ax.legend()
save(fig, "fig_5_4_imbalance_strategies.png")

# Figure 5.5 – SHAP global importance (top 15)
shap_imp = pd.read_csv(RES / "tables" / "shap_importance.csv")
shap_col = ("mean_abs_shap" if "mean_abs_shap" in shap_imp.columns
            else shap_imp.columns[1])
shap_imp = shap_imp.sort_values(shap_col, ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(shap_imp.iloc[:, 0], shap_imp[shap_col], color="#e67e22",
        edgecolor="black", alpha=0.85)
ax.set(xlabel="Mean |SHAP value|",
       title="Figure 5.5 – SHAP global feature importance (top 15)")
save(fig, "fig_5_5_shap_importance.png")

import shutil
for src_name, dst_name in [("shap_beeswarm.png", "fig_5_6_shap_beeswarm.png"),
                           ("shap_summary.png", "fig_5_7_shap_summary.png")]:
    src = RES / "figures" / src_name
    if src.exists():
        shutil.copy(src, FIG / dst_name)
        saved.append(str((FIG / dst_name).relative_to(BASE)))
        print(f"  [figure] {dst_name} (copied)")

# ---------------------------------------------------------------- Tables
print("== Thesis tables ==")

predictors = [c for c in df.columns if c not in
              ["serno", "year", "weight", "operational_risk_score", TARGET,
               "employee_turnover_rate", "absenteeism_rate", "grievance_rate",
               "disputes_raised", "dispute_risk_level", "dismissal_rate",
               "disciplinary_warnings", "suspensions",
               "other_industrial_action"]]

# Table 3.2 – dataset characteristics
chars = pd.DataFrame({
    "Attribute": ["Unit of analysis", "Observations", "Total variables",
                  "Predictive features", "Target variable", "Target classes",
                  "Class balance (Low/Med/High)", "Imbalance ratio (Low:High)",
                  "Missing values (predictors)", "Score range", "Year span"],
    "Value": [
        "Workplace (organization)",
        f"{len(df):,}",
        f"{df.shape[1]}",
        f"{len(predictors)}",
        TARGET,
        "3 (Low / Medium / High)",
        " / ".join(f"{counts[c]:,}" for c in ["Low", "Medium", "High"]),
        f"{counts['Low'] / max(counts['High'], 1):.0f}:1",
        f"{df[predictors].isna().sum().sum():,}",
        "0 – 100",
        f"{int(df['year'].min())} – {int(df['year'].max())}",
    ]})
chars.to_csv(TAB / "table_3_2_dataset_characteristics.csv", index=False)

# Table 4.1 – environment versions
pkgs = ["python", "flask", "pandas", "numpy", "scikit-learn", "xgboost",
        "lightgbm", "shap", "matplotlib", "pydantic", "flask-cors", "pytest"]
env_rows = []
for p in pkgs:
    try:
        from importlib.metadata import version
        v = (sys.version.split()[0] if p == "python" else version(p))
    except Exception:
        v = "not installed"
    env_rows.append({"Package": p, "Version": v})
pd.DataFrame(env_rows).to_csv(TAB / "table_4_1_environment.csv", index=False)

# Table 4.2 – excluded (leakage) features
excl = pd.DataFrame([
    ("operational_risk_score", "Composite target precursor — direct leakage"),
    ("employee_turnover_rate", "Target dimension component (25 pts)"),
    ("employee_turnover_rate_5yr", "Target dimension component (turnover)"),
    ("absenteeism_rate", "Target dimension component (25 pts)"),
    ("grievance_rate", "Target dimension component (25 pts)"),
    ("disputes_raised", "Target dimension component (industrial action)"),
    ("dispute_risk_level", "Target dimension component (industrial action)"),
    ("dismissal_rate", "Target dimension component (industrial action)"),
    ("disciplinary_warnings", "Target dimension component (industrial action)"),
    ("suspensions", "Target dimension component (industrial action)"),
    ("other_industrial_action", "Target dimension component (industrial action)"),
    ("serno", "Survey identifier — non-predictive"),
    ("year", "Survey wave identifier — administrative"),
    ("weight", "Survey sampling weight — administrative"),
], columns=["Excluded feature", "Rationale"])
excl.to_csv(TAB / "table_4_2_excluded_features.csv", index=False)

# Table 5.1 – model comparison
mc_out = mc[mc["model"].isin(NICE)].copy()[
    ["model", "test_accuracy", "test_balanced_accuracy", "test_f1_macro",
     "cv_test_balanced_accuracy"]]
mc_out.columns = ["Model", "Accuracy", "Balanced accuracy", "Macro-F1",
                  "CV balanced accuracy"]
mc_out["Model"] = mc_out["Model"].map(NICE).str.replace("\n", " ")
mc_out.to_csv(TAB / "table_5_1_model_comparison.csv", index=False)

# Table 5.2 – full evaluation summary
ev.to_csv(TAB / "table_5_2_evaluation_summary.csv", index=False)

# Tables 5.3-5.7 – per-model metrics from evaluation summary
for i, m in enumerate(NICE, start=3):
    row = ev[ev["model"] == m]
    if not row.empty:
        row.to_csv(TAB / f"table_5_{i}_{m}_metrics.csv", index=False)

# Tables 5.8-5.9 – confusion matrices
for i, m in enumerate(["logreg", "random_forest"], start=8):
    pd.read_csv(RES / "tables" / f"confusion_matrix_{m}.csv").to_csv(
        TAB / f"table_5_{i}_confusion_matrix_{m}.csv", index=False)

# Table 5.10 – feature importance (top 20)
pd.read_csv(RES / "tables" / "feature_importance.csv").head(20).to_csv(
    TAB / "table_5_10_feature_importance_top20.csv", index=False)

# Table 5.11 – SHAP importance (top 20)
pd.read_csv(RES / "tables" / "shap_importance.csv").head(20).to_csv(
    TAB / "table_5_11_shap_importance_top20.csv", index=False)

# Imbalance strategy table
mc[~mc["model"].isin(NICE)].to_csv(
    TAB / "table_5_12_imbalance_strategies.csv", index=False)

# Dataset characteristics JSON (machine-readable)
meta = {
    "n_observations": int(len(df)),
    "n_variables": int(df.shape[1]),
    "n_predictive_features": int(len(predictors)),
    "target": TARGET,
    "class_counts": {k: int(v) for k, v in counts.items()},
    "class_pct": {k: round(v / len(df) * 100, 2) for k, v in counts.items()},
    "imbalance_ratio": round(counts["Low"] / max(counts["High"], 1), 1),
}
(TAB / "dataset_characteristics.json").write_text(json.dumps(meta, indent=2))

print(f"\nSaved {len(saved)} figures + tables -> research/")
for s in saved:
    print("  ", s)
print("DONE")

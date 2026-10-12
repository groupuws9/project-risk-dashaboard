"""05_model_evaluation.py — Deep evaluation of trained models (RO2/RQ3)."""
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import efa_tools  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "saved_models"
TAB_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"
REP_DIR = BASE / "results" / "evaluation_reports"
DATA = BASE / "data" / "processed" / "wers2011_operational_risk_merged.csv"
for d in (TAB_DIR, FIG_DIR, REP_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("MODEL EVALUATION")
print("=" * 78)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, roc_auc_score,
    RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

META = json.loads((MODEL_DIR / "model_metadata.json").read_text())
FEATS = META["features"]
LABELS = META["labels"]
TARGET = META["target"]

df = pd.read_csv(DATA)
df, _ = efa_tools.ensure(df, MODEL_DIR / "factor_scoring.json")  # EFA scores
X = df[FEATS].copy()
y = df[TARGET].astype(str).str.strip().map({"Low": 0, "Medium": 1, "High": 2})
m = y.notna()
X, y = X.loc[m].reset_index(drop=True), y.loc[m].astype(int).reset_index(drop=True)

# ---- SAME cleaning as 04_model_training.py (else median-imputer chokes) ----
NUM_FEATS = META.get("numeric_features", [])
CAT_FEATS = META.get("categorical_features", [])
MISS_TOKENS = ["", " ", "NA", "N/A", "nan", "None", "-9", "-8"]
for c in NUM_FEATS:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c].replace(MISS_TOKENS, np.nan), errors="coerce")
for c in CAT_FEATS:
    if c in X.columns:
        X[c] = X[c].replace(MISS_TOKENS, np.nan).astype(str).str.strip().replace(
            {"": "MISSING", "nan": "MISSING", "None": "MISSING"})
_, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

summary = []

for pkl in sorted(MODEL_DIR.glob("*.pkl")):
    if pkl.name == "best_model.pkl":
        continue
    name = pkl.stem
    try:
        pipe = joblib.load(pkl)
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)
        bal = balanced_accuracy_score(yte, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yte, pred, average="macro", zero_division=0)
        try:
            proba = pipe.predict_proba(Xte)
            auc = roc_auc_score(yte, proba, multi_class="ovr")
        except Exception:
            proba, auc = None, float("nan")
        cm = confusion_matrix(yte, pred)
        summary.append({"model": name, "accuracy": acc,
                        "balanced_accuracy": bal, "macro_precision": prec,
                        "macro_recall": rec, "macro_f1": f1, "roc_auc_ovr": auc})
        print(f"{name:15s} acc={acc:.3f} bal={bal:.3f} f1m={f1:.3f} auc={auc:.3f}")
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
            TAB_DIR / f"confusion_matrix_{name}.csv")
        if HAS_PLT:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Confusion matrix — {name}")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i, j], ha="center", va="center")
            fig.tight_layout(); fig.savefig(FIG_DIR / f"cm_{name}.png", dpi=130)
            plt.close(fig)


            if proba is not None:
                try:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    for k, lab in enumerate(LABELS):
                        RocCurveDisplay.from_predictions(
                            (yte == k).astype(int), proba[:, k],
                            name=lab, ax=ax2)
                    ax2.plot([0, 1], [0, 1], "k--", lw=1)
                    ax2.set_title(f"ROC (one-vs-rest) — {name}")
                    fig2.tight_layout()
                    fig2.savefig(FIG_DIR / f"roc_{name}.png", dpi=130)
                    plt.close(fig2)
                except Exception as e:
                    print(f"  ROC skipped: {e}")
        try:
            samp = Xte.sample(min(400, len(Xte)), random_state=0)
            ysamp = yte.loc[samp.index]
            r = permutation_importance(pipe, samp, ysamp, n_repeats=5,
                                       random_state=0, n_jobs=-1)
            pi = pd.DataFrame({"feature": FEATS,
                               "importance": r.importances_mean,
                               "std": r.importances_std}).sort_values(
                "importance", ascending=False)
            pi.to_csv(TAB_DIR / f"permutation_importance_{name}.csv", index=False)
            if HAS_PLT:
                top = pi.head(15).iloc[::-1]
                fig3, ax3 = plt.subplots(figsize=(9, 6))
                ax3.barh(top["feature"].astype(str), top["importance"],
                         xerr=top["std"])
                ax3.set_title(f"Permutation importance — {name}")
                fig3.tight_layout()
                fig3.savefig(FIG_DIR / f"perm_importance_{name}.png", dpi=130)
                plt.close(fig3)
        except Exception as e:
            print(f"  permutation importance skipped: {e}")
    except Exception as e:
        print(f"{name}: FAILED ({e})")


if summary:
    res = pd.DataFrame(summary).sort_values("balanced_accuracy", ascending=False)
else:
    res = pd.DataFrame(columns=["model", "accuracy", "balanced_accuracy",
                                "macro_precision", "macro_recall", "macro_f1",
                                "roc_auc_ovr"])
res.to_csv(TAB_DIR / "evaluation_summary.csv", index=False)
res.to_csv(REP_DIR / "evaluation_summary.csv", index=False)

md = ["# Model Evaluation Report — WERS 2011 Operational Risk", "",
      f"Best model (training): **{META['best_model']}**", "",
      "## Test metrics (held-out 20%, stratified)", ""]
if not res.empty:
    try:
        md.append(res.round(3).to_markdown(index=False))
    except Exception:
        md.append(res.round(3).to_string(index=False))
else:
    md.append("_No models evaluated successfully — see console output._")
md += ["", "## Notes",
       "- Features exclude raw risk indicators (no leakage).",
       "- High class is tiny (~0.4%); use balanced_accuracy / macro-F1.",
       "- Artefacts: confusion matrices + ROC + permutation importance.",
       "", f"Labels: {LABELS}"]
(REP_DIR / "evaluation_report.md").write_text("\n".join(md))
print("\nSaved evaluation_summary.csv + evaluation_report.md + figures")
print("DONE.")


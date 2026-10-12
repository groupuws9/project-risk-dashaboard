"""06_shap_explanation.py — SHAP global + local explanations (spec 2 §3.6).

Outputs:
  results/tables/shap_importance.csv   mean |SHAP| aggregated to original features
  results/tables/shap_local.json       local explanation for one example org
  results/figures/shap_summary.png     horizontal bar chart (global)
  results/figures/shap_beeswarm.png    distribution of SHAP values (summary plot)
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "wers2011_operational_risk_merged.csv"
MODEL_DIR = BASE / "models" / "saved_models"
TAB_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_BG, N_EVAL = 80, 300   # keep KernelExplainer tractable

print("=" * 78)
print("SHAP EXPLANATIONS — WERS 2011 operational risk")
print("=" * 78)

meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
best = meta["best_model"]
FEATS, NUM, CAT = meta["features"], meta["numeric_features"], meta["categorical_features"]
LABELS = meta["labels"]
pipe = joblib.load(MODEL_DIR / f"{best}.pkl")
prep = pipe.named_steps["prep"]
clf = pipe.named_steps["clf"]
print(f"Best model: {best} ({type(clf).__name__})")

# ---- data: same cleaning as 04/05 ----------------------------------------
df = pd.read_csv(DATA)
X = df[FEATS].copy()
MISS = ["", " ", "NA", "N/A", "nan", "None", "-9", "-8"]
for c in NUM:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c].replace(MISS, np.nan), errors="coerce")
for c in CAT:
    if c in X.columns:
        X[c] = X[c].replace(MISS, np.nan).astype(str).str.strip().replace(
            {"": "MISSING", "nan": "MISSING", "None": "MISSING"})
X = X.reset_index(drop=True)
rng = np.random.RandomState(42)
bg_idx = rng.choice(len(X), size=min(N_BG, len(X)), replace=False)
ev_idx = rng.choice(len(X), size=min(N_EVAL, len(X)), replace=False)
X_bg_t = prep.transform(X.loc[bg_idx])
X_ev_t = prep.transform(X.loc[ev_idx])
t_names = list(prep.get_feature_names_out())


def to_original(tname):
    """num__pct_female / cat__sector_1 -> original feature name."""
    raw = tname.split("__", 1)[-1]
    for f in sorted(FEATS, key=len, reverse=True):
        if raw == f or raw.startswith(f + "_") or raw.startswith(f):
            return f
    return raw


# ---- explainer (spec 3.6: TreeExplainer for trees, else model-specific) ---
kind = type(clf).__name__
if hasattr(clf, "estimators_") or kind in ("XGBClassifier", "LGBMClassifier"):
    import shap
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_ev_t)
    print(f"TreeExplainer on {kind}")
elif hasattr(clf, "coef_"):
    import shap
    explainer = shap.LinearExplainer(clf, X_bg_t)
    sv = explainer.shap_values(X_ev_t)
    print(f"LinearExplainer on {kind}")
else:
    import shap
    explainer = shap.Explainer(clf.predict_proba, X_bg_t)
    sv = explainer(X_ev_t).values
    print(f"Kernel/Sampling Explainer on {kind}")

# normalise to 3-D (n_eval, n_features, n_classes)
arr = np.asarray(sv, dtype=float)
if arr.ndim == 2:
    arr = arr[:, :, None]
print(f"SHAP array: {arr.shape}")

# ---- global importance aggregated to original features -------------------
orig = [to_original(t) for t in t_names]
mean_abs = np.abs(arr).mean(axis=(0, 2))           # per transformed feature
agg = {}
for o, v in zip(orig, mean_abs):
    agg[o] = agg.get(o, 0.0) + float(v)
imp = (pd.DataFrame({"feature": list(agg), "mean_abs_shap": list(agg.values())})
       .sort_values("mean_abs_shap", ascending=False))
imp.to_csv(TAB_DIR / "shap_importance.csv", index=False)
print("Top SHAP drivers:")
print(imp.head(10).to_string(index=False))

# ---- plots ----------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

top = imp.head(20).iloc[::-1]
plt.figure(figsize=(9, 7))
plt.barh(top["feature"].astype(str), top["mean_abs_shap"], color="steelblue")
plt.xlabel("mean |SHAP value|")
plt.title(f"SHAP global feature importance — {best}")
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary.png", dpi=130)
plt.close()
print("Saved results/figures/shap_summary.png")

try:
    import shap as shap_mod
    plt.figure(figsize=(10, 7))
    shap_mod.summary_plot(sv if arr.shape[2] > 1 else arr[:, :, 0],
                          features=X_ev_t, feature_names=t_names,
                          max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=130)
    plt.close()
    print("Saved results/figures/shap_beeswarm.png")
except Exception as e:
    print(f"Beeswarm skipped: {e}")

# ---- local explanation for one REAL example org (dashboard fallback) ------
# Pick a well-populated workplace (valid numeric percentages, few imputed
# categoricals): an all-missing / sentinel-999 row explains nothing because
# every SHAP contribution collapses to ~0.
num_ok = X[NUM].notna().all(axis=1) & (X[NUM] <= 100).all(axis=1)
n_missing = (X[CAT].astype(str) == "MISSING").sum(axis=1)
cands = X.index[num_ok & (n_missing <= 2)]
row = int(cands[0]) if len(cands) else int(X.index[0])
row_df = X.iloc[[row]]
k_pred = int(pipe.predict_proba(row_df)[0].argmax())   # class being explained

sv_row = explainer.shap_values(prep.transform(row_df))
if isinstance(sv_row, list):
    rarr = np.stack(sv_row, axis=-1)[0]                # (f, C)
else:
    rarr = np.asarray(sv_row, dtype=float)
    if rarr.ndim == 3:
        rarr = rarr[0]                                 # (1,f,C) -> (f,C)
    elif rarr.ndim == 2 and rarr.shape[0] == 1:
        rarr = rarr[0][:, None]                        # (1,f)   -> (f,1)
contrib = rarr[:, k_pred] if rarr.shape[1] > 1 else rarr[:, 0]
ev = np.asarray(getattr(explainer, "expected_value", 0.0)).reshape(-1)
base = float(ev[k_pred]) if ev.size > k_pred else float(ev[0])

loc = {}
for o, c in zip(orig, contrib):
    loc[o] = loc.get(o, 0.0) + float(c)
items = sorted(loc.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
example = X.iloc[row].to_dict()
payload = {
    "model": best,
    "class_explained": LABELS[k_pred],
    "base_value": round(base, 5),
    "example": {k: (None if (isinstance(v, float) and v != v) else v)
                for k, v in example.items()},
    "contributions": [{"feature": k, "shap": round(v, 5),
                       "value": (None if (isinstance(example.get(k), float)
                                          and example.get(k) != example.get(k))
                                 else str(example.get(k)))}
                      for k, v in items],
}
(TAB_DIR / "shap_local.json").write_text(json.dumps(payload, indent=2))
print("Saved results/tables/shap_local.json "
      f"(row {row}, class {LABELS[k_pred]}, "
      f"max |shap| {max(abs(v) for _, v in items):.4f})")
print("DONE.")

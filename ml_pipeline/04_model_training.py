"""04_model_training.py — Train & validate operational-risk classifiers (RQ2/RQ3).

Uses ONLY pre-existing organisational characteristics + HR/ER practices as
features (excludes raw risk indicators used to build the target => no leakage).
Handles severe class imbalance (High n=28) via stratified splits + balanced
class weights. Saves pipelines, metrics and metadata for dashboard + 05 script.
"""
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "wers2011_operational_risk_merged.csv"
META_JSON = BASE / "data" / "processed" / "dataset_metadata.json"
MODEL_DIR = BASE / "models" / "saved_models"
TABLE_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "operational_risk_category"
SCORE_COL = "operational_risk_score"
ID_COLS = ["serno", "year", "weight"]
# Leakage columns: raw risk indicators that CONSTRUCT the target score.
LEAKAGE = ["employee_turnover_rate", "employee_turnover_rate_5yr",
    "absenteeism_rate", "grievance_rate", "disputes_raised",
    "dispute_risk_level", "other_industrial_action", "dismissal_rate",
    "disciplinary_warnings", "suspensions", SCORE_COL]
LABELS = ["Low", "Medium", "High"]
RANDOM_STATE = 42
N_SPLITS = 5           # stratified k-fold CV folds (spec 3.4)

print("=" * 78)
print("MODEL TRAINING — WERS 2011 operational risk (Low/Medium/High)")
print("=" * 78)

df = pd.read_csv(DATA)
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
print(df[TARGET].value_counts().to_string())

# ---- feature list: org + HR + ER + labour (no leakage, no IDs/target) ----
if META_JSON.exists():
    meta = json.loads(META_JSON.read_text())
    cats = meta["variable_categories"]
    feats = (cats["organizational"] + cats["hr_practices"]
             + cats["employee_relations"] + cats["labor_market"])
else:
    feats = [c for c in df.columns if c not in ID_COLS + LEAKAGE + [TARGET]]
feats = [c for c in feats if c in df.columns]
# Drop ultra-high-cardinality raw SIC codes (500+/200+ levels -> huge one-hot);
# sector_detailed retains industry signal at tractable dimensionality.
for _drop in ("industry_code", "industry_code_80"):
    if _drop in feats:
        feats.remove(_drop)

# ---- EFA factor scores (spec 1 §3.7): latent HR-practice constructs --------
# Deterministic full-sample EFA over the 16 HR/ER practice items (union
# coverage, collective bargaining, JCC, information sharing, consultation
# depth, ...). The identical scores are re-applied at inference from
# models/saved_models/factor_scoring.json => train/serve parity.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import efa_tools  # noqa: E402

efa_artifact, _efa_diag = efa_tools.fit(df)
factors = efa_tools.transform(df, efa_artifact)
efa_tools.save(efa_artifact, MODEL_DIR / "factor_scoring.json")
raw_count = len(feats)
feats = feats + [c for c in factors.columns if c not in feats]
print(f"EFA: {len(efa_artifact['items'])} items -> {list(factors.columns)} "
      f"(KMO={efa_artifact['kmo_overall']:.3f}, "
      f"PA retains {efa_artifact['parallel_analysis']['n_factors']}, "
      f"retained {efa_artifact['n_factors']})")
print(f"Features: {len(feats)} ({raw_count} raw + "
      f"{len(factors.columns)} EFA factor scores)")

X = pd.concat([df[[c for c in feats if c in df.columns]], factors],
              axis=1)[feats]
y = df[TARGET].astype(str).str.strip().map({"Low": 0, "Medium": 1, "High": 2})
mask = y.notna()
X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].astype(int).reset_index(drop=True)

# ---- auto-detect numeric vs categorical ----
MISS_TOKENS = {"", " ", "NA", "N/A", "nan", "None", "-9", "-8"}
num_cols, cat_cols = [], []
for c in X.columns:
    s = X[c].replace(list(MISS_TOKENS), np.nan)
    coerced = pd.to_numeric(s, errors="coerce")
    valid = s.notna().sum()
    numeric_ratio = (coerced.notna().sum() / valid) if valid else 0
    if numeric_ratio > 0.85 and coerced.nunique(dropna=True) > 12:
        num_cols.append(c)
        X[c] = coerced
    else:
        cat_cols.append(c)
        X[c] = s.astype(str).str.strip().replace(
            {"": "MISSING", " ": "MISSING", "nan": "MISSING", "None": "MISSING"})
        # group ultra-rare categories (<1%) to reduce dimensionality
        vc = X[c].value_counts(normalize=True)
        rare = set(vc[vc < 0.01].index)
        if rare:
            X[c] = X[c].apply(lambda v: "OTHER_RARE" if v in rare else v)
print(f"Numeric ({len(num_cols)}): {num_cols}")
print(f"Categorical ({len(cat_cols)}): {len(cat_cols)} cols")

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), cat_cols),
], verbose_feature_names_out=False)

def build_models():
    """Return {name: estimator} with imbalance-aware settings."""
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=150,
            min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=300, early_stopping=True, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=5,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, eval_metric="mlogloss", tree_method="hist",
            random_state=RANDOM_STATE)
    if HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(n_estimators=200, num_leaves=63,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", verbose=-1, random_state=RANDOM_STATE)
    return models

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain {Xtr.shape}  Test {Xte.shape}")
print("Train dist:", ytr.value_counts().to_dict())
print("Test  dist:", yte.value_counts().to_dict())

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
print(f"CV: stratified {N_SPLITS}-fold (shuffle=True, random_state={RANDOM_STATE})")
results, fitted = {}, {}


for name, est in build_models().items():
    pipe = Pipeline([("prep", preprocess), ("clf", est)])
    print(f"\n--- {name} ---")
    try:
        sc = cross_validate(pipe, Xtr, ytr, cv=cv,
            scoring=["accuracy", "balanced_accuracy", "f1_macro"],
            n_jobs=-1, error_score=np.nan)
        cv_sum = {k: float(np.nanmean(v)) for k, v in sc.items()
                  if k.startswith("test_")}
        print("CV acc=%.3f bal=%.3f f1m=%.3f" % (
            cv_sum.get("test_accuracy", np.nan),
            cv_sum.get("test_balanced_accuracy", np.nan),
            cv_sum.get("test_f1_macro", np.nan)))
    except Exception as e:
        print(f"CV failed: {e}"); cv_sum = {}
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    bal = balanced_accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    print(f"TEST acc={acc:.3f} bal_acc={bal:.3f} f1_macro={f1m:.3f}")
    print(classification_report(yte, pred, target_names=LABELS, zero_division=0))
    fitted[name] = pipe
    results[name] = {"cv": cv_sum,
        "test": {"accuracy": acc, "balanced_accuracy": bal, "f1_macro": f1m,
                 "report": classification_report(yte, pred, target_names=LABELS,
                     zero_division=0, output_dict=True),
                 "confusion_matrix": confusion_matrix(yte, pred).tolist()}}

best_name = max(results, key=lambda k: results[k]["test"]["balanced_accuracy"])
print(f"\n>>> BEST: {best_name} "
      f"(bal_acc={results[best_name]['test']['balanced_accuracy']:.3f})")


for name, pipe in fitted.items():
    joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")
joblib.dump(fitted[best_name], MODEL_DIR / "best_model.pkl")
with open(MODEL_DIR / "best_model_name.txt", "w") as f:
    f.write(best_name)
rows = []
for n, r in results.items():
    row = {"model": n}
    row.update({f"test_{k}": v for k, v in r["test"].items()
                if isinstance(v, float)})
    row.update({f"cv_{k}": v for k, v in r["cv"].items()})
    rows.append(row)
pd.DataFrame(rows).to_csv(TABLE_DIR / "model_comparison.csv", index=False)
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump({"best_model": best_name, "target": TARGET, "labels": LABELS,
               "features": feats, "numeric_features": num_cols,
               "categorical_features": cat_cols, "leakage_excluded": LEAKAGE,
               "factor_features": list(factors.columns),
               "cv_folds": N_SPLITS,
               "all_test_balanced_accuracy": {n: r["test"]["balanced_accuracy"]
                                              for n, r in results.items()},
               "random_state": RANDOM_STATE}, f, indent=2)
print("Saved models + model_comparison.csv + model_metadata.json")


imp_model = next((m for m in ("xgboost", "lightgbm", "random_forest")
                  if m in fitted), best_name)
try:
    pipe = fitted[imp_model]
    feat_names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None and hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_)).mean(axis=0)
    if imp is not None:
        fi = pd.DataFrame({"feature": feat_names,
                           "importance": np.asarray(imp, float)})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv(TABLE_DIR / "feature_importance.csv", index=False)
        print(f"Importance ({imp_model}) top 10:")
        print(fi.head(10).to_string(index=False))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = fi.head(20).iloc[::-1]
        plt.figure(figsize=(9, 7))
        plt.barh(top["feature"].astype(str), top["importance"])
        plt.title(f"Top 20 features — {imp_model}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "feature_importance.png", dpi=130)
        plt.close()
        print("Saved results/figures/feature_importance.png")
except Exception as e:
    print(f"Importance export skipped: {e}")
print("\nDONE. Next: python 05_model_evaluation.py")


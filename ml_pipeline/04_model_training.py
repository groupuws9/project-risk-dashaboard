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
from sklearn.model_selection import StratifiedKFold, train_test_split
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import efa_tools  # noqa: E402
import data_contract as dc  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin, clone

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"
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
# Leakage columns. The target is now the employee-turnover band derived from
# ``tempc`` (see 01_build_dataset.py), so the turnover rate itself plus the other raw
# risk indicators, the two duplicate-derived predictors (tprocgrv ->
# grievance_procedure_formal; ttrnprob -> training_problems /
# skill_shortage_reported) and the retired composite score are all excluded.
LEAKAGE = ["employee_turnover_rate", "employee_turnover_rate_5yr",
    "absenteeism_rate", "grievance_rate", "grievance_procedure_formal",
    "training_problems", "skill_shortage_reported", "disputes_raised",
    "dispute_risk_level", "other_industrial_action", "dismissal_rate",
    "disciplinary_warnings", "suspensions", SCORE_COL]
LABELS = ["Low", "Medium", "High"]
RANDOM_STATE = 42
N_SPLITS = 5           # stratified k-fold CV folds (spec 3.4)

print("=" * 78)
print("MODEL TRAINING — WERS/WIRS workplace operational risk (Low/Medium/High)")
print("Target = employee turnover band (tempc), sentinel-cleaned")
print("=" * 78)

df = pd.read_csv(DATA)
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
print(df[TARGET].value_counts().to_string())

# ---- feature list: leakage-safe predictor columns from 01_build_dataset ----
# Prefer the single-source-of-truth predictor contract written by
# 01_build_dataset.build() into dataset_metadata.json. Those columns are already
# verified leakage-free (raw risk indicators, the two tprocgrv/ttrnprob
# duplicates and the score are excluded there). Fall back to a column filter
# only if the metadata is absent.
if META_JSON.exists():
    meta = json.loads(META_JSON.read_text())
    feats = [c for c in meta.get("predictor_columns", []) if c in df.columns]
else:
    feats = [c for c in df.columns if c not in ID_COLS + LEAKAGE + [TARGET]]
# Fail fast if any target-derived / duplicate-derived column leaked in.
_leak_set = set(LEAKAGE) | set(meta.get("leakage_excluded", [])) if META_JSON.exists() else set(LEAKAGE)
assert not (set(feats) & _leak_set), (
    f"leakage columns present in features: {set(feats) & _leak_set}")
assert len(feats) == len(set(feats)), "duplicate feature names detected"
# Drop ultra-high-cardinality raw SIC codes (500+/200+ levels -> huge one-hot);
# sector_detailed retains industry signal at tractable dimensionality.
for _drop in ("industry_code", "industry_code_80"):
    if _drop in feats:
        feats.remove(_drop)
print(f"Verified {len(feats)} unique leakage-free predictors")

# ---- canonical split (data_contract) ------------------------------------
# The split is the single source of truth for the held-out test partition and
# is shared by 05/06/07 so every reported metric refers to the same rows.
Xtr_raw, Xte_raw, ytr, yte, feats = dc.split_dataset(df, feats)
print(f"\nTrain {Xtr_raw.shape}  Test {Xte_raw.shape}")
print("Train dist:", ytr.value_counts().to_dict())
print("Test  dist:", yte.value_counts().to_dict())

# ---- EFA factor scores: custom transformer for CV-fold-level fitting ----
# CRITICAL: EFA must be fitted WITHIN each CV fold to prevent leakage.
# The target is NEVER passed to EFA - only predictor features are used.
# This custom transformer fits EFA on training data only during each fold.
class EFATransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that fits EFA on training data only (no target leakage).

    Fits EFA on the 16 HR/ER practice items within each CV fold,
    then applies the same transformation to validation/test data.
    The target variable is never used in EFA fitting.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.artifact_ = None
        self.factor_names_ = None

    def fit(self, X, y=None):
        """Fit EFA on training data (y is ignored - no target leakage).

        EFA is unsupervised: ``y`` is never used and ``efa_tools.fit`` reads
        only the HR/ER practice items, so the target can never influence the
        factor solution.
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Defensive: never let the target column leak into the EFA item set.
        X = X.drop(columns=[TARGET], errors="ignore")
        # Fit EFA on predictors only (y is intentionally ignored)
        self.artifact_, _ = efa_tools.fit(X)
        self.factor_names_ = self.artifact_["factor_names"]
        return self

    def transform(self, X):
        """Transform using the fitted EFA artifact.

        Adds (or overwrites) the factor_* score columns on X and returns the
        frame, so raw predictors and factor scores flow to downstream steps
        together and frames that already carry factor columns are not
        duplicated.
        """
        if self.artifact_ is None:
            raise RuntimeError("EFATransformer must be fitted before transform.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        factors = efa_tools.transform(X, self.artifact_)
        for c in factors.columns:
            X[c] = factors[c].values
        return X

# Fit EFA on full training set once to determine factor structure for metadata
# (This is for artifact persistence only; CV uses fold-level fitting)
print("\nFitting EFA on training data (no target) to determine factor structure...")
efa_artifact_full, _efa_diag = efa_tools.fit(Xtr_raw)
factors_tr_full = efa_tools.transform(Xtr_raw, efa_artifact_full)
efa_tools.save(efa_artifact_full, MODEL_DIR / "factor_scoring.json")
raw_count = len(feats)
factor_names = list(factors_tr_full.columns)
feats = feats + factor_names
print(f"EFA: {len(efa_artifact_full['items'])} items -> {factor_names} "
      f"(KMO={efa_artifact_full['kmo_overall']:.3f}, "
      f"PA retains {efa_artifact_full['parallel_analysis']['n_factors']}, "
      f"retained {efa_artifact_full['n_factors']})")
print(f"Features: {len(feats)} ({raw_count} raw + "
      f"{len(factor_names)} EFA factor scores)")
print("NOTE: EFA will be refitted within each CV fold to prevent leakage.")

# Full-train EFA factor scores for the final model frame (production contract:
# 33 raw predictors + EFA factor scores, scored with the saved artifact).
# NOTE: cross-validation below refits EFA within each fold, so validation-fold
# data never influences the factor scores used for training.
factors_te_full = efa_tools.transform(Xte_raw, efa_artifact_full)
Xtr = pd.concat([Xtr_raw, factors_tr_full], axis=1)[feats]
Xte = pd.concat([Xte_raw, factors_te_full], axis=1)[feats]

# ---- auto-detect numeric vs categorical (on training data only) ----
# Cleaning is applied to the RAW predictor frames; factor columns are added
# afterwards (they are already clean floats from the EFA artifact).  Cleaning
# Xtr_raw/Xte_raw here means CV fold subsets below are already leakage-safe.
# The contract owns this transformation so 04/05/06/07 cannot drift apart.
num_cols, cat_cols, rare_map = dc.clean_frames(Xtr_raw, Xte_raw)
# EFA factor scores are clean floats -> numeric branch
num_cols = num_cols + [c for c in factor_names if c not in num_cols]
print(f"Numeric ({len(num_cols)}): {num_cols}")
print(f"Categorical ({len(cat_cols)}): {len(cat_cols)} cols")

# Final model frames: cleaned raw predictors + full-train EFA factor scores.
# CV (below) refits EFA within each fold on cleaned fold-train subsets only.
Xtr = pd.concat([Xtr_raw, factors_tr_full], axis=1)[feats]
Xte = pd.concat([Xte_raw, factors_te_full], axis=1)[feats]

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

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
print(f"CV: stratified {N_SPLITS}-fold (shuffle=True, random_state={RANDOM_STATE})")
results, fitted = {}, {}
fold_records = []   # per-model, per-fold CV scores (published in Table 5.11)


for name, est in build_models().items():
    print(f"\n--- {name} ---")
    try:
        # ---- CV with EFA refitted WITHIN each fold -------------------------
        # Factor scores for fold train/val come from an EFA fitted on that
        # fold's training partition only: validation rows never influence the
        # factor solution and the target is never an EFA input.
        fold_scores = {"accuracy": [], "balanced_accuracy": [], "f1_macro": []}
        for fold, (tr_idx, val_idx) in enumerate(cv.split(Xtr_raw, ytr), 1):
            Xf_tr = Xtr_raw.iloc[tr_idx].copy()
            Xf_val = Xtr_raw.iloc[val_idx].copy()
            yf_tr = ytr.iloc[tr_idx]
            yf_val = ytr.iloc[val_idx]
            efa = EFATransformer(random_state=RANDOM_STATE)
            efa.fit(Xf_tr)                        # EFA on fold-train only
            Xf_tr_e = efa.transform(Xf_tr)        # raw + fold-fit factors
            Xf_val_e = efa.transform(Xf_val)      # val scored with fold artifact
            fold_pipe = Pipeline([
                ("prep", clone(preprocess)),
                ("clf", clone(est)),
            ])
            fold_pipe.fit(Xf_tr_e, yf_tr)
            pred = fold_pipe.predict(Xf_val_e)
            fold_scores["accuracy"].append(accuracy_score(yf_val, pred))
            fold_scores["balanced_accuracy"].append(
                balanced_accuracy_score(yf_val, pred))
            fold_scores["f1_macro"].append(
                f1_score(yf_val, pred, average="macro", zero_division=0))
            # Per-fold scores are recorded so the published fold-level
            # performance table (Table 5.11) is generated from this very run
            # instead of being hand-maintained.
            fold_records.append({
                "model": name, "fold": fold,
                "accuracy": fold_scores["accuracy"][-1],
                "balanced_accuracy": fold_scores["balanced_accuracy"][-1],
                "f1_macro": fold_scores["f1_macro"][-1],
            })
        cv_sum = {f"test_{k}": float(np.nanmean(v))
                  for k, v in fold_scores.items()}
        # Export fold-level CV records so Table 5.11 is reproducible from this run
        fold_df = pd.DataFrame(fold_records)
        fold_df.to_csv(RESULTS_DIR / "tables" / "table_5_11_fold_level_performance.csv", index=False)
        print("CV acc=%.3f bal=%.3f f1m=%.3f" % (
            cv_sum.get("test_accuracy", np.nan),
            cv_sum.get("test_balanced_accuracy", np.nan),
            cv_sum.get("test_f1_macro", np.nan)))
    except Exception as e:
        print(f"CV failed: {e}"); cv_sum = {}
    # ---- final model on the full training frame ------------------------------
    # Preprocess + classifier only: factor columns come from the full-train
    # EFA artifact (identical contract to inference); EFA refit-per-fold above
    # guarantees unbiased CV estimates without leaking into the saved pipe.
    pipe = Pipeline([("prep", preprocess), ("clf", est)])
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

# Model selection is driven by cross-validated balanced accuracy on the
# TRAINING set only. The held-out test set is never used to choose a model —
# it is reserved for the single final evaluation (reviewer Fix 3). Models that
# produced no finite CV score are not eligible for selection.
valid_models = {n: r for n, r in results.items()
                if np.isfinite(r["cv"].get("test_balanced_accuracy", np.nan))}
if not valid_models:
    raise RuntimeError(
        "No model has a valid cross-validated balanced-accuracy score.")
best_name = max(valid_models,
                key=lambda n: valid_models[n]["cv"]["test_balanced_accuracy"])
print(f"\n>>> BEST (by training CV balanced accuracy): {best_name} "
      f"(cv_bal_acc={valid_models[best_name]['cv']['test_balanced_accuracy']:.3f})")
print(f"    held-out test bal_acc (reported, not used for selection): "
      f"{results[best_name]['test']['balanced_accuracy']:.3f}")



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
               "factor_features": factor_names,
               "rare_category_map": rare_map,
               "rare_threshold": dc.RARE_THRESHOLD,
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
print("\nDONE. Next: python 06_model_evaluation.py")


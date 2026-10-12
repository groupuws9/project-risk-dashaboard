"""04b_imbalance_tuning.py — advanced imbalance handling & tuning (spec 2).

Adds to the base comparison from 04_model_training.py:
  * logreg_smoteenn  — SMOTEENN resampling + balanced logistic regression
  * rf_smoteenn      — SMOTEENN + Random Forest
  * xgboost_tuned    — RandomizedSearchCV (stratified 5-fold) over an
                       imblearn pipeline (prep -> SMOTEENN -> XGB)
  * stacked_ensemble — StackingClassifier (RF + XGB + LGBM -> logistic meta)
                       inside the same resampling pipeline

Evaluates on the SAME stratified 80/20 split as 04 (random_state=42) so rows
are directly comparable.  Appends to results/tables/model_comparison.csv,
saves pkls, and promotes a new best model if balanced accuracy improves.
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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier  # noqa: F401 (kept for parity)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

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

TARGET = "operational_risk_category"
SCORE_COL = "operational_risk_score"
ID_COLS = ["serno", "year", "weight"]
LEAKAGE = ["employee_turnover_rate", "employee_turnover_rate_5yr",
           "absenteeism_rate", "grievance_rate", "disputes_raised",
           "dispute_risk_level", "other_industrial_action", "dismissal_rate",
           "disciplinary_warnings", "suspensions", SCORE_COL]
LABELS = ["Low", "Medium", "High"]
RS = 42

print("=" * 78)
print("IMBALANCE HANDLING + HYPERPARAMETER TUNING + STACKING (spec 2)")
print("=" * 78)

df = pd.read_csv(DATA)
meta = json.loads(META_JSON.read_text())
cats = meta["variable_categories"]
feats = (cats["organizational"] + cats["hr_practices"]
         + cats["employee_relations"] + cats["labor_market"])
feats = [c for c in feats if c in df.columns]
for _d in ("industry_code", "industry_code_80"):
    if _d in feats:
        feats.remove(_d)

# EFA factor scores — identical contract to 04_model_training.py (same saved
# scoring artifact, models/saved_models/factor_scoring.json).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import efa_tools  # noqa: E402

df, _art = efa_tools.ensure(df, MODEL_DIR / "factor_scoring.json")
feats = feats + [c for c in _art["factor_names"] if c not in feats]
factors = _art["factor_names"]
print(f"EFA factor scores appended: {factors}")

X = df[feats].copy()
y = df[TARGET].astype(str).str.strip().map({"Low": 0, "Medium": 1, "High": 2})
m = y.notna()
X, y = X.loc[m].reset_index(drop=True), y.loc[m].astype(int).reset_index(drop=True)

# ---- identical cleaning to 04 (median/mode imputation lives in `prep`) ----
MISS = ["", " ", "NA", "N/A", "nan", "None", "-9", "-8"]
num_cols, cat_cols = [], []
for c in feats:
    if c in X.columns:
        X[c] = X[c].replace(MISS, np.nan)
    if pd.api.types.is_numeric_dtype(X[c]):
        X[c] = pd.to_numeric(X[c], errors="coerce")
        num_cols.append(c)
    else:
        X[c] = X[c].astype(str).str.strip().replace({"": "MISSING",
                                                      "nan": "MISSING",
                                                      "None": "MISSING"})
        cat_cols.append(c)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), cat_cols),
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                      random_state=RS)
print(f"Train {len(Xtr)} / Test {len(Xte)} | features {len(feats)} "
      f"({len(num_cols)} num, {len(cat_cols)} cat)")

results, fitted = {}, {}

models = {}
models["logreg_smoteenn"] = ImbPipeline([
    ("prep", preprocess), ("resample", SMOTEENN(random_state=RS)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RS))])
models["rf_smoteenn"] = ImbPipeline([
    ("prep", preprocess), ("resample", SMOTEENN(random_state=RS)),
    ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                   random_state=RS, n_jobs=-1))])
if HAS_XGB:
    models["xgboost_tuned"] = ImbPipeline([
        ("prep", preprocess), ("resample", SMOTEENN(random_state=RS)),
        ("clf", XGBClassifier(n_estimators=300, eval_metric="mlogloss",
                              random_state=RS, n_jobs=2))])
if HAS_XGB and HAS_LGBM:
    stacking = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100,
                                          class_weight="balanced",
                                          random_state=RS, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=150, eval_metric="mlogloss",
                                  random_state=RS, n_jobs=2)),
            ("lgbm", LGBMClassifier(n_estimators=150, random_state=RS,
                                    verbose=-1)),
        ],
        final_estimator=LogisticRegression(max_iter=2000,
                                           class_weight="balanced",
                                           random_state=RS),
        cv=5, n_jobs=-1)          # stratified 5-fold (spec 3.4)
    models["stacked_ensemble"] = ImbPipeline([
        ("prep", preprocess), ("resample", SMOTEENN(random_state=RS)),
        ("clf", stacking)])

# plain SMOTE variant (spec 3.2 mentions both SMOTE and SMOTEENN)
models["logreg_smote"] = ImbPipeline([
    ("prep", preprocess),
    ("resample", SMOTE(random_state=RS, k_neighbors=3)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RS))])

# ---- RandomizedSearchCV (spec 3.4: stratified 5-fold, balanced accuracy) --
if HAS_XGB:
    print("\nRandomizedSearchCV on xgboost_tuned (iter=8, cv=5, stratified)...")
    search_pipe = ImbPipeline([
        ("prep", preprocess), ("resample", SMOTEENN(random_state=RS)),
        ("clf", XGBClassifier(eval_metric="mlogloss", random_state=RS,
                              n_jobs=2))])
    grid = {
        "clf__n_estimators": [150, 300, 450],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.03, 0.1, 0.2],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.9, 1.0],
    }
    try:
        search = RandomizedSearchCV(
            search_pipe, grid, n_iter=8,
            cv=StratifiedKFold(5, shuffle=True, random_state=RS),
            scoring="balanced_accuracy", random_state=RS, n_jobs=2,
            refit=True, verbose=0)
        search.fit(Xtr, ytr)
        models["xgboost_tuned"] = search.best_estimator_
        print(f"  best params: {json.dumps(search.best_params_, default=str)}")
        print(f"  best CV balanced_accuracy: {search.best_score_:.3f}")
    except Exception as e:
        print(f"  search failed ({e}); keeping default xgboost_tuned")

# ---- fit + evaluate on the shared held-out split --------------------------
for name, pipe in models.items():
    print(f"\n--- {name} ---")
    try:
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)
        bal = balanced_accuracy_score(yte, pred)
        f1m = f1_score(yte, pred, average="macro", zero_division=0)
        print(f"TEST acc={acc:.3f} bal_acc={bal:.3f} f1_macro={f1m:.3f}")
        print(classification_report(yte, pred, target_names=LABELS,
                                    zero_division=0))
        fitted[name] = pipe
        results[name] = {"accuracy": acc, "balanced_accuracy": bal,
                         "f1_macro": f1m,
                         "confusion_matrix": confusion_matrix(yte, pred).tolist()}
        joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")
    except Exception as e:
        print(f"FAILED: {e}")

if not results:
    raise SystemExit("No models trained successfully.")

# ---- append to the 04 comparison table ------------------------------------
comp_path = TABLE_DIR / "model_comparison.csv"
comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
new_rows = []
for n, r in results.items():
    new_rows.append({"model": n, "test_accuracy": r["accuracy"],
                     "test_balanced_accuracy": r["balanced_accuracy"],
                     "test_f1_macro": r["f1_macro"],
                     "cv_test_accuracy": np.nan,
                     "cv_test_balanced_accuracy": np.nan,
                     "cv_test_f1_macro": np.nan})
comp = pd.concat([comp, pd.DataFrame(new_rows)], ignore_index=True)
comp = comp.drop_duplicates(subset=["model"], keep="last")
comp = comp.sort_values("test_balanced_accuracy", ascending=False)
comp.to_csv(comp_path, index=False)
print(f"\nUpdated {comp_path}")

# ---- promote best model if improved ---------------------------------------
mdl_path = MODEL_DIR / "model_metadata.json"
mdl = json.loads(mdl_path.read_text())
mdl.setdefault("all_test_balanced_accuracy", {}).update(
    {n: r["balanced_accuracy"] for n, r in results.items()})
cur_best = mdl["best_model"]
cur_best_score = mdl["all_test_balanced_accuracy"].get(cur_best, -1)
winner = max(mdl["all_test_balanced_accuracy"],
             key=lambda k: mdl["all_test_balanced_accuracy"][k])
if mdl["all_test_balanced_accuracy"][winner] > cur_best_score:
    joblib.dump(joblib.load(MODEL_DIR / f"{winner}.pkl"),
                MODEL_DIR / "best_model.pkl")
    (MODEL_DIR / "best_model_name.txt").write_text(winner)
    mdl["best_model"] = winner
    print(f">>> NEW BEST MODEL: {winner} "
          f"(bal_acc={mdl['all_test_balanced_accuracy'][winner]:.3f})")
else:
    print(f">>> best model unchanged: {cur_best} "
          f"(bal_acc={cur_best_score:.3f})")
mdl_path.write_text(json.dumps(mdl, indent=2))
print("DONE. Next: python 05_model_evaluation.py && python 06_shap_explanation.py")

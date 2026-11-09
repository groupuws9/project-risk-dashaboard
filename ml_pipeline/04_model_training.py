﻿"""
04_model_training.py — Train and validate operational-risk classifiers.

Implements a leakage-safe, fold-safe machine-learning pipeline for
classifying workplaces into Low, Medium and High employee-turnover bands.

The outer 80/20 train-test partition is created once. Five-fold stratified
cross-validation is then applied only within the outer training partition.
Rare-category grouping, EFA, imputation, scaling and categorical encoding
are fitted within each cross-validation training fold.

The held-out test partition is used only after model selection.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_contract as dc  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"
META_JSON = BASE / "data" / "processed" / "dataset_metadata.json"

MODEL_DIR = BASE / "models" / "saved_models"
TABLE_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = dc.TARGET
LABELS = dc.LABELS
RANDOM_STATE = dc.RANDOM_STATE

# Aligns with the five-fold stratified cross-validation protocol in Section 3.6.
N_SPLITS = dc.N_SPLITS
TEST_SIZE = dc.TEST_SIZE


print("=" * 78)
print("MODEL TRAINING — WERS/WIRS workplace operational risk (Low/Medium/High)")
print("Target = employee turnover band (Low / Medium / High)")
print("=" * 78)


# ---------------------------------------------------------------------------
# Load data and define leakage-safe predictors
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA)

print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
print(df[TARGET].value_counts().to_string())

predictor_columns = dc.predictor_columns()

prohibited_columns = (
    set(dc.TARGET_COLUMNS)
    | set(dc.DIRECT_TARGET_LEAKAGE)
    | set(dc.PRECAUTIONARY_EXCLUSIONS)
    | set(dc.ADMINISTRATIVE_EXCLUSIONS)
    | set(dc.HIGH_CARDINALITY_EXCLUSIONS)
)

leaked_columns = set(predictor_columns) & prohibited_columns

if leaked_columns:
    raise ValueError(
        f"Leakage-prone columns detected: {sorted(leaked_columns)}"
    )

if len(predictor_columns) != len(set(predictor_columns)):
    raise ValueError("Duplicate predictor columns detected.")

print(f"Verified {len(predictor_columns)} unique leakage-free predictors")


# ---------------------------------------------------------------------------
# Canonical outer train-test partition
# ---------------------------------------------------------------------------

Xtr_raw, Xte_raw, ytr, yte = dc.split_dataset(
    df,
    predictor_columns,
    target_column=TARGET,
)

Xtr_raw = Xtr_raw.copy()
Xte_raw = Xte_raw.copy()

print(f"\nOuter training partition: {Xtr_raw.shape}")
print(f"Outer held-out test partition: {Xte_raw.shape}")
print("Training class distribution:", ytr.value_counts().to_dict())
print("Test class distribution:", yte.value_counts().to_dict())


# ---------------------------------------------------------------------------
# Identify numeric and categorical columns from outer training data
# ---------------------------------------------------------------------------

numeric_candidates, categorical_candidates = dc.classify_columns(Xtr_raw)

print(f"Numeric candidate columns: {len(numeric_candidates)}")
print(f"Categorical candidate columns: {len(categorical_candidates)}")

# For the pipeline, we need to know which columns will be numeric after EFA
# EFA will add factor_1, factor_2, factor_3, factor_4 which are always numeric
# So we need to update the numeric columns list to include a placeholder for factors
# The actual factor columns will be added by EFA, so we don't need to list them here
# The ColumnTransformer will handle them dynamically when they appear


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

def build_models():
    """Return candidate estimators with imbalance-aware settings."""
    models = {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
        )

    if HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            verbose=-1,
            random_state=RANDOM_STATE,
        )

    return models


# ---------------------------------------------------------------------------
# Fold-safe cross-validation
# ---------------------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print(
    f"\nCross-validation: stratified {N_SPLITS}-fold "
    f"(shuffle=True, random_state={RANDOM_STATE})"
)
print(
    "CV uses a Pipeline containing fold-specific rare-category grouping, "
    "EFA, imputation, scaling and encoding."
)

results = {}
fitted_pipelines = {}
fold_records = []

for model_name, estimator in build_models().items():
    print(f"\n--- {model_name} ---")

    try:
        cv_pipeline = dc.build_cv_pipeline(
            model=estimator,
        )

        cv_results = cross_validate(
            cv_pipeline,
            Xtr_raw,
            ytr,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1_macro": "f1_macro",
            },
            return_train_score=False,
            n_jobs=-1,
            error_score=np.nan,
        )

        cv_summary = {
            "test_accuracy": float(
                np.nanmean(cv_results["test_accuracy"])
            ),
            "test_balanced_accuracy": float(
                np.nanmean(cv_results["test_balanced_accuracy"])
            ),
            "test_f1_macro": float(
                np.nanmean(cv_results["test_f1_macro"])
            ),
        }

        for fold_index in range(N_SPLITS):
            fold_records.append(
                {
                    "model": model_name,
                    "fold": fold_index + 1,
                    "accuracy": cv_results["test_accuracy"][fold_index],
                    "balanced_accuracy": cv_results[
                        "test_balanced_accuracy"
                    ][fold_index],
                    "f1_macro": cv_results["test_f1_macro"][fold_index],
                }
            )

        print(
            "CV accuracy={:.3f}, balanced accuracy={:.3f}, "
            "macro-F1={:.3f}".format(
                cv_summary["test_accuracy"],
                cv_summary["test_balanced_accuracy"],
                cv_summary["test_f1_macro"],
            )
        )

        results[model_name] = {"cv": cv_summary}

    except Exception as error:
        print(f"Cross-validation failed for {model_name}: {error}")
        results[model_name] = {"cv": {}}


# ---------------------------------------------------------------------------
# Final refitting on the full outer training partition
# ---------------------------------------------------------------------------

valid_models = {
    model_name: result
    for model_name, result in results.items()
    if np.isfinite(
        result["cv"].get("test_balanced_accuracy", np.nan)
    )
}

if not valid_models:
    raise RuntimeError(
        "No model produced a valid cross-validated balanced-accuracy score."
    )

best_model_name = max(
    valid_models,
    key=lambda model_name: valid_models[model_name]["cv"][
        "test_balanced_accuracy"
    ],
)

print(
    f"\nBest model selected using training cross-validated balanced "
    f"accuracy: {best_model_name}"
)

# Refit the best model on the full outer training partition using the same pipeline
best_estimator = build_models()[best_model_name]
final_pipeline = dc.build_cv_pipeline(model=best_estimator)
final_pipeline.fit(Xtr_raw, ytr)

# Evaluate on the held-out test partition
test_predictions = final_pipeline.predict(Xte_raw)
test_probabilities = final_pipeline.predict_proba(Xte_raw)

test_accuracy = accuracy_score(yte, test_predictions)
test_balanced_accuracy = balanced_accuracy_score(yte, test_predictions)
test_f1_macro = f1_score(yte, test_predictions, average="macro", zero_division=0)

print(
    f"\nTest metrics for {best_model_name}:"
    f"\n  Accuracy: {test_accuracy:.4f}"
    f"\n  Balanced accuracy: {test_balanced_accuracy:.4f}"
    f"\n  Macro-F1: {test_f1_macro:.4f}"
)

print(
    classification_report(
        yte,
        test_predictions,
        target_names=LABELS,
        zero_division=0,
    )
)

results[best_model_name]["test"] = {
    "accuracy": test_accuracy,
    "balanced_accuracy": test_balanced_accuracy,
    "f1_macro": test_f1_macro,
    "report": classification_report(
        yte,
        test_predictions,
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    ),
    "confusion_matrix": confusion_matrix(
        yte,
        test_predictions
    ).tolist(),
}

fitted_pipelines[best_model_name] = final_pipeline


# ---------------------------------------------------------------------------
# Save artefacts and metadata
# ---------------------------------------------------------------------------

for model_name, pipeline in fitted_pipelines.items():
    joblib.dump(pipeline, MODEL_DIR / f"{model_name}.pkl")

joblib.dump(
    fitted_pipelines[best_model_name],
    MODEL_DIR / "best_model.pkl",
)

with open(MODEL_DIR / "best_model_name.txt", "w") as file:
    file.write(best_model_name)

comparison_rows = []

for model_name, result in results.items():
    row = {"model": model_name}

    for metric, value in result["cv"].items():
        row[f"cv_{metric}"] = value

    # Only add test metrics if they exist (only for best model)
    if "test" in result:
        for metric, value in result["test"].items():
            if isinstance(value, float):
                row[f"test_{metric}"] = value

    comparison_rows.append(row)

pd.DataFrame(comparison_rows).to_csv(
    TABLE_DIR / "model_comparison.csv",
    index=False,
)

pd.DataFrame(fold_records).to_csv(
    TABLE_DIR / "cross_validation_fold_results.csv",
    index=False,
)

best_pipeline = fitted_pipelines[best_model_name]

# Persist the canonical train-derived rare-category map and the derived factor
# contract, so the metadata always matches data_contract.build_model_frames().
_, _, rare_category_map = dc.clean_frames(
    Xtr_raw.copy(),
    Xte_raw.copy(),
)

factor_features = list(dc.load_factor_artifact()["factor_names"])

metadata = {
    "best_model": best_model_name,
    "target": TARGET,
    "labels": LABELS,
    "features": predictor_columns,
    "numeric_features": numeric_candidates,
    "categorical_features": categorical_candidates,
    "factor_features": factor_features,
    "leakage_excluded": dc.LEAKAGE,
    "high_cardinality_excluded": dc.HIGH_CARDINALITY_EXCLUSIONS,
    "rare_category_map": rare_category_map,
    "rare_threshold": dc.RARE_THRESHOLD,
    "cv_folds": N_SPLITS,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "all_test_balanced_accuracy": {
        model_name: result.get("test", {}).get("balanced_accuracy", np.nan)
        for model_name, result in results.items()
    },
}

with open(MODEL_DIR / "model_metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

print("\nSaved models, comparison table, fold results and metadata.")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

importance_model = next(
    (
        model_name
        for model_name in (
            "xgboost",
            "lightgbm",
            "random_forest",
        )
        if model_name in fitted_pipelines
    ),
    best_model_name,
)

try:
    pipeline = fitted_pipelines[importance_model]
    classifier = pipeline.named_steps["classifier"]
    preprocessing_step = pipeline.named_steps["preprocessing"]

    feature_names = preprocessing_step.get_feature_names_out()
    importances = getattr(classifier, "feature_importances_", None)

    if importances is None and hasattr(classifier, "coef_"):
        importances = np.abs(
            np.asarray(classifier.coef_)
        ).mean(axis=0)

    if importances is not None:
        importance_frame = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.asarray(importances, dtype=float),
            }
        )

        importance_frame = importance_frame.sort_values(
            by="importance",
            ascending=False,
        )

        importance_frame.to_csv(
            TABLE_DIR / "feature_importance.csv",
            index=False,
        )

        print(f"Feature importance saved for {importance_model}.")

except Exception as error:
    print(f"Feature-importance export skipped: {error}")


print("\nDONE. Next: python 05_imbalance_tuning.py")
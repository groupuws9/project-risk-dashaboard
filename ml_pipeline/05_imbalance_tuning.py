"""
05_imbalance_tuning.py — Advanced imbalance handling and hyperparameter tuning.

This script evaluates imbalance-aware variants using the same canonical
outer train-test partition as 04_model_training.py.

The saved pipelines contain the complete inference workflow:

    Categorical cleaning -> numeric coercion -> rare-category grouping
    -> EFA -> preprocessing -> SMOTEENN/SMOTE -> classifier

Therefore, this script passes raw predictor matrices directly to the
pipelines. All preprocessing and resampling are fitted only on the relevant
training data.

The held-out test partition is used only for reporting. It is not used to
select or promote a deployed model.
"""

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbalancedPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
)
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
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier  # noqa: F401
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


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"
MODEL_DIR = BASE / "models" / "saved_models"
META_JSON = MODEL_DIR / "model_metadata.json"
TABLE_DIR = BASE / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = dc.TARGET
LABELS = dc.LABELS
RANDOM_STATE = dc.RANDOM_STATE
N_SPLITS = dc.N_SPLITS
TEST_SIZE = dc.TEST_SIZE


print("=" * 78)
print("IMBALANCE HANDLING, HYPERPARAMETER TUNING AND STACKING")
print("=" * 78)


# ---------------------------------------------------------------------------
# Load metadata and reconstruct the canonical outer partition
# ---------------------------------------------------------------------------

if not META_JSON.exists():
    raise FileNotFoundError(
        f"Model metadata not found: {META_JSON}. "
        "Run 04_model_training.py before running imbalance tuning."
    )

metadata = json.loads(META_JSON.read_text())

TARGET = metadata["target"]
LABELS = metadata["labels"]
PREDICTOR_COLUMNS = metadata["features"]

df = pd.read_csv(DATA)

Xtr_raw, Xte_raw, ytr, yte = dc.split_dataset(
    df,
    PREDICTOR_COLUMNS,
    target_column=TARGET,
)

Xtr_raw = Xtr_raw.copy()
Xte_raw = Xte_raw.copy()

print(f"Outer training partition: {Xtr_raw.shape}")
print(f"Outer held-out test partition: {Xte_raw.shape}")
print("Training class distribution:", ytr.value_counts().to_dict())
print("Test class distribution:", yte.value_counts().to_dict())

if list(Xtr_raw.columns) != list(PREDICTOR_COLUMNS):
    raise ValueError(
        "Predictor columns in the reconstructed training partition do not "
        "match the feature contract in model_metadata.json."
    )


# ---------------------------------------------------------------------------
# Identify numeric and categorical columns from outer training data
# ---------------------------------------------------------------------------

numeric_candidates = []
categorical_candidates = []

for column in Xtr_raw.columns:
    series = Xtr_raw[column].replace(dc.MISS_TOKENS, np.nan)
    coerced = pd.to_numeric(series, errors="coerce")

    valid_count = series.notna().sum()
    numeric_ratio = (
        coerced.notna().sum() / valid_count
        if valid_count > 0
        else 0
    )

    if numeric_ratio > 0.85 and coerced.nunique(dropna=True) > 12:
        numeric_candidates.append(column)
    else:
        categorical_candidates.append(column)

print(f"Numeric candidate columns: {len(numeric_candidates)}")
print(f"Categorical candidate columns: {len(categorical_candidates)}")


# ---------------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------------

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_candidates,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            categorical_candidates,
        ),
    ],
    verbose_feature_names_out=False,
)

# Replace ColumnTransformer with DynamicColumnTransformer for consistency
preprocess = dc.DynamicColumnTransformer()


# ---------------------------------------------------------------------------
# Imbalance-aware pipelines
# ---------------------------------------------------------------------------

def build_imbalanced_pipeline(classifier, resampler):
    """Build a complete imbalance-aware pipeline using raw input data."""
    return ImbalancedPipeline(
        steps=[
            (
                "categorical_cleaning",
                dc.CategoricalCleaningTransformer(),
            ),
            (
                "numeric_coercion",
                dc.NumericCoercionTransformer(),
            ),
            (
                "rare_category",
                dc.RareCategoryTransformer(
                    threshold=dc.RARE_THRESHOLD,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "efa",
                dc.EFATransformer(
                    random_state=RANDOM_STATE,
                ),
            ),
            ("preprocessing", preprocess),
            ("resampler", resampler),
            ("classifier", classifier),
        ]
    )


models = {
    "logreg_smoteenn": build_imbalanced_pipeline(
        classifier=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        resampler=SMOTEENN(random_state=RANDOM_STATE),
    ),
    "rf_smoteenn": build_imbalanced_pipeline(
        classifier=RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        resampler=SMOTEENN(random_state=RANDOM_STATE),
    ),
    "logreg_smote": build_imbalanced_pipeline(
        classifier=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        resampler=SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=3,
        ),
    ),
}

if HAS_XGB:
    models["xgboost_tuned"] = build_imbalanced_pipeline(
        classifier=XGBClassifier(
            n_estimators=300,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
        resampler=SMOTEENN(random_state=RANDOM_STATE),
    )

if HAS_XGB and HAS_LGBM:
    stacking_classifier = StackingClassifier(
        estimators=[
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=150,
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=2,
                ),
            ),
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=150,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                ),
            ),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        n_jobs=-1,
    )

    models["stacked_ensemble"] = build_imbalanced_pipeline(
        classifier=stacking_classifier,
        resampler=SMOTEENN(random_state=RANDOM_STATE),
    )


# ---------------------------------------------------------------------------
# Randomized search for XGBoost
# ---------------------------------------------------------------------------

if HAS_XGB:
    print(
        "\nRandomizedSearchCV for XGBoost: "
        f"{N_SPLITS}-fold stratified CV, balanced accuracy objective"
    )

    search_pipeline = build_imbalanced_pipeline(
        classifier=XGBClassifier(
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
        resampler=SMOTEENN(random_state=RANDOM_STATE),
    )

    parameter_distributions = {
        "classifier__n_estimators": [150, 300, 450],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.03, 0.10, 0.20],
        "classifier__subsample": [0.70, 0.90, 1.00],
        "classifier__colsample_bytree": [0.70, 0.90, 1.00],
    }

    try:
        random_search = RandomizedSearchCV(
            estimator=search_pipeline,
            param_distributions=parameter_distributions,
            n_iter=8,
            cv=StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE,
            ),
            scoring="balanced_accuracy",
            random_state=RANDOM_STATE,
            n_jobs=2,
            refit=True,
            verbose=0,
        )

        random_search.fit(Xtr_raw, ytr)

        models["xgboost_tuned"] = random_search.best_estimator_

        print(
            "Best parameters: "
            f"{json.dumps(random_search.best_params_, default=str)}"
        )
        print(
            "Best cross-validated balanced accuracy: "
            f"{random_search.best_score_:.3f}"
        )

    except Exception as error:
        print(f"Randomized search failed: {error}")
        print("Retaining the default XGBoost configuration.")


# ---------------------------------------------------------------------------
# Fit and evaluate imbalance-aware variants
# ---------------------------------------------------------------------------

results = {}
fitted_pipelines = {}

for model_name, pipeline in models.items():
    print(f"\n--- {model_name} ---")

    try:
        pipeline.fit(Xtr_raw, ytr)
        predictions = pipeline.predict(Xte_raw)

        accuracy = accuracy_score(yte, predictions)
        balanced_accuracy = balanced_accuracy_score(yte, predictions)
        macro_f1 = f1_score(
            yte,
            predictions,
            average="macro",
            zero_division=0,
        )

        print(
            f"Held-out test accuracy={accuracy:.3f}, "
            f"balanced accuracy={balanced_accuracy:.3f}, "
            f"macro-F1={macro_f1:.3f}"
        )

        print(
            classification_report(
                yte,
                predictions,
                target_names=LABELS,
                zero_division=0,
            )
        )

        fitted_pipelines[model_name] = pipeline

        results[model_name] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1_macro": macro_f1,
            "confusion_matrix": confusion_matrix(
                yte,
                predictions
            ).tolist(),
        }

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{model_name}.pkl",
        )

    except Exception as error:
        print(f"{model_name} failed: {error}")

if not results:
    raise SystemExit("No imbalance-aware models trained successfully.")


# ---------------------------------------------------------------------------
# Append results to the model-comparison table
# ---------------------------------------------------------------------------

comparison_path = TABLE_DIR / "model_comparison.csv"

if comparison_path.exists():
    comparison_table = pd.read_csv(comparison_path)
else:
    comparison_table = pd.DataFrame()

new_rows = []

for model_name, result in results.items():
    new_rows.append(
        {
            "model": model_name,
            "test_accuracy": result["accuracy"],
            "test_balanced_accuracy": result["balanced_accuracy"],
            "test_f1_macro": result["f1_macro"],
            "cv_test_accuracy": np.nan,
            "cv_test_balanced_accuracy": np.nan,
            "cv_test_f1_macro": np.nan,
        }
    )

comparison_table = pd.concat(
    [comparison_table, pd.DataFrame(new_rows)],
    ignore_index=True,
)

comparison_table = comparison_table.drop_duplicates(
    subset=["model"],
    keep="last",
)

comparison_table = comparison_table.sort_values(
    by="test_balanced_accuracy",
    ascending=False,
)

comparison_table.to_csv(comparison_path, index=False)

print(f"\nUpdated {comparison_path}")


# ---------------------------------------------------------------------------
# Update metadata without changing the deployed model
# ---------------------------------------------------------------------------

metadata_path = MODEL_DIR / "model_metadata.json"
updated_metadata = json.loads(metadata_path.read_text())

updated_metadata.setdefault(
    "all_test_balanced_accuracy",
    {}
).update(
    {
        model_name: result["balanced_accuracy"]
        for model_name, result in results.items()
    }
)

deployed_model = updated_metadata["best_model"]

best_variant = max(
    results,
    key=lambda model_name: results[model_name]["balanced_accuracy"],
)

print(
    f"\nDeployed model unchanged: {deployed_model} "
    "(selected using training cross-validated balanced accuracy in "
    "04_model_training.py)"
)

print(
    f"Best imbalance-aware variant by held-out test balanced accuracy: "
    f"{best_variant} "
    f"({results[best_variant]['balanced_accuracy']:.3f})"
)

with open(metadata_path, "w") as file:
    json.dump(updated_metadata, file, indent=2)

print("\nDONE. Next: python 06_model_evaluation.py")
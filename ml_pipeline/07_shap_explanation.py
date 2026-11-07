"""
07_shap_explanation.py — SHAP global and local explanations.

Produces SHAP-based explanations for the selected employee-turnover-band
classification model.

The saved model artefact contains the complete inference pipeline:

    Categorical cleaning
    -> numeric coercion
    -> rare-category grouping
    -> EFA
    -> preprocessing
    -> classifier

Therefore, this script passes raw predictor data through the saved pipeline's
preprocessing steps before generating SHAP explanations.

SHAP values explain how the fitted model attributes its predictions to the
retained predictor features. They do not establish causal effects or the
consequences of organisational interventions.
"""

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_contract as dc  # noqa: E402


BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"
MODEL_DIR = BASE / "models" / "saved_models"
TAB_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"

TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Background and evaluation sample sizes are deliberately limited to keep
# KernelExplainer computationally tractable where required.
N_BACKGROUND = 100
N_EVALUATION = 300

RANDOM_STATE = dc.RANDOM_STATE


print("=" * 78)
print("SHAP EXPLANATIONS — workplace operational-risk classification")
print("=" * 78)


# ---------------------------------------------------------------------------
# Load metadata and reconstruct the canonical partitions
# ---------------------------------------------------------------------------

metadata_path = MODEL_DIR / "model_metadata.json"

if not metadata_path.exists():
    raise FileNotFoundError(
        f"Model metadata not found: {metadata_path}. "
        "Run 04_model_training.py before running SHAP explanation."
    )

metadata = json.loads(metadata_path.read_text())

best_model_name = metadata["best_model"]
TARGET = metadata["target"]
LABELS = metadata["labels"]
PREDICTOR_COLUMNS = metadata["features"]
NUMERIC_COLUMNS = metadata["numeric_features"]
CATEGORICAL_COLUMNS = metadata["categorical_features"]

model_path = MODEL_DIR / f"{best_model_name}.pkl"

if not model_path.exists():
    raise FileNotFoundError(
        f"Selected model artefact not found: {model_path}. "
        "Run 04_model_training.py before running SHAP explanation."
    )

pipeline = joblib.load(model_path)

print(f"Selected model: {best_model_name}")
print(f"Pipeline type: {type(pipeline).__name__}")

df = pd.read_csv(DATA)

Xtr_raw, Xte_raw, ytr, yte = dc.split_dataset(
    df,
    PREDICTOR_COLUMNS,
    target_column=TARGET,
)

Xtr_raw = Xtr_raw.copy()
Xte_raw = Xte_raw.copy()

print(f"SHAP background sample drawn from training partition: {Xtr_raw.shape}")
print(f"SHAP evaluation sample drawn from held-out test partition: {Xte_raw.shape}")


# ---------------------------------------------------------------------------
# Transform raw data using the saved pipeline
# ---------------------------------------------------------------------------

# pipeline[:-1] applies every preprocessing step but not the final classifier:
# categorical cleaning -> numeric coercion -> rare-category grouping -> EFA
# -> imputation, scaling and encoding.
preprocessing_pipeline = pipeline[:-1]

background_size = min(N_BACKGROUND, len(Xtr_raw))
evaluation_size = min(N_EVALUATION, len(Xte_raw))

rng = np.random.default_rng(RANDOM_STATE)

background_indices = rng.choice(
    len(Xtr_raw),
    size=background_size,
    replace=False,
)

evaluation_indices = rng.choice(
    len(Xte_raw),
    size=evaluation_size,
    replace=False,
)

X_background_raw = Xtr_raw.iloc[background_indices].copy()
X_evaluation_raw = Xte_raw.iloc[evaluation_indices].copy()

X_background_transformed = preprocessing_pipeline.transform(
    X_background_raw
)

X_evaluation_transformed = preprocessing_pipeline.transform(
    X_evaluation_raw
)

transformed_feature_names = list(
    pipeline.named_steps["preprocessing"].get_feature_names_out()
)

print(f"Transformed feature matrix: {X_evaluation_transformed.shape}")
print(f"Transformed feature count: {len(transformed_feature_names)}")


# ---------------------------------------------------------------------------
# Map transformed features to original predictors
# ---------------------------------------------------------------------------

def map_transformed_feature_to_original(
    transformed_feature_name,
    numeric_columns,
    categorical_columns,
):
    """
    Map a transformed feature name to its original predictor column.

    Numeric transformed features retain their original names. One-hot encoded
    categorical features use the form `column_value`.
    """
    if transformed_feature_name in numeric_columns:
        return transformed_feature_name

    for column in sorted(
        categorical_columns,
        key=len,
        reverse=True,
    ):
        if transformed_feature_name == column:
            return column

        if transformed_feature_name.startswith(f"{column}_"):
            return column

    return transformed_feature_name


original_feature_mapping = [
    map_transformed_feature_to_original(
        transformed_feature_name,
        NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
    )
    for transformed_feature_name in transformed_feature_names
]


# ---------------------------------------------------------------------------
# Generate SHAP values
# ---------------------------------------------------------------------------

classifier = pipeline.named_steps["classifier"]
classifier_name = type(classifier).__name__

import shap  # noqa: E402

print(f"Generating SHAP values using {classifier_name}.")

if hasattr(classifier, "estimators_") or classifier_name in (
    "XGBClassifier",
    "LGBMClassifier",
):
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_evaluation_transformed)
    print("Used TreeExplainer.")

elif hasattr(classifier, "coef_"):
    explainer = shap.LinearExplainer(
        classifier,
        X_background_transformed,
    )
    shap_values = explainer.shap_values(X_evaluation_transformed)
    print("Used LinearExplainer.")

else:
    explainer = shap.Explainer(
        classifier.predict_proba,
        X_background_transformed,
    )
    shap_values = explainer(X_evaluation_transformed).values
    print("Used model-specific SHAP explainer.")

# Normalise SHAP output to a three-dimensional array:
# (n_evaluation_records, n_transformed_features, n_classes)
shap_array = np.asarray(shap_values, dtype=float)

if shap_array.ndim == 2:
    shap_array = shap_array[:, :, np.newaxis]

print(f"SHAP array shape: {shap_array.shape}")

n_classes = shap_array.shape[2]


# ---------------------------------------------------------------------------
# Global SHAP importance
# ---------------------------------------------------------------------------

mean_absolute_shap = np.abs(shap_array).mean(axis=(0, 2))

global_importance = {}

for original_feature, shap_value in zip(
    original_feature_mapping,
    mean_absolute_shap,
):
    global_importance[original_feature] = (
        global_importance.get(original_feature, 0.0)
        + float(shap_value)
    )

global_importance_frame = pd.DataFrame(
    {
        "feature": list(global_importance.keys()),
        "mean_abs_shap": list(global_importance.values()),
    }
).sort_values(
    by="mean_abs_shap",
    ascending=False,
)

global_importance_frame.to_csv(
    TAB_DIR / "shap_importance.csv",
    index=False,
)

print("\nTop SHAP drivers:")
print(global_importance_frame.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Global SHAP plots
# ---------------------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


top_features = global_importance_frame.head(20).iloc[::-1]

plt.figure(figsize=(9, 7))
plt.barh(
    top_features["feature"].astype(str),
    top_features["mean_abs_shap"],
    color="steelblue",
)
plt.xlabel("Mean absolute SHAP value")
plt.title(
    f"SHAP global feature importance — {best_model_name}"
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "shap_summary.png",
    dpi=130,
)
plt.close()

print("Saved results/figures/shap_summary.png")

try:
    plt.figure(figsize=(10, 7))

    if n_classes > 1:
        shap_values_for_plot = shap_array
    else:
        shap_values_for_plot = shap_array[:, :, 0]

    shap.summary_plot(
        shap_values_for_plot,
        features=X_evaluation_transformed,
        feature_names=transformed_feature_names,
        max_display=20,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "shap_beeswarm.png",
        dpi=130,
    )
    plt.close()

    print("Saved results/figures/shap_beeswarm.png")

except Exception as error:
    print(f"Beeswarm plot skipped: {error}")


# ---------------------------------------------------------------------------
# Local SHAP explanation
# ---------------------------------------------------------------------------

# Select a held-out test record with relatively complete predictor values.
numeric_complete = Xte_raw[NUMERIC_COLUMNS].notna().all(axis=1)

numeric_within_range = Xte_raw[NUMERIC_COLUMNS].apply(
    lambda column: pd.to_numeric(column, errors="coerce")
).le(100).all(axis=1)

missing_categorical_count = (
    Xte_raw[CATEGORICAL_COLUMNS]
    .astype(str)
    .eq(dc.MISSING_LEVEL)
    .sum(axis=1)
)

candidate_mask = (
    numeric_complete
    & numeric_within_range
    & (missing_categorical_count <= 2)
)

candidate_indices = Xte_raw.index[candidate_mask]

local_row_position = (
    int(candidate_indices[0])
    if len(candidate_indices) > 0
    else int(Xte_raw.index[0])
)

local_row_raw = Xte_raw.iloc[[local_row_position]].copy()
local_row_transformed = preprocessing_pipeline.transform(local_row_raw)

predicted_class_index = int(
    np.argmax(
        pipeline.predict_proba(local_row_raw)[0]
    )
)

local_shap_values = explainer.shap_values(local_row_transformed)

if isinstance(local_shap_values, list):
    local_shap_array = np.stack(
        local_shap_values,
        axis=-1,
    )[0]
else:
    local_shap_array = np.asarray(
        local_shap_values,
        dtype=float,
    )

    if local_shap_array.ndim == 3:
        local_shap_array = local_shap_array[0]
    elif local_shap_array.ndim == 2:
        if local_shap_array.shape[0] == 1:
            local_shap_array = local_shap_array[0][:, np.newaxis]

if local_shap_array.shape[1] > 1:
    class_contributions = local_shap_array[:, predicted_class_index]
else:
    class_contributions = local_shap_array[:, 0]

expected_value = np.asarray(
    getattr(explainer, "expected_value", 0.0)
).reshape(-1)

if expected_value.size > predicted_class_index:
    base_value = float(expected_value[predicted_class_index])
else:
    base_value = float(expected_value[0])

local_contributions = {}

for original_feature, contribution in zip(
    original_feature_mapping,
    class_contributions,
):
    local_contributions[original_feature] = (
        local_contributions.get(original_feature, 0.0)
        + float(contribution)
    )

top_local_contributions = sorted(
    local_contributions.items(),
    key=lambda item: abs(item[1]),
    reverse=True,
)[:12]

example_record = Xte_raw.iloc[local_row_position].to_dict()

local_explanation = {
    "model": best_model_name,
    "class_explained": LABELS[predicted_class_index],
    "base_value": round(base_value, 5),
    "example": {
        column: (
            None
            if isinstance(value, float) and value != value
            else value
        )
        for column, value in example_record.items()
    },
    "contributions": [
        {
            "feature": feature,
            "shap": round(contribution, 5),
            "value": (
                None
                if isinstance(example_record.get(feature), float)
                and example_record.get(feature) != example_record.get(feature)
                else str(example_record.get(feature))
            ),
        }
        for feature, contribution in top_local_contributions
    ],
}

(TAB_DIR / "shap_local.json").write_text(
    json.dumps(local_explanation, indent=2)
)

print(
    "\nSaved results/tables/shap_local.json "
    f"(test row position {local_row_position}, "
    f"class {LABELS[predicted_class_index]}, "
    f"maximum absolute SHAP contribution "
    f"{max(abs(value) for _, value in top_local_contributions):.4f})"
)

print("\nDONE.")
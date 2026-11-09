"""
data_contract.py — shared data and preprocessing contract.

This module is the single source of truth for:

* The canonical outer train-test partition.
* Leakage-safe predictor selection.
* Target-construction rules.
* Fold-safe rare-category grouping.
* Fold-safe exploratory factor analysis (EFA).
* The complete preprocessing and inference pipeline.

The canonical preprocessing sequence is:

    Raw predictors
    -> categorical cleaning
    -> numeric coercion
    -> fold-specific rare-category grouping
    -> fold-specific EFA
    -> imputation, scaling and encoding
    -> classifier

The outer 80/20 train-test partition is created before model development.
Five-fold stratified cross-validation is applied only within the outer
training partition. The held-out test partition is used only after model
selection.

Saved model artefacts contain the complete inference pipeline. Therefore,
evaluation, SHAP analysis and dashboard inference must pass raw predictor
data directly to the saved pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "ml_pipeline"))

import efa_tools  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"
META_JSON = BASE / "data" / "processed" / "dataset_metadata.json"
MODEL_DIR = BASE / "models" / "saved_models"
FACTOR_ARTIFACT = MODEL_DIR / "factor_scoring.json"
MODEL_METADATA = MODEL_DIR / "model_metadata.json"

TARGET = "operational_risk_category"
SCORE_COL = "operational_risk_score"

ID_COLS = ["serno", "year", "weight"]

DIRECT_TARGET_LEAKAGE = [
    "employee_turnover_rate",
    "employee_turnover_rate_5yr",
    SCORE_COL,
]

PRECAUTIONARY_EXCLUSIONS = [
    "absenteeism_rate",
    "grievance_rate",
    "grievance_procedure_formal",
    "training_problems",
    "skill_shortage_reported",
    "disputes_raised",
    "dispute_risk_level",
    "other_industrial_action",
    "dismissal_rate",
    "disciplinary_warnings",
    "suspensions",
]

ADMINISTRATIVE_EXCLUSIONS = ID_COLS

HIGH_CARDINALITY_EXCLUSIONS = [
    "industry_code",
    "industry_code_80",
]

# Leakage-family exclusions only (14 = 3 directly target-derived + 11
# precautionary risk indicators). The three administrative identifiers
# (serno / year / weight) are excluded separately and are deliberately NOT
# counted as leakage, matching the dataset contract in 01_build_dataset.py.
LEAKAGE = (
    DIRECT_TARGET_LEAKAGE
    + PRECAUTIONARY_EXCLUSIONS
)

TARGET_COLUMNS = {
    TARGET,
    "employee_turnover_rate",
    "employee_turnover_rate_5yr",
    SCORE_COL,
}

LABELS = ["Low", "Medium", "High"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

MISS_TOKENS = [
    "",
    " ",
    "NA",
    "N/A",
    "nan",
    "None",
    "-9",
    "-8",
    "MISSING",
]

RARE_THRESHOLD = 0.01
NUMERIC_RATIO_THRESHOLD = 0.85
CARDINALITY_THRESHOLD = 12
MISSING_LEVEL = "MISSING"
RARE_LEVEL = "OTHER_RARE"

TURNOVER_SENTINELS = [999, 9999.99, 99999, -9, -8]


# ---------------------------------------------------------------------------
# Dataset and predictor contract
# ---------------------------------------------------------------------------

def load_dataset():
    """Load the processed dataset and return leakage-safe predictors."""
    df = pd.read_csv(DATA)

    if META_JSON.exists():
        metadata = json.loads(META_JSON.read_text())

        predictor_columns = [
            column
            for column in metadata.get("predictor_columns", [])
            if column in df.columns
        ]
    else:
        excluded = (
            set(DIRECT_TARGET_LEAKAGE)
            | set(PRECAUTIONARY_EXCLUSIONS)
            | set(ADMINISTRATIVE_EXCLUSIONS)
            | set(HIGH_CARDINALITY_EXCLUSIONS)
            | set(TARGET_COLUMNS)
        )

        predictor_columns = [
            column
            for column in df.columns
            if column not in excluded
        ]

    for column in HIGH_CARDINALITY_EXCLUSIONS:
        if column in predictor_columns:
            predictor_columns.remove(column)

    return df, predictor_columns


def predictor_columns():
    """Return the leakage-safe predictor columns."""
    _, columns = load_dataset()

    prohibited = (
        set(TARGET_COLUMNS)
        | set(DIRECT_TARGET_LEAKAGE)
        | set(PRECAUTIONARY_EXCLUSIONS)
        | set(ADMINISTRATIVE_EXCLUSIONS)
        | set(HIGH_CARDINALITY_EXCLUSIONS)
    )

    leaked_columns = set(columns) & prohibited

    if leaked_columns:
        raise ValueError(
            f"Leakage-prone columns detected: {sorted(leaked_columns)}"
        )

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate predictor columns detected.")

    return columns


# ---------------------------------------------------------------------------
# Canonical outer train-test partition
# ---------------------------------------------------------------------------

def split_dataset(
    df,
    predictor_columns,
    target_column=TARGET,
):
    """
    Create the canonical outer train-test partition.

    The split is performed before rare-category grouping, EFA, imputation,
    scaling or encoding. The held-out test partition therefore remains
    untouched during cross-validation and model selection.

    Returns:
        Xtr_raw, Xte_raw, ytr, yte
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    missing_predictors = [
        column
        for column in predictor_columns
        if column not in df.columns
    ]

    if missing_predictors:
        raise ValueError(
            f"Missing predictor columns: {missing_predictors}"
        )

    X = df[predictor_columns].copy()
    y = df[target_column].astype(str).str.strip()

    valid_labels = set(LABELS)
    invalid_labels = set(y.dropna().unique()) - valid_labels

    if invalid_labels:
        raise ValueError(
            f"Unexpected target labels detected: {sorted(invalid_labels)}"
        )

    target_map = {"Low": 0, "Medium": 1, "High": 2}
    y = y.map(target_map)

    valid_mask = y.notna()

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True).astype(int)

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return (
        Xtr_raw.copy(),
        Xte_raw.copy(),
        ytr.copy(),
        yte.copy(),
    )


# ---------------------------------------------------------------------------
# Canonical cleaning helpers
# ---------------------------------------------------------------------------

def clean_categorical(series):
    """Canonicalise a categorical column for training and inference."""
    series = series.replace(MISS_TOKENS, np.nan)
    series = series.astype("string").str.strip()
    series = series.replace(
        {
            "": MISSING_LEVEL,
            " ": MISSING_LEVEL,
            "nan": MISSING_LEVEL,
            "None": MISSING_LEVEL,
        }
    )
    series = series.fillna(MISSING_LEVEL)

    return series.astype(str)


def clean_turnover(series):
    """Clean the turnover-rate measure used to construct the target."""
    value = pd.to_numeric(series, errors="coerce")
    value = value.mask(value.isin(TURNOVER_SENTINELS))
    value = value.mask((value < 0) | (value > 100))

    return value


def turnover_to_category(value):
    """Map a cleaned turnover rate to Low, Medium or High."""
    if pd.isna(value):
        return np.nan

    if value <= 10.0:
        return "Low"

    if value <= 20.0:
        return "Medium"

    return "High"


def classify_columns(frame):
    """Split a raw predictor frame into numeric and categorical columns.

    A column is numeric when almost all of its non-sentinel values parse as
    numbers (``NUMERIC_RATIO_THRESHOLD``) AND it carries more than
    ``CARDINALITY_THRESHOLD`` distinct values. Everything else is categorical.
    This is the single classification rule used by 04, 05 and the inference
    wrapper, so the persisted metadata always matches the live contract.
    """
    numeric_columns = []
    categorical_columns = []

    for column in frame.columns:
        series = frame[column].replace(MISS_TOKENS, np.nan)
        coerced = pd.to_numeric(series, errors="coerce")

        valid_count = series.notna().sum()
        numeric_ratio = (
            coerced.notna().sum() / valid_count
            if valid_count > 0
            else 0.0
        )

        if (
            numeric_ratio > NUMERIC_RATIO_THRESHOLD
            and coerced.nunique(dropna=True) > CARDINALITY_THRESHOLD
        ):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def clean_frames(train, test):
    """Canonically clean the train and test partitions in place.

    The categorical levels are canonicalised on BOTH partitions (so a missing
    value is ``MISSING`` everywhere) and the rare-category map is derived from
    the TRAIN partition only, then applied to both partitions. The canonical
    missing level is never folded into ``OTHER_RARE``.

    Returns ``(train, categorical_columns, rare_map)``; ``train`` and ``test``
    are mutated in place.
    """
    numeric_columns, categorical_columns = classify_columns(train)

    for column in categorical_columns:
        train[column] = clean_categorical(train[column])
        test[column] = clean_categorical(test[column])

    for column in numeric_columns:
        train[column] = pd.to_numeric(
            train[column].replace(MISS_TOKENS, np.nan), errors="coerce"
        )
        test[column] = pd.to_numeric(
            test[column].replace(MISS_TOKENS, np.nan), errors="coerce"
        )

    rare_map = {}

    for column in categorical_columns:
        frequencies = train[column].value_counts(normalize=True)

        rare_values = sorted(
            value
            for value in frequencies[frequencies < RARE_THRESHOLD].index
            if value != MISSING_LEVEL
        )

        if rare_values:
            rare_map[column] = rare_values
            train[column] = train[column].apply(
                lambda value: RARE_LEVEL if value in rare_values else value
            )
            test[column] = test[column].apply(
                lambda value: RARE_LEVEL if value in rare_values else value
            )

    return train, categorical_columns, rare_map


def apply_serve_transform(frame, numeric_columns, categorical_columns, rare_map):
    """Apply the canonical cleaning to a raw serving frame.

    This reproduces the train/test cleaning of :func:`clean_frames` for a
    single frame: categorical levels are canonicalised, persisted rare levels
    are folded to ``OTHER_RARE`` (never ``MISSING``) and numeric columns are
    coerced. The EFA factor scores are NOT added here; the caller derives them
    from the saved scoring artifact.
    """
    frame = frame.copy()
    numeric_columns = list(numeric_columns or [])
    categorical_columns = list(categorical_columns or [])
    rare_map = dict(rare_map or {})

    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = clean_categorical(frame[column])

    for column, rare_values in rare_map.items():
        if column in frame.columns:
            frame[column] = clean_categorical(frame[column])
            rare_values = [v for v in rare_values if v != MISSING_LEVEL]
            frame[column] = frame[column].apply(
                lambda value: RARE_LEVEL if value in rare_values else value
            )

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column].replace(MISS_TOKENS, np.nan), errors="coerce"
            )

    return frame


def load_factor_artifact(artifact_path=None):
    """Load the persisted EFA scoring artifact (fit + persist if absent)."""
    artifact_path = Path(artifact_path or FACTOR_ARTIFACT)

    if not artifact_path.exists():
        df, _ = load_dataset()
        artifact, _ = efa_tools.fit(df)
        efa_tools.save(artifact, artifact_path)

    return efa_tools.load(artifact_path)


def derive_factor_scores(frame, artifact_path=None):
    """Return the ``factor_1..factor_k`` scores for a raw predictor frame."""
    artifact = load_factor_artifact(artifact_path)
    return efa_tools.transform(frame, artifact)


def build_model_frames(artifact_path=None):
    """Rebuild the canonical training / held-out contract frames.

    Returns a dictionary with the cleaned + rare-grouped + factor-scored
    frames (``Xtr`` / ``Xte``), the untouched raw partitions
    (``Xtr_raw`` / ``Xte_raw``), the integer-encoded targets, the feature
    contract and the train-derived rare-category map. Every script and the
    inference wrapper build their frames through this function so the split,
    the cleaning and the rare map can never drift apart.
    """
    df, predictor_columns = load_dataset()

    Xtr_raw, Xte_raw, ytr, yte = split_dataset(
        df, predictor_columns, target_column=TARGET
    )

    Xtr_raw = Xtr_raw.copy()
    Xte_raw = Xte_raw.copy()

    Xtr = Xtr_raw.copy()
    Xte = Xte_raw.copy()

    _, cat_cols, rare_map = clean_frames(Xtr, Xte)

    num_cols = [column for column in Xtr.columns if column not in cat_cols]

    factor_names = list(load_factor_artifact(artifact_path)["factor_names"])

    if factor_names:
        train_factors = derive_factor_scores(Xtr, artifact_path)
        test_factors = derive_factor_scores(Xte, artifact_path)

        for name in factor_names:
            Xtr[name] = train_factors[name].to_numpy()
            Xte[name] = test_factors[name].to_numpy()

        num_cols = num_cols + factor_names

    return {
        "Xtr": Xtr,
        "Xte": Xte,
        "ytr": ytr,
        "yte": yte,
        "Xtr_raw": Xtr_raw,
        "Xte_raw": Xte_raw,
        "features": list(predictor_columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "rare_map": rare_map,
    }


# ---------------------------------------------------------------------------
# Fold-safe transformers
# ---------------------------------------------------------------------------

class CategoricalCleaningTransformer(BaseEstimator, TransformerMixin):
    """
    Apply canonical categorical cleaning without learning parameters.

    This transformer cleans object and string columns only. Numeric columns
    are left unchanged so that NumericCoercionTransformer can convert
    placeholder values such as 'MISSING' to NaN before imputation.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        for column in X.columns:
            if (
                pd.api.types.is_object_dtype(X[column])
                or pd.api.types.is_string_dtype(X[column])
            ):
                X[column] = clean_categorical(X[column])

        return X


class DynamicColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Dynamically detect numeric vs categorical columns and apply appropriate preprocessing.

    This transformer fits after EFA and rare-category grouping, when the final
    column schema is known. It separates numeric columns (imputation + scaling)
    from categorical columns (imputation + one-hot encoding).
    """

    def __init__(self):
        self.numeric_columns_ = None
        self.categorical_columns_ = None
        self.numeric_pipeline_ = None
        self.categorical_pipeline_ = None

    def fit(self, X, y=None):
        """Detect column types and fit preprocessing pipelines."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numeric_columns_ = []
        self.categorical_columns_ = []

        for column in X.columns:
            # Factor columns are always numeric
            if column.startswith("factor_"):
                self.numeric_columns_.append(column)
            # Check if column is already numeric dtype
            elif pd.api.types.is_numeric_dtype(X[column]):
                self.numeric_columns_.append(column)
            # Otherwise treat as categorical (after cleaning, these should be proper categories)
            else:
                self.categorical_columns_.append(column)

        # Build numeric pipeline
        if self.numeric_columns_:
            self.numeric_pipeline_ = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            self.numeric_pipeline_.fit(X[self.numeric_columns_])

        # Build categorical pipeline
        if self.categorical_columns_:
            self.categorical_pipeline_ = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            self.categorical_pipeline_.fit(X[self.categorical_columns_])

        return self

    def transform(self, X):
        """Apply preprocessing and concatenate results."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        transformed_parts = []

        if self.numeric_columns_ and self.numeric_pipeline_:
            numeric_transformed = self.numeric_pipeline_.transform(X[self.numeric_columns_])
            numeric_df = pd.DataFrame(
                numeric_transformed,
                columns=self.numeric_pipeline_.get_feature_names_out(),
                index=X.index
            )
            transformed_parts.append(numeric_df)

        if self.categorical_columns_ and self.categorical_pipeline_:
            categorical_transformed = self.categorical_pipeline_.transform(X[self.categorical_columns_])
            categorical_df = pd.DataFrame(
                categorical_transformed,
                columns=self.categorical_pipeline_.get_feature_names_out(),
                index=X.index
            )
            transformed_parts.append(categorical_df)

        if transformed_parts:
            return pd.concat(transformed_parts, axis=1)
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        feature_names = []
        if self.numeric_columns_ and self.numeric_pipeline_:
            feature_names.extend(self.numeric_pipeline_.get_feature_names_out())
        if self.categorical_columns_ and self.categorical_pipeline_:
            feature_names.extend(self.categorical_pipeline_.get_feature_names_out())
        return feature_names


class NumericCoercionTransformer(BaseEstimator, TransformerMixin):
    """
    Coerce numeric columns to numeric values dynamically.

    Detects numeric columns based on data type and content ratio.
    The transformation is deterministic and learns no statistics from the data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        missing_tokens = {
            token.strip().lower()
            for token in MISS_TOKENS
        }

        for column in X.columns:
            values = X[column].copy()

            # Only process object/string columns
            if not (
                pd.api.types.is_object_dtype(values)
                or pd.api.types.is_string_dtype(values)
            ):
                continue

            # Skip columns that are clearly categorical (low cardinality)
            unique_vals = values.dropna().unique()
            if len(unique_vals) <= 15:  # Likely categorical
                continue

            # Skip if contains categorical markers
            if any("MISSING" in str(v) or "OTHER_RARE" in str(v) for v in unique_vals):
                continue

            values = (
                values.astype(str)
                .str.strip()
                .str.lower()
            )

            values = values.replace(missing_tokens, np.nan)

            coerced = pd.to_numeric(values, errors="coerce")

            # If >85% of values can be coerced to numeric, use the coerced version
            valid_count = values.notna().sum()
            if valid_count > 0:
                numeric_ratio = coerced.notna().sum() / valid_count
                if numeric_ratio > NUMERIC_RATIO_THRESHOLD:
                    X[column] = coerced

        return X


class RareCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Fold-safe rare-category grouping transformer.

    The rare-category map is fitted using only the training observations
    supplied to fit(). During cross-validation, a separate map is fitted for
    each CV training fold.
    """

    def __init__(
        self,
        threshold=RARE_THRESHOLD,
        random_state=RANDOM_STATE,
    ):
        self.threshold = threshold
        self.random_state = random_state
        self.rare_category_map_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.rare_category_map_ = {}

        for column in X.columns:
            cleaned = clean_categorical(X[column])
            frequencies = cleaned.value_counts(normalize=True)

            rare_values = frequencies[
                frequencies < self.threshold
            ].index.tolist()

            rare_values = [
                value
                for value in rare_values
                if value != MISSING_LEVEL
            ]

            if rare_values:
                self.rare_category_map_[column] = sorted(rare_values)

        return self

    def transform(self, X):
        if self.rare_category_map_ is None:
            raise RuntimeError(
                "RareCategoryTransformer must be fitted before transform()."
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        for column, rare_values in self.rare_category_map_.items():
            if column in X.columns:
                X[column] = clean_categorical(X[column])
                X[column] = X[column].apply(
                    lambda value: (
                        RARE_LEVEL
                        if value in rare_values
                        else value
                    )
                )

        return X


class EFATransformer(BaseEstimator, TransformerMixin):
    """
    Fold-safe EFA transformer.

    EFA is fitted using only the training observations supplied to fit().
    During cross-validation, a separate EFA model is fitted for each CV
    training fold.
    """

    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.artifact_ = None
        self.factor_names_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.drop(columns=[TARGET], errors="ignore")

        self.artifact_, _ = efa_tools.fit(X)
        self.factor_names_ = list(self.artifact_["factor_names"])

        return self

    def transform(self, X):
        if self.artifact_ is None:
            raise RuntimeError(
                "EFATransformer must be fitted before transform()."
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()
        factors = efa_tools.transform(X, self.artifact_)

        for column in factors.columns:
            X[column] = factors[column].to_numpy()

        return X


# ---------------------------------------------------------------------------
# Fold-safe pipeline builder
# ---------------------------------------------------------------------------

def build_cv_pipeline(
    model,
):
    """
    Build the complete fold-safe preprocessing and classification pipeline.

    The pipeline applies the following sequence:

        1. Canonical categorical cleaning
        2. Numeric coercion (dynamic detection)
        3. Fold-specific rare-category grouping
        4. Fold-specific EFA
        5. Dynamic column detection and preprocessing
        6. Classifier

    The pipeline must receive raw predictor data. It must not receive data
    that has already been rare-grouped, factor-scored, imputed, scaled or
    encoded.
    """
    return Pipeline(
        steps=[
            (
                "categorical_cleaning",
                CategoricalCleaningTransformer(),
            ),
            (
                "numeric_coercion",
                NumericCoercionTransformer(),
            ),
            (
                "rare_category",
                RareCategoryTransformer(
                    threshold=RARE_THRESHOLD,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "efa",
                EFATransformer(
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "preprocessing",
                DynamicColumnTransformer(),
            ),
            (
                "classifier",
                model,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def payload_to_frame(
    payload,
    predictor_columns,
):
    """
    Convert an API payload to a one-row raw DataFrame.

    The returned frame must be passed directly to a saved full pipeline.
    The pipeline applies the same transformations used during training.
    """
    frame = pd.DataFrame([payload])

    missing_columns = [
        column
        for column in predictor_columns
        if column not in frame.columns
    ]

    for column in missing_columns:
        frame[column] = np.nan

    return frame[predictor_columns]
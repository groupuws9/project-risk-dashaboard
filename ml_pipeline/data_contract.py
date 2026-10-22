"""data_contract.py — shared data / preprocessing contract (single source of truth).

Every downstream script (04 training, 05 imbalance tuning, 06 evaluation,
07 SHAP) must build model frames through this module so that the held-out
test set, the EFA factor scores, the numeric/categorical typing and the
rare-category grouping are IDENTICAL across all of them.

Why this module exists
----------------------
``04_model_training.py`` performs, in this exact order:

  1. select the leakage-free predictor contract from
     ``data/processed/dataset_metadata.json``;
  2. encode the target (turnover band) and drop unlabelled rows;
  3. split 80/20, stratified, ``random_state=42``;
  4. fit EFA on the TRAIN frame only and score both frames with that
     artifact (``models/saved_models/factor_scoring.json``);
  5. clean the raw frames: numeric coercion, sentinel-token handling,
     categorical string coercion and *rare-category grouping* (categories
     below 1% of the TRAIN frame are folded into ``OTHER_RARE``);
  6. concatenate the cleaned raw predictors with the EFA factor scores.

Step 5 is the subtle one: the rare-category map is derived from the TRAIN
frame and then applied to BOTH train and test. A script that cleans the
whole frame and then splits -- or that skips rare-category grouping --
produces a test frame the saved pipelines were never fitted on, which
silently changes every reported metric.  This module reproduces that
contract exactly so evaluation numbers always match the training run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "ml_pipeline"))
import efa_tools  # noqa: E402

DATA = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"
META_JSON = BASE / "data" / "processed" / "dataset_metadata.json"
MODEL_DIR = BASE / "models" / "saved_models"

TARGET = "operational_risk_category"
SCORE_COL = "operational_risk_score"
ID_COLS = ["serno", "year", "weight"]
LEAKAGE = ["employee_turnover_rate", "employee_turnover_rate_5yr",
    "absenteeism_rate", "grievance_rate", "grievance_procedure_formal",
    "training_problems", "skill_shortage_reported", "disputes_raised",
    "dispute_risk_level", "other_industrial_action", "dismissal_rate",
    "disciplinary_warnings", "suspensions", SCORE_COL]
LABELS = ["Low", "Medium", "High"]
RANDOM_STATE = 42
MISS_TOKENS = ["", " ", "NA", "N/A", "nan", "None", "-9", "-8"]
RARE_THRESHOLD = 0.01          # categories below this share -> OTHER_RARE
MISSING_LEVEL = "MISSING"      # canonical level for a missing category
RARE_LEVEL = "OTHER_RARE"      # canonical level for a rare category
_CAT_MISSING_MAP = {"": MISSING_LEVEL, " ": MISSING_LEVEL,
                    "nan": MISSING_LEVEL, "None": MISSING_LEVEL}


def clean_categorical(series):
    """Canonicalise a categorical column to the train/serve contract.

    Every missing category becomes the literal ``"MISSING"`` level:

      * sentinel tokens (``MISS_TOKENS``) are treated as missing;
      * whitespace-only strings are treated as missing;
      * real nulls (``NaN`` / ``pd.NA`` / ``None``) are treated as missing;
      * missing is then mapped to ``"MISSING"``.

    The ORDER matters and is the whole point of this helper: the
    missing -> ``"MISSING"`` substitution happens BEFORE any string
    coercion, so a null can never survive as the literal string ``"nan"``
    and can never be left as a null either.

    Training, evaluation and serving all call this one function, which is
    what makes the train/serve parity exact.  The result never contains a
    null, so the downstream ``SimpleImputer``/``OneHotEncoder`` see the same
    levels regardless of where the frame came from.
    """
    s = series.replace(list(MISS_TOKENS), np.nan)
    s = s.astype("string").str.strip()
    s = s.replace(_CAT_MISSING_MAP)
    s = s.fillna(MISSING_LEVEL)
    return s.astype(str)



def load_dataset():
    """Return (df, predictor_columns) using the leakage-safe contract."""
    df = pd.read_csv(DATA)
    if META_JSON.exists():
        meta = json.loads(META_JSON.read_text())
        feats = [c for c in meta.get("predictor_columns", []) if c in df.columns]
    else:
        feats = [c for c in df.columns if c not in ID_COLS + LEAKAGE + [TARGET]]
    for _drop in ("industry_code", "industry_code_80"):
        if _drop in feats:
            feats.remove(_drop)
    return df, feats


def clean_frames(Xtr_raw, Xte_raw):
    """Reproduce 04's cleaning: typing, sentinels, rare-category grouping.

    Returns (num_cols, cat_cols, rare_map). Mutates the frames in place,
    exactly as the training script does.

    ``rare_map`` maps a categorical column to the list of TRAIN-derived
    categories that were folded into ``OTHER_RARE``. It is persisted in
    ``model_metadata.json`` so that inference can reproduce the identical
    transformation (train/serve parity).
    """
    num_cols, cat_cols = [], []
    rare_map = {}
    for c in Xtr_raw.columns:
        s = Xtr_raw[c].replace(list(MISS_TOKENS), np.nan)
        coerced = pd.to_numeric(s, errors="coerce")
        valid = s.notna().sum()
        numeric_ratio = (coerced.notna().sum() / valid) if valid else 0
        if numeric_ratio > 0.85 and coerced.nunique(dropna=True) > 12:
            num_cols.append(c)
            Xtr_raw[c] = coerced
            Xte_raw[c] = pd.to_numeric(
                Xte_raw[c].replace(list(MISS_TOKENS), np.nan), errors="coerce")
        else:
            cat_cols.append(c)
            # SAME canonical cleaning for train and test: every missing
            # category becomes the literal "MISSING" level on both sides.
            Xtr_raw[c] = clean_categorical(Xtr_raw[c])
            Xte_raw[c] = clean_categorical(Xte_raw[c])
            # rare-category grouping, derived on the TRAIN frame only.
            # "MISSING" is the canonical missing level and must never be
            # folded away, otherwise a missing value at serving time would
            # be encoded as "OTHER_RARE" instead of "MISSING".
            vc = Xtr_raw[c].value_counts(normalize=True)
            rare = set(vc[vc < RARE_THRESHOLD].index) - {MISSING_LEVEL}
            if rare:
                rare_map[c] = sorted(rare)
                Xtr_raw[c] = Xtr_raw[c].apply(
                    lambda v: RARE_LEVEL if v in rare else v)
                Xte_raw[c] = Xte_raw[c].apply(
                    lambda v: RARE_LEVEL if v in rare else v)
    return num_cols, cat_cols, rare_map


# Backwards-compatible alias used by earlier call sites.
_clean_raw_frames = clean_frames


def split_dataset(df=None, feats=None):
    """Return the canonical train/test split (before EFA / cleaning).

    Splitting happens BEFORE the EFA fit and BEFORE cleaning, so the EFA
    artifact and the rare-category map are derived from the training
    partition only.  Every pipeline stage must use this function so that the
    held-out rows are identical across training, evaluation and tuning.

    Returns (Xtr_raw, Xte_raw, ytr, yte, feats).
    """
    if df is None or feats is None:
        df, feats = load_dataset()
    X_raw = df[[c for c in feats if c in df.columns]]
    y = df[TARGET].astype(str).str.strip().map({"Low": 0, "Medium": 1, "High": 2})
    mask = y.notna()
    X_raw = X_raw.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return Xtr_raw, Xte_raw, ytr, yte, list(Xtr_raw.columns)


def build_full_frame():
    """Return the cleaned frame for the WHOLE dataset (train + test rows).

    The rows are cleaned with the *training-derived* rare-category map, so the
    categorical levels are identical to the ones the saved model was fitted
    on.  This is what explanation scripts (07 SHAP) and the inference wrapper
    must use: a frame that skipped the rare-category grouping would expose
    levels the one-hot encoder never saw, silently dropping their effect.

    Returns (X_full, y_full, rare_map) where X_full contains the cleaned raw
    predictors plus the EFA factor scores.
    """
    fr = build_model_frames()
    X_full = pd.concat([fr["Xtr"], fr["Xte"]], axis=0)
    y_full = pd.concat([fr["ytr"], fr["yte"]], axis=0)
    X_full = X_full.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)
    return X_full, y_full, fr["rare_map"]


def build_model_frames():
    """Build the authoritative train/test frames for every pipeline stage.

    Returns a dict with:
      df          : source dataframe
      feats       : raw predictor columns (leakage-free, no industry codes)
      factor_names: EFA factor score column names
      Xtr / Xte   : cleaned raw predictors (train / test)
      ytr / yte   : encoded target (train / test)
      Xtr_full / Xte_full : raw predictors + EFA factor scores
      num_cols / cat_cols : column typing used by the ColumnTransformer
      train_dist / test_dist : class counts for the report

    The EFA artifact is the one persisted by the training run
    (``models/saved_models/factor_scoring.json``); it was fitted on the
    training partition only, so scoring the test frame with it leaks
    nothing.
    """
    df, feats = load_dataset()
    X_raw = df[[c for c in feats if c in df.columns]]
    y = df[TARGET].astype(str).str.strip().map({"Low": 0, "Medium": 1, "High": 2})
    mask = y.notna()
    X_raw = X_raw.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)

    # split BEFORE EFA / cleaning (prevents leakage) -- identical to 04
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # EFA factor scores from the persisted training artifact
    art_path = MODEL_DIR / "factor_scoring.json"
    artifact = efa_tools.load(art_path) if art_path.exists() else None
    if artifact is None:
        artifact, _ = efa_tools.fit(Xtr_raw)
        efa_tools.save(artifact, art_path)
    factor_names = list(artifact["factor_names"])
    f_tr = efa_tools.transform(Xtr_raw, artifact)
    f_te = efa_tools.transform(Xte_raw, artifact)

    # cleaning (typing + rare-category grouping) -- identical to 04
    num_cols, cat_cols, rare_map = clean_frames(Xtr_raw, Xte_raw)
    num_cols = num_cols + [c for c in factor_names if c not in num_cols]

    full_feats = list(Xtr_raw.columns) + factor_names
    Xtr_full = pd.concat([Xtr_raw, f_tr], axis=1)[full_feats]
    Xte_full = pd.concat([Xte_raw, f_te], axis=1)[full_feats]

    return {
        "df": df,
        "feats": list(Xtr_raw.columns),
        "factor_names": factor_names,
        "features": full_feats,
        "Xtr_raw": Xtr_raw, "Xte_raw": Xte_raw,
        "Xtr": Xtr_full, "Xte": Xte_full,
        "ytr": ytr, "yte": yte,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "rare_map": rare_map,
        "train_dist": ytr.value_counts().to_dict(),
        "test_dist": yte.value_counts().to_dict(),
    }


def apply_serve_transform(df, numeric_features, categorical_features,
                          rare_map=None):
    """Reproduce the training-time cleaning for an arbitrary (serve) frame.

    This is the single implementation of the "raw frame -> model frame"
    transformation used at inference time, so a payload is always preprocessed
    exactly like the training rows were:

      * numeric columns -> ``pd.to_numeric`` with the sentinel tokens dropped;
      * categorical columns -> string coercion, strip, missing-token folding;
      * TRAIN-derived rare categories -> ``OTHER_RARE`` (from ``rare_map``).

    ``rare_map`` must come from the training run (``model_metadata.json``).
    When it is ``None`` no rare-category folding is applied, which reproduces
    the pre-contract behaviour of the inference wrapper.
    """
    out = df.copy()
    rare_map = rare_map or {}
    for c in numeric_features:
        if c in out.columns:
            out[c] = pd.to_numeric(
                out[c].replace(list(MISS_TOKENS), np.nan), errors="coerce")
    for c in categorical_features:
        if c in out.columns:
            # SAME canonical cleaning as training/evaluation, so a missing
            # category is encoded as "MISSING" at serving time too.
            out[c] = clean_categorical(out[c])
            # "MISSING" is the canonical missing level and is never folded
            # into OTHER_RARE, so a missing value keeps its own encoding.
            rare = set(rare_map.get(c, [])) - {MISSING_LEVEL}
            if rare:
                out[c] = out[c].apply(
                    lambda v: RARE_LEVEL if v in rare else v)
    return out



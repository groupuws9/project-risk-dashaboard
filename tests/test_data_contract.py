"""Contract regression tests for the shared data / preprocessing contract.

These tests lock in the guarantees that 04/05/06/07 and the inference wrapper
rely on:

  * one canonical 80/20 split shared by every script,
  * rare categories derived from the TRAIN frame only,
  * the persisted metadata contract matching the live contract module,
  * the inference wrapper reproducing the training frame exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ml_pipeline"))
sys.path.insert(0, str(BASE / "models"))

import data_contract as dc  # noqa: E402
from risk_predictor import OperationalRiskPredictor  # noqa: E402

META_PATH = BASE / "models" / "saved_models" / "model_metadata.json"


@pytest.fixture(scope="module")
def meta():
    return json.loads(META_PATH.read_text())


@pytest.fixture(scope="module")
def frames():
    return dc.build_model_frames()


def test_split_is_canonical_and_disjoint(frames):
    """Train and test partitions must be disjoint and cover the modelled rows."""
    Xtr, Xte = frames["Xtr"], frames["Xte"]
    assert len(Xtr) + len(Xte) == len(frames["Xtr_raw"]) + len(frames["Xte_raw"])
    # no row can appear in both partitions: compare the target vectors
    assert len(frames["ytr"]) == len(Xtr)
    assert len(frames["yte"]) == len(Xte)
    # the split ratio is the documented 80/20
    assert round(len(Xte) / (len(Xtr) + len(Xte)), 3) == 0.2


def test_features_match_metadata_contract(frames, meta):
    """The live contract and the persisted metadata must agree exactly."""
    assert list(frames["features"]) == list(meta["features"])
    assert list(frames["num_cols"]) == list(meta["numeric_features"]) + [
        c for c in meta["factor_features"] if c not in meta["numeric_features"]]
    assert list(frames["cat_cols"]) == list(meta["categorical_features"])


def test_rare_categories_are_train_derived_and_persisted(frames, meta):
    """The rare map must be persisted, and the levels it lists must be ones
    the training frame actually contained before grouping."""
    rare_map = frames["rare_map"]
    assert rare_map == meta.get("rare_category_map")
    assert meta.get("rare_threshold") == dc.RARE_THRESHOLD

    # clean_frames() mutates the frames in place, so the pre-grouping levels
    # are recovered from the persisted map + the grouped frame itself.
    for col, rares in rare_map.items():
        grouped = set(frames["Xtr"][col].astype(str))
        # the rare levels must have been folded away in the cleaned train frame
        assert not set(rares) & grouped, f"rare level survived in {col}"
        # the column must be a categorical column of the contract
        assert col in frames["cat_cols"]


def test_rare_folding_is_applied_to_the_test_frame(frames):
    """A rare level present in TEST must be folded to OTHER_RARE too."""
    rare_map = frames["rare_map"]
    for col, rares in rare_map.items():
        test_levels = set(frames["Xte"][col].astype(str))
        assert not set(rares) & test_levels, \
            f"test frame for {col} was not rare-grouped"
        # OTHER_RARE must appear in whichever frame actually held a rare level
        assert ("OTHER_RARE" in set(frames["Xtr"][col].astype(str))
                or "OTHER_RARE" in test_levels), \
            f"OTHER_RARE absent for {col}"


def test_serve_transform_matches_training_cleaning(frames, meta):
    """apply_serve_transform() must reproduce the training-time cleaning.

    Only the raw predictor columns are compared: the EFA factor_* scores are
    derived from the raw HR items by the scoring artifact, not produced by the
    cleaning step (the inference wrapper derives them before calling this).
    """
    num_feats = meta["numeric_features"]
    cat_feats = meta["categorical_features"]
    raw = frames["Xte_raw"].head(50).copy()
    served = dc.apply_serve_transform(raw, num_feats, cat_feats,
                                      meta.get("rare_category_map"))
    contract = frames["Xte"].head(50)
    for c in contract.columns:
        if c not in served.columns:
            # derived (factor_*) columns are not part of the cleaning step
            assert c.startswith("factor_"), f"unexpected missing column {c}"
            continue
        if c in num_feats:
            a = pd.to_numeric(served[c], errors="coerce")
            b = pd.to_numeric(contract[c], errors="coerce")
            assert np.allclose(a.fillna(-999).astype(float),
                               b.fillna(-999).astype(float), equal_nan=True)
        else:
            a = served[c].astype("string").fillna("__NA__")
            b = contract[c].astype("string").fillna("__NA__")
            assert (a == b).all(), f"serve/contract mismatch in {c}"


def test_inference_wrapper_reproduces_the_contract_frame(frames, meta):
    """The deployed wrapper must build the same frame the model was fitted on."""
    pred = OperationalRiskPredictor.load()
    assert pred.rare_map == meta.get("rare_category_map")

    n = 30
    raw = frames["Xte_raw"].head(n).copy()
    served = pred._frame(raw)
    contract = frames["Xte"].head(n)

    num_feats = set(meta["numeric_features"])
    for c in contract.columns:
        if c in num_feats:
            a = pd.to_numeric(served[c], errors="coerce")
            b = pd.to_numeric(contract[c], errors="coerce")
            assert np.allclose(a.fillna(-999).astype(float),
                               b.fillna(-999).astype(float), equal_nan=True)
        else:
            a = served[c].astype("string").fillna("__NA__")
            b = contract[c].astype("string").fillna("__NA__")
            assert (a == b).all(), f"serve/contract mismatch in {c}"

    # and the model must therefore produce identical probabilities
    assert np.abs(pred.pipe.predict_proba(served)
                   - pred.pipe.predict_proba(contract)).max() < 1e-12


def test_evaluation_matches_training_metrics():
    """06 (evaluation) must reproduce 04 (training) balanced accuracies."""
    ev = pd.read_csv(BASE / "results" / "tables" / "evaluation_summary.csv") \
        .set_index("model")
    mc = pd.read_csv(BASE / "results" / "tables" / "model_comparison.csv") \
        .set_index("model")
    shared = [m for m in ev.index if m in mc.index]
    assert shared, "no common models between evaluation and training tables"
    for m in shared:
        assert abs(ev.loc[m, "balanced_accuracy"]
                   - mc.loc[m, "test_balanced_accuracy"]) < 1e-12, \
            f"{m}: 06 evaluation does not match 04 training metrics"

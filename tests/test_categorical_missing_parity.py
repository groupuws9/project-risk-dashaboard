"""Regression tests for categorical missing-value parity.

The contract is that EVERY path -- training, evaluation and serving --
encodes a missing categorical value as the literal ``"MISSING"`` level.

Historically the paths disagreed:

  * the TRAIN branch of ``clean_frames`` stringified the frame, so a real
    null could survive as ``NaN`` / ``<NA>``;
  * the TEST branch folded the missing tokens into ``"MISSING"``;
  * ``apply_serve_transform`` (serving) followed the TRAIN branch.

So the same logical row was encoded differently at training and serving
time.  These tests lock in the canonical behaviour.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ml_pipeline"))
sys.path.insert(0, str(BASE / "models"))

import data_contract as dc  # noqa: E402

MISSING = "MISSING"

# every spelling of "this value is absent" the contract must recognise
MISSING_INPUTS = [
    np.nan,
    pd.NA,
    None,
    "",
    "   ",
    "NA",
    "N/A",
    "nan",
    "None",
    "-9",
    "-8",
]


@pytest.fixture
def raw_series():
    """A categorical column mixing real values with every missing spelling."""
    return pd.Series(["A", "B"] + MISSING_INPUTS, dtype="object")


def test_clean_categorical_maps_every_missing_spelling(raw_series):
    """Every missing spelling collapses to the single "MISSING" level."""
    out = dc.clean_categorical(raw_series)
    assert out.iloc[0] == "A"
    assert out.iloc[1] == "B"
    for i in range(2, len(raw_series)):
        assert out.iloc[i] == MISSING, (
            f"value {raw_series.iloc[i]!r} (index {i}) was not mapped to MISSING")


def test_clean_categorical_never_leaves_a_null(raw_series):
    """The cleaned column must contain no nulls at all."""
    out = dc.clean_categorical(raw_series)
    assert out.isna().sum() == 0
    assert not out.isnull().any()


def test_clean_categorical_strips_whitespace(raw_series):
    """Surrounding whitespace is stripped from real values."""
    out = dc.clean_categorical(pd.Series(["  Large  ", "\tSmall\t"]))
    assert out.tolist() == ["Large", "Small"]


def test_clean_categorical_preserves_the_real_value_case():
    """Cleaning must not fold distinct real categories together."""
    out = dc.clean_categorical(pd.Series(["Yes", "No", "yes"]))
    assert sorted(out.tolist()) == ["No", "Yes", "yes"]


def test_clean_categorical_handles_categorical_dtype():
    """A pandas Categorical column is cleaned the same way."""
    raw = pd.Series(pd.Categorical(["A", None, "B"]))
    out = dc.clean_categorical(raw)
    assert out.tolist() == ["A", MISSING, "B"]
    assert out.isna().sum() == 0


def test_clean_categorical_handles_all_null_column():
    """A column that is entirely missing is all "MISSING"."""
    out = dc.clean_categorical(pd.Series([None, np.nan, pd.NA]))
    assert out.tolist() == [MISSING] * 3


# ---------------------------------------------------------------------------
# train / test / serve parity
# ---------------------------------------------------------------------------

def test_clean_frames_maps_missing_identically_on_train_and_test():
    """The TRAIN and TEST branches of clean_frames must agree on missing."""
    tr = pd.DataFrame({"cat": ["A"] * 95 + ["B"] * 4 + [np.nan]})
    te = pd.DataFrame({"cat": ["A", np.nan, None, "", "  "]})
    _, cat_cols, _ = dc.clean_frames(tr, te)
    assert cat_cols == ["cat"]
    assert tr["cat"].isna().sum() == 0
    assert te["cat"].isna().sum() == 0
    # the missing row on the train side is "MISSING" ...
    assert tr["cat"].iloc[-1] == MISSING
    # ... and every missing spelling on the test side is "MISSING" too
    assert te["cat"].tolist() == ["A", MISSING, MISSING, MISSING, MISSING]


def test_clean_frames_train_and_test_use_the_same_level_set():
    """Both partitions must expose the same levels for shared categories.

    A category that only occurs in the test partition is legitimately kept
    (the one-hot encoder simply ignores unseen levels), but a level that
    occurs in BOTH partitions must be spelled identically on both sides --
    that is the parity the old train/test asymmetry broke.
    """
    tr = pd.DataFrame({"cat": ["A"] * 90 + ["B"] * 10})
    te = pd.DataFrame({"cat": ["A", "B", "C"]})
    dc.clean_frames(tr, te)
    train_levels = set(tr["cat"])
    test_levels = set(te["cat"])
    # every level known to the training frame is known to the test frame
    assert train_levels <= test_levels
    # ... and they are spelled the same way on both sides
    for level in train_levels:
        assert level in test_levels


def test_serve_transform_matches_the_training_frame():
    """apply_serve_transform must reproduce the training-time cleaning.

    The serve path is fed the SAME raw rows the contract cleaned for
    training; the resulting categorical column must be identical.
    """
    tr = pd.DataFrame({"cat": ["A"] * 95 + ["B"] * 4 + [np.nan]})
    te = pd.DataFrame({"cat": ["A", np.nan, None, "", "  ", "B"]})
    dc.clean_frames(tr, te)

    served = dc.apply_serve_transform(te, [], ["cat"], {})
    assert served["cat"].tolist() == te["cat"].tolist()
    assert served["cat"].isna().sum() == 0


def test_serve_transform_maps_missing_to_missing_level():
    """A missing categorical at serving time becomes "MISSING"."""
    frame = pd.DataFrame({"cat": [np.nan, pd.NA, None, "", "  ", "NA"]})
    served = dc.apply_serve_transform(frame, [], ["cat"], {})
    assert served["cat"].tolist() == [MISSING] * 6


def test_missing_level_is_never_folded_into_other_rare():
    """The canonical missing level keeps its own encoding.

    "MISSING" may be a rare level (< 1% of the train frame).  It must not
    be folded into "OTHER_RARE", otherwise a missing value would be
    encoded as a rare category instead of as missing.
    """
    tr = pd.DataFrame({"cat": ["A"] * 199 + [np.nan]})      # 0.5% missing
    te = pd.DataFrame({"cat": [np.nan, "A"]})
    _, _, rare_map = dc.clean_frames(tr, te)
    assert MISSING not in rare_map.get("cat", [])
    assert tr["cat"].iloc[-1] == MISSING
    assert te["cat"].iloc[0] == MISSING


def test_serve_transform_keeps_missing_even_with_a_stale_rare_map():
    """A persisted rare map that lists "MISSING" must not swallow it.

    Older training runs persisted the missing level inside the rare map.
    Serving must still encode a missing value as "MISSING".
    """
    frame = pd.DataFrame({"cat": [np.nan, "A"]})
    served = dc.apply_serve_transform(frame, [], ["cat"], {"cat": [MISSING]})
    assert served["cat"].iloc[0] == MISSING


def test_rare_categories_are_still_folded():
    """The rare-category grouping itself must keep working."""
    tr = pd.DataFrame({"cat": ["A"] * 199 + ["B"]})          # "B" is 0.5%
    te = pd.DataFrame({"cat": ["A", "B"]})
    _, _, rare_map = dc.clean_frames(tr, te)
    assert rare_map.get("cat") == ["B"]
    assert tr["cat"].iloc[-1] == "OTHER_RARE"
    assert te["cat"].iloc[1] == "OTHER_RARE"

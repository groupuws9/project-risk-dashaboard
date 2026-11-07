"""Unit tests for the Pydantic input contract (spec 2 §3.7.1)."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dashboard.validation import (MODEL_FEATURES, REQUIRED_FIELDS,
                                  OperationalRiskInput, model_to_payload,
                                  validate_payload)


def _model_raw_features():
    """Raw (non-factor) features the trained model actually consumes."""
    meta = json.loads(
        (Path(__file__).resolve().parent.parent / "models" / "saved_models"
         / "model_metadata.json").read_text())
    factors = set(meta.get("factor_features", []))
    return [f for f in meta["features"] if f not in factors]


def test_schema_matches_trained_model_features():
    # The API contract must expose exactly the model's raw inputs (the 4 EFA
    # factor scores are derived server-side, never supplied by the caller).
    raw = _model_raw_features()
    assert set(MODEL_FEATURES) == set(raw)
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))  # no duplicates
    assert len(MODEL_FEATURES) == len(raw)
    for f in REQUIRED_FIELDS:
        assert f in MODEL_FEATURES



def test_extra_fields_ignored():
    m = OperationalRiskInput(sector=1, org_size_category_3=2, multinational=2,
                             union_coverage_pct=30, collective_bargaining=2,
                             performance_appraisal_pct=3, jcc_exists=2,
                             information_sharing_score=3, pct_part_time=20,
                             pct_temporary=5, unknown_field="x")
    assert m.sector == 1


def test_string_numbers_coerce(valid_payload):
    m = OperationalRiskInput(**valid_payload)
    assert m.union_coverage_pct == 30.0
    assert m.pct_part_time == 20.0


def test_range_bounds_enforced(valid_payload):
    with pytest.raises(ValidationError):
        OperationalRiskInput(**dict(valid_payload, pct_female=101))
    with pytest.raises(ValidationError):
        OperationalRiskInput(**dict(valid_payload, union_coverage_pct=-1))


def test_validate_payload_returns_extras(valid_payload):
    model, err, status, extras = validate_payload(
        dict(valid_payload, surprise=1))
    assert err is None and status is None
    assert extras == ["surprise"]
    payload = model_to_payload(model)
    assert all(v is not None for v in payload.values())


def test_validate_payload_non_dict():
    _, body, status, _ = validate_payload(["nope"])
    assert status == 400 and body["code"] == "MALFORMED_JSON"


def test_invalid_category_is_rejected():
    """Invalid categorical values should be rejected with OUT_OF_RANGE (reviewer Fix 12)."""
    with pytest.raises(ValidationError):
        OperationalRiskInput(
            sector=1, org_size_category_3=2, multinational=2,
            union_coverage_pct=30, collective_bargaining=999,  # Invalid: out of range
            performance_appraisal_pct=3, jcc_exists=2,
            information_sharing_score=3, pct_part_time=20,
            pct_temporary=5)


def test_validation_error_returns_400():
    """Validation errors should return 400 status (reviewer Fix 13)."""
    from dashboard.validation import error_body
    body, status = error_body("MISSING_FIELD", "Missing required field", field="sector")
    assert status == 400
    assert body["code"] == "MISSING_FIELD"


def test_feature_names_match_training():
    """Feature names in validation must match training features (reviewer Fix 14)."""
    raw = _model_raw_features()
    # Ensure MODEL_FEATURES matches the training contract
    assert set(MODEL_FEATURES) == set(raw)
    # Ensure no features are silently dropped
    for f in raw:
        assert f in MODEL_FEATURES, f"Training feature {f} missing from validation schema"

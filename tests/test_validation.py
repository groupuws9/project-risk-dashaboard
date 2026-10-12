"""Unit tests for the Pydantic input contract (spec 2 §3.7.1)."""
import pytest
from pydantic import ValidationError

from dashboard.validation import (MODEL_FEATURES, REQUIRED_FIELDS,
                                  OperationalRiskInput, model_to_payload,
                                  validate_payload)


def test_all_36_model_features_covered():
    assert len(MODEL_FEATURES) == 36
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

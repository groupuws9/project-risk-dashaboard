"""Runtime inference validation for the model artefact (spec 2 §3.7.4)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "models"))

from risk_predictor import OperationalRiskPredictor  # noqa: E402

_ORG = {"sector": "1", "org_size_category_3": "2", "multinational": "2",
        "union_coverage_pct": 30.0, "collective_bargaining": 2,
        "performance_appraisal_pct": 3, "jcc_exists": 2,
        "information_sharing_score": 3, "pct_part_time": 20.0,
        "pct_temporary": 5}


def test_load_best_model():
    p = OperationalRiskPredictor.load()
    assert p.labels == ["Low", "Medium", "High"]
    # Feature count is defined by the trained model contract
    # (models/saved_models/model_metadata.json): 33 leakage-free raw HR/org/ER
    # predictors + 4 EFA factor scores derived at inference. Assert the contract
    # structurally rather than a hard-coded number so the test tracks the model.
    factors = [f for f in p.features if f.startswith("factor_")]
    raw = [f for f in p.features if not f.startswith("factor_")]
    assert factors == ["factor_1", "factor_2", "factor_3", "factor_4"]
    assert len(raw) == 33
    assert len(p.features) == len(raw) + len(factors)
    # The target basis (turnover) must never be an input feature.
    assert "employee_turnover_rate" not in p.features


def test_factor_scores_derived_at_inference():
    """factor_* must be computed from raw HR items, never left NaN."""
    p = OperationalRiskPredictor.load()
    frame = p._frame(_ORG)
    assert not frame[[c for c in frame.columns
                      if c.startswith("factor_")]].isna().any().any()


def test_predict_returns_valid_label_and_probability():
    p = OperationalRiskPredictor.load()
    label, prob = p.predict(_ORG)
    assert label in p.labels
    assert 0.0 <= prob <= 1.0


def test_predict_proba_sums_to_one():
    p = OperationalRiskPredictor.load()
    d = p.predict_proba_dict(_ORG)
    assert abs(sum(d.values()) - 1.0) < 1e-6


def test_partial_input_defaults():
    p = OperationalRiskPredictor.load()
    label, _ = p.predict({"sector": "2"})
    assert label in p.labels


def test_batch_predict():
    import pandas as pd
    p = OperationalRiskPredictor.load()
    df = pd.DataFrame([_ORG, dict(_ORG, sector="3")])
    out = p.predict_batch(df)
    assert len(out) == 2 and all(l in p.labels for l in out)

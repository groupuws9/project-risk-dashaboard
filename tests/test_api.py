"""Edge-case API tests — spec 1 §3.10.5, spec 2 §3.7.5, spec 3 §3.6.6."""
import json


def test_valid_input_returns_prediction(client, valid_payload):
    r = client.post("/api/predict", json=valid_payload)
    assert r.status_code == 200
    j = r.get_json()
    assert j["risk_level"] in ("Low", "Medium", "High")
    assert 0.0 <= j["probability"] <= 1.0
    assert abs(sum(j["probabilities"].values()) - 1.0) < 1e-6
    assert "X-Request-ID" in r.headers


def test_missing_required_field(client):
    r = client.post("/api/predict", json={"sector": "1"})
    assert r.status_code == 400
    j = r.get_json()
    assert j["code"] == "MISSING_FIELD"
    assert "union_coverage_pct" in j["error"]


def test_empty_payload_lists_all_missing_fields(client):
    r = client.post("/api/predict", json={})
    assert r.status_code == 400
    j = r.get_json()
    assert j["code"] == "MISSING_FIELD"
    assert "sector" in j["error"] and "pct_temporary" in j["error"]


def test_wrong_type_rejected(client, valid_payload):
    payload = dict(valid_payload, sector="abc")
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 400
    j = r.get_json()
    assert j["code"] == "INVALID_TYPE"
    assert j["field"] == "sector"


def test_out_of_range_rejected(client, valid_payload):
    payload = dict(valid_payload, union_coverage_pct=150)
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 400
    j = r.get_json()
    assert j["code"] == "OUT_OF_RANGE"
    assert j["field"] == "union_coverage_pct"


def test_extra_field_ignored_but_prediction_returned(client, valid_payload):
    payload = dict(valid_payload, foo="bar")
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    assert r.get_json()["risk_level"] in ("Low", "Medium", "High")


def test_null_optional_field_is_imputed(client, valid_payload):
    payload = dict(valid_payload, pct_female=None)
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    assert r.get_json()["risk_level"] in ("Low", "Medium", "High")


def test_null_required_field_is_missing(client, valid_payload):
    payload = dict(valid_payload, union_coverage_pct=None)
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 400
    assert r.get_json()["code"] == "MISSING_FIELD"


def test_malformed_json(client, valid_payload):
    r = client.post("/api/predict", data="{not json",
                    content_type="application/json")
    assert r.status_code == 400
    assert r.get_json()["code"] == "MALFORMED_JSON"


def test_non_object_body_is_malformed(client):
    r = client.post("/api/predict", json=[1, 2, 3])
    assert r.status_code == 400
    assert r.get_json()["code"] == "MALFORMED_JSON"


def test_model_error_returns_422(client, valid_payload, monkeypatch):
    from dashboard import app as app_module

    def boom(_):
        raise ValueError("incomplete feature vector")

    monkeypatch.setattr(app_module.predictor, "predict", boom)
    r = client.post("/api/predict", json=valid_payload)
    assert r.status_code == 422
    assert r.get_json()["code"] == "MODEL_ERROR"


def test_internal_error_returns_500(client, valid_payload, monkeypatch):
    from dashboard import app as app_module

    def boom(_):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(app_module, "model_to_payload", boom)
    r = client.post("/api/predict", json=valid_payload)
    assert r.status_code == 500
    assert r.get_json()["code"] == "INTERNAL_ERROR"


def test_rate_limit_returns_429(client, valid_payload, monkeypatch):
    from dashboard import app as app_module
    monkeypatch.setattr(app_module, "RATE_LIMIT", 3)
    app_module._hits.clear()
    codes = [client.post("/api/predict", json=valid_payload).status_code
             for _ in range(6)]
    assert codes[-1] == 429
    r = client.post("/api/predict", json=valid_payload)
    assert r.status_code == 429
    assert r.get_json()["code"] == "RATE_LIMITED"
    assert "Retry-After" in r.headers
    app_module._hits.clear()


def test_spec_endpoint_aliases(client, valid_payload):
    assert client.post("/predict", json=valid_payload).status_code == 200
    assert client.get("/data").status_code == 200
    r = client.post("/simulate", json={"baseline": valid_payload,
                                       "scenario": valid_payload})
    assert r.status_code == 200
    assert "baseline" in r.get_json() and "scenario" in r.get_json()


def test_whatif_missing_scenario_block(client, valid_payload):
    r = client.post("/api/whatif", json={"baseline": valid_payload})
    assert r.status_code == 400
    assert r.get_json()["code"] == "MALFORMED_JSON"


def test_predict_returns_live_local_shap(client, valid_payload):
    """Live local SHAP for the EXACT input (not the static example file)."""
    r = client.post("/api/predict", json=valid_payload)
    assert r.status_code == 200
    j = r.get_json()
    loc = j.get("local_explanation")
    assert loc and loc.get("contributions"), "local_explanation missing"
    assert loc["class_explained"] == j["risk_level"]
    for item in loc["contributions"]:
        assert "feature" in item and isinstance(item["shap"], (int, float))
    # a real (non-missing) input must produce non-trivial drivers
    assert max(abs(i["shap"]) for i in loc["contributions"]) > 1e-6
    # what-if stays lean: no per-block explanations
    w = client.post("/api/whatif", json={"baseline": valid_payload,
                                         "scenario": valid_payload}).get_json()
    assert "local_explanation" not in w["baseline"]


def test_local_shap_failure_does_not_break_prediction(client, valid_payload,
                                                      monkeypatch):
    from dashboard.app import predictor

    def boom(*a, **k):
        raise RuntimeError("shap unavailable")

    monkeypatch.setattr(predictor, "explain", boom)
    r = client.post("/api/predict", json=valid_payload)
    assert r.status_code == 200
    j = r.get_json()
    assert "probabilities" in j
    assert "local_explanation" not in j


def test_features_endpoint(client):
    r = client.get("/api/features")
    assert r.status_code == 200
    j = r.get_json()
    assert "best_model" in j and "feature_importance" in j


def test_data_endpoints(client):
    assert client.get("/api/data/summary").status_code == 200
    r = client.get("/api/data/export?sector=1")
    assert r.status_code == 200
    assert "text/csv" in r.headers["Content-Type"]
    assert "attachment" in r.headers["Content-Disposition"]


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    # The dashboard is titled "Business Risk Dashboard"; assert on the content
    # the template actually renders rather than a stale product name.
    assert b"Business Risk Dashboard" in r.data
    assert b"Predict Low / Medium / High risk" in r.data

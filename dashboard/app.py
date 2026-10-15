"""Flask dashboard — risk prediction, data explorer, model insights, what-if lab.

Runtime guarantees (spec 1 §3.10, spec 2 §3.7, spec 3 §3.6):
  * Pydantic input contract before any model inference
  * Structured error responses (MISSING_FIELD / INVALID_TYPE / OUT_OF_RANGE /
    MALFORMED_JSON / MODEL_ERROR / INTERNAL_ERROR / RATE_LIMITED)
  * request-ID logging of every failed request (no payload contents)
  * per-IP rate limiting, CORS, spec-named endpoint aliases

Run:  python dashboard/app.py   ->  http://localhost:5000
"""
from pathlib import Path
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict, deque

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "models"))
sys.path.insert(0, str(BASE))

from flask import Flask, jsonify, render_template, request, Response
from flask_cors import CORS

from risk_predictor import OperationalRiskPredictor
from dashboard.validation import error_body, model_to_payload, validate_payload

app = Flask(__name__)
CORS(app)  # spec 3.9: Flask-CORS

MODEL_DIR = BASE / "models" / "saved_models"
TAB_DIR = BASE / "results" / "tables"
DATA_PATH = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dashboard")

# ---- rate limiting (spec 3.6.3: HTTP 429 RATE_LIMITED) --------------------
RATE_LIMIT = int(os.environ.get("DASH_RATE_LIMIT", "120"))
RATE_WINDOW = 60
_hits = defaultdict(deque)


def _rate_limited():
    now = time.time()
    q = _hits[request.remote_addr or "?"]
    while q and q[0] < now - RATE_WINDOW:
        q.popleft()
    q.append(now)
    return len(q) > RATE_LIMIT


@app.before_request
def _guard_rate_limit():
    request.rid = uuid.uuid4().hex[:8]
    if _rate_limited():
        body, status = error_body(
            "RATE_LIMITED",
            f"Too many requests. Please retry in {RATE_WINDOW} seconds.",
            request_id=request.rid)
        resp = jsonify(body)
        resp.status_code = status
        resp.headers["Retry-After"] = str(RATE_WINDOW)
        return resp
    return None


@app.after_request
def _log_request(resp):
    rid = getattr(request, "rid", "-")
    resp.headers["X-Request-ID"] = rid
    if resp.status_code >= 400:
        try:
            code = resp.get_json().get("code", "?")
        except Exception:
            code = "?"
        LOG.warning("rid=%s %s %s -> %s (%s)", rid, request.method,
                    request.path, resp.status_code, code)
    return resp


@app.errorhandler(400)
def _bad_request(_e):
    body, status = error_body("MALFORMED_JSON", "Invalid JSON format",
                              request_id=getattr(request, "rid", None))
    return jsonify(body), status


@app.errorhandler(500)
def _server_error(_e):
    body, status = error_body("INTERNAL_ERROR", "An unexpected error occurred",
                              request_id=getattr(request, "rid", None))
    return jsonify(body), status


predictor, META, DF = None, {}, None
try:
    predictor = OperationalRiskPredictor.load()
    META = json.loads((MODEL_DIR / "model_metadata.json").read_text())
except Exception as e:
    print(f"WARNING: model not loaded yet ({e}). Train first: 04_model_training.py")
try:
    DF = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"WARNING: dataset not loaded ({e})")


@app.route("/")
def index():
    feats = META.get("features", [])
    return render_template("index.html", features=feats,
                           labels=META.get("labels", ["Low", "Medium", "High"]),
                           best_model=META.get("best_model", "—"))


@app.route("/api/metadata")
def api_metadata():
    out = dict(META)
    for k in ("features", "numeric_features", "categorical_features"):
        out[k] = out.get(k, [])
    return jsonify(out)


def _read_json():
    """Decode JSON body or return the MALFORMED_JSON error response."""
    payload = request.get_json(silent=True)
    if payload is None and request.data:
        body, status = error_body("MALFORMED_JSON", "Invalid JSON format",
                                  request_id=request.rid)
        return None, (jsonify(body), status)
    if payload is None:
        payload = {}
    return payload, None


def _run_prediction(payload, request_id, explain=True):
    """Validate one org dict then predict. Returns (dict, None) or (None, resp).

    With explain=True the response carries a live local SHAP explanation for
    the predicted class (computed fail-safe; never breaks the prediction).
    What-if calls pass explain=False — it compares outcomes, not reasons.
    """
    model, err, status, extras = validate_payload(payload, request_id)
    if err is not None:
        return None, (jsonify(err), status)
    if extras:
        LOG.warning("rid=%s ignored unexpected field(s): %s",
                    request_id, ", ".join(extras))  # spec 3.7.2 step 5
    data = model_to_payload(model)
    try:
        label, prob = predictor.predict(data)
        probs = predictor.predict_proba_dict(data)
    except Exception as e:
        LOG.warning("rid=%s MODEL_ERROR: %s", request_id, e)
        body, st = error_body(
            "MODEL_ERROR",
            "Model inference failed due to incomplete feature vector",
            request_id=request_id)
        return None, (jsonify(body), st)
    if any(v != v for v in probs.values()):  # NaN check
        body, st = error_body(
            "MODEL_ERROR",
            "Model inference failed due to incomplete feature vector",
            request_id=request_id)
        return None, (jsonify(body), st)
    out = {"risk_level": label, "probability": prob, "probabilities": probs}
    if explain:
        try:
            loc = predictor.explain(data)
        except Exception as e:            # explainability is optional decoration
            LOG.warning("rid=%s local SHAP skipped: %s", request_id, e)
            loc = None
        if loc is not None:
            out["local_explanation"] = loc
    return out, None


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])  # spec 3.2 alias
def api_predict():
    if predictor is None:
        body, st = error_body("SERVICE_UNAVAILABLE",
                              "Model not trained yet.", request_id=request.rid)
        return jsonify(body), st
    payload, resp = _read_json()
    if resp:
        return resp
    out, err = _run_prediction(payload, request.rid)
    if err:
        return err
    return jsonify(out)


def _tag_block(err, block):
    """Tag a structured error with which what-if block failed (baseline/scenario).

    Keeps the error-code protocol intact while making per-block validation
    failures unambiguous for both the UI (inline highlighting) and API users.
    """
    resp, status = err
    body = resp.get_json() or {}
    body["block"] = block
    body["error"] = f"[{block}] " + str(body.get("error", ""))
    return jsonify(body), status


@app.route("/api/whatif", methods=["POST"])
@app.route("/simulate", methods=["POST"])   # spec 3.2 alias
def api_whatif():
    """Compare baseline vs modified scenario side-by-side."""
    if predictor is None:
        body, st = error_body("SERVICE_UNAVAILABLE",
                              "Model not trained yet.", request_id=request.rid)
        return jsonify(body), st
    payload, resp = _read_json()
    if resp:
        return resp
    base = payload.get("baseline") if isinstance(payload, dict) else None
    alt = payload.get("scenario") if isinstance(payload, dict) else None
    if not isinstance(base, dict) or not isinstance(alt, dict):
        body, st = error_body(
            "MALFORMED_JSON",
            "Request body must contain 'baseline' and 'scenario' objects",
            request_id=request.rid)
        return jsonify(body), st
    base_out, err = _run_prediction(base, request.rid, explain=False)
    if err:
        return _tag_block(err, "baseline")
    alt_out, err = _run_prediction(alt, request.rid, explain=False)
    if err:
        return _tag_block(err, "scenario")
    return jsonify({"baseline": base_out, "scenario": alt_out})



@app.route("/api/data/summary")
@app.route("/data")  # spec 3.2 alias (data explorer)
def api_data_summary():
    if DF is None:
        return jsonify({"error": "Dataset unavailable"}), 503
    sector = request.args.get("sector", "")
    size = request.args.get("size", "")
    d = DF
    if sector and "sector" in d.columns:
        d = d[d["sector"].astype(str) == str(sector)]
    if size and "org_size_category_3" in d.columns:
        d = d[d["org_size_category_3"].astype(str) == str(size)]
    dist = d["operational_risk_category"].value_counts().to_dict() if \
        "operational_risk_category" in d.columns else {}
    out = {"n": int(len(d)), "risk_distribution": dist,
           "mean_score": float(d["operational_risk_score"].mean())
           if "operational_risk_score" in d.columns else None}
    if "operational_risk_score" in d.columns:
        out["score_histogram"] = d["operational_risk_score"].value_counts(
            bins=10, sort=False).tolist()
    try:
        out["by_sector"] = {
            (str(k).strip() or "Unknown"): v for k, v in d.groupby("sector")[
                "operational_risk_score"].mean().round(2).to_dict().items()
        }
    except Exception:
        out["by_sector"] = {}
    return jsonify(out)


@app.route("/api/data/export")
def api_data_export():
    """CSV export of the currently filtered data (spec 3.4 Data Explorer)."""
    if DF is None:
        body, st = error_body("SERVICE_UNAVAILABLE", "Dataset unavailable",
                              request_id=request.rid)
        return jsonify(body), st
    d = DF
    sector = request.args.get("sector", "")
    size = request.args.get("size", "")
    if sector and "sector" in d.columns:
        d = d[d["sector"].astype(str) == str(sector)]
    if size and "org_size_category_3" in d.columns:
        d = d[d["org_size_category_3"].astype(str) == str(size)]
    csv = d.to_csv(index=False)
    return Response(csv, mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=wers_panel_filtered.csv"})


@app.route("/api/data/upload", methods=["POST"])
def api_data_upload():
    """Upload a cleaned CSV file to replace the current dataset."""
    global DF
    if "file" not in request.files:
        body, st = error_body("MISSING_FIELD", "No file provided",
                              request_id=request.rid)
        return jsonify(body), st

    file = request.files["file"]
    if file.filename == "":
        body, st = error_body("MISSING_FIELD", "No file selected",
                              request_id=request.rid)
        return jsonify(body), st

    if not file.filename.endswith(".csv"):
        body, st = error_body("INVALID_TYPE", "Only CSV files are supported",
                              request_id=request.rid)
        return jsonify(body), st

    try:
        df = pd.read_csv(file)
        DF = df
        LOG.info("Data uploaded successfully: %d rows, %d columns", len(df), len(df.columns))
        return jsonify({"n": len(df), "columns": list(df.columns)})
    except Exception as e:
        LOG.error("Data upload error: %s", e)
        body, st = error_body("INTERNAL_ERROR", f"Failed to parse CSV: {str(e)}",
                              request_id=request.rid)
        return jsonify(body), st


@app.route("/api/data/load-sample", methods=["POST"])
def api_load_sample():
    """Load sample data from the processed directory."""
    global DF
    try:
        df = pd.read_csv(DATA_PATH)
        DF = df
        LOG.info("Sample data loaded: %d rows", len(df))
        return jsonify({"n": len(df), "columns": list(df.columns)})
    except Exception as e:
        LOG.error("Failed to load sample data: %s", e)
        body, st = error_body("SERVICE_UNAVAILABLE",
                              f"Could not load sample data: {str(e)}",
                              request_id=request.rid)
        return jsonify(body), st


@app.route("/api/features")
def api_features():
    """Feature importance + SHAP explanation data (spec 3.2 /features)."""
    out = {"best_model": META.get("best_model"), "shap": None,
           "local_explanation": None}
    fi = TAB_DIR / "feature_importance.csv"
    if fi.exists():
        try:
            out["feature_importance"] = json.loads(
                pd.read_csv(fi).head(30).to_json(orient="records"))
        except Exception:
            pass
    shap_p = TAB_DIR / "shap_importance.csv"
    if shap_p.exists():
        try:
            out["shap"] = json.loads(
                pd.read_csv(shap_p).head(30).to_json(orient="records"))
        except Exception:
            pass
    loc_p = TAB_DIR / "shap_local.json"
    if loc_p.exists():
        try:
            out["local_explanation"] = json.loads(loc_p.read_text())
        except Exception:
            pass
    return jsonify(out)


@app.route("/api/insights")
def api_insights():
    out = {}
    for name in ("model_comparison.csv", "feature_importance.csv",
                 "evaluation_summary.csv", "shap_importance.csv"):
        p = TAB_DIR / name
        if p.exists():
            try:
                out[name] = json.loads(
                    pd.read_csv(p).head(50).to_json(orient="records"))
            except Exception:
                pass
    # Always return valid JSON with defaults for missing model data
    out["best_model"] = META.get("best_model") if META else None
    out["metrics"] = META.get("all_test_balanced_accuracy", {}) if META else {}
    return jsonify(out)


if __name__ == "__main__":
    # Run without debug mode to avoid file watcher reload instability
    app.run(host="0.0.0.0", port=5000, debug=False)


"""
Flask dashboard — risk prediction, data explorer, model insights and what-if lab.

Runtime guarantees:

* Pydantic input validation occurs before model inference.
* Structured error responses are returned for validation, inference and
  server failures.
* Request identifiers are logged for failed requests without logging payload
  contents.
* Per-IP rate limiting and CORS are enabled.
* The prediction endpoint passes raw input data to the saved full pipeline.

Run:

    python dashboard/app.py

Then open:

    http://localhost:5000
"""

from pathlib import Path
import json
import logging
import math
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
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from pydantic import ValidationError

from risk_predictor import OperationalRiskPredictor
from dashboard.validation import (
    error_body,
    model_to_payload,
    validate_payload,
    pydantic_error_to_response,
)


def json_safe(value):
    """Recursively map non-finite floats to ``None``.

    Python's ``json`` emits bare ``NaN`` / ``Infinity`` tokens, which are not
    valid strict JSON and are rejected by browsers — this previously broke
    ``/api/metadata`` and surfaced as "Unable to load insights".  Replacing
    such values with ``null`` keeps every response valid (system test ST19).
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class StrictJSONProvider(DefaultJSONProvider):
    """JSON provider that never writes invalid (NaN/Infinity) tokens."""

    def dumps(self, obj, **kwargs):
        return super().dumps(json_safe(obj), **kwargs)


app = Flask(__name__)
app.json = StrictJSONProvider(app)
CORS(app)

MODEL_DIR = BASE / "models" / "saved_models"
TAB_DIR = BASE / "results" / "tables"
DATA_PATH = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LOG = logging.getLogger("dashboard")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT = int(os.environ.get("DASH_RATE_LIMIT", "120"))
RATE_WINDOW = 60

# Per-IP rolling window of request timestamps. Exposed as ``_hits`` so the
# rate-limit behaviour can be exercised by the test-suite.
_hits = defaultdict(deque)
request_timestamps = _hits


def is_rate_limited():
    """Return True if the requesting IP exceeds the configured rate limit."""
    now = time.time()
    client_key = request.remote_addr or "unknown"

    timestamps = _hits[client_key]

    while timestamps and timestamps[0] < now - RATE_WINDOW:
        timestamps.popleft()

    timestamps.append(now)

    return len(timestamps) > RATE_LIMIT


@app.before_request
def apply_rate_limit():
    """Assign a request identifier and apply per-IP rate limiting."""
    request.rid = uuid.uuid4().hex[:8]

    if is_rate_limited():
        body, status = error_body(
            "RATE_LIMITED",
            f"Too many requests. Please retry in {RATE_WINDOW} seconds.",
            request_id=request.rid,
        )

        response = jsonify(body)
        response.status_code = status
        response.headers["Retry-After"] = str(RATE_WINDOW)

        return response

    return None


@app.after_request
def log_request(response):
    """Add a request identifier and log failed requests without payloads."""
    request_id = getattr(request, "rid", "-")

    response.headers["X-Request-ID"] = request_id

    if response.status_code >= 400:
        try:
            error_code = response.get_json().get("code", "UNKNOWN")
        except Exception:
            error_code = "UNKNOWN"

        LOG.warning(
            "rid=%s %s %s -> %s (%s)",
            request_id,
            request.method,
            request.path,
            response.status_code,
            error_code,
        )

    return response


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def handle_bad_request(_error):
    """Handle malformed JSON and other bad requests."""
    body, status = error_body(
        "MALFORMED_JSON",
        "Invalid JSON format",
        request_id=getattr(request, "rid", None),
    )

    return jsonify(body), status


@app.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle Pydantic validation errors with structured responses."""
    body, status = pydantic_error_to_response(
        error,
        request_id=getattr(request, "rid", None),
    )

    return jsonify(body), status


@app.errorhandler(500)
def handle_server_error(_error):
    """Handle unexpected server errors."""
    body, status = error_body(
        "INTERNAL_ERROR",
        "An unexpected error occurred",
        request_id=getattr(request, "rid", None),
    )

    return jsonify(body), status


# ---------------------------------------------------------------------------
# Model and dataset loading
# ---------------------------------------------------------------------------

predictor = None
metadata = {}
dataset = None

# Metadata (feature contract, labels and best-model name) is loaded first and
# independently of the saved pipeline. The index page, /api/metadata and
# /api/insights depend only on metadata, so the dashboard still renders model
# insights even when no .pkl artefact has been generated yet.
METADATA_PATH = MODEL_DIR / "model_metadata.json"

if METADATA_PATH.exists():
    try:
        metadata = json.loads(METADATA_PATH.read_text())
    except Exception as error:
        print(f"WARNING: model metadata could not be parsed ({error}).")
else:
    print(
        f"WARNING: model metadata not found ({METADATA_PATH}). "
        "Run 04_model_training.py first."
    )

# The predictor needs both the metadata contract and the saved pipeline, so a
# missing artefact disables live prediction only — never the insights views.
if metadata:
    try:
        predictor = OperationalRiskPredictor.load()
        LOG.info("Loaded best model: %s", metadata.get("best_model"))
    except Exception as error:
        print(
            f"WARNING: model not loaded yet ({error}). "
            "Run 04_model_training.py first."
        )

try:
    dataset = pd.read_csv(DATA_PATH)
except Exception as error:
    print(f"WARNING: dataset not loaded ({error}).")


# ---------------------------------------------------------------------------
# Dashboard pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the dashboard home page."""
    return render_template(
        "index.html",
        features=metadata.get("features", []),
        labels=metadata.get(
            "labels",
            ["Low", "Medium", "High"],
        ),
        best_model=metadata.get("best_model", "Not available"),
    )


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

@app.route("/api/metadata")
def api_metadata():
    """Return model metadata for the dashboard."""
    response = dict(metadata)

    for key in (
        "features",
        "numeric_features",
        "categorical_features",
    ):
        response[key] = response.get(key, [])

    return jsonify(response)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def read_json_payload():
    """Decode the JSON request body or return a structured error."""
    payload = request.get_json(silent=True)

    if payload is None and request.data:
        body, status = error_body(
            "MALFORMED_JSON",
            "Invalid JSON format",
            request_id=request.rid,
        )

        return None, (jsonify(body), status)

    if payload is None:
        payload = {}

    return payload, None


def run_prediction(
    payload,
    request_id,
    explain=True,
):
    """
    Validate one organisation record and generate a prediction.

    The raw payload is passed to the predictor. The saved full pipeline is
    responsible for categorical cleaning, numeric coercion, rare-category
    grouping, EFA, imputation, scaling, encoding and classification.
    """
    validated_model, error, status, extra_fields = validate_payload(
        payload,
        request_id,
    )

    if error is not None:
        return None, (jsonify(error), status)

    if extra_fields:
        LOG.warning(
            "rid=%s ignored unexpected field(s): %s",
            request_id,
            ", ".join(extra_fields),
        )

    raw_record = model_to_payload(validated_model)

    try:
        predicted_label, probability = predictor.predict(raw_record)
        class_probabilities = predictor.predict_proba_dict(raw_record)
    except Exception as error:
        LOG.warning(
            "rid=%s MODEL_ERROR: %s",
            request_id,
            error,
        )

        body, status = error_body(
            "MODEL_ERROR",
            "Model inference failed",
            request_id=request_id,
        )

        return None, (jsonify(body), status)

    if any(
        value != value
        for value in class_probabilities.values()
    ):
        body, status = error_body(
            "MODEL_ERROR",
            "Model inference failed",
            request_id=request_id,
        )

        return None, (jsonify(body), status)

    response = {
        "risk_level": predicted_label,
        "probability": probability,
        "probabilities": class_probabilities,
    }

    if explain:
        try:
            local_explanation = predictor.explain(raw_record)
        except Exception as error:
            LOG.warning(
                "rid=%s local SHAP explanation skipped: %s",
                request_id,
                error,
            )
            local_explanation = None

        if local_explanation is not None:
            response["local_explanation"] = local_explanation

    return response, None


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def api_predict():
    """Return a turnover-band prediction for one organisation record."""
    if predictor is None:
        body, status = error_body(
            "SERVICE_UNAVAILABLE",
            "Model not trained yet.",
            request_id=request.rid,
        )

        return jsonify(body), status

    payload, error_response = read_json_payload()

    if error_response:
        return error_response

    response, error_response = run_prediction(
        payload,
        request.rid,
    )

    if error_response:
        return error_response

    return jsonify(response)


def tag_error_block(error_response, block_name):
    """Tag a structured validation error with the failing what-if block."""
    response, status = error_response
    body = response.get_json() or {}

    body["block"] = block_name
    body["error"] = f"[{block_name}] {body.get('error', '')}"

    return jsonify(body), status


@app.route("/api/whatif", methods=["POST"])
@app.route("/simulate", methods=["POST"])
def api_whatif():
    """Compare a baseline organisation record with a modified scenario."""
    if predictor is None:
        body, status = error_body(
            "SERVICE_UNAVAILABLE",
            "Model not trained yet.",
            request_id=request.rid,
        )

        return jsonify(body), status

    payload, error_response = read_json_payload()

    if error_response:
        return error_response

    baseline = (
        payload.get("baseline")
        if isinstance(payload, dict)
        else None
    )

    scenario = (
        payload.get("scenario")
        if isinstance(payload, dict)
        else None
    )

    if not isinstance(baseline, dict) or not isinstance(scenario, dict):
        body, status = error_body(
            "MALFORMED_JSON",
            "Request body must contain 'baseline' and 'scenario' objects",
            request_id=request.rid,
        )

        return jsonify(body), status

    baseline_response, baseline_error = run_prediction(
        baseline,
        request.rid,
        explain=False,
    )

    if baseline_error:
        return tag_error_block(baseline_error, "baseline")

    scenario_response, scenario_error = run_prediction(
        scenario,
        request.rid,
        explain=False,
    )

    if scenario_error:
        return tag_error_block(scenario_error, "scenario")

    return jsonify(
        {
            "baseline": baseline_response,
            "scenario": scenario_response,
        }
    )


# ---------------------------------------------------------------------------
# Data explorer
# ---------------------------------------------------------------------------

@app.route("/api/data/summary")
@app.route("/data")
def api_data_summary():
    """Return descriptive summaries from the uploaded or sample dataset."""
    if dataset is None:
        body, status = error_body(
            "SERVICE_UNAVAILABLE",
            "Dataset unavailable",
            request_id=request.rid,
        )

        return jsonify(body), status

    filtered_data = dataset.copy()

    sector = request.args.get("sector", "")
    organisation_size = request.args.get("size", "")

    if sector and "sector" in filtered_data.columns:
        filtered_data = filtered_data[
            filtered_data["sector"].astype(str) == str(sector)
        ]

    if (
        organisation_size
        and "org_size_category_3" in filtered_data.columns
    ):
        filtered_data = filtered_data[
            filtered_data["org_size_category_3"].astype(str)
            == str(organisation_size)
        ]

    risk_distribution = {}

    if "operational_risk_category" in filtered_data.columns:
        risk_distribution = (
            filtered_data["operational_risk_category"]
            .value_counts()
            .to_dict()
        )

    response = {
        "n": int(len(filtered_data)),
        "risk_distribution": risk_distribution,
        "descriptive_note": (
            "The turnover-rate distribution is descriptive and is not used "
            "as a model input."
        ),
    }

    if "employee_turnover_rate" in filtered_data.columns:
        turnover_rate = pd.to_numeric(
            filtered_data["employee_turnover_rate"],
            errors="coerce",
        )

        response["turnover_rate_mean"] = (
            float(turnover_rate.mean())
            if turnover_rate.notna().any()
            else None
        )

        response["turnover_rate_histogram"] = (
            turnover_rate.dropna()
            .value_counts(bins=10, sort=False)
            .tolist()
        )

    try:
        response["turnover_rate_by_sector"] = {
            str(sector_value).strip() or "Unknown": float(mean_value)
            for sector_value, mean_value in filtered_data.groupby("sector")[
                "employee_turnover_rate"
            ].mean().items()
        }
    except Exception:
        response["turnover_rate_by_sector"] = {}

    return jsonify(response)


@app.route("/api/data/export")
def api_data_export():
    """Export the currently filtered descriptive dataset as CSV."""
    if dataset is None:
        body, status = error_body(
            "SERVICE_UNAVAILABLE",
            "Dataset unavailable",
            request_id=request.rid,
        )

        return jsonify(body), status

    filtered_data = dataset.copy()

    sector = request.args.get("sector", "")
    organisation_size = request.args.get("size", "")

    if sector and "sector" in filtered_data.columns:
        filtered_data = filtered_data[
            filtered_data["sector"].astype(str) == str(sector)
        ]

    if (
        organisation_size
        and "org_size_category_3" in filtered_data.columns
    ):
        filtered_data = filtered_data[
            filtered_data["org_size_category_3"].astype(str)
            == str(organisation_size)
        ]

    csv_content = filtered_data.to_csv(index=False)

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": (
                "attachment; filename=wers_panel_filtered.csv"
            )
        },
    )


@app.route("/api/data/upload", methods=["POST"])
def api_data_upload():
    """Upload a cleaned CSV file for descriptive exploration."""
    global dataset

    if "file" not in request.files:
        body, status = error_body(
            "MISSING_FIELD",
            "No file provided",
            request_id=request.rid,
        )

        return jsonify(body), status

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        body, status = error_body(
            "MISSING_FIELD",
            "No file selected",
            request_id=request.rid,
        )

        return jsonify(body), status

    if not uploaded_file.filename.endswith(".csv"):
        body, status = error_body(
            "INVALID_TYPE",
            "Only CSV files are supported",
            request_id=request.rid,
        )

        return jsonify(body), status

    try:
        uploaded_data = pd.read_csv(uploaded_file)
        dataset = uploaded_data

        LOG.info(
            "Data uploaded successfully: %d rows, %d columns",
            len(uploaded_data),
            len(uploaded_data.columns),
        )

        return jsonify(
            {
                "n": len(uploaded_data),
                "columns": list(uploaded_data.columns),
            }
        )

    except Exception as error:
        LOG.error("Data upload error: %s", error)

        body, status = error_body(
            "INTERNAL_ERROR",
            f"Failed to parse CSV: {error}",
            request_id=request.rid,
        )

        return jsonify(body), status


@app.route("/api/data/load-sample", methods=["POST"])
def api_load_sample():
    """Load the processed sample dataset for descriptive exploration."""
    global dataset

    try:
        sample_data = pd.read_csv(DATA_PATH)
        dataset = sample_data

        LOG.info(
            "Sample data loaded: %d rows",
            len(sample_data),
        )

        return jsonify(
            {
                "n": len(sample_data),
                "columns": list(sample_data.columns),
            }
        )

    except Exception as error:
        LOG.error(
            "Failed to load sample data: %s",
            error,
        )

        body, status = error_body(
            "SERVICE_UNAVAILABLE",
            f"Could not load sample data: {error}",
            request_id=request.rid,
        )

        return jsonify(body), status


# ---------------------------------------------------------------------------
# Model insights
# ---------------------------------------------------------------------------

@app.route("/api/features")
def api_features():
    """Return feature importance and SHAP explanation data."""
    response = {
        "best_model": metadata.get("best_model"),
        "shap": None,
        "local_explanation": None,
        "feature_importance": [],
    }

    feature_importance_path = TAB_DIR / "feature_importance.csv"

    if feature_importance_path.exists():
        try:
            response["feature_importance"] = json.loads(
                pd.read_csv(feature_importance_path)
                .head(30)
                .to_json(orient="records")
            )
        except Exception:
            pass

    shap_importance_path = TAB_DIR / "shap_importance.csv"

    if shap_importance_path.exists():
        try:
            response["shap"] = json.loads(
                pd.read_csv(shap_importance_path)
                .head(30)
                .to_json(orient="records")
            )
        except Exception:
            pass

    local_explanation_path = TAB_DIR / "shap_local.json"

    if local_explanation_path.exists():
        try:
            response["local_explanation"] = json.loads(
                local_explanation_path.read_text()
            )
        except Exception:
            pass

    return jsonify(response)


@app.route("/api/insights")
def api_insights():
    """Return model-comparison, evaluation and explanation outputs."""
    response = {}

    for file_name in (
        "model_comparison.csv",
        "feature_importance.csv",
        "evaluation_summary.csv",
        "shap_importance.csv",
    ):
        file_path = TAB_DIR / file_name

        if file_path.exists():
            try:
                response[file_name] = json.loads(
                    pd.read_csv(file_path)
                    .head(50)
                    .to_json(orient="records")
                )
            except Exception:
                pass

    response["best_model"] = (
        metadata.get("best_model")
        if metadata
        else None
    )

    response["metrics"] = {
        # Held-out balanced accuracy, or null where a model has no saved
        # artefact. NaN is not valid strict JSON (system test ST19).
        model: (
            float(score)
            if isinstance(score, (int, float)) and math.isfinite(score)
            else None
        )
        for model, score in (
            metadata.get("all_test_balanced_accuracy", {})
            if metadata
            else {}
        ).items()
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
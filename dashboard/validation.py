"""Runtime input validation (Pydantic v2) — the API's strict input contract.

Implements the structured error-code protocol shared by all three
specifications:

    MISSING_FIELD   400   required field absent (or null)
    INVALID_TYPE    400   value cannot be coerced to the declared type
    OUT_OF_RANGE    400   value outside the allowed domain (ge/le)
    MALFORMED_JSON  400   body is not valid JSON / not a JSON object
    MODEL_ERROR     422   inference failed (NaN / incomplete vector)
    INTERNAL_ERROR  500   unexpected server failure
    RATE_LIMITED    429   too many requests

Constraint ranges follow the real WERS 2011 feature semantics (coded survey
answers -> loose integer bands; percentages -> 0-100).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ERROR_HTTP_STATUS = {
    "MISSING_FIELD": 400,
    "INVALID_TYPE": 400,
    "OUT_OF_RANGE": 400,
    "MALFORMED_JSON": 400,
    "MODEL_ERROR": 422,
    "INTERNAL_ERROR": 500,
    "RATE_LIMITED": 429,
    "SERVICE_UNAVAILABLE": 503,
}

# pydantic error-type (v2) -> our machine-readable code
_TYPE_MAP = {
    "missing": "MISSING_FIELD",
    "none_not_allowed": "MISSING_FIELD",
    "int_parsing": "INVALID_TYPE",
    "int_from_float": "INVALID_TYPE",
    "float_parsing": "INVALID_TYPE",
    "string_type": "INVALID_TYPE",
    "bool_parsing": "INVALID_TYPE",
    "model_attributes_type": "INVALID_TYPE",
    "greater_than": "OUT_OF_RANGE",
    "greater_than_equal": "OUT_OF_RANGE",
    "less_than": "OUT_OF_RANGE",
    "less_than_equal": "OUT_OF_RANGE",
    "too_long": "OUT_OF_RANGE",
}


def error_body(code: str, message: str, field: Optional[str] = None,
               request_id: Optional[str] = None):
    """Build (payload, http_status) for the structured error protocol."""
    body: Dict[str, Any] = {"error": message, "code": code}
    if field:
        body["field"] = field
    if request_id:
        body["request_id"] = request_id
    return body, ERROR_HTTP_STATUS.get(code, 400)


def pydantic_error_to_response(exc: ValidationError,
                               request_id: Optional[str] = None):
    """Translate the first Pydantic failure into the structured protocol."""
    errs = exc.errors()
    if not errs:
        return error_body("INVALID_TYPE", "Invalid input", request_id=request_id)
    first = errs[0]
    loc = [str(p) for p in first.get("loc", []) if p != "body"]
    field = ".".join(loc) if loc else None
    etype = str(first.get("type", ""))
    # explicit null on a required (non-Optional) field counts as missing
    if first.get("input", ...) is None:
        code = "MISSING_FIELD"
    else:
        code = _TYPE_MAP.get(etype, "INVALID_TYPE")
    msg = str(first.get("msg", "Invalid input"))
    if code == "MISSING_FIELD" and field:
        message = f"Missing required field: {field}"
    elif field:
        message = f"Field '{field}': {msg}"
    else:
        message = msg
    # list every missing field at once (empty-payload test case)
    if code == "MISSING_FIELD":
        missing = [str(e.get("loc", ())[-1]) for e in errs
                   if str(e.get("type", "")) in ("missing", "none_not_allowed")
                   and e.get("loc")]
        if missing:
            message = ("Missing required field(s): "
                       + ", ".join(dict.fromkeys(missing)))
    return error_body(code, message, field=field, request_id=request_id)


# --------------------------------------------------------------------------
# Input contract — 36 user-supplied model features (5 numeric + 31
# categorical). The pipeline additionally derives 4 EFA factor scores
# (factor_1..factor_4) server-side from these HR items, so the trained model
# consumes 40 features while the API schema stays at the 36 raw inputs.
# --------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "sector",
    "org_size_category_3",
    "multinational",
    "union_coverage_pct",
    "collective_bargaining",
    "performance_appraisal_pct",
    "jcc_exists",
    "information_sharing_score",
    "pct_part_time",
    "pct_temporary",
]

# Coded survey answers: loose 0-99 integer domain (9/99 are survey
# missing-codes; anything above 99 is certainly a data-entry error).
_CODE = dict(ge=0, le=99)


class OperationalRiskInput(BaseModel):
    """Strict request schema for /predict, /simulate, /whatif."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    # ---- required (spec 3.6.1) ----
    sector: int = Field(..., ge=1, le=10,
                        description="Sector classification (1-10)")
    org_size_category_3: int = Field(..., ge=1, le=3,
                                     description="Org size (3-cat: 1 small-3 large)")
    multinational: int = Field(..., ge=0, le=2,
                               description="Multinational status (coded)")
    union_coverage_pct: float = Field(..., ge=0, le=100,
                                      description="Union coverage percentage (0-100)")
    collective_bargaining: int = Field(..., **_CODE,
                                       description="Collective bargaining code")
    performance_appraisal_pct: int = Field(..., **_CODE,
                                           description="Appraisal coverage code")
    jcc_exists: int = Field(..., **_CODE,
                            description="Joint consultative committee (0/1/2)")
    information_sharing_score: int = Field(..., **_CODE,
                                           description="Information sharing score (0-10)")
    pct_part_time: float = Field(..., ge=0, le=100,
                                 description="Percentage part-time workers (0-100)")
    pct_temporary: int = Field(..., **_CODE,
                               description="Temporary-worker code / percentage")

    # ---- optional numeric ----
    sector_detailed: Optional[float] = Field(None, ge=0, le=99)
    pct_female: Optional[float] = Field(None, ge=0, le=100)
    pct_part_time_12: Optional[float] = Field(None, ge=0, le=100)

    # ---- optional coded categoricals ----
    org_size_category: Optional[int] = Field(None, **_CODE)
    org_size_category_4: Optional[int] = Field(None, **_CODE)
    org_age_years: Optional[int] = Field(None, **_CODE)
    workplace_age: Optional[int] = Field(None, **_CODE)
    trading_status: Optional[int] = Field(None, **_CODE)
    union_coverage_pct_6: Optional[int] = Field(None, **_CODE)
    union_coverage_pct_7: Optional[int] = Field(None, **_CODE)
    bargaining_level_org: Optional[int] = Field(None, **_CODE)
    bargaining_level_est: Optional[int] = Field(None, **_CODE)
    formal_hr_strategy: Optional[int] = Field(None, **_CODE)
    training_budget_exists: Optional[int] = Field(None, **_CODE)
    training_problems: Optional[int] = Field(None, **_CODE)
    job_security_policy: Optional[int] = Field(None, **_CODE)
    jcc_frequency: Optional[int] = Field(None, **_CODE)
    jcc_function: Optional[int] = Field(None, **_CODE)
    jcc_union_chair: Optional[int] = Field(None, **_CODE)
    info_chain_exist: Optional[int] = Field(None, **_CODE)
    newsletter_exists: Optional[int] = Field(None, **_CODE)
    investment_plans_shared: Optional[int] = Field(None, **_CODE)
    consultation_depth: Optional[int] = Field(None, **_CODE)
    employee_voice_channels: Optional[int] = Field(None, **_CODE)
    grievance_procedure_formal: Optional[int] = Field(None, **_CODE)
    skill_shortage_reported: Optional[int] = Field(None, **_CODE)


MODEL_FEATURES = list(OperationalRiskInput.model_fields)


def validate_payload(payload: Any, request_id: Optional[str] = None):
    """Validate a decoded JSON body.

    Returns (model, None, None, extras) on success or
    (None, error_dict, http_status, extras) on failure.  Unknown fields are
    ignored (spec 3.7.2 step 5) but returned so the caller can log them.
    """
    if not isinstance(payload, dict):
        body, status = error_body(
            "MALFORMED_JSON",
            "Invalid JSON format: request body must be a JSON object",
            request_id=request_id)
        return None, body, status, []
    try:
        model = OperationalRiskInput(**payload)
    except ValidationError as exc:
        body, status = pydantic_error_to_response(exc, request_id)
        return None, body, status, []
    extras = sorted(set(payload) - set(MODEL_FEATURES))
    return model, None, None, extras


def model_to_payload(model: OperationalRiskInput) -> Dict[str, Any]:
    """Drop Nones so absent optionals stay absent for the pipeline."""
    return {k: v for k, v in model.model_dump().items() if v is not None}

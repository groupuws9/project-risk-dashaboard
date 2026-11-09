# Important Code Blocks

## Code Block 3.1: Target Construction

```python
# Turnover-based operational-risk target
TURNOVER_COL = "employee_turnover_rate"
TURNOVER_SENTINELS = [999, 9999.99, 99999, -9, -8]
TURNOVER_BANDS = [(10.0, "Low"), (20.0, "Medium")]
HIGH_LABEL = "High"

def clean_turnover(series):
    """Return turnover as float with UKDS sentinels / impossible rates -> NaN."""
    v = pd.to_numeric(series, errors="coerce")
    v = v.mask(v.isin(TURNOVER_SENTINELS))
    return v.mask((v < 0) | (v > 100))

def turnover_to_category(v):
    """Map a clean turnover rate (0-100) to Low / Medium / High."""
    if pd.isna(v):
        return np.nan
    if v <= TURNOVER_BANDS[0][0]:
        return "Low"
    if v <= TURNOVER_BANDS[1][0]:
        return "Medium"
    return HIGH_LABEL
```

## Code Block 3.2: Leakage-Safe Feature Selection

```python
# ml_pipeline/data_contract.py
# Target-constructing and risk-indicator columns can NEVER be candidate
# features (data leakage). Leakage = 14 = 3 directly target-derived + 11
# precautionary risk indicators; the 3 administrative IDs are excluded
# separately and are deliberately NOT counted as leakage.
ID_COLS = ["serno", "year", "weight"]

DIRECT_TARGET_LEAKAGE = [
    "employee_turnover_rate", "employee_turnover_rate_5yr", SCORE_COL,
]
PRECAUTIONARY_EXCLUSIONS = [
    "absenteeism_rate", "grievance_rate", "grievance_procedure_formal",
    "training_problems", "skill_shortage_reported", "disputes_raised",
    "dispute_risk_level", "other_industrial_action", "dismissal_rate",
    "disciplinary_warnings", "suspensions",
]
HIGH_CARDINALITY_EXCLUSIONS = ["industry_code", "industry_code_80"]

LEAKAGE = DIRECT_TARGET_LEAKAGE + PRECAUTIONARY_EXCLUSIONS  # 14 columns

def predictor_columns():
    """Mapped columns usable as predictors (not IDs, not leakage/target)."""
    return [c for c in ALL_VARIABLES if c not in LEAKAGE and c not in ID_COLS]
```

## Code Block 3.3: Stratified Train-Test Split (shared contract)

```python
# ml_pipeline/data_contract.py -- one split for the whole pipeline
Xtr_raw, Xte_raw, ytr, yte = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# every downstream stage builds its frames through the same entry point
FRAMES = dc.build_model_frames()
Xtr, Xte = FRAMES["Xtr"], FRAMES["Xte"]
ytr, yte = FRAMES["ytr"], FRAMES["yte"]

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
print(f"CV: stratified {N_SPLITS}-fold (shuffle=True, random_state={RANDOM_STATE})")
```

## Code Block 4.1: Canonical Categorical Cleaning (via data_contract)

```python
# ml_pipeline/data_contract.py -- clean_frames()
# The typing rules and the rare-category map are derived on the TRAIN frame
# only and then applied to both partitions.
# Canonical missing-tokens are defined in data_contract.clean_categorical();
# fill nulls with "MISSING" at inference time.
        Xtr_raw[c] = # Canonical cleaning via clean_categorical()
            {"": "MISSING", " ": "MISSING", "nan": "MISSING", "None": "MISSING"})
        Xte_raw[c] = Xte_raw[c].astype(str).str.strip().replace(
            {"": "MISSING", " ": "MISSING", "nan": "MISSING", "None": "MISSING"})
        # Rare categories (< 1% in train) are labeled "OTHER_RARE"
        rare = set(vc[vc < 0.01].index)
        if rare:
            X[c] = X[c].apply(lambda v: "OTHER_RARE" if v in rare else v)

# The preprocessing pipeline uses data_contract.clean_categorical() for
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), cat_cols),
], verbose_feature_names_out=False)
```

## Code Block 4.2: Candidate Model Definition

```python
def build_models():
    """Return {name: estimator} with imbalance-aware settings."""
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=150,
            min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=300, early_stopping=True, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=5,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, eval_metric="mlogloss", tree_method="hist",
            random_state=RANDOM_STATE)
    if HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(n_estimators=200, num_leaves=63,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", verbose=-1, random_state=RANDOM_STATE)
    return models
```

## Code Block 4.3: Cross-Validation

```python
for name, est in build_models().items():
    pipe = Pipeline([("prep", preprocess), ("clf", est)])
    print(f"\n--- {name} ---")
    try:
        sc = cross_validate(pipe, Xtr, ytr, cv=cv,
            scoring=["accuracy", "balanced_accuracy", "f1_macro"],
            n_jobs=-1, error_score=np.nan)
        cv_sum = {k: float(np.nanmean(v)) for k, v in sc.items()
                  if k.startswith("test_")}
        print("CV acc=%.3f bal=%.3f f1m=%.3f" % (
            cv_sum.get("test_accuracy", np.nan),
            cv_sum.get("test_balanced_accuracy", np.nan),
            cv_sum.get("test_f1_macro", np.nan)))
    except Exception as e:
        print(f"CV failed: {e}"); cv_sum = {}
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    bal = balanced_accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    print(f"TEST acc={acc:.3f} bal_acc={bal:.3f} f1_macro={f1m:.3f}")
```

## Code Block 4.4: Pydantic Validation Schema

```python
class OperationalRiskInput(BaseModel):
    """Strict request schema for /predict, /simulate, /whatif."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    # ---- required ----
    sector: int = Field(..., ge=1, le=10, description="Sector classification (1-10)")
    org_size_category_3: int = Field(..., ge=1, le=3, description="Org size (3-cat)")
    multinational: int = Field(..., ge=0, le=2, description="Multinational status")
    union_coverage_pct: float = Field(..., ge=0, le=100, description="Union coverage %")
    collective_bargaining: int = Field(..., **_CODE, description="Collective bargaining code")
    performance_appraisal_pct: int = Field(..., **_CODE, description="Appraisal coverage code")
    jcc_exists: int = Field(..., **_CODE, description="Joint consultative committee")
    information_sharing_score: int = Field(..., **_CODE, description="Information sharing score")
    pct_part_time: float = Field(..., ge=0, le=100, description="Percentage part-time workers")
    pct_temporary: int = Field(..., **_CODE, description="Temporary-worker code")

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
    job_security_policy: Optional[int] = Field(None, **_CODE)
    jcc_frequency: Optional[int] = Field(None, **_CODE)
    jcc_function: Optional[int] = Field(None, **_CODE)
    jcc_union_chair: Optional[int] = Field(None, **_CODE)
    info_chain_exist: Optional[int] = Field(None, **_CODE)
    newsletter_exists: Optional[int] = Field(None, **_CODE)
    investment_plans_shared: Optional[int] = Field(None, **_CODE)
    consultation_depth: Optional[int] = Field(None, **_CODE)
    employee_voice_channels: Optional[int] = Field(None, **_CODE)
```

## Code Block 4.5: Flask Prediction Endpoint

```python
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

def _run_prediction(payload, request_id, explain=True):
    """Validate one org dict then predict. Returns (dict, None) or (None, resp)."""
    model, err, status, extras = validate_payload(payload, request_id)
    if err is not None:
        return None, (jsonify(err), status)
    if extras:
        LOG.warning("rid=%s ignored unexpected field(s): %s", request_id, ", ".join(extras))
    data = model_to_payload(model)
    try:
        label, prob = predictor.predict(data)
        probs = predictor.predict_proba_dict(data)
    except Exception as e:
        LOG.warning("rid=%s MODEL_ERROR: %s", request_id, e)
        body, st = error_body("MODEL_ERROR", "Model inference failed", request_id=request_id)
        return None, (jsonify(body), st)
    if any(v != v for v in probs.values()):  # NaN check
        body, st = error_body("MODEL_ERROR", "Model inference failed", request_id=request_id)
        return None, (jsonify(body), st)
    out = {"risk_level": label, "probability": prob, "probabilities": probs}
    if explain:
        try:
            loc = predictor.explain(data)
        except Exception as e:
            LOG.warning("rid=%s local SHAP skipped: %s", request_id, e)
            loc = None
        if loc is not None:
            out["local_explanation"] = loc
    return out, None
```

## Code Block 5.1: Performance Evaluation

```python
for pkl in sorted(MODEL_DIR.glob("*.pkl")):
    if pkl.name == "best_model.pkl":
        continue
    name = pkl.stem
    try:
        pipe = joblib.load(pkl)
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)
        bal = balanced_accuracy_score(yte, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yte, pred, average="macro", zero_division=0)
        try:
            proba = pipe.predict_proba(Xte)
            auc = roc_auc_score(yte, proba, multi_class="ovr")
        except Exception:
            proba, auc = None, float("nan")
        cm = confusion_matrix(yte, pred)
        summary.append({"model": name, "accuracy": acc,
                        "balanced_accuracy": bal, "macro_precision": prec,
                        "macro_recall": rec, "macro_f1": f1, "roc_auc_ovr": auc})
        print(f"{name:15s} acc={acc:.3f} bal={bal:.3f} f1m={f1:.3f} auc={auc:.3f}")
```

## Code Block 5.2: SHAP Explanation

```python
# Explainer selection based on model type
kind = type(clf).__name__
if hasattr(clf, "estimators_") or kind in ("XGBClassifier", "LGBMClassifier"):
    import shap
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_ev_t)
    print(f"TreeExplainer on {kind}")
elif hasattr(clf, "coef_"):
    import shap
    explainer = shap.LinearExplainer(clf, X_bg_t)
    sv = explainer.shap_values(X_ev_t)
    print(f"LinearExplainer on {kind}")
else:
    import shap
    explainer = shap.Explainer(clf.predict_proba, X_bg_t)
    sv = explainer(X_ev_t).values
    print(f"Kernel/Sampling Explainer on {kind}")

# Global importance aggregated to original features
orig = [to_original(t) for t in t_names]
mean_abs = np.abs(arr).mean(axis=(0, 2))
agg = {}
for o, v in zip(orig, mean_abs):
    agg[o] = agg.get(o, 0.0) + float(v)
imp = (pd.DataFrame({"feature": list(agg), "mean_abs_shap": list(agg.values())})
       .sort_values("mean_abs_shap", ascending=False))
```

## Code Block 5.3: Valid Runtime Test

```python
def test_valid_input_returns_prediction(client, valid_payload):
    r = client.post("/api/predict", json=valid_payload)
    assert r.status_code == 200
    j = r.get_json()
    assert j["risk_level"] in ("Low", "Medium", "High")
    assert 0.0 <= j["probability"] <= 1.0
    assert abs(sum(j["probabilities"].values()) - 1.0) < 1e-6
    assert "X-Request-ID" in r.headers
```

## Code Block 5.4: Invalid-Input Test

```python
def test_missing_required_field(client):
    r = client.post("/api/predict", json={"sector": "1"})
    assert r.status_code == 400
    j = r.get_json()
    assert j["code"] == "MISSING_FIELD"
    assert "union_coverage_pct" in j["error"]

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
```

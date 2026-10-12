# TRUE README — Operational Risk Prediction & Interactive Dashboard (WERS 2011)

> This file describes **what the project actually is and does today**, verified
> against the real code, data and model artifacts on 2026-10-11. It replaces
> the aspirational `README.md` (which mentioned React/Vue, TensorFlow and cloud
> deployment — **none of which were used**). Every number below was produced by
> executing the pipeline, not copied from a plan.

---

## 1. One-paragraph summary

This dissertation project uses the ** Workplace Employment Relations Survey 2011
(WERS 2011, UKDA Study 4511)** to predict how operational risk — measured by a
composite of employee turnover, absenteeism, grievances and industrial
disputes — varies across UK workplaces. A target variable
`operational_risk_category` (**Low / Medium / High**) is built from a 0–100
composite risk score, then predicted **only from pre-existing organisational
and HR characteristics** (no leakage). Five machine-learning models are trained
and compared; a **logistic regression** wins on the primary metric (balanced
accuracy, because the target is extremely imbalanced) and is deployed inside a
**Flask web dashboard** with four tabs: Risk Prediction, Data Explorer, Model
Insights (SHAP) and a **What-If Simulator** for decision support.

---

## 2. Project at a glance (verified values)

| Fact | Value |
|---|---|
| Dataset | 6,635 workplaces × 52 variables (`data/processed/wers2011_operational_risk_merged.csv`) |
| Target classes | Low 5,502 (82.92%) · Medium 1,105 (16.65%) · High 28 (0.42%) — imbalance 196.5 : 1 |
| Model inputs | 36 features (5 numeric + 31 categorical) |
| Leakage exclusions | 10 risk-indicator columns + `operational_risk_score` (plus identifiers `serno`, `year`, `weight`) |
| Best model | `logreg` (logistic regression, class_weight=balanced) — balanced accuracy **0.550** |
| Test set | 1,327 (20% stratified hold-out, random_state=42) |
| Tests | pytest **29/29 passed** · system tests ST1–ST19 **19/19 passed** · browser E2E (Playwright) all passed |
| Dashboard | Flask 3.0 + vanilla JS + Plotly, runs locally on `python dashboard/app.py` → http://localhost:5000 |

---

## 3. Research questions

1. **RQ1 — Determinants.** Which organisational, HR and employee-relations
   predictors significantly influence operational risk levels (turnover,
   absenteeism, grievances, industrial disputes) in UK workplaces?
2. **RQ2 — Configurations.** How do different configurations of HR management
   practices and employee-relations strategies relate to the predictive
   probability of high operational risk across organisational sizes and sectors?
3. **RQ3 — Prediction & visualization.** To what extent can machine-learning
   models accurately predict risk levels from pre-existing characteristics, and
   how can predictions be effectively visualised in an interactive dashboard
   for decision-makers?

## 4. Research objectives → what was delivered

| Objective | Delivered artifact |
|---|---|
| **RO1** Identify key risk determinants | Correlation heatmap (Fig 4.3), MDI + SHAP importance (tables 5.10/5.11), factor analysis (`scripts/07_factor_analysis.py`) |
| **RO2** Develop predictive ML models | 5 models + 5 imbalance variants (`scripts/04*, 05`), comparison tables 5.1–5.12 |
| **RO3** Interactive dashboard | Flask app (`dashboard/`) with prediction, explorer, SHAP insights and what-if simulation; browser-verified E2E |

---

## 5. Data source

- **Workplace Employment Relations Survey 2011** — UK Data Archive Study 4511,
  cross-sectional workplace study, UK-wide, principal funder: Dept. for
  Business, Innovation and Skills.
- **The raw `.tab` files are NOT in this repository.** They must be obtained
  from the UK Data Archive under its End User Licence. The project works from
  the processed CSV `data/processed/wers2011_operational_risk_merged.csv`.
- **Processed dataset:** 6,635 rows × 52 columns; **0 missing values**,
  0 duplicates (verified in `scripts/02_exploratory_analysis.py` output).
- Column groups: 3 identifiers (`serno`, `year`, `weight`) · 10 risk
  indicators · 11 organisational · 11 HR practices · 11 employee relations ·
  5 labour-market · 2 derived (score + category).


---

## 6. How the target is built (Chapter 3 §3.4)

`operational_risk_score` is a **composite 0–100 index** — four dimensions, up
to 25 points each, computed in `scripts/01_data_preparation.py`:

| Dimension | Rule (points) |
|---|---|
| Employee turnover | >20% → 25 · >15% → 20 · >10% → 15 · >5% → 10 · else 5 |
| Absenteeism | >5% → 25 · >4% → 20 · >3% → 15 · >2% → 10 · else 5 |
| Industrial action | dispute risk = 1 → 25 · other action = 1 → 15 |
| Grievances | >10 → 25 · >5 → 20 · >2 → 15 · else 5 |

**Thresholds:** High ≥ 60 · Medium 35–59 · Low < 35.

Score distribution (verified): mean 20.20, median 15.00, std 10.35,
min 10, max 80 — which is why 82.9% of workplaces are Low risk.

---

## 7. Leakage control (the project's central methodological rule)

The target is *made from* risk indicators, so those indicators must never be
predictors — otherwise the model would just re-read the answer. Eleven columns
are hard-excluded in `scripts/04_model_training.py` and recorded in
`models/saved_models/model_metadata.json → leakage_excluded`:

```
employee_turnover_rate      absenteeism_rate           grievance_rate
employee_turnover_rate_5yr  disputes_raised            dispute_risk_level
other_industrial_action     dismissal_rate
disciplinary_warnings       suspensions
operational_risk_score          ← the score itself
```

Identifiers (`serno`, `year`, `weight`) are also excluded. Any artefact that
still lists risk indicators among "features" (e.g. an old
`feature_config.json` from 2026-10-05) is **stale and superseded** — the
authoritative feature list is `model_metadata.json` (36 features).

---

## 8. The 36 model features

**Numeric (5):** `sector_detailed`, `union_coverage_pct`, `pct_part_time`,
`pct_part_time_12`, `pct_female`

**Categorical (31):** organisation (`org_size_category`, `_3`, `_4`, `sector`,
`multinational`, `org_age_years`, `workplace_age`, `trading_status`) · HR
practices (`union_coverage_pct_6/_7`, `collective_bargaining`,
`bargaining_level_org/_est`, `formal_hr_strategy`, `performance_appraisal_pct`,
`training_budget_exists`, `training_problems`, `job_security_policy`) ·
employee relations (`jcc_exists`, `jcc_frequency`, `jcc_function`,
`jcc_union_chair`, `information_sharing_score`, `info_chain_exist`,
`newsletter_exists`, `investment_plans_shared`, `consultation_depth`,
`employee_voice_channels`, `grievance_procedure_formal`) · labour market
(`pct_temporary`, `skill_shortage_reported`).

Missing/unknown values are imputed inside the scikit-learn pipeline
(numeric → median, categorical → literal `"MISSING"`), so **partial inputs are
allowed** — only 10 fields are required by the API contract.

---

## 9. The pipeline, step by step (how to re-run everything, in order)

| Step | Script | What it produces |
|---|---|---|
| 1. Prepare data | `scripts/01_data_preparation.py` | merged CSV, risk score/category, `dataset_metadata.json` |
| 2. Explore | `scripts/02_exploratory_analysis.py` | EDA report (missing values, distributions) |
| 3. Features | `scripts/03_feature_engineering.py` | leakage-safe feature config |
| 4. Train | `scripts/04_model_training.py` | 5 model pipelines + `model_metadata.json` |
| 4b. Imbalance | `scripts/04b_imbalance_tuning.py` | SMOTE / SMOTE-ENN / tuned / stacked variants |
| 5. Evaluate | `scripts/05_model_evaluation.py` | metrics, confusion matrices, ROC, `results/` |
| 6. Explain | `scripts/06_shap_explanation.py` | SHAP importance + beeswarm plots |
| 7. Factors | `scripts/07_factor_analysis.py` | HR-practice factor analysis |
| 8. Dashboard | `dashboard/app.py` | interactive web app (see §12) |
| 9. Thesis artefacts | `scripts/research/00_thesis_outputs.py` | `research/figures/` (11) + `research/tables/` (16) |
| 10. System tests | `scripts/research/01_system_tests.py` | ST1–ST19 → `table_5_13` + report |

Run each with the **system Python**: `python scripts/01_data_preparation.py`
(see §14 for the environment note).

---

## 10. Verified results (Chapter 5 — real executed numbers)

### 10.1 Model comparison (held-out test set, n = 1,327)

| Model | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC (OvR) |
|---|---|---|---|---|
| **Logistic Regression (deployed)** | 0.649 | **0.550** | 0.410 | 0.746 |
| LightGBM | 0.797 | 0.441 | 0.436 | 0.782 |
| Random Forest | 0.822 | 0.435 | 0.442 | 0.806 |
| XGBoost | 0.829 | 0.404 | 0.417 | 0.804 |
| MLP (64, 32) | 0.826 | 0.376 | 0.381 | 0.751 |

**Why logistic regression wins despite the lowest accuracy:** always answering
"Low" already scores ≈ 83% accuracy, so raw accuracy is misleading under
196.5:1 imbalance. The research's primary metric is **balanced accuracy**
(mean of per-class recalls); logreg (0.550) beats every ensemble because its
balanced class weights force attention to the rare Medium/High classes. It is
saved as `models/saved_models/logreg.pkl` and is the model the API serves.

### 10.2 Imbalance-handling strategies (scripts/04b)

| Strategy | Balanced accuracy |
|---|---|
| logreg + class_weight=balanced | **0.550** |
| logreg + SMOTE-ENN | 0.502 |
| logreg + SMOTE | 0.497 |
| rf + SMOTE-ENN | 0.468 |
| stacked ensemble | 0.464 |
| xgboost tuned | 0.461 |

Cost-sensitive learning beat resampling — a clean dissertation finding (§5.5).

### 10.3 Confusion matrix — logistic regression

| Actual \\ Predicted | Low | Medium | High |
|---|---|---|---|
| Low | 714 | 332 | 54 |
| Medium | 61 | 146 | 14 |
| High | 1 | 3 | 2 |

Per-class recall: **Low 64.9% · Medium 66.1% · High 33.3%**. The High class
is genuinely hard: only 28 High workplaces exist in the entire dataset
(6 in the test set). This limitation is discussed honestly in §5.9 and in the
dashboard's own disclaimer text.

### 10.4 What drives predictions (SHAP, top drivers)

`employee_voice_channels`, `sector`, `trading_status`, `org_size_category_3`,
`union_coverage_pct_6/_7`, `collective_bargaining`, `formal_hr_strategy`,
`information_sharing_score` — i.e. employee-relations voice and
organisational identity features dominate, which directly answers **RQ1**.
Full table: `research/tables/table_5_11_shap_importance_top20.csv`.

---

## 11. Factor analysis of HR practices (supplementary, RO1)

Bartlett's test χ² = 15,924 (df = 120, p < 0.001) confirms the correlation
matrix is factorable; 4 factors retained (Kaiser criterion), tentatively:
collective bargaining, org-level bargaining, establishment-level bargaining,
and "HR formality" (appraisal/training/information-sharing).
**Honest caveat:** overall KMO = 0.491 (below the conventional 0.50 bar), so
these factors are reported as *exploratory* only and are **not** used as model
inputs. Output: `results/tables/factor_loadings.csv`,
`results/figures/factor_scree.png`.

---

## 12. The dashboard (RO3 — what it actually is)

**Stack:** Flask 3.0 backend · Pydantic v2 validation · vanilla JavaScript +
Plotly (CDN) frontend · no build step, no framework, runs locally.

### Tabs
1. **Risk Prediction** — 20-input grouped form (org / HR / labour), gauge +
   class-probability chart, and a **live local SHAP explanation computed
   server-side for the exact input** (exact linear SHAP, aggregated to the 36
   original features; falls back gracefully if explainability fails).
   Screenshot: `research/figures/dashboard_predict_local_shap.png`.
2. **Data Explorer** — sector/size filters, distribution charts, CSV export,
   optional CSV upload.
3. **Model Insights** — best-model card, balanced-accuracy comparison,
   confusion matrix, top-20 MDI and SHAP charts, all loaded live from
   `results/tables/`.
4. **What-If Simulator** — baseline vs scenario side-by-side forms; returns
   both probability vectors, a per-class **Δ table**, a grouped bar chart, an
   automatic **changed-fields diff** (e.g. `jcc_exists: 2 → 1`), a direction
   verdict (▲/▼), and a causality disclaimer ("model-learned associations,
   not causal effects"). Screenshot:
   `research/figures/dashboard_whatif_simulator.png`.

### API (all browser-verified live)
| Endpoint | Purpose |
|---|---|
| `POST /api/predict` (`/predict`) | single prediction; returns risk level + 3-class probabilities + `X-Request-ID` |
| `POST /api/whatif` (`/simulate`) | baseline vs scenario comparison; errors tagged `block: baseline\|scenario` |
| `GET /api/insights`, `GET /api/features` | model metrics, importance, SHAP (strict JSON, NaN-free) |
| `GET /api/data/summary` (`/data`), `/api/data/export`, `/api/data/upload`, `/api/data/load-sample` | explorer backend |
| `GET /api/metadata` | features, labels, best model |

### Error protocol (structured, tested ST6–ST15)
`MISSING_FIELD`/`INVALID_TYPE`/`OUT_OF_RANGE`/`MALFORMED_JSON` → 400 ·
`MODEL_ERROR` → 422 · `INTERNAL_ERROR` → 500 · `RATE_LIMITED` → 429
(per-IP, with `Retry-After`) · `SERVICE_UNAVAILABLE` → 503.

---

## 13. Verification status (Chapter 5 §5.11)

| Level | Scope | Result |
|---|---|---|
| Unit/integration (`pytest tests/`) | 29 cases: predictor, validation, API, aliases | **29/29 passed** |
| System tests ST1–ST19 | dataset facts, leakage, model load, all 8 error codes, rate limit, what-if, JSON validity | **19/19 passed** → `research/system_test_report.md`, `research/tables/table_5_13_system_tests.csv` |
| Live HTTP | all endpoints on `localhost:5000` | all 200, correct payloads |
| Browser E2E (Playwright) | forms auto-build (20/20/20), what-if delta chart + diff + validation gate, zero JS errors | **all passed** |

Regenerate everything deterministically:

```bash
python scripts/research/00_thesis_outputs.py   # 11 figures + 16 tables
python scripts/research/01_system_tests.py     # ST1-ST19 report
python -m pytest tests/ -q                     # 29 tests
```

---

## 14. Setup & run (the truth about this machine's environment)

**Use the system Python 3.13.3** — the repo's `.venv` only contains IPython
tooling, NOT the ML/Flask stack. All commands below assume `python` resolves
to 3.13.3 with the packages installed.

```bash
# 1. (new machine only) install dependencies
pip install -r requirements.txt

# 2. (only if data/processed CSV is missing) rebuild the dataset
python scripts/01_data_preparation.py

# 3. launch the dashboard
python dashboard/app.py          # → http://localhost:5000
```

Models are already trained and committed under `models/saved_models/`
(`logreg.pkl` + siblings + `model_metadata.json`), so steps 2–3 are usually
all you need.

### Environment (Table 4.1, executed)
Python 3.13.3 · Flask 3.0.0 · pandas 3.0.1 · numpy 2.4.2 · scikit-learn 1.8.0 ·
XGBoost 3.2.0 · LightGBM 4.6.0 · SHAP 0.51.0 · matplotlib 3.10.8 ·
pydantic 2.13.5 · flask-cors 6.0.2 · pytest 9.0.2

---

## 15. Project structure (as it really exists)

```
project-risk-dashboard/
├── data/processed/           wers2011_operational_risk_merged.csv (+ metadata)
├── scripts/                  01_data_preparation … 07_factor_analysis,
│                             04b_imbalance_tuning, research/ (thesis outputs)
├── models/saved_models/      logreg.pkl, random_forest.pkl, xgboost.pkl,
│                             lightgbm.pkl, mlp.pkl, model_metadata.json
├── dashboard/                app.py, validation.py, templates/, static/ (css+js)
├── tests/                    test_api.py, test_validation.py, test_predictor.py …
├── research/                 figures/ (12 PNG) , tables/ (16 CSV/JSON),
│                             README.md (chapter→artifact map), system_test_report.md
├── results/                  evaluation tables/figures from scripts 05–07
├── notebooks/                exploratory notebooks (optional)
├── requirements.txt          actual dependency list
├── README.md                 original project proposal (kept for history)
└── TRUE_README.md            this file — the verified, current state
```

Helper scripts `check_*.py`, `start_app.py`, `test_api.py` (root) are quick
manual diagnostics; the maintained suite lives in `tests/`.

---

## 16. What this project is NOT (correcting the original README)

- **Not** React/Vue/D3 — the frontend is one HTML template + one JS file + Plotly.
- **Not** TensorFlow/PyTorch — the "neural network" is sklearn's `MLPClassifier`.
- **Not** FastAPI — it is Flask with a Pydantic validation layer.
- **Not** Docker/cloud-deployed — it runs locally; deployment was out of scope.
- **Not** leakage-free by accident — exclusions are enforced in code and tested (ST3).

---

## 17. Artefact index (where to look for the dissertation)

| Need | Location |
|---|---|
| Chapter → artifact map (§3/4/5) | `research/README.md` |
| Figures 4.1–4.4, 5.1–5.7 + dashboard screenshot | `research/figures/` |
| Tables 3.2, 4.1–4.2, 5.1–5.13 | `research/tables/` |
| System-test evidence (ST1–ST19) | `research/system_test_report.md` |
| Per-model metrics/CMs | `results/tables/`, `models/saved_models/model_metadata.json` |

---

## 18. Limitations & responsible use

1. **Extreme rarity of High risk** (28/6,635) bounds achievable recall; treat
   High predictions as triage signals, not ground truth.
2. **Associations ≠ causation** — the what-if simulator shows how the fitted
   model responds to input changes; it does not prove that changing an HR
   practice will change real-world risk.
3. **Cross-sectional 2011 data** — relationships may have drifted; retrain
   with newer WERS waves before operational use.
4. **Survey weights** (`weight`) are excluded from modelling; population-level
   claims should use weighted descriptive statistics instead.
5. **Data licence** — WERS 2011 is distributed under the UKDA End User
   Licence; the raw data must never be redistributed with this repository.

---

## 19. Citation

```bibtex
@mastersthesis{operational_risk_wers2011,
  title  = {Predictive Modeling of Business Operational Risk Levels
            and an Interactive Web Dashboard},
  author = {Your Name},
  school = {Your Institution},
  year   = {2026},
  note   = {WERS 2011, UK Data Archive Study 4511}
}
```

**Data acknowledgement:** UK Data Archive — Workplace Employment Relations
Survey 2011 (Study 4511); funded by the UK Department for Business,
Innovation and Skills.

---

*This TRUE README reflects the executed pipeline state of 2026-10-11:
19/19 system tests, 29/29 pytest, browser-verified dashboard.*

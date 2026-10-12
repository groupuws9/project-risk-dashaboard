# Operational Risk Prediction & Interactive Dashboard

**Machine-learning classification of business operational risk from the UK Workplace
Employment Relations Survey 2011 (WERS 2011, UKDA Study 4511), served through an
interactive Flask dashboard with live SHAP explainability and a what-if scenario
simulator.**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)]()
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)]()
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)]()
[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Contents

1. [Overview](#overview)
2. [Key findings](#key-findings)
3. [Dataset & methodology](#dataset--methodology)
4. [Packages used](#packages-used)
5. [Project structure](#project-structure)
6. [Installation](#installation)
7. [Running the project](#running-the-project)
8. [Dashboard & API](#dashboard--api)
9. [Python usage example](#python-usage-example)
10. [Testing](#testing)
11. [Limitations & responsible use](#limitations--responsible-use)
12. [Contributing](#contributing)
13. [License & data acknowledgement](#license--data-acknowledgement)
14. [Citation](#citation)

---

## Overview

Operational risk — employee turnover, absenteeism, grievances and industrial
disputes — is costly and unevenly distributed across UK workplaces. This project:

- builds a **leakage-controlled target** (`Low` / `Medium` / `High`) from a
  composite 0–100 risk index computed over four risk dimensions;
- trains and compares **five machine-learning models** (logistic regression,
  random forest, XGBoost, LightGBM, MLP) plus five imbalance-handling variants,
  selecting on **balanced accuracy** — not raw accuracy — because the target is
  imbalanced 196.5 : 1;
- deploys the winning model (**logistic regression**) behind a structured,
  rate-limited REST API and a four-tab **Flask dashboard**: risk prediction with
  **live per-input SHAP explanations**, a data explorer, global model insights,
  and a **what-if simulator** that compares baseline vs scenario organisations
  side-by-side (delta table, probability chart, changed-fields diff).

The full pipeline is deterministic (`random_state = 42`) and verified by
**31 pytest tests**, including a 19-point system-test plan (ST1–ST19).

---

## Key findings

All numbers below come from the executed pipeline on the held-out test set
(n = 1,327 stratified, 20 %).

| Model | Accuracy | **Balanced accuracy** | Macro-F1 | ROC-AUC (OvR) |
|---|---|---|---|---|
| **Logistic Regression (deployed)** | 0.649 | **0.550** | 0.410 | 0.746 |
| LightGBM | 0.797 | 0.441 | 0.436 | 0.782 |
| Random Forest | 0.822 | 0.435 | 0.442 | 0.806 |
| XGBoost | 0.829 | 0.404 | 0.417 | 0.804 |
| MLP (64, 32) | 0.826 | 0.376 | 0.381 | 0.751 |

**Insights**

1. **Accuracy is misleading under imbalance.** Always predicting "Low" already
   scores ≈ 83 % accuracy. Tree ensembles cluster on the majority class; the
   logistic regression with balanced class weights generalises far better across
   all three classes and therefore wins the research's primary metric.
2. **Cost-sensitive learning beats resampling.** `class_weight="balanced"`
   (0.550) outperformed SMOTE (0.497), SMOTE-ENN (0.502), tuned XGBoost (0.461)
   and a stacked ensemble (0.464).
3. **Employee-relations "voice" drives predictions.** The strongest SHAP
   drivers are `employee_voice_channels`, `sector`, `trading_status`,
   `org_size_category_3`, union-coverage bands, `collective_bargaining` and
   `formal_hr_strategy` — organisational identity and employee-relations
   practices, not the risk indicators themselves (which are excluded by design).
4. **The rare High class is genuinely hard.** Only 28 of 6,635 workplaces are
   High risk (0.42 %); High-class recall is 33 % — a data-scarcity bound that is
   disclosed in the dashboard itself, not hidden.

Per-class recall (logistic regression): Low 64.9 % · Medium 66.1 % · High 33.3 %.

---

## Dataset & methodology

- **Source:** Workplace Employment Relations Survey 2011, UK Data Archive
  Study 4511. **Raw files are not redistributed with this repository** (see
  [License & data acknowledgement](#license--data-acknowledgement)).
- **Processed dataset:** 6,635 workplaces × 52 variables, zero missing values,
  zero duplicates.
- **Target:** `operational_risk_score` = four dimensions × 25 points
  (turnover, absenteeism, industrial action, grievances); categories
  High ≥ 60, Medium 35–59, Low < 35. Class counts: Low 5,502 (82.92 %) ·
  Medium 1,105 (16.65 %) · High 28 (0.42 %).
- **Leakage control:** the ten risk-indicator columns used to *construct* the
  target, plus `operational_risk_score` and the identifiers, are hard-excluded
  from modelling (enforced in `scripts/04_model_training.py` and regression-
  tested). The model sees **36 pre-existing features** (5 numeric +
  31 categorical) only.
- **Explainability:** global SHAP importance (tree/linear explainers) and a
  **live local explanation per prediction** — exact linear SHAP computed
  server-side for the exact input, aggregated back to the original features.

---

## Packages used

Exact versions from the executed environment (Table 4.1 of the dissertation);
`requirements.txt` carries the compatible minimum ranges.

| Package | Verified version | Role in this project |
|---|---|---|
| Python | 3.13.3 | runtime |
| pandas | 3.0.1 | data wrangling, feature frames, CSV I/O |
| numpy | 2.4.2 | numerical arrays, SHAP maths |
| scipy | ≥ 1.10 | statistical tests (Bartlett, KMO) |
| statsmodels | ≥ 0.14 | factor analysis (EDA) |
| scikit-learn | 1.8.0 | models, preprocessing pipeline, metrics, MLP |
| imbalanced-learn | ≥ 0.11 | SMOTE / SMOTE-ENN resampling variants |
| XGBoost | 3.2.0 | gradient-boosting model |
| LightGBM | 4.6.0 | gradient-boosting model |
| SHAP | 0.51.0 | global + local model explainability |
| matplotlib | 3.10.8 | static dissertation figures |
| seaborn | ≥ 0.12 | EDA plots |
| joblib | ≥ 1.3 | model serialisation (`.pkl` pipelines) |
| Flask | 3.0.0 | web backend / REST API |
| Flask-Cors | 6.0.2 | cross-origin support |
| pydantic | 2.13.5 | strict request validation & error protocol |
| tqdm | ≥ 4.65 | progress bars in long scripts |
| pytest | 9.0.2 | test suite (31 tests) |
| Plotly.js | 2.27.0 (CDN) | interactive dashboard charts (gauge, SHAP, what-if) |

---

## Project structure

```
project-risk-dashboard/
├── README.md                  # this file
├── TRUE_README.md             # deep-dive verification report (dissertation map)
├── LICENSE                    # MIT (code)
├── requirements.txt           # dependency ranges
│
├── data/
│   ├── wirsts10_july2014.csv        # raw WERS 2011 main survey (NOT in git — UKDA licence)
│   ├── wirsts25_july2014.csv        # raw WERS 2011 panel file  (NOT in git — UKDA licence)
│   └── processed/
│       ├── wers2011_operational_risk_merged.csv   # analysis-ready dataset
│       ├── wers2011_features_engineered.csv
│       ├── dataset_metadata.json
│       └── processing_summary.txt
│
├── scripts/                   # numbered pipeline (run in order)
│   ├── 01_data_preparation.py        # merge, clean, build risk score/category
│   ├── 02_exploratory_analysis.py    # EDA report
│   ├── 03_feature_engineering.py     # leakage-safe feature config
│   ├── 04_model_training.py          # 5 models + model_metadata.json
│   ├── 04b_imbalance_tuning.py       # SMOTE / SMOTE-ENN / tuned / stacked
│   ├── 05_model_evaluation.py        # metrics, confusion matrices, ROC
│   ├── 06_shap_explanation.py        # global + local SHAP artefacts
│   └── 07_factor_analysis.py         # HR-practice factor analysis
│
├── models/
│   └── saved_models/                 # *.pkl produced locally (NOT in git)
│       └── model_metadata.json       # best model, 36 features, metrics
│
├── dashboard/
│   ├── app.py                        # Flask app + REST API
│   ├── validation.py                 # Pydantic schema & error codes
│   ├── templates/index.html          # single-page frontend
│   └── static/
│       ├── css/style.css
│       └── js/app.js                 # tabs, forms, Plotly charts, what-if
│
├── tests/
│   ├── conftest.py                   # fixtures (client, valid payload)
│   ├── test_api.py                   # endpoints, error codes, SHAP contract
│   ├── test_validation.py            # schema edge cases
│   └── test_predictor.py             # inference wrapper
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_feature_analysis.ipynb
    └── 03_model_development.ipynb
```

---

## Installation

**Prerequisites:** Python 3.11+ (developed and verified on 3.13.3), pip, git.

```bash
git clone <your-fork-url>
cd project-risk-dashboard

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Data:** the raw WERS 2011 files must be obtained separately from the
[UK Data Archive](https://www.ukdataservice.ac.uk/) (Study 4511) under its
End User Licence, then placed in `data/`. If `data/processed/` already contains
`wers2011_operational_risk_merged.csv`, you can skip straight to
[Running](#running-the-project).

---

## Running the project

### Full pipeline (from raw data to dashboard)

```bash
python scripts/01_data_preparation.py        # merge + risk score/category
python scripts/02_exploratory_analysis.py    # EDA report
python scripts/03_feature_engineering.py     # leakage-safe features
python scripts/04_model_training.py          # train 5 models -> models/saved_models/
python scripts/04b_imbalance_tuning.py       # imbalance variants (optional)
python scripts/05_model_evaluation.py        # metrics + confusion matrices
python scripts/06_shap_explanation.py        # SHAP artefacts
python scripts/07_factor_analysis.py         # factor analysis (optional)
```

### Launch the dashboard

```bash
python dashboard/app.py
# -> http://localhost:5000
```

Model binaries (`*.pkl`) are generated by step 04 and are not committed; a
fresh clone must run the pipeline (at minimum steps 01 and 04) before the
dashboard will start.

---

## Dashboard & API

### Web UI (four tabs)

| Tab | What it does |
|---|---|
| **Risk Prediction** | 20-input form → gauge, class probabilities, and a **live local SHAP chart for the exact input** |
| **Data Explorer** | filter by sector/size, distribution charts, CSV export/upload |
| **Model Insights** | best-model card, balanced-accuracy comparison, confusion matrix, top-20 MDI + SHAP importance |
| **What-If Simulator** | baseline vs scenario side-by-side: per-class Δ table, grouped bar chart, changed-fields diff, direction verdict, causality disclaimer |

### REST API

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` (`/predict`) | POST | risk level + 3-class probabilities + live `local_explanation` |
| `/api/whatif` (`/simulate`) | POST | baseline vs scenario comparison (errors tagged `baseline`/`scenario`) |
| `/api/insights`, `/api/features` | GET | model metrics, confusion matrix, importance, SHAP tables |
| `/api/data/summary` (`/data`), `/api/data/export`, `/api/data/upload`, `/api/data/load-sample` | GET/POST | explorer backend |
| `/api/metadata` | GET | features, labels, best model |

Requests are validated with Pydantic and return a structured error protocol:

| Code | HTTP | Meaning |
|---|---|---|
| `MISSING_FIELD` | 400 | required field absent/null (all missing fields listed) |
| `INVALID_TYPE` | 400 | value cannot be coerced |
| `OUT_OF_RANGE` | 400 | outside the allowed domain |
| `MALFORMED_JSON` | 400 | body is not a JSON object |
| `MODEL_ERROR` | 422 | inference failed (incomplete vector) |
| `INTERNAL_ERROR` | 500 | unexpected failure |
| `RATE_LIMITED` | 429 | per-IP limit exceeded (`Retry-After` header) |
| `SERVICE_UNAVAILABLE` | 503 | model not loaded |

Ten fields are required; everything else is optional and safely imputed
(numeric → median, categorical → `"MISSING"`) inside the scikit-learn pipeline.
Every response carries an `X-Request-ID` for traceability.

---

## Python usage example

```python
from models.risk_predictor import OperationalRiskPredictor

predictor = OperationalRiskPredictor.load()   # loads models/saved_models/logreg.pkl

org = {
    "org_size_category_3": 2,
    "sector": 1,
    "union_coverage_pct": 30,
    "collective_bargaining": 2,
    "performance_appraisal_pct": 3,
    "jcc_exists": 2,
    "information_sharing_score": 3,
    "pct_part_time": 20,
    "pct_temporary": 5,
}

risk_level, probability = predictor.predict(org)
print(f"Predicted risk: {risk_level} ({probability:.1%})")
print(predictor.predict_proba_dict(org))     # {'Low': ..., 'Medium': ..., 'High': ...}

why = predictor.explain(org)                 # live local SHAP for this input
for c in why["contributions"][:5]:
    print(f"  {c['feature']:>28s}  shap={c['shap']:+.4f}  input={c['value']}")
```

---

## Testing

```bash
python -m pytest tests/ -q          # 31 tests, ~3 s
```

The suite covers the predictor wrapper, the Pydantic validation contract
(all eight error codes, edge cases, imputation), every API endpoint and
alias, the rate limiter, the what-if block protocol, and the live local-SHAP
response contract (including fail-safety: a broken explainer must never break
a prediction).

---

## Limitations & responsible use

1. **High-risk scarcity.** 28 High observations (0.42 %) bound achievable
   High-class recall; treat High predictions as triage signals, not ground truth.
2. **Associations ≠ causation.** The what-if simulator shows how the fitted
   model responds to input changes — it does not prove that changing an HR
   practice will change real-world risk. The UI states this explicitly.
3. **Cross-sectional 2011 data.** Relationships may have drifted; retrain with
   newer WERS waves before any operational use.
4. **Survey weights excluded.** `weight` is not a model feature; use weighted
   descriptive statistics for population-level claims.
5. **Development server.** `dashboard/app.py` runs Flask's dev server — put a
   production WSGI server (gunicorn/uWSGI) in front of it for any real
   deployment.

---

## Contributing

Contributions are welcome:

1. Fork the repository and create a feature branch
   (`git checkout -b feature/my-feature`).
2. Follow **PEP 8**; add docstrings to new functions.
3. Add or extend tests in `tests/` — the suite must stay green
   (`python -m pytest tests/ -q`).
4. Open a pull request describing the change and its motivation.

Please keep contributions consistent with the project's scope (leakage-safe
operational-risk modelling + decision-support dashboard) and do not commit
WERS raw data or large model binaries (both are git-ignored for licence and
size reasons).

---

## License & data acknowledgement

- **Code:** released under the [MIT License](LICENSE).
- **Data:** the Workplace Employment Relations Survey 2011 is provided by the
  **UK Data Archive** (Study 4511) under its **End User Licence**. The raw
  data files are excluded from this repository; you must register with the UK
  Data Archive and accept the licence to obtain them. Do not redistribute the
  raw or processed survey data with derived repositories.
- WERS 2011 was funded by the UK Department for Business, Innovation and
  Skills and conducted by the Institute for Employment Studies, the National
  Institute of Economic and Social Research, and the Policy Studies Institute.

---

## Citation

If you use this work, please cite:

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


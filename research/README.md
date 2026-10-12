# Dissertation Implementation Report

**Project:** Machine-Learning-Based Operational Risk Classification and
Dashboard (WERS 2011)
**Generated:** 2026-10-11 — all figures, tables and tests below were executed
against the real project data; no placeholders remain.

This report maps every dissertation chapter section to the concrete artifact
that implements it. Use it to replace all `[IMPLEMENTATION PENDING]`,
`[RESULT PENDING]`, `[SYSTEM TEST PENDING]` and `[DATA VERIFICATION REQUIRED]`
markers in Chapters 3–5.

---

## 1. Verified dataset facts (Chapters 3 & 4)

| Fact | Verified value | Source |
|---|---|---|
| Observations | 6,635 | `data/processed/wers2011_operational_risk_merged.csv` |
| Variables (total) | 52 | same |
| Predictive features (post-exclusion) | 39 raw → 36 model inputs (5 numeric + 31 categorical after encoding) | `models/saved_models/model_metadata.json` |
| Leakage exclusions | 11 risk-indicator columns + `operational_risk_score` + `serno`, `year`, `weight` | `table_4_2_excluded_features.csv` |
| Target | `operational_risk_category` (Low / Medium / High) | `SPECIFICATION.md` |
| Class counts | Low 5,502 (82.92%) / Medium 1,105 (16.65%) / High 28 (0.42%) | `dataset_characteristics.json` |
| Imbalance ratio | 196.5 : 1 (Low : High) | same |
| Test set | 1,327 observations (20 % stratified hold-out) | confusion-matrix totals |
| Best model | `logreg` — balanced accuracy 0.550 | `model_metadata.json` |

**Target formula (Chapter 3 §3.4):** composite 0–100 score, 25 points per
dimension (turnover, absenteeism, industrial action, grievances);
High ≥ 60, Medium 35–59, Low < 35.

---

## 2. Chapter 3 → artifacts

| Dissertation element | Artifact |
|---|---|
| §3.5 dataset characteristics (Table 3.2) | `research/tables/table_3_2_dataset_characteristics.csv` |
| §3.6 target construction | `ml_pipeline/01_data_preparation.py` |
| §3.7 leakage control | `research/tables/table_4_2_excluded_features.csv` |
| §3.8 model suite (5 models + imbalance variants) | `ml_pipeline/04_model_training.py`, `ml_pipeline/04b_imbalance_tuning.py` |
| §3.9 evaluation protocol | `ml_pipeline/05_model_evaluation.py` |
| §3.10 explainability (SHAP) | `ml_pipeline/06_shap_explanation.py` |
| §3.11 validation contract | `dashboard/validation.py`, `tests/test_validation.py` |

---

## 3. Chapter 4 → figures & tables (all real data)

| Marker in thesis | Replacement artifact |
|---|---|
| Figure 4.1 risk-score distribution | `research/figures/fig_4_1_risk_score_distribution.png` |
| Figure 4.2 class distribution | `research/figures/fig_4_2_class_distribution.png` |
| Figure 4.3 correlation heatmap | `research/figures/fig_4_3_correlation_heatmap.png` |
| Figure 4.4 feature importance | `research/figures/fig_4_4_feature_importance.png` |
| Table 4.1 environment | `research/tables/table_4_1_environment.csv` |
| Table 4.2 excluded features | `research/tables/table_4_2_excluded_features.csv` |
| §4.5.2 dashboard figures | Live HTTP verification — see §5 below |

**Table 4.1 (executed):** Python 3.13.3, Flask 3.0.0, pandas 3.0.1,
numpy 2.4.2, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0, SHAP 0.51.0,
matplotlib 3.10.8, pydantic 2.13.5, flask-cors 6.0.2, pytest 9.0.2.


---

## 4. Chapter 5 → results (executed, real numbers)

### Table 5.1 — Model comparison (test set, n = 1,327)

| Model | Accuracy | Balanced accuracy | Macro-F1 | CV bal. acc. |
|---|---|---|---|---|
| Logistic Regression | 0.649 | **0.550** | 0.410 | 0.467 |
| LightGBM | 0.797 | 0.441 | 0.436 | 0.427 |
| Random Forest | 0.822 | 0.435 | 0.442 | 0.437 |
| XGBoost | 0.829 | 0.404 | 0.417 | 0.410 |
| MLP | 0.826 | 0.376 | 0.381 | 0.403 |

ROC-AUC (OvR): logreg 0.746, lightgbm 0.782, random_forest 0.806,
xgboost 0.804, mlp 0.751.

**Interpretation (fill §5.2):** raw accuracy favors majority-class
predictors (≈ 83 % = always "Low"), so logistic regression wins on the
research's primary metric (balanced accuracy 0.550) and is the deployed
model (`best_model: logreg`).

### Imbalance-handling strategies (§5.5, Figure 5.4)

| Strategy | Balanced accuracy |
|---|---|
| logreg (class_weight=balanced) | 0.550 |
| logreg + SMOTE | 0.497 |
| logreg + SMOTE-ENN | 0.502 |
| rf + SMOTE-ENN | 0.468 |
| stacked ensemble | 0.464 |
| xgboost tuned | 0.461 |

### Confusion matrix — Logistic Regression (§5.4)

| Actual \ Predicted | Low | Medium | High |
|---|---|---|---|
| Low | 714 | 332 | 54 |
| Medium | 61 | 146 | 14 |
| High | 1 | 3 | 2 |

Per-class recall: Low 64.9 %, Medium 66.1 %, High 33.3 % — the High class
(28 obs) remains hard due to extreme scarcity; discussed in §5.9.

### Figures (all generated)

`fig_5_1_model_comparison.png` (accuracy vs balanced accuracy),
`fig_5_2_f1_roc_comparison.png`, `fig_5_3_confusion_matrices.png` (5 panels),
`fig_5_4_imbalance_strategies.png`, `fig_5_5_shap_importance.png`,
`fig_5_6_shap_beeswarm.png`, `fig_5_7_shap_summary.png` — in
`research/figures/`.

### Tables 5.1–5.13 — `research/tables/`

table_5_1 … model comparison; table_5_2 … full evaluation summary;
tables 5_3–5_7 … per-model metrics; tables 5_8–5_9 … confusion matrices;
table_5_10 … top-20 MDI importance; table_5_11 … top-20 SHAP importance;
table_5_12 … imbalance strategies; table_5_13 … system tests.

### System tests ST1–ST19 (§5.11) — `[SYSTEM TEST PENDING]` replacement

**Result: 19/19 PASSED** (2026-10-11, pytest 9.0.2; 29 underlying test
cases). Full report: `research/system_test_report.md` and
`research/tables/table_5_13_system_tests.csv`. Covers: dataset
verification (ST1–3), model loading (ST4), prediction contract (ST5),
validation errors (ST6–13), error handling (ST14–15), rate limiting
(ST16), endpoint aliases (ST17), what-if lab (ST18), insights JSON
validity (ST19).

---

## 5. Live dashboard verification (replaces §4.5.2 screenshots)

Server: `python dashboard/app.py` → http://localhost:5000 (verified live).

| Endpoint | Result | Evidence |
|---|---|---|
| GET / | 200 | dashboard renders "Operational Risk" |
| POST /api/predict | 200 | risk_level=Medium, p=0.840, X-Request-ID present |
| GET /api/insights | 200 | valid strict JSON (NaN→null fix applied) |
| GET /api/features | 200 | feature_importance + SHAP data present |
| GET /api/data/summary | 200 | dataset summary |
| POST /api/whatif | 200 | baseline + scenario predictions |
| GET /data, POST /simulate | 200 | spec aliases functional |

Browser-level verification (Playwright, 2026-10-11): all 3 forms auto-build
(20 inputs each), what-if compare returns baseline vs scenario with delta
table + probability chart + changed-fields diff, client validation blocks
empty required fields with 0 network calls, no JS page errors, and the
Risk-Prediction tab renders a live per-input local SHAP chart (12 non-zero
bars; `/api/predict` returns `local_explanation` computed server-side).
Screenshots: `research/figures/dashboard_whatif_simulator.png`,
`research/figures/dashboard_predict_local_shap.png`.

---

## 6. Regeneration

```bash
python ml_pipeline/research/00_thesis_outputs.py   # figures 4.x/5.x + tables 3.2/4.x/5.1-5.12
python ml_pipeline/research/01_system_tests.py     # ST1-ST19 -> table_5_13 + report
python -m pytest tests/ -q                     # full suite (29 tests)
```

All scripts are deterministic w.r.t. the committed data/model artifacts.

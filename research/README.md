# Dissertation Implementation Report

**Project:** Machine-Learning-Based Operational Risk Classification and
Dashboard (WERS time-series, 1980–2011)
**Generated:** 2026-10-13 — all figures, tables and tests below were executed
against the real project data; no placeholders remain.

This report maps every dissertation chapter section to the concrete artifact
that implements it. Use it to replace all `[IMPLEMENTATION PENDING]`,
`[RESULT PENDING]`, `[SYSTEM TEST PENDING]` and `[DATA VERIFICATION REQUIRED]`
markers in Chapters 3–5.

---

## 1. Verified dataset facts (Chapters 3 & 4)

| Fact | Verified value | Source |
|---|---|---|
| Observations | 14,448 (7 waves: 1980/1984/1990/1991/1998/2004/2011) | `data/processed/wers_operational_risk_merged.csv` |
| Variables (total) | 53 | same |
| Modelled rows (target available) | 6,385 (turnover observed in 1998/2004/2011 waves) | same |
| Predictive features (post-exclusion) | 33 raw + 4 EFA factors = 37 model inputs (9 numeric + 28 categorical) | `models/saved_models/model_metadata.json` |
| Leakage exclusions | 14 columns (turnover family, risk indicators, `tprocgrv`/`ttrnprob` duplicates, score alias) | `table_4_2_excluded_features.csv` |
| Target | `operational_risk_category` (Low / Medium / High) | `SPECIFICATION.md` |
| Class counts | Low 5,077 (79.51%) / Medium 621 (9.73%) / High 687 (10.76%) of 6,385 | `dataset_characteristics.json` |
| Imbalance ratio | 7.4 : 1 (Low : High) | same |
| Test set | 1,277 observations (20 % stratified hold-out) | confusion-matrix totals |
| Best model | `logreg` — CV balanced accuracy 0.385 (test 0.389, reported only) | `model_metadata.json` |

**Target formula (Chapter 3 §3.4):** employee turnover rate (`tempc`),
sentinel-cleaned; Low ≤ 10 %, Medium 10–20 %, High > 20 %. Rows without an
observed turnover rate are excluded from training (6,385 of 14,448).

---

## 2. Chapter 3 → artifacts

| Dissertation element | Artifact |
|---|---|
| §3.5 dataset characteristics (Table 3.2) | `research/tables/table_3_2_dataset_characteristics.csv` |
| §3.6 target construction | `ml_pipeline/01_build_dataset.py` |
| §3.7 leakage control | `research/tables/table_4_2_excluded_features.csv` |
| §3.8 model suite (5 models + imbalance variants) | `ml_pipeline/04_model_training.py`, `ml_pipeline/05_imbalance_tuning.py` |
| §3.9 evaluation protocol | `ml_pipeline/06_model_evaluation.py` |
| §3.10 explainability (SHAP) | `ml_pipeline/07_shap_explanation.py` |
| §3.11 validation contract | `dashboard/validation.py`, `tests/test_validation.py` |

---

## 3. Chapter 4 → figures & tables (all real data)

| Marker in thesis | Replacement artifact |
|---|---|
| Figure 4.1 turnover-rate distribution (target basis) | `research/figures/fig_4_1_turnover_distribution.png` |
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

### Table 5.1 — Model comparison (test set, n = 1,277)

| Model | Accuracy | Balanced accuracy | Macro-F1 | CV bal. acc. |
|---|---|---|---|---|
| Logistic Regression | 0.422 | **0.389** | 0.317 | **0.385** |
| LightGBM | 0.739 | 0.385 | 0.391 | 0.366 |
| Random Forest | 0.764 | 0.358 | 0.355 | 0.345 |
| XGBoost | 0.792 | 0.334 | 0.300 | 0.340 |
| MLP | 0.796 | 0.333 | 0.295 | 0.333 |

ROC-AUC (OvR): logreg 0.585, lightgbm 0.614, random_forest 0.616,
xgboost 0.638, mlp 0.539.

**Interpretation (fill §5.2):** raw accuracy rewards majority-class guessing
(an always-"Low" classifier scores 79.5 %), so the boosted/tree models reach
0.74–0.80 accuracy while collapsing Medium and High. Logistic regression wins
the research's primary metric — it has the highest **training-CV** balanced
accuracy (0.385) and the highest held-out balanced accuracy (0.389) among the
five CV-evaluated candidates, so it is the deployed model
(`best_model: logreg`). The held-out test score is reported but never used to
pick the winner.

### Imbalance-handling strategies (§5.5, Figure 5.4)

Test balanced accuracy (05_imbalance_tuning variants are evaluated on the same 80/20 split but
carry **no CV score**, so they are ineligible for model selection):

| Strategy | Balanced accuracy |
|---|---|
| rf + SMOTE-ENN | 0.414 |
| xgboost tuned | 0.413 |
| stacked ensemble | 0.396 |
| logreg + SMOTE | 0.393 |
| logreg (class_weight=balanced) — **deployed** | 0.389 |
| logreg + SMOTE-ENN | 0.375 |

### Confusion matrix — Logistic Regression (§5.4)

| Actual \ Predicted | Low | Medium | High |
|---|---|---|---|
| Low (1,016) | 442 | 286 | 288 |
| Medium (124) | 35 | 49 | 40 |
| High (137) | 36 | 54 | 47 |

Per-class recall: Low 43.5 %, Medium 39.5 %, High 34.3 % (balanced accuracy
0.389). The turnover-band target is genuinely hard without the leaked
indicators — Medium (621) and High (687) are rare and overlap Low heavily;
discussed in §5.9.

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

**Result: 19/19 PASSED** (2026-10-13, pytest 9.0.2; 32 underlying test
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
python ml_pipeline/research/thesis_outputs.py   # figures 4.x/5.x + tables 3.2/4.x/5.1-5.12
python ml_pipeline/research/system_tests.py     # ST1-ST19 -> table_5_13 + report
python -m pytest tests/ -q                     # full suite (32 tests)
```

All scripts are deterministic w.r.t. the committed data/model artifacts.

# Dissertation Implementation Report - Chapter-by-Chapter Reference

**Project:** Machine-Learning-Based Operational Risk Classification and Dashboard (WERS time-series, 1980-2011)
**Author:** Lawrencia Appiah (Machine-learning lead, group project)
**Date:** 2026-11-07
**Status:** Verified against executed code, live dashboard output, and ground-truth pytest results (65 tests: 65 passed; system tests ST1–ST19: 19/19 passed)

---

## >>> COMPENDIUM OF TABLES, FIGURES, CODE BLOCKS & SCREENSHOTS

This README contains the **complete compendium** of all:

- ✅ **TABLES** - Every dataset characteristic, model comparison, confusion matrix, SHAP table, and system test table
- ✅ **FIGURES** - All statistical plots, architecture diagrams, and dashboard screenshots
- ✅ **CODE BLOCKS** - All key Python code from preprocessing to SHAP explanation
- ✅ **SCREENSHOTS** - Live dashboard screenshots (What-If Simulator, Local SHAP)

---

# TABLES - Quick Reference Index

| # | Table | File | Chapter | Description |
|---|-------|------|---------|-------------|
| 3.1 | Dataset characteristics | research/tables/table_3_2_dataset_characteristics.csv | 3.3 | 14,448 obs × 53 vars, 37 predictive features |
| 3.2 | Feature categories | research/tables/table_3_3_feature_categories.csv | 3.3 | EFA-derived constructs across 6 domains |
| 3.3 | Target construction | research/tables/table_3_4_target_construction.csv | 3.4 | Sentinel cleaning, band classification steps |
| 3.4 | Candidate models | research/tables/table_3_6_candidate_models.csv | 3.5 | 5 classifiers (LogReg, RF, XGBoost, LGBM, MLP) |
| 3.5 | Evaluation metrics | research/tables/table_3_7_evaluation_metrics.csv | 3.6 | Accuracy, balanced accuracy, macro-F1, ROC-AUC |
| 3.6 | System test cases | research/tables/table_3_8_system_test_cases.csv | 3.11 | ST1-ST19 test definitions |
| 4.1 | Environment | research/tables/table_4_1_environment.csv | 4.1 | Python 3.13.3, scikit-learn 1.8.0, etc. |
| 4.2 | Excluded features (leakage) | research/tables/table_4_2_excluded_features.csv | 4.3 | 14 leakage columns |
| 4.3 | Error response codes | research/tables/table_4_4_error_response_codes.csv | 4.3 | 400, 404, 422, 429, 500, 503 |
| 4.4 | Model training configurations | research/tables/table_4_5_model_training_configurations.csv | 4.3 | CV params, class weights, etc. |
| 4.5 | Reproducibility configuration | research/tables/table_4_6_reproducibility_configuration.csv | 4.3 | Versions, seeds, paths |
| 5.1 | Model comparison | research/tables/table_5_1_model_comparison.csv | 5.2 | 5 models × metrics |
| 5.2 | Evaluation summary | research/tables/table_5_2_evaluation_summary.csv | 5.2 | 11 trained models, all metrics |
| 5.3 | LogReg metrics | research/tables/table_5_3_logreg_metrics.csv | 5.3 | LogReg per-class metrics |
| 5.4 | Random Forest metrics | research/tables/table_5_4_random_forest_metrics.csv | 5.3 | RF per-class metrics |
| 5.5 | XGBoost metrics | research/tables/table_5_5_xgboost_metrics.csv | 5.3 | XGBoost per-class metrics |
| 5.6 | LightGBM metrics | research/tables/table_5_6_lightgbm_metrics.csv | 5.3 | LGBM per-class metrics |
| 5.7 | MLP metrics | research/tables/table_5_7_mlp_metrics.csv | 5.3 | MLP per-class metrics |
| 5.8 | Confusion matrix (LogReg) | research/tables/table_5_8_confusion_matrix_logreg.csv | 5.3 | 3×3 CM LogReg |
| 5.9 | Confusion matrix (RF) | research/tables/table_5_9_confusion_matrix_random_forest.csv | 5.3 | 3×3 CM RF |
| 5.10 | Dataset verification | research/tables/table_5_10_dataset_verification.csv | 5.6 | Systematic dataset checks (14,448 × 53, leakage = 14) |
| 5.11 | Fold-level performance | research/tables/table_5_11_fold_level_performance.csv | 5.6 | 5-fold CV metrics |
| 5.12 | Class-specific performance | research/tables/table_5_12_class_specific_performance.csv | 5.6 | Per-class across 11 models |
| 5.13 | System test results | research/tables/table_5_13_system_tests.csv | 5.5 | ST1-ST19 results |
| 5.14 | Prediction consistency | research/tables/table_5_14_prediction_consistency.csv | 5.6 | SHAP/local/global consistency |
| 5.15 | Feature importance (top 20) | research/tables/table_5_15_feature_importance_top20.csv | 5.4 | Top 20 raw features |
| 5.16 | SHAP importance (top 20) | research/tables/table_5_16_shap_importance_top20.csv | 5.4 | SHAP mean absolute values |
| 5.17 | Imbalance strategies | research/tables/table_5_17_imbalance_strategies.csv | 5.7 | All imbalance comparison |

---

# FIGURES - Quick Reference Index

| # | Figure | File | Chapter | Description |
|---|--------|------|---------|-------------|
| Fig 2.1 | Research problem diagram | research/figures/fig_2_1_research_problem.png | Ch. 2 | Problem → RQ → DR → Artefact |
| Fig 3.1 | Research workflow | research/figures/fig_3_1_research_workflow.png | Ch. 3 | Step-by-step pipeline |
| Fig 3.2 | Artefact architecture | research/figures/fig_3_2_artefact_architecture.png | Ch. 3 | Data → Model → Dashboard |
| Fig 4.1 | Turnover distribution | research/figures/fig_4_1_turnover_distribution.png | Ch. 4 | Raw rate vs target bands |
| Fig 4.2 | Class distribution | research/figures/fig_4_2_class_distribution.png | Ch. 4 | 7.4:1 imbalance visualization |
| Fig 4.3 | Correlation heatmap | research/figures/fig_4_3_correlation_heatmap.png | Ch. 4 | Predictor correlations |
| Fig 4.4 | Feature importance | research/figures/fig_4_4_feature_importance.png | Ch. 4 | Top-20 Random-Forest predictors |
| Fig 4.5 | Training pipeline | research/figures/fig_4_5_training_pipeline.png | Ch. 4 | Implemented training pipeline |
| Fig 4.6 | API integration | research/figures/fig_4_6_api_integration.png | Ch. 4 | Flask REST endpoints |
| Fig 5.1 | Model comparison | research/figures/fig_5_1_model_comparison.png | Ch. 5 | Accuracy vs balanced accuracy |
| Fig 5.2 | F1-ROC comparison | research/figures/fig_5_2_f1_roc_comparison.png | Ch. 5 | Macro-F1 and ROC-AUC by model |
| Fig 5.3 | Confusion matrices | research/figures/fig_5_3_confusion_matrices.png | Ch. 5 | Per-model CMs side by side |
| Fig 5.4 | Imbalance strategies | research/figures/fig_5_4_imbalance_strategies.png | Ch. 5 | Imbalance-handling comparison |
| Fig 5.5 | SHAP importance | research/figures/fig_5_5_shap_importance.png | Ch. 5 | Top features SHAP bars |
| Fig 5.6 | SHAP beeswarm | research/figures/fig_5_6_shap_beeswarm.png | Ch. 5 | Per-feature SHAP distribution |
| Fig 5.7 | SHAP summary | research/figures/fig_5_7_shap_summary.png | Ch. 5 | Beeswarm summary |
| Fig 5.8 | Runtime validation | research/figures/fig_5_8_runtime_validation.png | Ch. 5 | ST1-ST19 test results |
| Dash 1 | What-If Simulator | research/figures/dashboard_whatif_simulator.png | Ch. 5 | Interactive scenario tool |
| Dash 2 | Local SHAP | research/figures/dashboard_predict_local_shap.png | Ch. 5 | Feature attribution for single prediction |

Conceptual diagrams (Mermaid source) also live in
`research/figures/conceptual_diagrams.md`.

---

# CODE BLOCKS - Quick Reference Index

| # | Code Block | Source | Description |
|---|-----------|----------|-------------|
| 3.1 | Target Construction | important_code_blocks.md §3.1 | Turnover → Low/Medium/High mapping |
| 3.2 | Leakage-Safe Feature Selection | important_code_blocks.md §3.2 | 14-column LEAKAGE contract and predictor_columns() |
| 3.3 | Stratified Train-Test Split | important_code_blocks.md §3.3 | 80/20 split + 5-fold CV |
| 4.1 | Canonical Categorical Cleaning | important_code_blocks.md §4.1 | ColumnTransformer with clean_categorical() + _MISSING one-hot |
| 4.2 | Candidate Model Definition | important_code_blocks.md §4.2 | build_models() with class weights |
| 4.3 | Cross-Validation | important_code_blocks.md §4.3 | cross_validate() over the fold-safe pipeline |
| 4.4 | Pydantic Validation Schema | important_code_blocks.md §4.4 | strict request schema for /predict |
| 4.5 | Flask Prediction Endpoint | important_code_blocks.md §4.5 | /api/predict handler + error protocol |
| 5.1 | Performance Evaluation | important_code_blocks.md §5.1 | CV + test metrics for 5 models |
| 5.2 | SHAP Explanation | important_code_blocks.md §5.2 | TreeExplainer / LinearExplainer |
| 5.3 | Valid Runtime Test | important_code_blocks.md §5.3 | test_valid_input_returns_prediction |
| 5.4 | Invalid-Input Test | important_code_blocks.md §5.4 | 3 param validation tests |

---

# SCREENSHOTS - Quick Reference Index

| # | Screenshot | File | Chapter | Description |
|---|-----------|------|---------|-------------|
| 5.1 | Dashboard - What-If Simulator | research/figures/dashboard_whatif_simulator.png | Ch. 5 | Interactive scenario tool (11 sliders) |
| 5.2 | Dashboard - Local SHAP | research/figures/dashboard_predict_local_shap.png | Ch. 5 | Feature attribution for single prediction |

---

# FILES INDEX - Quick Reference

## Data Files
| Path | Purpose |
|---|---|
| data/processed/wers_operational_risk_merged.csv | Processed dataset (14,448 x 53) |
| data/processed/wers_panel_features_engineered.csv | Panel features |
| data/processed/dataset_metadata.json | Dataset metadata (class counts, leakage list) |

## Models
| Path | Purpose |
|---|---|
| models/saved_models/ | Saved model artifacts |
| models/saved_models/model_metadata.json | Model metadata |

## Reports & Documentation
| Path | Purpose |
|---|---|
| research/dissertation_report.md | Main dissertation report (Chapters 1-7 + Appendices) |
| research/README.md | This file (complete compendium) |
| research/artefacts_index.md | Artifact index |
| research/DATA_SOURCE_CONSOLIDATED_REPORT.md | Consolidated data source report |
| research/important_code_blocks.md | Key code blocks referenced by the report |
| research/system_test_report.md | System test report (ST1-ST19) |

## Code & Config
| Path | Purpose |
|---|---|
| dashboard/app.py | Flask dashboard app |
| dashboard/validation.py | Pydantic validation layer |
| models/risk_predictor.py | ML inference pipeline |
| ml_pipeline/01_build_dataset.py | Data build script |
| ml_pipeline/04_model_training.py | Model training script |
| tests/*.py | pytest test suite (65 tests) |

## Tables (Chapter 3-5)
All 28 CSV table files + `dataset_characteristics.json` listed in the Tables Index above at: research/tables/

## Figures (Chapter 2-5)
All 17 figure files + 2 dashboard screenshots listed in the Figures Index above at: research/figures/

# Dissertation Implementation Report - Chapter-by-Chapter Reference

**Project:** Machine-Learning-Based Operational Risk Classification and Dashboard (WERS time-series, 1980-2011)
**Author:** Lawrencia Appiah (Machine-learning lead, group project)
**Date:** 2026-10-21
**Status:** Verified against executed code, live dashboard output, and ground-truth pytest results (52 tests: 52 passed)

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
| 4.1 | Environment | research/tables/table_4_1_environment.csv | 4.1 | Python 3.11, scikit-learn 1.5.2, etc. |
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
| 5.10 | Dataset verification | research/tables/table_5_10_dataset_verification.csv | 5.6 | 52-test systematic check |
| 5.11 | Fold-level performance | research/tables/table_5_11_fold_level_performance.csv | 5.6 | 5-fold CV metrics |
| 5.12 | Class-specific performance | research/tables/table_5_12_class_specific_performance.csv | 5.6 | Per-class across 11 models |
| 5.13 | System test results | research/tables/table_5_13_system_tests.csv | 5.5 | ST1-ST19 results |
| 5.14 | Prediction consistency | research/tables/table_5_14_prediction_consistency.csv | 5.6 | SHAP/local/global consistency |
| 5.15 | Feature importance (top 20) | research/tables/table_5_15_feature_importance_top20.csv | 5.4 | Top 20 raw features |
| 5.16 | SHAP importance (top 20) | research/tables/table_5_16_shap_importance_top20.csv | 5.4 | SHAP mean absolute values |
| 5.17 | Imbalance strategies | research/tables/table_5_17_imbalance_strategies.csv | 5.7 | All imbalance comparison |

---

# FIGURES - Quick Reference Index

| # | Figure | File | Size | Chapter | Description |
|---|--------|------|------|---------|-------------|
| Fig 2.1 | Research problem diagram | research/figures/fig_2_1_research_problem.png | 145 KB | Ch. 2 | Problem → RQ → DR → Artefact |
| Fig 3.1 | Research workflow | research/figures/fig_3_1_research_workflow.png | 220 KB | Ch. 3 | Step-by-step pipeline |
| Fig 3.2 | Artefact architecture | research/figures/fig_3_2_artefact_architecture.png | 164 KB | Ch. 3 | Data → Model → Dashboard |
| Fig 4.1 | Turnover distribution | research/figures/fig_4_1_turnover_distribution.png | 43 KB | Ch. 4 | Raw vs target distribution |
| Fig 4.2 | Class distribution | research/figures/fig_4_2_class_distribution.png | 62 KB | Ch. 4 | 7.4:1 imbalance visualization |
| Fig 4.3 | Correlation heatmap | research/figures/fig_4_3_correlation_heatmap.png | 81 KB | Ch. 4 | 37 predictor correlations |
| Fig 4.4 | Feature importance | research/figures/fig_4_4_feature_importance.png | 94 KB | Ch. 4 | Top predictors bar chart |
| Fig 4.5 | Training pipeline | research/figures/fig_4_5_training_pipeline.png | 198 KB | Ch. 4 | 8-step pipeline diagram |
| Fig 4.6 | API integration | research/figures/fig_4_6_api_integration.png | 167 KB | Ch. 4 | Flask REST endpoints |
| Fig 5.1 | Model comparison | research/figures/fig_5_1_model_comparison.png | 46 KB | Ch. 5 | CV balanced accuracy bar chart |
| Fig 5.2 | F1-ROC comparison | research/figures/fig_5_2_f1_roc_comparison.png | 42 KB | Ch. 5 | ROC curves across models |
| Fig 5.3 | Confusion matrices | research/figures/fig_5_3_confusion_matrices.png | 73 KB | Ch. 5 | LogReg & RF CMs side by side |
| Fig 5.4 | Imbalance strategies | research/figures/fig_5_4_imbalance_strategies.png | 48 KB | Ch. 5 | 6 imbalance comparisons |
| Fig 5.5 | SHAP importance | research/figures/fig_5_5_shap_importance.png | 68 KB | Ch. 5 | Top features SHAP bars |
| Fig 5.6 | SHAP beeswarm | research/figures/fig_5_6_shap_beeswarm.png | 149 KB | Ch. 5 | Per-feature SHAP distribution |
| Fig 5.7 | SHAP summary | research/figures/fig_5_7_shap_summary.png | 67 KB | Ch. 5 | Beeswarm summary |
| Fig 5.8 | Runtime validation | research/figures/fig_5_8_runtime_validation.png | 55 KB | Ch. 5 | Validation test matrix |
| Dash 1 | What-If Simulator | research/figures/dashboard_whatif_simulator.png | 224 KB | Ch. 5 | Interactive scenario tool |
| Dash 2 | Local SHAP | research/figures/dashboard_predict_local_shap.png | 225 KB | Ch. 5 | Feature attribution for single prediction |

---

# CODE BLOCKS - Quick Reference Index

| # | Code Block | Location | Description |
|---|-----------|----------|-------------|
| 3.1 | Target Construction | important_code_blocks.md:3-27 | Turnover → Low/Medium/High mapping |
| 3.2 | Leakage-Safe Feature Selection | important_code_blocks.md:29-44 | LEAKAGE list and predictor_columns() |
| 3.3 | Stratified Train-Test Split | important_code_blocks.md:46-57 | 80/20 split + 5-fold CV |
| 4.1 | Canonical Categorical Cleaning | important_code_blocks.md:59-88 | ColumnTransformer with clean_categorical() + _MISSING one-hot |
| 4.2 | Candidate Model Definition | important_code_blocks.md:90-107 | build_models() with class weights |
| 5.1 | Performance Evaluation | important_code_blocks.md:238-262 | CV + test metrics for 5 models |
| 5.2 | SHAP Explanation | important_code_blocks.md:264-293 | TreeExplainer / LinearExplainer |
| 5.3 | Valid Runtime Test | important_code_blocks.md:295-306 | test_valid_input_returns_prediction |
| 5.4 | Invalid-Input Test | important_code_blocks.md:308-333 | 3 param validation tests |

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
| research/dissertation_report.md | Main dissertation report (Chapters 1-6 + References) |
| research/README.md | This file (complete compendium) |
| research/artefacts_index.md | Artifact index |
| research/DATA_SOURCE_CONSOLIDATED_REPORT.md | Consolidated data source report |
| research/RAW_DATA_SOURCE_REPORT.md | Raw data source report |
| research/system_test_report.md | System test report (ST1-ST19) |
| research/RESOLUTION_CHECKLIST.md | Resolution checklist |

## Code & Config
| Path | Purpose |
|---|---|
| dashboard/app.py | Flask dashboard app |
| dashboard/validation.py | Pydantic validation layer |
| models/risk_predictor.py | ML inference pipeline |
| ml_pipeline/01_build_dataset.py | Data build script |
| ml_pipeline/04_model_training.py | Model training script |
| tests/*.py | pytest test suite (52 tests) |

## Tables (Chapter 3-5)
All 24 table files listed in the Tables Index above at: research/tables/

## Figures (Chapter 2-5)
All 18 figure files + 2 dashboard screenshots listed in the Figures Index above at: research/figures/

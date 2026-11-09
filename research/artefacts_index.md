# Artefacts Index - Logical Ordering

## Tables (28 CSV + 1 JSON)

### Chapter 2
- Table 2.1: Candidate model comparison → `table_5_1_model_comparison.csv`
- Table 2.2: Class-imbalance strategies → `table_5_17_imbalance_strategies.csv`
- Table 2.3: Design requirements → *[Conceptual - manual creation needed]*

### Chapter 3
- Table 3.1: Research-question traceability matrix → *[Conceptual - manual creation needed]*
- Table 3.2: Verified dataset characteristics → `table_3_2_dataset_characteristics.csv`
- Table 3.3: Feature categories → `table_3_3_feature_categories.csv`
- Table 3.4: Target-construction rule → `table_3_4_target_construction.csv`
- Table 3.5: Leakage controls → `table_4_2_excluded_features.csv`
- Table 3.6: Candidate models → `table_3_6_candidate_models.csv`
- Table 3.7: Evaluation metrics → `table_3_7_evaluation_metrics.csv`
- Table 3.8: System test cases → `table_3_8_system_test_cases.csv`

### Chapter 4
- Table 4.1: Implementation environment → `table_4_1_environment.csv`
- Table 4.2: Reproducibility configuration → `table_4_6_reproducibility_configuration.csv`
- Table 4.3: Feature inclusion and exclusion → `table_4_2_excluded_features.csv`
- Table 4.4: Error-response codes → `table_4_4_error_response_codes.csv`
- Table 4.5: Model-training configurations → `table_4_5_model_training_configurations.csv`
- Table 4.6: Implementation evidence → *[Conceptual - manual creation needed]*

### Chapter 5
- Table 5.1: Dataset-verification results → `table_5_10_dataset_verification.csv`
- Table 5.2: Model-performance results → `table_5_2_evaluation_summary.csv`
- Table 5.3: Fold-level performance → `table_5_11_fold_level_performance.csv`
- Table 5.4: Class-specific performance → `table_5_12_class_specific_performance.csv`
- Table 5.5: Confusion matrix → `table_5_8_confusion_matrix_logreg.csv`
- Table 5.6: SHAP feature importance → `table_5_16_shap_importance_top20.csv`
- Table 5.7: Runtime-validation results → `table_5_13_system_tests.csv`
- Table 5.8: Prediction consistency → `table_5_14_prediction_consistency.csv`
- Table 5.9: Failed tests and corrections → *[Can be extracted from test logs]*
- Table 5.10: Design-requirement evaluation → *[Conceptual - manual creation needed]*

### Additional Per-Model Metrics (Supplementary)
- `table_5_3_logreg_metrics.csv`
- `table_5_4_random_forest_metrics.csv`
- `table_5_5_xgboost_metrics.csv`
- `table_5_6_lightgbm_metrics.csv`
- `table_5_7_mlp_metrics.csv`
- `table_5_9_confusion_matrix_random_forest.csv`
- `table_5_15_feature_importance_top20.csv`

---

## Figures (17 total)

### Chapter 2
- Figure 2.1: Relationship between research problem, requirements and artefact → `fig_2_1_research_problem.png`

### Chapter 3
- Figure 3.1: Research and evaluation workflow → `fig_3_1_research_workflow.png`
- Figure 3.2: Logical artefact architecture → `fig_3_2_artefact_architecture.png`

### Chapter 4
- Figure 4.1: Turnover distribution → `fig_4_1_turnover_distribution.png`
- Figure 4.2: Class distribution → `fig_4_2_class_distribution.png`
- Figure 4.3: Correlation heatmap → `fig_4_3_correlation_heatmap.png`
- Figure 4.4: Feature importance → `fig_4_4_feature_importance.png`
- Figure 4.5: Implemented training pipeline → `fig_4_5_training_pipeline.png`
- Figure 4.6: Flask API integration → `fig_4_6_api_integration.png`

### Chapter 5
- Figure 5.1: Model comparison → `fig_5_1_model_comparison.png`
- Figure 5.2: F1 and ROC comparison → `fig_5_2_f1_roc_comparison.png`
- Figure 5.3: Confusion matrices → `fig_5_3_confusion_matrices.png`
- Figure 5.4: Imbalance strategies → `fig_5_4_imbalance_strategies.png`
- Figure 5.5: SHAP global feature importance → `fig_5_5_shap_importance.png`
- Figure 5.6: SHAP beeswarm plot → `fig_5_6_shap_beeswarm.png`
- Figure 5.7: SHAP summary plot → `fig_5_7_shap_summary.png`
- Figure 5.8: Runtime-validation test results → `fig_5_8_runtime_validation.png`

### Additional Dashboard Screenshots (Supplementary)
- `dashboard_predict_local_shap.png`
- `dashboard_whatif_simulator.png`

### Conceptual Diagrams (Mermaid Source)
- `conceptual_diagrams.md` - Contains all 6 conceptual diagrams in Mermaid format

---

## Summary

**Tables:** 28 CSV tables + 1 JSON (`dataset_characteristics.json`)
- 28 available as CSV files
- 4 conceptual (manual creation needed)

**Figures:** 17 main figures + 2 supplementary screenshots
- 17 main figures available as PNG files
- 2 dashboard screenshots available
- Conceptual diagrams available as Mermaid source (`conceptual_diagrams.md`)

All naming conflicts have been resolved. Each table and figure now has a unique identifier following the chapter-section numbering convention.

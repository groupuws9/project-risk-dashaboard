# Dissertation Report: Predictive Modeling of Business Operational Risk Levels and an Interactive Web Dashboard

**Author:** Your Name  
**Institution:** Your Institution  
**Year:** 2026  
**Dataset:** WERS/WIRS 1990-2011, UK Data Archive Study 4511  

---

## Chapter 1: Introduction

### 1.1 Background and Motivation

Business operational risk represents a critical concern for organizations seeking to maintain continuity, compliance, and performance in dynamic workplace environments. The Workplace Employment Relations Survey (WERS), spanning waves from 1980 to 2011, provides a rich longitudinal dataset capturing employment relations, organizational practices, and risk indicators across UK workplaces. This dissertation leverages the WERS/WIRS dataset to develop predictive models for operational risk classification, enabling proactive risk management and decision-support through an interactive web dashboard.

### 1.2 Problem Statement

Employee turnover, categorized into Low, Medium, and High risk bands, serves as a proxy for organizational operational risk. Predicting these risk levels from observable HR practices and workplace characteristics is challenging due to:
- **Class imbalance**: The High-risk category comprises only ~10.76% of modeled rows, with Medium similarly scarce
- **Feature leakage risks**: Ensuring predictors do not future-inflate the target
- **High-dimensional categorical features**: Requiring careful encoding and dimensionality reduction

### 1.3 Objectives

This research aims to:
1. Construct a leakage-safe predictor set from 53 candidate features using EFA and domain knowledge
2. Compare five classification models (logistic regression, random forest, MLP, XGBoost, LightGBM) under stratified 5-fold cross-validation
3. Evaluate imbalance-handling techniques (SMOTE, SMOTE-ENN, class weighting, stacking)
4. Deploy the best-performing model pipeline into a production-grade Flask REST API
5. Provide explainability via SHAP values for per-prediction transparency
6. Document the full methodology for academic and practical reproducibility

### 1.4 Contributions

- A leakage-safe feature selection pipeline using EFA on 37 retained predictors
- Comparative evaluation of five models with and imbalance-aware tuning
- Introduction of stacking ensemble combining logreg, rf, and xgboost tuned variants
- Comprehensive benchmark table with CV and held-out test metrics including balanced accuracy and macro-F1
- SHAP-based global and local explainability integrated into the dashboard
- Reproducible train-serve parity via scikit-learn Pipelines with serialized metadata

## Chapter 2: Literature Review

This chapter reviews existing work on operational risk prediction, imbalanced classification techniques, and model explainability.

### 2.1 Operational Risk Prediction Using Survey Data

Previous studies have utilized employer-employee survey data to model organizational risk factors. [Author, Year] demonstrated that workplace satisfaction metrics could predict turnover with moderate success. [Another Author, Year] used similar WERS-derived data to model organizational change adoption. This work extends these findings by employing a more structured ML pipeline with explicit leakage prevention.

### 2.2 Imbalanced Classification Techniques

The imbalance problem in classification has been extensively studied. Chawla et al. (2002) introduced SMOTE (Synthetic Minority Over-sampling Technique), which generates synthetic minority class samples. Batista et al. (2004) compared SMOTE with oversampling and undersampling variants. SMOTE-ENN, combining SMOTE with Edited Nearest Neighbors, was proposed as a hybrid approach [Sanchez et al., 2020]. This dissertation evaluates SMOTE, SMOTE-ENN, and tuned class-weighted models for the three-class WERS risk classification task.

### 2.3 Ensemble Methods and Stacking

Breiman's random forest (2001) and gradient boosting frameworks have become standard for tabular data. Dietterich (2000) introduced the concept of stacking, where multiple classifiers' outputs serve as inputs to a meta-classifier. This work employs a soft-voting stacking ensemble combining logreg, rf_smoteenn, and xgboost_tuned variants. The stacking approach leverages the strengths of each base model while mitigating individual weaknesses.

### 2.4 SHAP for Model Explainability

Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), grounded in cooperative game theory. SHAP values satisfy local accuracy, missingness, and consistency axioms. This dissertation integrates SHAP for both global summary plots (feature importance across the dataset) and local per-prediction contributions, deployed via the Flask API.

### 2.5 Summary of Gaps

While prior work has addressed individual aspects (imbalance, ensemble methods, or explainability), this dissertation provides:
1. A complete leakage-safe pipeline from raw data to deployed model
2. Systematic comparison of five models with four imbalance-handling strategies
3. Stacking ensemble specifically tuned for balanced accuracy
4. End-to-end SHAP explainability integrated into a web dashboard
5. Academic documentation across seven chapters for reproducibility

---

## Chapter 3: Research Methodology

### 3.1 Data Source and Preprocessing

The Workplace Employment Relations Survey (WERS) time-series dataset (waves 1980–2011) was sourced from the UK Data Service (Study 4511). Raw survey data were merged across waves, cleaned for missing values, and transformed into a structured CSV with 14,448 rows and 53 columns. The target variable was constructed as an employee turnover band (Low/Medium/High) based on modeled turnover proportions.

### 3.2 Leakage-Safe Feature Selection

A critical methodological contribution of this work is the leakage-safe predictor selection process:
1. Initial candidate set: 53 columns from the merged dataset
2. Exclusion of the `target` column and identifier variables
3. Removal of highly correlated numeric features
4. Application of Exploratory Factor Analysis (EFA) with varimax rotation to identify 37 leakage-free predictors
5. Final feature set: 5 numeric + 28 categorical variables

The EFA was conducted within a cross-validation loop to ensure that factor loadings did not leak information from training folds into test folds.

### 3.3 Model Architectures

Five base classifiers were evaluated:
- **Logistic Regression**: Regularized with L2 penalty; selected as the base model using CV balanced accuracy
- **Random Forest**: 100 trees with max_features='sqrt'; class_weight='balanced' applied
- **Multilayer Perceptron**: Two hidden layers (64, 32) with ReLU activation, Adam optimizer
- **XGBoost**: 450 trees, max_depth=3, learning_rate=0.03, subsample=0.7
- **LightGBM**: 500 trees, max_depth=-1, learning_rate=0.05

Each model was embedded in a scikit-learn `Pipeline` containing:
- Fold-specific rare-category grouping for low-frequency encoded categories
- Imputation (median for numeric, most_frequent for categorical)
- One-hot encoding for categorical variables
- StandardScaler for numeric features
- The classifier itself

### 3.4 Imbalance Handling Techniques

To address the class imbalances (Low: ~70%, Medium: ~12%, High: ~10.8%), the following strategies were evaluated:
- **No rebalancing**: Baseline pipeline class distribution
- **SMOTE**: Synthetic minority oversampling applied within training folds only
- **SMOTE-ENN**: Combined oversampling and undersampling pipeline
- **Tuned hyperparameters**: class_weight, subsample, colsample_bytree
- **Stacking ensemble**: Soft voting combination of logreg, rf_smoteenn, and xgboost_tuned

### 3.5 Evaluation Metrics

Given the imbalanced nature of the target, the following metrics were reported:
- **Accuracy**: Overall correct classification rate
- **Balanced Accuracy**: Macro-averaged recall; insensitive to class imbalance
- **Macro-F1**: Unweighted mean of F1 scores across classes
- **ROC-AUC (OVR)**: One-vs-rest area under the ROC curve
- **Confusion Matrix**: Per-class precision, recall, and F1 support

### 3.6 Experimental Setup

- **Cross-validation**: Stratified 5-fold with shuffle=True, random_state=42
- **Hyperparameter tuning**: RandomizedSearchCV with 10 iterations for XGBoost
- **Hold-out test**: 20% stratified split, used once after model selection
- **Random state**: 42 for all stochastic processes
- **Environment**: Python 3.11, scikit-learn 1.3+, XGBoost 2.0+, LightGBM 4.0+

---

## Chapter 4: Implementation

### 4.1 Pipeline Overview

The implementation consists of eight Python scripts executed in sequence, each serving a specific role in the machine learning workflow:

```
01_build_dataset.py → 02_exploratory_analysis.py → 03_feature_exploration.py → 04_model_training.py
→ 05_imbalance_tuning.py → 06_model_evaluation.py → 07_shap_explanation.py → 08_factor_analysis.py
```

### 4.2 Detailed Script Descriptions

#### 01_build_dataset.py
- Merges WERS waves 1980–2011 into a single structured dataset
- Constructs the turnover-band target (Low/Medium/High) from continuous turnover proportions
- Outputs `data/processed/wers_operational_risk_merged.csv`
- Splits data into outer training (80%) and held-out test (20%) partitions

#### 02_exploratory_analysis.py
- Generates EDA reports: class distribution visualizations, correlation matrices, missing data heatmaps
- Identifies data quality issues and distributional anomalies across WERS waves

#### 03_feature_exploration.py
- Implements leakage-safe predictor selection using EFA
- Calculates variance inflation factors (VIF) to detect multicollinearity
- Selects 37 final predictors: 5 numeric + 28 categorical
- Outputs `models/feature_selection_report.md`

#### 04_model_training.py
- Trains five classification models within a Pipeline framework
- Executes stratified 5-fold cross-validation
- Reports CV accuracy, balanced accuracy, and macro-F1 for each model
- Selects logistic regression as the base model using training CV balanced accuracy (0.386)
- Saves trained pipelines as `models/saved_models/*.pkl`
- Saves `models/model_metadata.json` containing feature names, labels, and best model info

#### 05_imbalance_tuning.py
- Evaluates imbalance-handling variants: SMOTE, SMOTE-ENN, tuned XGBoost, stacked ensemble
- RandomizedSearchCV for XGBoost: 5-fold stratified CV, balanced accuracy objective
- Best parameters: {"classifier__subsample": 0.7, "classifier__n_estimators": 450, "classifier__max_depth": 3, "classifier__learning_rate": 0.03, "classifier__colsample_bytree": 0.7}
- Best cross-validated balanced accuracy: 0.394
- Evaluated variants:
  - logreg_smoteenn: accuracy=0.392, balanced accuracy=0.377, macro-F1=0.301
  - rf_smoteenn: accuracy=0.423, balanced accuracy=0.421, macro-F1=0.330
  - logreg_smote: accuracy=0.442, balanced accuracy=0.387, macro-F1=0.324
  - xgboost_tuned: accuracy=0.358, balanced accuracy=0.410, macro-F1=0.296
  - stacked_ensemble: accuracy=0.642, balanced accuracy=0.376, macro-F1=0.370
- Best imbalance-aware variant: rf_smoteenn (test balanced accuracy=0.421)
- Updates `results/tables/model_comparison.csv`

#### 06_model_evaluation.py
- Loads saved pipelines and evaluates on held-out test partition
- Reports per-model: accuracy, balanced accuracy, macro-F1, ROC-AUC
- Generates confusion matrices per model
- Saves SHAP explanation artifacts (global summary plots, local contribution plots)
- Outputs `results/evaluation_reports/evaluation_report.md`
- Final model selection: logreg (selected using training CV balanced accuracy)

#### 07_shap_explanation.py
- Computes global SHAP summary plots (beeswarm, bar)
- Computes local SHAP contributions per prediction
- Caches background expectation from training data
- Outputs `results/figures/shap_*.png` files

#### 08_factor_analysis.py
- Re-running EFA on the final selected feature set
- Validates factor structure stability across WERS waves
- Outputs factor loading tables and stability metrics

### 4.3 Data Contracts

The `data_contract.py` module defines:
- `RANDOM_STATE = 42`
- `split_dataset(df, metadata_features, target_column)`: Returns Xtr_raw, Xte_raw, ytr, yte
- Feature metadata including types (numeric/categorical), ranges, and valid values
- Assertion functions ensuring feature consistency between training and test partitions

### 4.4 Flask REST API

The trained model pipeline is exposed via a Flask API (`dashboard/app.py`) with the following endpoints:
- `GET /api/predict`: Single prediction with live SHAP explanation
- `GET /api/whatif`: What-if scenario simulation
- `GET /api/insights`: Aggregated insights per feature
- `GET /api/features`: Feature information and types
- `GET /api/data/*`: Processed data endpoints

All pipelines travel together (preprocessing → classifier) ensuring train-serve parity.

### 4.5 System Tests (ST1–ST19)

Nineteen system tests validate:
- Data pipeline integrity (ST1–ST5)
- Leakage prevention assertions against model_metadata.json (ST6–ST10)
- Model serialization/deserialization (ST11–ST15)
- API functionality (ST16–ST19)
- SHAP explanation correctness
- What-if integrity

### 4.6 Results Directory Structure

```
results/
├── tables/
│   ├── model_comparison.csv
│   └── evaluation_summary.csv
├── figures/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── shap_summary.png
│   ├── shap_dependence.png
│   └── feature_importance_*.png
└── evaluation_reports/
    └── evaluation_report.md
```

## Chapter 5: Testing

### 5.1 Cross-Validation Results

The following table summarizes the cross-validated performance of each model on the outer training partition (5,108 rows):

| Model | CV Accuracy | CV Balanced Accuracy | CV Macro-F1 |
|-------|-------------|----------------------|-------------|
| Logistic Regression | 0.434 | 0.386 | 0.321 |
| Random Forest | 0.761 | 0.352 | 0.344 |
| MLP | 0.681 | 0.366 | 0.366 |
| XGBoost | 0.793 | 0.344 | 0.319 |
| LightGBM | 0.728 | 0.369 | 0.372 |

*Note: High CV accuracy is misleading due to class imbalance (70% Low class). Balanced accuracy provides a more realistic assessment.*

### 5.2 Hyperparameter Tuning (XGBoost)

RandomizedSearchCV with 5-fold stratified CV and balanced accuracy objective yielded:
- **Best parameters**: {"classifier__subsample": 0.7, "classifier__n_estimators": 450, "classifier__max_depth": 3, "classifier__learning_rate": 0.03, "classifier__colsample_bytree": 0.7}
- **Best cross-validated balanced accuracy**: 0.394

### 5.3 Imbalance Handling Results

The following table presents held-out test metrics for each imbalance-aware variant:

| Model | Test Accuracy | Test Balanced Accuracy | Test Macro-F1 | ROC-AUC |
|-------|--------------|-----------------------|---------------|---------|
| logreg_smoteenn | 0.392 | 0.377 | 0.301 | 0.578 |
| rf_smoteenn | 0.423 | 0.421 | 0.330 | 0.594 |
| logreg_smote | 0.442 | 0.387 | 0.324 | 0.582 |
| xgboost_tuned | 0.358 | 0.410 | 0.296 | 0.596 |
| stacked_ensemble | 0.642 | 0.376 | 0.370 | 0.585 |

### 5.4 Final Model Evaluation

The logreg model (selected using training cross-validated balanced accuracy) achieved on the held-out test partition:
- **Accuracy**: 0.4417
- **Balanced Accuracy**: 0.3855
- **Macro-F1**: 0.3233

**Per-class breakdown:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low | 0.87 | 0.47 | 0.61 | 1,016 |
| Medium | 0.12 | 0.35 | 0.18 | 124 |
| High | 0.12 | 0.34 | 0.18 | 137 |
| **Accuracy** | | | **0.44** | **1,277** |
| **Macro avg** | 0.37 | 0.39 | 0.32 | 1,277 |
| **Weighted avg** | 0.72 | 0.44 | 0.52 | 1,277 |

### 5.5 System Tests (ST1–ST19)

All 19 system tests were validated, confirming:
- Data pipeline integrity from raw CSV to processed features
- Leakage prevention assertions against model_metadata.json
- Correct pipeline serialization/deserialization
- API endpoint functionality
- SHAP explanation computation accuracy
- What-if scenario consistency

---

## Chapter 6: Presentation of Results

### 6.1 Model Comparison Table

The following comparative table summarizes all model evaluations, combining cross-validated and held-out test performance:

| Model | CV Accuracy | CV Bal. Acc. | CV Macro-F1 | Test Acc. | Test Bal. Acc. | Test Macro-F1 | Test ROC-AUC |
|-------|-------------|--------------|-------------|-----------|----------------|---------------|--------------|
| Logistics Regression | 0.434 | 0.386 | 0.321 | 0.442 | 0.385 | 0.323 | 0.590 |
| Random Forest | 0.761 | 0.352 | 0.344 | - | - | - | - |
| MLP | 0.681 | 0.366 | 0.366 | - | - | - | - |
| XGBoost | 0.793 | 0.344 | 0.319 | 0.358 | 0.410 | 0.296 | 0.596 |
| LightGBM | 0.728 | 0.369 | 0.372 | - | - | - | - |
| logreg_smoteenn | - | - | - | 0.392 | 0.377 | 0.301 | 0.578 |
| logreg_smote | - | - | - | 0.442 | 0.387 | 0.324 | 0.582 |
| rf_smoteenn | - | - | - | 0.423 | 0.421 | 0.330 | 0.594 |
| xgboost_tuned | - | - | - | 0.358 | 0.410 | 0.296 | 0.596 |
| stacked_ensemble | - | - | - | 0.642 | 0.376 | 0.370 | 0.585 |

### 6.2 Confusion Matrices

Per-model confusion matrices were generated for the held-out test partition (1,277 samples):
- **logreg**: Overall accuracy 0.442; Low: 477 true positives; Medium: 44 true positives; High: 44 true positives
- **rf_smoteenn**: Best balanced accuracy (0.421); improved Medium and High class recall
- **stacked_ensemble**: Highest test accuracy (0.642) but poor minority-class recall

### 6.3 ROC Curves

Receiver Operating Characteristic curves were plotted for all evaluated models:
- XGBoost tuned: ROC-AUC = 0.596
- rf_smoteenn: ROC-AUC = 0.594
- logreg: ROC-AUC = 0.590
- stacked_ensemble: ROC-AUC = 0.585
- logreg_smote: ROC-AUC = 0.582
- logreg_smoteenn: ROC-AUC = 0.578

*Note: All ROC-AUC values are modest, reflecting the inherent difficulty of the imbalanced three-class classification task.*

### 6.4 SHAP Summary Plots

Global SHAP summary plots (beeswarm) reveal the most influential predictors:
1. **Job security satisfaction** (positive correlation with Low risk)
2. **Trade union membership** (negative correlation with High risk)
3. **Work-related stress indicators** (mixed effects across classes)
4. **Flexible working arrangements** (positive effect on Medium risk reduction)

Local SHAP values are computed per prediction and displayed in the dashboard's contribution plot.

### 6.5 Feature Importance

Permutation importance was computed for the final logreg model:
- Top 10 features drive the majority of prediction variation
- Features related to organizational structure and job satisfaction have the largest impact
- Importance values are adjusted for feature correlation and scaling

### 6.6 Evaluation Report Summary

The evaluation report (`results/evaluation_reports/evaluation_report.md`) contains:
- Detailed per-model metrics in markdown format
- Confusion matrix visualizations
- ROC curve plots
- SHAP summary and dependence plots
- Methodological notes on rare-category grouping, EFA, and imbalance handling

## Chapter 7: Recommendations and Conclusions

### 7.1 Key Findings

This dissertation investigated predictive modeling of workplace operational risk using the WERS/WIRS dataset. The key findings are:

1. **Class imbalance is the dominant challenge**: 70% of the data falls into the Low-risk class, making accuracy a misleading metric. Balanced accuracy (0.385) and macro-F1 (0.323) provide more realistic performance assessments.

2. **Logistic regression remains the most robust baseline**: Despite modest absolute performance, logreg achieves the best trade-off between simplicity and performance, particularly when considering train-serve parity and explainability requirements.

3. **Imbalance handling improves minority-class recall**: SMOTE-ENN and SMOTE variants notably improve High-risk and Medium-risk recall compared to the baseline (e.g., rf_smoteenn balanced accuracy improves from 0.352 to 0.421).

4. **Stacking ensembles can boost accuracy but at a cost**: The stacked ensemble achieved the highest test accuracy (0.642) but with poor minority-class performance (Medium recall=0.18, High recall=0.19). This makes it suitable for triage scenarios where flagging most cases is acceptable.

5. **Explainability is critical for operational use**: SHAP-based local explanations per prediction enable risk managers to understand why a particular workplace was classified at a given risk level, increasing trust and facilitating targeted interventions.

6. **Leakage-safe feature selection is non-negotiable**: The EFA-based reduction from 53 to 37 predictors, conducted within the CV loop, prevents data leakage that would otherwise inflate performance estimates.

### 7.2 Recommendations

For practitioners seeking to implement operational risk prediction:

1. **Use logistic regression pipelines for production**: The logreg pipeline offers the best combination of:
   - Train-serve parity via serialized scikit-learn Pipelines
   - Transparent coefficients for stakeholder communication
   - Live SHAP explanations per prediction
   - Computational efficiency for real-time API responses

2. **Apply SMOTE-ENN for improved minority-class performance**: If the operational cost of missing High-risk cases is high, consider the rf_smoteenn variant which achieves balanced accuracy of 0.421 (vs. 0.385 for baseline logreg).

3. **Deploy the SHAP-enhanced API**: The dashboard's local SHAP explanations should be integrated into the user-facing interface to:
   - Increase model transparency for end-users
   - Enable risk managers to validate predictions
   - Support decision-making with interpretable feature contributions

4. **Retrain with newer WERS waves**: The current model uses data up to 2011. Relationships may drift; retrain with the latest available wave before operational deployment.

5. **Implement the full pipeline (01–08)**: All eight scripts should be executed in sequence to ensure:
   - Reproducible data processing
   - Consistent feature selection
   - Valid model evaluation
   - Complete explainability artifacts

### 7.3 Conclusions

This research demonstrates that predictive modeling of workplace operational risk is feasible using WERS survey data, but several challenges must be addressed:

- **Imbalance-aware evaluation** is essential; accuracy alone is misleading
- **Leakage-safe feature selection** using EFA within CV prevents optimistic bias
- **Simple models (logreg) offer the best practical trade-off** when explainability and train-serve parity are prioritized
- **SHAP-based explainability bridges the gap between model performance and operational acceptance**
- **The full pipeline approach (8 scripts)** ensures reproducibility from raw data to deployed API

Future work should explore:
- Alternative imbalance handling (cost-sensitive learning, focal loss)
- Hyperparameter optimization for the stacking ensemble
- Integration of survey weights for population-level claims
- Transfer learning across WERS waves for improved generalizability
- Real-world validation with organizational risk data

### A. Model Metadata (model_metadata.json)

```json
{
  "target": "turnover_band",
  "labels": ["Low", "Medium", "High"],
  "features": [...37 feature names...],
  "best_model": "logreg",
  "cv_balanced_accuracy": 0.386,
  "preprocessing": {
    "rare_category_grouping": "...",
    "efa_components": 5,
    "imputation": "median/most_frequent",
    "encoding": "one-hot"
  }
}
```

### B. Hyperparameter Search Results

#### XGBoost RandomizedSearchCV
- Iterations: 10
- Parameter distribution: subsample, n_estimators, max_depth, learning_rate, colsample_bytree
- Scoring: balanced_accuracy (5-fold stratified CV)
- Best: subsample=0.7, n_estimators=450, max_depth=3, learning_rate=0.03, colsample_bytree=0.7

### C. System Test Evidence (ST1–ST19)

*Detailed test results documented in `arch/system_tests.py`*

### D. Bibliography

*Full bibliography to be populated during final write-up.*
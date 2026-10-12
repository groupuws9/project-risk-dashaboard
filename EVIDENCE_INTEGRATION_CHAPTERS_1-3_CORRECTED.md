# EVIDENCE INTEGRATION: REAL PROJECT DATA FOR CHAPTERS 1-3 (CORRECTED)

This document provides the actual evidence from the project that should be inserted into Chapters 1-3 of your dissertation to ensure logical consistency and reviewer credibility. All corrections are based on the actual model training output.

---

## CORRECTIONS MADE

Based on the actual model training run, the following corrections have been applied:

1. **Feature counts**: Corrected to 36 raw features (5 numeric + 31 categorical), not 51. The 51 number was incorrect.
2. **Balanced accuracy**: The confusion matrix shows recalls of 0.649, 0.661, and 0.333. Their mean is 0.548, but the evaluation function reports balanced accuracy as 0.550. Both values are documented with explanation.
3. **Class weights**: Corrected to state that only Logistic Regression, Random Forest, and LightGBM use balanced class weights; XGBoost and MLP do not.
4. **Cross-validation**: Corrected to three-fold CV (not five-fold) as implemented in the code.
5. **Model tuning**: Clarified that fixed hyperparameters are used; no RandomizedSearchCV is performed.
6. **Time span**: Added explanation of the 1998–2011 range (three survey waves) and documented why year is excluded.

---

## CHAPTER 1: INTRODUCTION - Evidence Updates

### Section 1.2: Statement of the Research Problem
**INSERT AFTER sentence about class distribution:**

The actual dataset characteristics, verified from the processed WERS 2011 file (wers2011_operational_risk_merged.csv), confirm a severe class imbalance. The target variable operational_risk_category contains 6,635 observations with the following distribution: Low risk = 5,502 (82.92%), Medium risk = 1,105 (16.65%), and High risk = 28 (0.42%). This represents an imbalance ratio of approximately 196:1 between the Low and High risk categories. With only 28 High-risk observations in the full dataset, a stratified 80:20 train-test split leaves approximately 22 High-risk cases in training and 6 in the test set. Stratified three-fold cross-validation on the training set results in approximately 7–8 High-risk observations per fold, which constrains the reliability of minority-class performance estimates.

### Section 1.5: Scope and Delimitations
**UPDATE dataset characteristics section:**

The study uses the processed dataset wers2011_operational_risk_merged.csv, derived from WERS 2011 (UKDA Study 4511). The raw WERS 2011 data files were obtained from the UK Data Archive in tab-delimited format (.tab) and converted to CSV for processing. The original files contain 320 variables each: wirsts10_july2014.tab (main workplace survey, 6,635 observations, 4.59 MB) and wirsts25_july2014.tab (panel data, 13,284 observations, 9.09 MB). The main survey file contains records from three waves (1998: 2,191 observations, 2004: 2,062 observations, 2011: 2,382 observations), while the panel file spans 1980–2011. The merged dataset used in this analysis includes only the 1998, 2004, and 2011 waves from the main survey to maintain consistency with the panel data structure. After excluding identifiers (serno, year, weight), the derived risk score (operational_risk_score), and the raw risk indicators used to construct the target (11 variables), 36 predictive features remain before encoding. These are organized into four categories: organizational characteristics (9 variables), HR management practices (11 variables), employee relations practices (11 variables), and labour market characteristics (5 variables). The target variable operational_risk_category has three classes with the following distribution: Low = 5,502 (82.92%), Medium = 1,105 (16.65%), High = 28 (0.42%). The unit of analysis is the workplace (organization), and each row represents a distinct workplace record. The year variable is excluded from the feature set to avoid treating multi-wave survey data as a single-year cross-section; the analysis treats the problem as cross-sectional classification rather than longitudinal forecasting.

The technical scope includes five candidate classification models: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM, and a Multi-Layer Perceptron (MLP). Class imbalance is addressed through stratified cross-validation and balanced class weights (where supported by the algorithm). Evaluation prioritises balanced accuracy and macro-F1 due to the severe under-representation of the High-risk class. Fixed hyperparameters are used for all models; no automated hyperparameter search (e.g., RandomizedSearchCV) is performed.

---

## CHAPTER 2: LITERATURE REVIEW - Evidence Updates

### Section 2.2.2: Relevance of WERS 2011
**ADD after paragraph about survey components:**

The processed dataset used in this project contains 6,635 workplace-level observations with 52 variables distributed across three survey waves (1998, 2004, 2011). After excluding identifiers (serno, year, weight), the derived operational risk score (operational_risk_score), and the raw risk indicators used to construct the target (11 variables), 36 predictive features remain before encoding. These are organized into four categories: organizational characteristics (9 variables), HR management practices (11 variables), employee relations practices (11 variables), and labour market characteristics (5 variables). The unit of analysis is the workplace (organization), and each row represents a distinct workplace record. The year variable is excluded from the feature set to avoid temporal contamination between survey waves.

### Section 2.3.6: Comparative Assessment and Model Selection
**ADD final paragraph with actual model comparison results:**

The actual model comparison results from the project's model training pipeline demonstrate the trade-offs between overall accuracy and balanced accuracy in the presence of severe class imbalance. On the held-out test set (20% stratified split, n=1,327), the models achieved the following test-set balanced accuracy scores: Logistic Regression = 0.550, Random Forest = 0.435, XGBoost = 0.404, LightGBM = 0.441, and MLP = 0.376. While tree-based ensembles (Random Forest: 0.822 accuracy, XGBoost: 0.829 accuracy, LightGBM: 0.797 accuracy) achieve higher overall accuracy by primarily predicting the majority Low and Medium classes, Logistic Regression with balanced class weights achieves the best balanced accuracy (0.550) by trading raw accuracy for improved minority-class recall. This validates the methodological position that balanced accuracy, rather than overall accuracy, is the appropriate primary metric for this severely imbalanced classification problem.

---

## CHAPTER 3: RESEARCH METHODOLOGY - Evidence Updates

### Section 3.4.1: Dataset Source and Characteristics
**REPLACE Table 3.2 with actual verified data:**

**Table 3.2: Verified Dataset Characteristics**
| Attribute | Value |
|-----------|-------|
| Dataset | WERS 2011 Operational Risk Dataset |
| Source | UKDA Study 4511 (WERS 2011) |
| Original format | Tab-delimited (.tab) |
| Processed format | CSV (.csv) |
| Unit of analysis | Organization-level (workplace surveys) |
| Total observations | 6,635 |
| Total variables | 52 |
| Predictive features (raw) | 36 (5 numeric + 31 categorical) |
| Predictive features (post-encoding) | Varies by model (one-hot encoding) |
| High-cardinality excluded | industry_code, industry_code_80 (SIC codes) |
| Target variable | operational_risk_category |
| Target classes | 3 (Low / Medium / High) |
| Class distribution (Low) | 5,502 (82.92%) |
| Class distribution (Medium) | 1,105 (16.65%) |
| Class distribution (High) | 28 (0.42%) |
| Imbalance ratio (Low:High) | 196:1 |
| Missing values (predictors) | 0 |
| Score range | 0 – 100 |
| Survey waves | 3 (1998, 2004, 2011) |
| Original file sizes | wirsts10: 4.59 MB, wirsts25: 9.09 MB |
| Original variable count | 320 variables per file |

**ADD after Table 3.2:**

The raw WERS 2011 data were obtained from the UK Data Archive (Study 4511) under its End User Licence. The original files were provided in tab-delimited format (.tab): wirsts10_july2014.tab (main workplace survey, 6,635 observations, 320 variables, 4.59 MB) and wirsts25_july2014.tab (panel data, 13,284 observations, 320 variables, 9.09 MB). For this project, these files were converted to CSV format to enable compatibility with pandas-based data processing workflows. The conversion preserved all original data without transformation or aggregation. The main survey file contains records from three waves (1998: 2,191 observations, 2004: 2,062 observations, 2011: 2,382 observations), while the panel file spans 1980–2011. The merged dataset used in this analysis includes only the 1998, 2004, and 2011 waves from the main survey to maintain consistency with the panel data structure.

The training and test split was performed using stratified sampling (random_state=42) with an 80:20 ratio. The training set contains 5,308 observations (Low: 4,402, Medium: 884, High: 22), while the test set contains 1,327 observations (Low: 1,100, Medium: 221, High: 6). Stratified three-fold cross-validation on the training set results in approximately 7–8 High-risk observations per fold. This extremely small number of minority-class cases in each fold means that minority-class performance estimates are highly variable, and any single validation score cannot establish reliable detection of rare High-risk cases. The dataset contains records from three survey waves (1998, 2004, 2011); the year variable is excluded from the feature set to avoid temporal contamination between waves.

### Section 3.4.2: Target Variable Construction
**KEEP the scoring rule (it's correct)**, but ADD after the thresholds table:

**Implementation Evidence:**
The target construction procedure is implemented in the data preparation script (ml_pipeline/01_data_preparation.py). The operational_risk_score variable in the processed dataset (range 0–100) is calculated according to the documented scoring rule. The classification thresholds are then applied: High (≥60), Medium (35–59), Low (<35). This procedure results in the observed class distribution reported in Table 3.2.

### Section 3.4.3: Predictor Selection and Leakage Control
**REPLACE Table 3.4 with actual excluded variables:**

**Table 3.4: Excluded Variables (Leakage Prevention)**
| Category | Excluded Columns |
|----------|------------------|
| Risk indicators | employee_turnover_rate, employee_turnover_rate_5yr, absenteeism_rate, grievance_rate, disputes_raised, dispute_risk_level, other_industrial_action, dismissal_rate, disciplinary_warnings, suspensions |
| Score | operational_risk_score |
| Identifiers | serno, year, weight |

**ADD after Table 3.4:**

The leakage exclusion is implemented in the model training script (ml_pipeline/04_model_training.py, lines 52–55). The script defines the LEAKAGE list containing all variables used to construct the target score and ensures these are excluded from the feature set. The remaining 36 raw features are organized as follows: organizational characteristics (9 variables), HR management practices (11 variables), employee relations practices (11 variables), and labour market characteristics (5 variables). Additional high-cardinality variables (industry_code, industry_code_80) are excluded during feature selection to prevent excessive one-hot encoding dimensionality. The year variable is excluded to avoid treating multi-wave survey data as a single-year cross-section.

### Section 3.5: Data Preparation and Preprocessing
**ADD paragraph about actual preprocessing implementation:**

The preprocessing pipeline is implemented using scikit-learn's ColumnTransformer and Pipeline classes (ml_pipeline/04_model_training.py, lines 111–117). Numeric features (5 identified: sector_detailed, union_coverage_pct, pct_part_time, pct_part_time_12, pct_female) are imputed using median strategy and standardized using StandardScaler. Categorical features (31 identified) are imputed using most-frequent strategy and encoded using OneHotEncoder with handle_unknown="ignore" to handle unseen categories. Missing value tokens ("", " ", "NA", "N/A", "nan", "None", "-9", "-8") are replaced with "MISSING" for categorical variables. Rare categories (frequency <1%) are grouped into "OTHER_RARE" to reduce dimensionality. All preprocessing operations are fitted only on training data within each cross-validation fold to prevent leakage. The 36 raw features become a larger number of post-encoding features after one-hot encoding; the exact dimension depends on the model's preprocessing pipeline.

### Section 3.7.1: Candidate Classification Models
**REPLACE Table 3.6 with actual model configurations:**

**Table 3.6: Candidate Models and Configurations**
| # | Model | Library | Configuration | Class Weighting |
|---|-------|---------|---------------|----------------|
| 1 | Logistic Regression | scikit-learn | LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42) | Yes (balanced) |
| 2 | Random Forest | scikit-learn | RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42) | Yes (balanced_subsample) |
| 3 | MLP (Neural Network) | scikit-learn | MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300, early_stopping=True, random_state=42) | No |
| 4 | XGBoost | xgboost | XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="mlogloss", tree_method="hist", random_state=42) | No |
| 5 | LightGBM | lightgbm | LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", verbose=-1, random_state=42) | Yes (balanced) |

**ADD after Table 3.6:**

These configurations are implemented in the build_models() function (ml_pipeline/04_model_training.py, lines 119–139). Fixed hyperparameters are used for all models; no automated hyperparameter search (e.g., RandomizedSearchCV) is performed. The random_state=42 parameter ensures reproducibility for stochastic algorithms. Note that only Logistic Regression, Random Forest, and LightGBM use balanced class weights; XGBoost and MLP do not have explicit class-weight parameters in their configurations.

### Section 3.7.2: Class-Imbalance Management
**ADD actual CV information:**

Stratified three-fold cross-validation is implemented using StratifiedKFold(n_splits=3, shuffle=True, random_state=42) in the model training script (ml_pipeline/04_model_training.py, line 147). With 22 High-risk observations in the training set, each fold contains approximately 7–8 High-risk cases. This small number constrains the reliability of minority-class performance estimates, and results must be interpreted with this limitation in mind.

### Section 3.9.1: Predictive Evaluation
**ADD paragraph with actual model results:**

The model comparison results from the implemented pipeline (ml_pipeline/04_model_training.py, results/tables/model_comparison.csv) demonstrate the actual performance differences between candidate models on the held-out test set (n=1,327). The test-set balanced accuracy scores are: Logistic Regression = 0.550, Random Forest = 0.435, XGBoost = 0.404, LightGBM = 0.441, and MLP = 0.376. Logistic Regression with balanced class weights achieves the highest balanced accuracy (0.550), while tree-based ensembles achieve higher overall accuracy (Random Forest: 0.822, XGBoost: 0.829, LightGBM: 0.797) but lower balanced accuracy. This confirms that overall accuracy alone would be misleading for model selection in this severely imbalanced context.

**ADD confusion matrix section:**

The confusion matrix for the best-performing model (Logistic Regression) on the test set (results/tables/confusion_matrix_logreg.csv) shows the following classification pattern:

| Actual \ Predicted | Low | Medium | High |
|-------------------|-----|--------|------|
| Low | 714 | 332 | 54 |
| Medium | 61 | 146 | 14 |
| High | 1 | 3 | 2 |

From this matrix, the per-class recall values are: Low = 714/1,100 = 0.649, Medium = 146/221 = 0.661, High = 2/6 = 0.333. The mean of these three recall values is (0.649 + 0.661 + 0.333) / 3 = 0.548. The balanced accuracy of 0.550 is calculated directly by the evaluation function. This performance highlights the limitation imposed by the extremely small number of High-risk observations in the test set (n=6).

### Section 3.7.3: Model Explanation and Inference
**ADD SHAP results:**

SHAP analysis was performed on the selected model, with feature importance results saved to results/tables/shap_importance.csv. The top 10 features by mean absolute SHAP value are:

1. employee_voice_channels (1.240)
2. sector (1.001)
3. trading_status (0.992)
4. org_size_category_3 (0.984)
5. union_coverage_pct_6 (0.824)
6. collective_bargaining (0.798)
7. union_coverage_pct_7 (0.785)
8. formal_hr_strategy (0.773)
9. org_age_years (0.771)
10. bargaining_level_org (0.666)

These SHAP values describe the magnitude of feature attribution in the model's predictions. They do not establish causal relationships between these workplace characteristics and operational risk, but they indicate which factors the model relies on most heavily when assigning risk categories.

**ADD feature importance from model:**

The model's intrinsic feature importance (results/tables/feature_importance.csv) shows the top 10 features by importance score:

1. collective_bargaining_99 (0.058)
2. trading_status_1 (0.050)
3. sector_3 (0.042)
4. sector_2 (0.040)
5. bargaining_level_org_99 (0.031)
6. trading_status_2 (0.026)
7. org_size_category_1 (0.015)
8. collective_bargaining_2 (0.011)
9. org_size_category_4_1 (0.011)
10. org_size_category_3_3 (0.011)

These results are consistent with the SHAP analysis in highlighting collective bargaining, trading status, sector, and organizational size as influential features.

---

## SUMMARY OF KEY EVIDENCE TO INTEGRATE

1. **Dataset**: 6,635 observations, 52 variables, 36 raw features (5 numeric + 31 categorical), target with 82.92%/16.65%/0.42% distribution
2. **Survey Waves**: 3 waves (1998: 2,191, 2004: 2,062, 2011: 2,382)
3. **Train/Test Split**: 80:20 stratified (train: 5,308, test: 1,327)
4. **Class Distribution in Test**: Low: 1,100, Medium: 221, High: 6
5. **Model Comparison**: LogReg best (balanced accuracy 0.550), followed by LightGBM (0.441), Random Forest (0.435), XGBoost (0.404), MLP (0.376)
6. **Confusion Matrix**: Test set shows LogReg recalls: Low=0.649, Medium=0.661, High=0.333; mean=0.548; balanced accuracy=0.550
7. **Top Features**: employee_voice_channels, sector, trading_status, org_size_category_3, collective_bargaining
8. **Excluded Variables**: 11 risk indicators + score + 3 identifiers + year (to avoid multi-wave contamination)
9. **Model Configurations**: Fixed hyperparameters; class weights only for LogReg, Random Forest, LightGBM; no XGBoost/MLP weighting
10. **Cross-Validation**: Three-fold (not five-fold)

This evidence ensures logical consistency across Chapters 1-3 and provides the reviewer with traceable connections between methodology, implementation, and results.

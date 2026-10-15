# EVIDENCE INTEGRATION: REAL PROJECT DATA FOR CHAPTERS 1-3 (CORRECTED)

This document provides the actual evidence from the project that should be inserted into Chapters 1-3 of your dissertation to ensure logical consistency and reviewer credibility. All corrections are based on actual pipeline execution against `data/processed/wers_operational_risk_merged.csv` (14,448 × 53).

---

## CORRECTIONS MADE

1. **Dataset**: 14,448 rows × 53 variables (7 waves: 1980/1984/1990/1991/1998/2004/2011); **6,385 modelled rows** where turnover is observed (1998/2004/2011 waves). Not 6,635 rows / 52 variables.
2. **Target**: employee turnover rate (`tempc`), sentinel-cleaned; Low ≤ 10 %, Medium 10–20 %, High > 20 %. The old composite 0–100 score is retired.
3. **Class counts**: Low 5,077 (79.51 %), Medium 621 (9.73 %), High 687 (10.76 %); imbalance 7.4 : 1. Not 5,502 / 1,105 / 28 and not 196 : 1.
4. **Feature count**: 37 model inputs = 33 raw + 4 EFA factors (9 numeric + 28 categorical). Not 36 (or 51).
5. **Leakage exclusions**: 14 columns (turnover family, risk indicators, `tprocgrv`/`ttrnprob` duplicates, score alias) + 3 identifiers. Not 11.
6. **Cross-validation**: five-fold stratified (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`). Not three-fold.
7. **Balanced accuracy**: logreg 0.389 held-out / 0.385 training-CV. Not 0.550 (that was the leaking composite target).
8. **Best-model selection**: highest *training-CV* balanced accuracy; held-out test is reported but never used to pick the winner.

---

## CHAPTER 1: INTRODUCTION - Evidence Updates

### Section 1.2: Statement of the Research Problem
**INSERT AFTER sentence about class distribution:**

The actual dataset characteristics, verified from the processed file `wers_operational_risk_merged.csv`, confirm a substantial class imbalance. The target variable `operational_risk_category` is defined for 6,385 of the 14,448 merged rows (turnover is only measured in the 1998/2004/2011 waves) with the following distribution: Low = 5,077 (79.51 %), Medium = 621 (9.73 %), High = 687 (10.76 %). This represents an imbalance ratio of approximately 7.4 : 1 between the Low and High classes. A stratified 80 : 20 train-test split leaves 550 High-risk cases in training and 137 in the test set. Stratified five-fold cross-validation on the training set yields roughly 110 High-risk observations per fold, which supports minority-class estimates but still constrains the reliability of rare-class recall.

### Section 1.5: Scope and Delimitations
**UPDATE dataset characteristics section:**

The study uses the processed dataset `wers_operational_risk_merged.csv`, derived from the WERS time-series (UKDA Study 4511, End User Licence — Safeguarded data). The raw files were obtained from the UK Data Service in tab-delimited format (.tab) and converted to CSV: `wirsts10` (main workplace survey, 6,635 rows, 320 variables, waves 1998/2004/2011) and `wirsts25` (panel data, 13,284 rows, 320 variables, 7 waves 1980–2011). The authoritative merge rule (`ml_pipeline/01_build_dataset.py`) unions all 6,635 `wirsts10` rows with the early-wave `wirsts25` rows (7,813), producing **14,448 rows across 7 waves**. After excluding identifiers (`serno`, `year`, `weight`), the turnover family, the raw risk indicators and their duplicate-mapped predictors (14 columns total), 33 leakage-safe raw predictors remain; with four EFA factor scores the model uses **37 inputs** (9 numeric + 28 categorical). The target has three classes: Low = 5,077 (79.51 %), Medium = 621 (9.73 %), High = 687 (10.76 %) of the 6,385 modelled rows. The unit of analysis is the workplace (organization). `year` is excluded from the feature set to avoid treating multi-wave survey data as a single-year cross-section; the analysis is cross-sectional classification, not longitudinal forecasting.

The technical scope includes five candidate classification models: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM, and a Multi-Layer Perceptron (MLP). Class imbalance is addressed through stratified five-fold cross-validation and balanced class weights where the algorithm supports them. Evaluation prioritises balanced accuracy and macro-F1. Fixed hyperparameters are used for the five candidates; the only automated search (`RandomizedSearchCV`) lives in the imbalance-variant script and never participates in model selection.

---

## CHAPTER 2: LITERATURE REVIEW - Evidence Updates

### Section 2.2.2: Relevance of the WERS time-series
**ADD after paragraph about survey components:**

The processed dataset used in this project contains 14,448 workplace-level rows with 53 variables distributed across seven survey waves (1980, 1984, 1990, 1991, 1998, 2004, 2011); the turnover-based target is defined for the 6,385 rows of the 1998/2004/2011 waves. After excluding identifiers (`serno`, `year`, `weight`), the turnover family, the raw risk indicators and their duplicate-derived predictors (14 columns), 33 leakage-safe raw predictors remain; adding four EFA factor scores gives 37 model inputs. The unit of analysis is the workplace (organization). `year` is excluded to avoid temporal contamination between survey waves.

### Section 2.3.6: Comparative Assessment and Model Selection
**ADD final paragraph with actual model comparison results:**

The actual model comparison results demonstrate the trade-offs between overall accuracy and balanced accuracy under class imbalance. On the held-out test set (20 % stratified split, n = 1,277), balanced accuracy scores are: Logistic Regression = 0.389, LightGBM = 0.385, Random Forest = 0.358, XGBoost = 0.334, MLP = 0.333. While tree-based ensembles (Random Forest 0.764 accuracy, XGBoost 0.792, LightGBM 0.739) achieve higher overall accuracy by leaning on the majority Low class, Logistic Regression with balanced class weights attains the best *training-CV* balanced accuracy (0.385) and the best held-out balanced accuracy (0.389) among the five candidates, so it is the deployed model. This validates that balanced accuracy, not overall accuracy, is the appropriate primary metric, and that model selection must be driven by cross-validation rather than the held-out test score.

---
## CHAPTER 3: RESEARCH METHODOLOGY - Evidence Updates

### Section 3.4.1: Dataset Source and Characteristics
**REPLACE Table 3.2 with actual verified data:**

**Table 3.2: Verified Dataset Characteristics**
| Attribute | Value |
|-----------|-------|
| Dataset | WERS/WIRS Operational Risk Dataset (turnover-based) |
| Source | UKDA Study 4511 (WERS time-series, End User Licence — Safeguarded) |
| Original format | Tab-delimited (.tab) |
| Processed format | CSV (.csv) |
| Unit of analysis | Workplace (organization) |
| Total observations | 14,448 (7 waves: 1980, 1984, 1990, 1991, 1998, 2004, 2011) |
| Total variables | 53 |
| Modelled rows (target available) | 6,385 (turnover observed in 1998/2004/2011) |
| Model inputs | 37 (33 raw + 4 EFA factors; 9 numeric + 28 categorical) |
| High-cardinality excluded | industry_code, industry_code_80 (SIC codes) |
| Target variable | operational_risk_category |
| Target classes | 3 (Low / Medium / High) |
| Class distribution (Low) | 5,077 (79.51 %) |
| Class distribution (Medium) | 621 (9.73 %) |
| Class distribution (High) | 687 (10.76 %) |
| Imbalance ratio (Low:High) | 7.4 : 1 |
| Missing values (predictors) | 0 |
| Turnover rate range (%) | 0.0 – 100.0 |
| Survey waves | 7 (target defined for 3: 1998, 2004, 2011) |

**ADD after Table 3.2:**

The raw WERS time-series data were obtained from the UK Data Service (Study 4511) under its End User Licence (Safeguarded data). The original files were provided in tab-delimited format (.tab): `wirsts10` (main workplace survey, 6,635 rows, 320 variables) and `wirsts25` (panel data, 13,284 rows, 320 variables), converted to CSV for pandas-based processing with all data preserved. The authoritative merge (`ml_pipeline/01_build_dataset.py`) unions all 6,635 `wirsts10` rows with the early-wave `wirsts25` rows (7,813), producing 14,448 rows across 7 waves (1980–2011). The turnover-based target is defined only for the 1998/2004/2011 waves (6,385 rows).

The training and test split uses stratified sampling (random_state=42, 80:20). The training set contains 5,108 rows and the test set 1,277 rows (test support: Low 1,016, Medium 124, High 137). Stratified five-fold cross-validation on the training set yields roughly 110 High-risk observations per fold, so minority-class estimates are usable but still noisy. `year` is excluded from the feature set to avoid temporal contamination between waves.

### Section 3.4.2: Target Variable Construction
**REPLACE the composite-score rule with the turnover-based target:**

**Implementation Evidence:**
The target is constructed in `ml_pipeline/01_build_dataset.py`. The employee turnover rate (`tempc`) is sentinel-cleaned (values outside 0–100 are treated as missing), then banded: **Low ≤ 10 %, Medium 10–20 %, High > 20 %**. Rows without an observed turnover rate are set to NaN and excluded from training (6,385 of 14,448 modelled). The retired composite 0–100 score (25 points per dimension, High ≥ 60) is no longer used — its inputs were categorical indicators, not rates, making the composite misleading. The `operational_risk_score` column retained in the CSV is only an alias of the cleaned turnover rate for backwards compatibility and is excluded from the model as leakage.

### Section 3.4.3: Predictor Selection and Leakage Control
**REPLACE Table 3.4 with actual excluded variables:**

**Table 3.4: Excluded Variables (Leakage Prevention)**
| Category | Excluded Columns |
|----------|------------------|
| Turnover family (target basis) | employee_turnover_rate, employee_turnover_rate_5yr |
| Risk indicators | absenteeism_rate, grievance_rate, disputes_raised, dispute_risk_level, other_industrial_action, dismissal_rate, disciplinary_warnings, suspensions |
| Duplicate-derived predictors | grievance_procedure_formal (tprocgrv), training_problems + skill_shortage_reported (ttrnprob) |
| Score alias | operational_risk_score |
| Identifiers | serno, year, weight |

**ADD after Table 3.4:**

The leakage exclusion is implemented in `ml_pipeline/04_model_training.py` (the `LEAKAGE` list, 14 columns) and asserted by system test ST3 against `model_metadata.json`. The 33 leakage-safe raw predictors (industry_code and industry_code_80 are additionally dropped for high cardinality) are joined by four EFA factor scores for 37 model inputs. `year` is excluded to avoid treating multi-wave survey data as a single-year cross-section.

### Section 3.5: Data Preparation and Preprocessing
**ADD paragraph about actual preprocessing implementation:**

Preprocessing uses scikit-learn's ColumnTransformer and Pipeline classes so train/serve parity is guaranteed: imputation, scaling and one-hot encoding travel inside the model `.pkl`. Numeric features (9) are median-imputed and standardised with StandardScaler; categorical features (28) are mode-imputed and one-hot encoded with `handle_unknown="ignore"`. Missing-value tokens ("", " ", "NA", "N/A", "nan", "None", "-9", "-8") become "MISSING" for categorical variables, and rare categories (<1 %) are grouped into "OTHER_RARE". All preprocessing is fitted only on training data within each cross-validation fold to prevent leakage. The 37 raw inputs expand to a larger post-encoding dimension; the exact count depends on the fitted vocabulary.

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

These configurations are implemented in `build_models()` (`ml_pipeline/04_model_training.py`). Fixed hyperparameters are used for the five candidates; `random_state=42` ensures reproducibility. Only Logistic Regression, Random Forest and LightGBM use balanced class weights; XGBoost and MLP have no class-weight parameter. A `RandomizedSearchCV` (n_iter=8, stratified 5-fold) exists only in `05_imbalance_tuning.py` for the `xgboost_tuned` imbalance variant and is never used for model selection.

### Section 3.7.2: Class-Imbalance Management
**ADD actual CV information:**

Stratified five-fold cross-validation is implemented with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` in `ml_pipeline/04_model_training.py`. With 550 High-risk rows in the training set, each fold contains roughly 110 High-risk cases, so minority-class estimates are reasonably supported. Best-model selection uses the mean **training-CV** balanced accuracy; the held-out test score is reported but never used to choose the winner.

---
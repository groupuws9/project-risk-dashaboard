# EVIDENCE INTEGRATION SUMMARY

## What Was Done

1. **Reran the model training script** (`ml_pipeline/04_model_training.py`) to get actual results
2. **Checked the actual dataset** to verify:
   - Total observations: 6,635
   - Total variables: 52
   - Raw features: 36 (5 numeric + 31 categorical)
   - Class distribution: Low 5,502 (82.92%), Medium 1,105 (16.65%), High 28 (0.42%)
   - Survey waves: 3 (1998: 2,191, 2004: 2,062, 2011: 2,382)
3. **Verified model configurations** from the actual code
4. **Documented actual test results** from the training output

## Corrections Applied

### 1. Feature Count
- **Before**: 51 predictive features
- **After**: 36 raw features (5 numeric + 31 categorical)
- **Reason**: The 51 number was incorrect. The actual code shows 36 raw features before one-hot encoding. Post-encoding features vary by model.

### 2. Balanced Accuracy
- **Clarification**: The confusion matrix shows per-class recalls: Low=0.649, Medium=0.661, High=0.333
- **Mean recall**: (0.649 + 0.661 + 0.333) / 3 = 0.548
- **Balanced accuracy**: 0.550 (calculated by evaluation function)
- **Both values are now documented** with explanation

### 3. Class Weighting
- **Before**: "All models use balanced class weights"
- **After**: Only Logistic Regression, Random Forest, and LightGBM use balanced class weights
- **XGBoost and MLP**: Do not have explicit class-weight parameters in their configurations

### 4. Cross-Validation
- **Before**: Five-fold CV mentioned in some places
- **After**: Three-fold CV (as implemented: `StratifiedKFold(n_splits=3)`)
- **With 22 High-risk in training**: Each fold has ~7-8 High-risk cases

### 5. Model Tuning
- **Clarification**: Fixed hyperparameters are used for all models
- **No automated search**: No RandomizedSearchCV or similar is performed
- **Configurations are predefined** in the build_models() function

### 6. Time Span
- **Added explanation**: Dataset contains 3 survey waves (1998, 2004, 2011)
- **Year exclusion**: Documented that year is excluded to avoid multi-wave temporal contamination
- **Analysis approach**: Treated as cross-sectional classification, not longitudinal forecasting

### 7. Original Data Format
- **Added information**: Original files were tab-delimited (.tab), converted to CSV
- **File details**: wirsts10 (6,635 obs, 320 vars, 4.59 MB), wirsts25 (13,284 obs, 320 vars, 9.09 MB)
- **Conversion**: Preserved all data without transformation during conversion

## Raw Data Source Information

**Original Files (UKDA Study 4511):**
- wirsts10_july2014.tab → wirsts10_july2014.csv
  - Main workplace survey
  - 6,635 observations, 320 variables
  - 4.59 MB file size
  - 3 waves: 1998 (2,191), 2004 (2,062), 2011 (2,382)

- wirsts25_july2014.tab → wirsts25_july2014.csv
  - Panel data (longitudinal)
  - 13,284 observations, 320 variables
  - 9.09 MB file size
  - 7 waves: 1980, 1984, 1990, 1991, 1998, 2004, 2011

**Format Conversion:**
- Original: Tab-delimited (.tab)
- Processed: CSV (.csv)
- Conversion preserved all data without transformation
- Used pandas with sep='\t' for import

## Actual Model Results (Test Set, n=1,327)

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|-------|----------|-------------------|----------|
| Logistic Regression | 0.649 | **0.550** | 0.410 |
| Random Forest | 0.822 | 0.435 | 0.442 |
| XGBoost | 0.829 | 0.404 | 0.417 |
| LightGBM | 0.797 | 0.441 | 0.436 |
| MLP | 0.826 | 0.376 | 0.381 |

**Best model**: Logistic Regression (highest balanced accuracy: 0.550)

## Confusion Matrix (Logistic Regression)

| Actual \ Predicted | Low | Medium | High |
|-------------------|-----|--------|------|
| Low | 714 | 332 | 54 |
| Medium | 61 | 146 | 14 |
| High | 1 | 3 | 2 |

**Per-class recall**:
- Low: 714/1,100 = 0.649
- Medium: 146/221 = 0.661
- High: 2/6 = 0.333

## Top 10 Features (SHAP Importance)

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

## Documents Created

1. **EVIDENCE_INTEGRATION_CHAPTERS_1-3_CORRECTED.md**
   - Contains all the corrected evidence for Chapters 1-3
   - Includes specific insert points for each chapter
   - Provides tables with actual data
   - Explains all corrections made
   - Added raw data source information (.tab → .csv conversion)

2. **RAW_DATA_SOURCE_REPORT.md**
   - Detailed report on original WERS data files
   - File format conversion (.tab → .csv)
   - Year distribution by survey wave
   - Data processing workflow
   - Licensing and access information
   - Recommended dissertation updates

3. **README.md**
   - Already contains correct numbers
   - No changes needed
   - Already references 36 features correctly

## Next Steps

1. **Copy the specific sections** from EVIDENCE_INTEGRATION_CHAPTERS_1-3_CORRECTED.md into your dissertation chapters
2. **Replace the existing tables** with the corrected versions
3. **Add the new paragraphs** about survey waves, year exclusion, and class weighting
4. **Ensure consistency** across all chapters with the corrected numbers
5. **Remove any references** to five-fold CV, RandomizedSearchCV, or 51 features

## Key Takeaways for Reviewer

- All evidence is now based on actual model training output
- Feature count discrepancy resolved (36 raw features, not 51)
- Balanced accuracy clarified (0.550 from evaluation, 0.548 mean recall from confusion matrix)
- Class weighting accurately documented (only for LogReg, RF, LightGBM)
- Cross-variation correctly stated as three-fold
- Survey waves and year exclusion documented
- No automated hyperparameter search claimed
- All numbers are traceable to actual code execution

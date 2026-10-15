# EVIDENCE INTEGRATION SUMMARY

## What Was Done

1. **Reran the full pipeline** (`ml_pipeline/01_build_dataset.py` → `04_model_training.py`) to get actual results
2. **Checked the actual dataset** to verify:
   - Total observations: 14,448 (7 waves: 1980/1984/1990/1991/1998/2004/2011)
   - Total variables: 53
   - Modelled rows (target available): 6,385 (turnover observed in 1998/2004/2011)
   - Model inputs: 37 features (33 raw + 4 EFA factors; 9 numeric + 28 categorical)
   - Class distribution: Low 5,077 (79.51%), Medium 621 (9.73%), High 687 (10.76%)
   - Imbalance ratio: 7.4 : 1 (Low : High)
3. **Verified model configurations** from the actual code
4. **Documented actual test results** from the training output

## Corrections Applied

### 1. Feature Count
- **Before**: 36 raw features / 51 predictive features
- **After**: 37 model inputs = 33 raw + 4 EFA factor scores (9 numeric + 28 categorical)
- **Reason**: The feature contract in `model_metadata.json` includes four EFA factor scores; leakage-safe raw predictors total 33 after excluding identifiers, the turnover family and risk indicators.

### 2. Target construction
- **Before**: composite 0–100 score (25 points per dimension), thresholds High ≥ 60 / Medium 35–59 / Low < 35
- **After**: employee turnover rate (`tempc`), sentinel-cleaned; **Low ≤ 10 %, Medium 10–20 %, High > 20 %**
- **Reason**: The composite score was retired as misleading (its inputs are categorical, not rates). The target is now the turnover band.

### 3. Balanced accuracy
- **Before**: 0.550 (composite target)
- **After**: 0.389 held-out (logreg), 0.385 training-CV — the turnover target is far harder without the leaked indicators
- **Mean recall**: (0.435 + 0.395 + 0.343) / 3 = 0.391 ≈ balanced accuracy 0.389

### 4. Class Weighting (verified)
- Logistic Regression, Random Forest (`balanced_subsample`) and LightGBM use balanced class weights
- XGBoost and MLP have no class-weight parameter

### 5. Cross-Validation
- **Before**: three-fold CV
- **After**: **five-fold** stratified CV (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) — spec 3.4
- With 687 High-risk rows (550 in training after the 80:20 split), each fold carries ~110 High-risk cases

### 6. Model Tuning
- Fixed hyperparameters in `04_model_training.py` (no search there)
- `05_imbalance_tuning.py` runs a `RandomizedSearchCV` (n_iter=8, stratified 5-fold) only for the `xgboost_tuned` imbalance variant; it is never eligible for model selection (no CV score; test set reported only)

### 7. Time Span
- Dataset contains **7 survey waves (1980–2011)**; the target is defined only for 1998/2004/2011 (6,385 of 14,448 rows)
- `year` is excluded from features to avoid multi-wave temporal contamination; treated as cross-sectional classification

### 8. Best-model selection protocol
- Best model = **highest training-CV balanced accuracy** (`04_model_training.py`)
- Held-out test balanced accuracy is **reported but never used to select** the winner
- `05_imbalance_tuning.py` records imbalance variants in the comparison table but must not re-select the deployed model

## Raw Data Source Information

**Original Files (UKDA Study 4511, End User Licence — Safeguarded data):**
- `wirsts10` — main workplace survey, 6,635 rows × 320 variables, waves 1998/2004/2011
- `wirsts25` — panel data, 13,284 rows × 320 variables, 7 waves 1980–2011
- **Merge rule** (`ml_pipeline/01_build_dataset.py`): all 6,635 `wirsts10` rows + early-wave `wirsts25` rows (7,813) = **14,448 rows over 7 waves**
- Raw files are tab-delimited (.tab) → converted to CSV; not redistributed (licence)

## Actual Model Results (Test Set, n=1,277)

| Model | Accuracy | Balanced Accuracy | Macro F1 | CV bal. acc. |
|-------|----------|-------------------|----------|--------------|
| Logistic Regression | 0.422 | **0.389** | 0.317 | **0.385** |
| LightGBM | 0.739 | 0.385 | 0.391 | 0.366 |
| Random Forest | 0.764 | 0.358 | 0.355 | 0.345 |
| XGBoost | 0.792 | 0.334 | 0.300 | 0.340 |
| MLP | 0.796 | 0.333 | 0.295 | 0.333 |

**Best model**: Logistic Regression (highest **training-CV** balanced accuracy 0.385 → deployed; held-out test 0.389 reported only)

## Confusion Matrix (Logistic Regression)

| Actual \ Predicted | Low | Medium | High |
|-------------------|-----|--------|------|
| Low (1,016) | 442 | 286 | 288 |
| Medium (124) | 35 | 49 | 40 |
| High (137) | 36 | 54 | 47 |

**Per-class recall**:
- Low: 442/1,016 = 0.435
- Medium: 49/124 = 0.395
- High: 47/137 = 0.343

## Top 10 Features (SHAP Importance)

1. pct_part_time_12 (0.716)
2. pct_part_time (0.671)
3. factor_4 (0.369)
4. sector (0.283)
5. employee_voice_channels (0.275)
6. information_sharing_score (0.257)
7. training_budget_exists (0.242)
8. collective_bargaining (0.235)
9. sector_detailed (0.200)
10. bargaining_level_est (0.198)

## Documents Created

1. **EVIDENCE_INTEGRATION_CHAPTERS_1-3_CORRECTED.md** — corrected evidence + insert points for Chapters 1–3
2. **RAW_DATA_SOURCE_REPORT.md** — raw WERS files, wave distribution, merge + target construction, licence
3. **README.md** / **research/README.md** — verified packages, structure and executed numbers

## Next Steps

1. Copy the specific sections from `EVIDENCE_INTEGRATION_CHAPTERS_1-3_CORRECTED.md` into your dissertation chapters
2. Replace the existing tables with the corrected versions
3. Ensure consistency across all chapters with the corrected numbers
4. Remove any references to: the composite 0–100 score, 6,635 rows / 52 variables as the *analysis* dataset, 196:1 imbalance, balanced accuracy 0.550, or three-fold CV

## Key Takeaways for Reviewer

- All evidence is based on actual pipeline execution against `wers_operational_risk_merged.csv` (14,448 × 53)
- Feature count: 37 model inputs (33 raw + 4 EFA factors), not 36/51
- Target: turnover band (10 % / 20 % thresholds), not a composite score
- Balanced accuracy 0.389 (test) / 0.385 (CV) on the honest turnover target
- Cross-validation: five-fold stratified (not three-fold)
- Best model selected by training CV; held-out test never used for selection
- Class weighting documented accurately (LogReg, RF, LightGBM only)
- No automated hyperparameter search for the deployed model
- Licence: UK Data Service End User Licence (Safeguarded), Study 4511

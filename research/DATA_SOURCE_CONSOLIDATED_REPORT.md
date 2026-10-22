# WERS Time-Series Data Source - Consolidated Research Report

## Executive Summary

The dataset used in this project is derived from the **WERS time-series** (UK Data Archive Study 4511), spanning **7 survey waves from 1980 to 2011**, not a single-year 2011 cross-section. The processed dataset contains **14,448 workplace-level observations** across all waves, with the turnover-based target defined for **6,385 rows** in the 1998/2004/2011 waves only.

---

## Data Source Details

### Original Files (UKDA Study 4511)

| File | Observations | Variables | Waves | Format |
|------|-------------|-----------|-------|--------|
| wirsts10_july2014.tab | 6,635 | 320 | 1998, 2004, 2011 | Tab-delimited |
| wirsts25_july2014.tab | 13,284 | 320 | 1980, 1984, 1990, 1991, 1998, 2004, 2011 | Tab-delimited |

**Licence:** UK Data Service End User Licence - Safeguarded Data
**Access:** Requires registration and acceptance of licence terms
**Restrictions:** Raw data files not redistributed; must cite UK Data Archive

### Merge Rule

The authoritative merge (`ml_pipeline/01_build_dataset.py`):
- Uses **wirsts10** for modern waves (1998/2004/2011): 6,635 rows
- Uses **wirsts25** for early waves (1980/1984/1990/1991): 7,813 rows
- **Total: 14,448 rows across 7 waves**
- No serno-year overlap (one coding per wave)
- Files converted from .tab to .csv for pandas compatibility

---

## Processed Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| Dataset name | WERS/WIRS Operational Risk Dataset (turnover-based) |
| Source | UKDA Study 4511 - WERS time-series |
| Total observations | 14,448 (7 waves: 1980, 1984, 1990, 1991, 1998, 2004, 2011) |
| Total variables | 53 |
| Modelled rows (target available) | 6,385 (turnover observed in 1998/2004/2011) |
| Model inputs | 37 (33 raw + 4 EFA factors; 9 numeric + 28 categorical) |
| High-cardinality excluded | industry_code, industry_code_80 (SIC codes) |
| Target variable | operational_risk_category |
| Target classes | 3 (Low / Medium / High) |
| Class distribution (Low) | 5,077 (79.51%) |
| Class distribution (Medium) | 621 (9.73%) |
| Class distribution (High) | 687 (10.76%) |
| Imbalance ratio (Low:High) | 7.4 : 1 |
| Missing values (predictors) | 0 |
| Turnover rate range (%) | 0.0 – 100.0 |
| Survey waves | 7 (target defined for 3: 1998, 2004, 2011) |

---

## Target Construction

**Method:** Employee turnover rate (`tempc`) banding

**Steps:**
1. Sentinel cleaning: UKDS missing codes (9/99/999/9999.99) and impossible rates (<0 or >100) → NaN
2. Band classification:
   - **Low:** turnover ≤ 10%
   - **Medium:** 10% < turnover ≤ 20%
   - **High:** turnover > 20%
3. Rows without observed turnover excluded from training (6,385 of 14,448 modelled)

**Note:** The earlier composite 0–100 score (25 points per dimension) was retired because its inputs (absenteeism, industrial action, grievances) are categorical survey codes, not rates, making numeric thresholds meaningless.

---

## Feature Selection (Leakage Control)

### Excluded Columns (14 total)

| Category | Columns |
|----------|---------|
| Turnover family (target basis) | employee_turnover_rate, employee_turnover_rate_5yr |
| Risk indicators | absenteeism_rate, grievance_rate, disputes_raised, dispute_risk_level, other_industrial_action, dismissal_rate, disciplinary_warnings, suspensions |
| Duplicate-derived predictors | grievance_procedure_formal (tprocgrv), training_problems, skill_shortage_reported (ttrnprob) |
| Score alias | operational_risk_score |
| Identifiers | serno, year, weight |

### Model Inputs (37 total)

- **33 leakage-safe raw predictors** (after dropping high-cardinality SIC codes)
- **4 EFA factor scores** (latent constructs from factor analysis)
- **9 numeric + 28 categorical** features

---

## Analysis Approach

**Unit of analysis:** Workplace (organization) level

**Temporal handling:**
- `year` variable excluded from feature set to avoid temporal contamination
- Analysis is **cross-sectional classification**, not longitudinal forecasting
- Dataset treated as logically unified WERS time-series report

**Train-test split:**
- Stratified 80:20 split (random_state=42)
- Training: 5,108 rows (Low 4,061, Medium 497, High 550)
- Test: 1,277 rows (Low 1,016, Medium 124, High 137)

**Cross-validation:**
- Stratified 5-fold CV (shuffle=True, random_state=42)
- ~110 High-risk observations per fold in training

---

## Key Clarifications for Research Reporting

1. **Not single-year 2011 data:** The dataset spans 7 waves (1980–2011). Only the target is restricted to 1998/2004/2011 due to turnover measurement availability.

2. **Logical time-series report:** The merge produces a unified WERS time-series dataset with 14,448 rows, representing the full historical scope of the WERS/WIRS surveys.

3. **Cross-sectional classification:** Despite multi-wave data, the analysis is cross-sectional (year excluded from features) to avoid temporal leakage between survey waves.

4. **Turnover-based target:** The target is derived solely from employee turnover rate, not a composite of multiple risk indicators.

5. **Leakage-safe features:** All risk indicators and turnover family members are excluded from model inputs to prevent target leakage.

---

## References for Dissertation

### Chapter 1 (Introduction)
- State that data comes from WERS time-series (UKDA Study 4511), not single-year 2011
- Clarify 14,448 rows over 7 waves, with 6,385 modelled rows
- Mention turnover-based target construction

### Chapter 2 (Literature Review)
- Reference WERS as a time-series study spanning multiple decades
- Note that this analysis uses the full merged time-series for feature context

### Chapter 3 (Methodology)
- Use Table 3.2 from `research/tables/table_3_2_dataset_characteristics.csv`
- Document merge rule: wirsts10 (modern waves) + wirsts25 (early waves)
- Explain why year is excluded from features (temporal contamination)
- Document leakage exclusion list (14 columns)

---


---

## File Locations

- Processed dataset: `data/processed/wers_operational_risk_merged.csv`
- Dataset metadata: `models/saved_models/model_metadata.json`
- Table 3.2: `research/tables/table_3_2_dataset_characteristics.csv`
- Build script: `ml_pipeline/01_build_dataset.py`
- Training script: `ml_pipeline/04_model_training.py`

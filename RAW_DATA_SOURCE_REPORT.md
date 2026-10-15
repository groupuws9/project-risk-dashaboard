# RAW WERS DATA SOURCE REPORT

## Original Data Files

The project uses two raw WERS 2011 data files obtained from the UK Data Archive (Study 4511):

### 1. wirsts10_july2014.csv (originally wirsts10_july2014.tab)

**File Details:**
- Original format: Tab-delimited (.tab)
- Current format: CSV (.csv)
- File size: 4.59 MB
- Observations: 6,635 rows
- Variables: 320 columns
- Survey component: Main workplace survey (cross-sectional)

**Year Distribution:**
| Year | Count | Percentage |
|------|-------|------------|
| 1998 | 2,191 | 33.03% |
| 2004 | 2,062 | 31.08% |
| 2011 | 2,382 | 35.89% |
| **Total** | **6,635** | **100%** |

**Key Variables (first 20):**
- serno (serial number)
- year (survey year)
- tallemps (total employees)
- tallemp3, tallemp4, tallemp5, tallemp6 (employee categories)
- temp1ago, temp5ago (employment change)
- tallfem (total female employees)
- tfempc, tfempc12, tfempc3, tfempc3b, tfempc5b (female employment percentages)
- tftmale, tftfem (full-time male/female)
- tallft, tallpt (full-time/part-time)
- tptpc (part-time percentage)

**Note:** This file contains 95 columns with mixed data types, requiring low_memory=False during import.

### 2. wirsts25_july2014.csv (originally wirsts25_july2014.tab)

**File Details:**
- Original format: Tab-delimited (.tab)
- Current format: CSV (.csv)
- File size: 9.09 MB
- Observations: 13,284 rows
- Variables: 320 columns
- Survey component: Panel data (longitudinal)

**Year Distribution:**
| Year | Count | Percentage |
|------|-------|------------|
| 1980 | 2,040 | 15.36% |
| 1984 | 2,019 | 15.20% |
| 1990 | 2,061 | 15.53% |
| 1991 | 1,693 | 12.75% |
| 1998 | 1,929 | 14.52% |
| 2004 | 1,648 | 12.41% |
| 2011 | 1,894 | 14.26% |
| **Total** | **13,284** | **100%** |

**Key Variables (first 20):**
- serno (serial number)
- year (survey year)
- tallemps (total employees)
- tallemp3, tallemp4, tallemp5, tallemp6 (employee categories)
- temp1ago, temp5ago (employment change)
- tallfem (total female employees)
- tfempc, tfempc12, tfempc3, tfempc3b, tfempc5b (female employment percentages)
- tftmale, tftfem (full-time male/female)
- tallft, tallpt (full-time/part-time)
- tptpc (part-time percentage)

**Note:** This file contains 216 columns with mixed data types, requiring low_memory=False during import.

## Data Processing Workflow

### Step 1: Load Raw Data
The data preparation script (`ml_pipeline/01_data_preparation.py`) loads the original tab-delimited files:
```python
df_main = pd.read_csv(
    os.path.join(RAW_DATA_PATH, 'wirsts10_july2014.tab'),
    sep='\t',
    low_memory=False
)
```

### Step 2: Merge and Clean
- The main survey (wirsts10) provides the primary workplace-level data
- The panel data (wirsts25) provides longitudinal information
- Files are merged on common identifiers (serno, year)
- The merged dataset is filtered to include only the 1998, 2004, and 2011 waves

### Step 3: Create Operational Risk Score
The operational risk score is calculated from raw risk indicators:
- Employee turnover rate
- Absenteeism rate
- Grievance rate
- Disputes raised
- Dispute risk level
- Other industrial action
- Dismissal rate
- Disciplinary warnings
- Suspensions

Each dimension contributes up to 25 points to a 0–100 score.

### Step 4: Create Risk Categories
Categories are assigned based on the risk score:
- High: Score ≥ 60
- Medium: Score 35–59
- Low: Score < 35

### Step 5: Select Features
After excluding:
- Identifiers (serno, year, weight)
- Risk indicators (11 variables)
- Derived score (operational_risk_score)

The remaining 36 raw features are organized into:
- Organizational characteristics (9 variables)
- HR management practices (11 variables)
- Employee relations practices (11 variables)
- Labour market characteristics (5 variables)

### Step 6: Output Processed Dataset
The final processed dataset is saved as:
`data/processed/wers2011_operational_risk_merged.csv`

**Final Dataset Characteristics:**
- Observations: 6,635
- Variables: 52
- Predictive features: 36 (raw)
- Target: operational_risk_category (3 classes)
- Class distribution: Low 5,502 (82.92%), Medium 1,105 (16.65%), High 28 (0.42%)

## File Format Conversion

The original WERS 2011 data files were provided in tab-delimited format (.tab) by the UK Data Archive. For this project, they were converted to CSV format (.csv) to:
- Enable easier handling with pandas
- Maintain compatibility with standard data science workflows
- Preserve all original data without loss

The conversion process:
1. Read original .tab file with `sep='\t'`
2. Write to .csv with standard comma separation
3. Preserve all variable names and data types
4. No data transformation or aggregation during conversion

## Data Access and Licensing

**Source:** UK Data Archive (UKDA), Study 4511
**Title:** Workplace Employment Relations Study 2011
**Access:** Requires registration and acceptance of End User Licence
**Restrictions:**
- Raw data files are not included in this repository
- Data cannot be redistributed
- Must cite the UK Data Archive in any publications
- Original licence terms must be respected

**Acknowledgement:**
WERS 2011 was funded by the UK Department for Business, Innovation and Skills and conducted by:
- Institute for Employment Studies
- National Institute of Economic and Social Research
- Policy Studies Institute

## Important Notes for Dissertation

1. **Original Format:** The raw data files were originally in tab-delimited format (.tab) and were converted to CSV for this project. This should be mentioned in the data sources section.

2. **Survey Waves:** The dataset contains three survey waves (1998, 2004, 2011) from the main survey, not just 2011 data. The year variable is excluded from the feature set to avoid treating multi-wave data as a single-year cross-section.

3. **Unit of Analysis:** The analysis is at the workplace (organization) level, not individual employee level. Each row represents a distinct workplace.

4. **Data Size:** The original files are substantial (4.59 MB and 9.09 MB) with 320 variables each, but only 52 variables are used in the final processed dataset after feature selection.

5. **Panel Data:** The wirsts25 file contains panel data spanning 1980–2011, but only the 1998, 2004, and 2011 waves are used in the merged dataset to align with the main survey waves.

## Recommended Dissertation Updates

### Chapter 3 (Research Methodology) - Section 3.4.1

**ADD paragraph about original data sources:**

The raw WERS 2011 data were obtained from the UK Data Archive (Study 4511) under its End User Licence. The original files were provided in tab-delimited format (.tab): wirsts10_july2014.tab (main workplace survey, 6,635 observations, 320 variables) and wirsts25_july2014.tab (panel data, 13,284 observations, 320 variables). For this project, these files were converted to CSV format to enable compatibility with pandas-based data processing workflows. The conversion preserved all original data without transformation or aggregation. The main survey file contains records from three waves (1998: 2,191 observations, 2004: 2,062 observations, 2011: 2,382 observations), while the panel file spans 1980–2011. The merged dataset used in this analysis includes only the 1998, 2004, and 2011 waves from the main survey to maintain consistency with the panel data structure.

### Chapter 3 (Research Methodology) - Section 3.4.1

**UPDATE Table 3.2 with additional row:**

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

### Chapter 1 (Introduction) - Section 1.5

**ADD sentence about data format:**

The raw WERS 2011 data files were obtained from the UK Data Archive (Study 4511) in tab-delimited format (.tab) and converted to CSV for processing. The original files contain 320 variables each, with the main survey (wirsts10) providing 6,635 workplace-level observations across three survey waves (1998, 2004, 2011) and the panel file (wirsts25) containing 13,284 observations spanning 1980–2011.

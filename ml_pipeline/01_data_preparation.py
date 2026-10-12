"""
Data Preparation Script for WERS 2011 Operational Risk Prediction

This script:
1. Loads WERS 2011 tab-delimited data files
2. Creates derived operational risk indicators
3. Selects relevant variables for analysis
4. Outputs a clean merged CSV for modeling

Author: Research Team
Date: October 2024
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_PATH = '../UKDA-4511-tab/tab/'
OUTPUT_PATH = '../data/processed/'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 80)
print("WERS 2011 DATA PREPARATION")
print("=" * 80)
print("\nLoading raw data files...")

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================

# Load main workplace survey (cross-sectional)
try:
    df_main = pd.read_csv(
        os.path.join(RAW_DATA_PATH, 'wirsts10_july2014.tab'),
        sep='\t',
        low_memory=False
    )
    print(f"✓ Loaded main survey: {df_main.shape[0]:,} rows, {df_main.shape[1]} columns")
except Exception as e:
    print(f"✗ Error loading main survey: {e}")
    df_main = None

# Load panel data (if available)
try:
    df_panel = pd.read_csv(
        os.path.join(RAW_DATA_PATH, 'wirsts25_july2014.tab'),
        sep='\t',
        low_memory=False
    )
    print(f"✓ Loaded panel data: {df_panel.shape[0]:,} rows, {df_panel.shape[1]} columns")
except Exception as e:
    print(f"⚠ Panel data not available or error: {e}")
    df_panel = None

if df_main is None:
    raise ValueError("Main survey data could not be loaded. Please check data file paths.")

# ============================================================================
# STEP 2: DEFINE VARIABLE MAPPINGS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: MAPPING VARIABLES")
print("=" * 80)

# Core identifier variables
ID_VARS = ['serno', 'year', 'weight']

# TARGET VARIABLES - Operational Risk Indicators
RISK_VARS = {
    # Employee Turnover
    'employee_turnover_rate': 'tempc',      # Employee turnover percentage
    'employee_turnover_rate_5yr': 'tempc5', # 5-year turnover rate
    
    # Absenteeism
    'absenteeism_rate': 'tsicklog',         # Sickness absence rate
    
    # Grievances and Disputes
    'grievance_rate': 'tprocgrv',           # Grievance procedures
    'disputes_raised': 'traised',           # Disputes raised
    'dispute_risk_level': 'tstrike',        # Industrial action (strike)
    'other_industrial_action': 'tothact',   # Other industrial action
    
    # Dismissals
    'dismissal_rate': 'tdismiss',           # Dismissals
    'disciplinary_warnings': 'twarning',    # Warnings issued
    'suspensions': 'tsuspend',              # Suspensions
}

# ORGANIZATIONAL CHARACTERISTICS
ORG_VARS = {
    'org_size_category': 'torgsize',        # Organization size
    'org_size_category_3': 'torgsiz3',      # Size (3 categories)
    'org_size_category_4': 'torgsiz4',      # Size (4 categories)
    'sector': 'tsector3',                   # Sector (public/private)
    'sector_detailed': 'tsect15',           # Sector (15 categories)
    'industry_code': 'tsic92',              # SIC 1992 industry code
    'industry_code_80': 'tsic80',           # SIC 1980 industry code
    'multinational': 'tmultnat',            # Multinational status
    'org_age_years': 'tage',                # Organization age
    'workplace_age': 'ttimhere',            # Years at current location
    'trading_status': 'ttrading',           # Trading status
}

# HR MANAGEMENT PRACTICES
HR_VARS = {
    'union_coverage_pct': 'tcovpc',         # Union coverage percentage
    'union_coverage_pct_6': 'tcovpc6',      # Union coverage (6 categories)
    'union_coverage_pct_7': 'tcovpc7',      # Union coverage (7 categories)
    'collective_bargaining': 'tnatbarg',    # Collective bargaining present
    'bargaining_level_org': 'torgbarg',     # Bargaining at organization level
    'bargaining_level_est': 'testbarg',     # Bargaining at establishment level
    'formal_hr_strategy': 'tstaffin',      # Formal HR/personnel strategy
    'performance_appraisal_pct': 'tapprais', # Staff with performance appraisal
    'training_budget_exists': 'tfinance',   # Training budget available
    'training_problems': 'ttrnprob',        # Training/skill shortage issues
    'job_security_policy': 'tjobsec',        # Job security policies
}

# EMPLOYEE RELATIONS PRACTICES
ER_VARS = {
    'jcc_exists': 'tjcc',                    # Joint consultative committee
    'jcc_frequency': 'tjccfreq',             # JCC meeting frequency
    'jcc_function': 'tjccf',                   # JCC function
    'jcc_union_chair': 'tjcctu',             # JCC chaired by union
    'information_sharing_score': 'tbrief',   # Briefing practices
    'info_chain_exist': 'tchain',            # Information chain
    'newsletter_exists': 'tnews',            # Newsletter to employees
    'investment_plans_shared': 'tinvplan',   # Investment plans shared
    'consultation_depth': 'tconsall',        # Depth of consultation
    'employee_voice_channels': 'tvoice1',    # Number of voice mechanisms
    'grievance_procedure_formal': 'tprocgrv', # Formal grievance procedure
}

# LABOR MARKET FACTORS
LABOR_VARS = {
    'pct_part_time': 'tptpc',               # Part-time employees %
    'pct_part_time_12': 'tptpc12',          # Part-time % (12 cat)
    'pct_female': 'tfempc',                  # Female employees %
    'pct_temporary': 'ttemplyr',             # Temporary/casual staff %
    'skill_shortage_reported': 'ttrnprob',  # Skill shortage reported
}

# Combine all variable mappings
ALL_VARIABLES = {}
ALL_VARIABLES.update(RISK_VARS)
ALL_VARIABLES.update(ORG_VARS)
ALL_VARIABLES.update(HR_VARS)
ALL_VARIABLES.update(ER_VARS)
ALL_VARIABLES.update(LABOR_VARS)

print(f"✓ Defined {len(ALL_VARIABLES)} variables across {len([RISK_VARS, ORG_VARS, HR_VARS, ER_VARS, LABOR_VARS])} categories")

# ============================================================================
# STEP 3: EXTRACT AND TRANSFORM VARIABLES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: EXTRACTING AND TRANSFORMING VARIABLES")
print("=" * 80)

# Create a new dataframe with selected variables
df_processed = pd.DataFrame()

# Add identifiers
for var in ID_VARS:
    if var in df_main.columns:
        df_processed[var] = df_main[var]
        
# Extract and rename variables based on mapping
for new_name, original_name in ALL_VARIABLES.items():
    if original_name in df_main.columns:
        df_processed[new_name] = df_main[original_name]
    else:
        # If variable doesn't exist, create with NaN
        df_processed[new_name] = np.nan
        print(f"⚠ Variable '{original_name}' not found in dataset")

print(f"✓ Extracted {df_processed.shape[1]} variables")
print(f"✓ Dataset shape: {df_processed.shape[0]:,} rows x {df_processed.shape[1]} columns")

# ============================================================================
# STEP 4: CREATE DERIVED RISK INDICATORS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: CREATING DERIVED RISK INDICATORS")
print("=" * 80)

# Create composite risk score
def calculate_risk_score(row):
    """Calculate composite operational risk score (0-100)"""
    score = 0
    
    # Helper function to safely convert to numeric
    def safe_numeric(value):
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    # Helper function to safely get numeric value from row
    def get_numeric(row, key):
        if key in row:
            return safe_numeric(row[key])
        return None
    
    # Employee turnover risk (0-25 points)
    turnover = get_numeric(row, 'employee_turnover_rate')
    if turnover is not None:
        if turnover > 20:
            score += 25
        elif turnover > 15:
            score += 20
        elif turnover > 10:
            score += 15
        elif turnover > 5:
            score += 10
        else:
            score += 5
    
    # Absenteeism risk (0-25 points)
    absence = get_numeric(row, 'absenteeism_rate')
    if absence is not None:
        if absence > 5:
            score += 25
        elif absence > 4:
            score += 20
        elif absence > 3:
            score += 15
        elif absence > 2:
            score += 10
        else:
            score += 5
    
    # Industrial action risk (0-25 points)
    dispute_risk = 0
    dispute_level = get_numeric(row, 'dispute_risk_level')
    other_action = get_numeric(row, 'other_industrial_action')
    
    if dispute_level is not None and dispute_level == 1:
        dispute_risk = 25
    elif other_action is not None and other_action == 1:
        dispute_risk = 15
    score += dispute_risk
    
    # Grievance risk (0-25 points)
    grievances = get_numeric(row, 'grievance_rate')
    if grievances is not None:
        if grievances > 10:
            score += 25
        elif grievances > 5:
            score += 20
        elif grievances > 2:
            score += 15
        else:
            score += 5
    
    return min(100, max(0, score))

# Apply risk score calculation
print("Calculating composite operational risk scores...")
df_processed['operational_risk_score'] = df_processed.apply(calculate_risk_score, axis=1)

# Create risk categories
def categorize_risk(score):
    """Categorize risk into Low, Medium, High"""
    if score >= 60:
        return 'High'
    elif score >= 35:
        return 'Medium'
    else:
        return 'Low'

df_processed['operational_risk_category'] = df_processed['operational_risk_score'].apply(categorize_risk)

print(f"✓ Risk score distribution:")
print(df_processed['operational_risk_category'].value_counts())
print(f"\n✓ Mean risk score: {df_processed['operational_risk_score'].mean():.2f}")

# ============================================================================
# STEP 5: DATA QUALITY CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: DATA QUALITY CHECKS")
print("=" * 80)

# Check for missing values
missing_pct = (df_processed.isnull().sum() / len(df_processed) * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

print(f"\nVariables with missing values:")
print(f"Total variables with missing: {len(missing_pct)}")
if len(missing_pct) > 0:
    print("\nTop 10 variables with most missing:")
    print(missing_pct.head(10))

# Check for duplicates
duplicates = df_processed.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Summary statistics
print(f"\nDataset summary:")
print(f"- Total observations: {df_processed.shape[0]:,}")
print(f"- Total variables: {df_processed.shape[1]}")
print(f"- Risk categories: {df_processed['operational_risk_category'].nunique()}")

# ============================================================================
# STEP 6: SAVE PROCESSED DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: SAVING PROCESSED DATA")
print("=" * 80)

# Save full dataset
output_file = os.path.join(OUTPUT_PATH, 'wers2011_operational_risk_merged.csv')
df_processed.to_csv(output_file, index=False)
print(f"✓ Saved full dataset: {output_file}")
print(f"  Shape: {df_processed.shape[0]:,} rows x {df_processed.shape[1]} columns")

# Save a metadata file
metadata = {
    'dataset_name': 'WERS 2011 Operational Risk Dataset',
    'source': 'UKDA Study 4511 - WERS 2011',
    'date_created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_observations': df_processed.shape[0],
    'n_variables': df_processed.shape[1],
    'target_variable': 'operational_risk_category',
    'risk_score_variable': 'operational_risk_score',
    'variable_categories': {
        'identifiers': ID_VARS,
        'risk_indicators': list(RISK_VARS.keys()),
        'organizational': list(ORG_VARS.keys()),
        'hr_practices': list(HR_VARS.keys()),
        'employee_relations': list(ER_VARS.keys()),
        'labor_market': list(LABOR_VARS.keys()),
        'derived': ['operational_risk_score', 'operational_risk_category']
    }
}

# Save metadata as JSON
import json
metadata_file = os.path.join(OUTPUT_PATH, 'dataset_metadata.json')
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata: {metadata_file}")

# Create a summary report
summary_report = f"""
WERS 2011 OPERATIONAL RISK DATASET - PROCESSING SUMMARY
{'=' * 80}

Dataset Information:
- Source: UKDA Study 4511 (WERS 2011)
- Processing Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Output File: wers2011_operational_risk_merged.csv

Dimensions:
- Total Observations: {df_processed.shape[0]:,}
- Total Variables: {df_processed.shape[1]}

Target Variable Distribution (Risk Categories):
{df_processed['operational_risk_category'].value_counts().to_string()}

Risk Score Statistics:
- Mean: {df_processed['operational_risk_score'].mean():.2f}
- Median: {df_processed['operational_risk_score'].median():.2f}
- Std Dev: {df_processed['operational_risk_score'].std():.2f}
- Min: {df_processed['operational_risk_score'].min():.2f}
- Max: {df_processed['operational_risk_score'].max():.2f}

Variable Categories:
- Identifiers: {len(ID_VARS)} variables
- Risk Indicators: {len(RISK_VARS)} variables
- Organizational: {len(ORG_VARS)} variables
- HR Practices: {len(HR_VARS)} variables
- Employee Relations: {len(ER_VARS)} variables
- Labor Market: {len(LABOR_VARS)} variables
- Derived Variables: 2 (risk score and category)

Data Quality:
- Duplicate Rows: {duplicates}
- Variables with Missing Values: {len(missing_pct)}

Files Created:
1. wers2011_operational_risk_merged.csv - Main dataset
2. dataset_metadata.json - Metadata and variable information
3. processing_summary.txt - This summary report

Next Steps:
1. Run exploratory data analysis: python 02_exploratory_analysis.py
2. Train predictive models: python 04_model_training.py
3. Launch dashboard: python ../dashboard/app.py

{'=' * 80}
"""

summary_file = os.path.join(OUTPUT_PATH, 'processing_summary.txt')
with open(summary_file, 'w') as f:
    f.write(summary_report)
print(f"✓ Saved summary report: {summary_file}")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)
print(f"\nOutput files created in: {OUTPUT_PATH}")
print("\nNext steps:")
print("  1. Review the processing summary: processing_summary.txt")
print("  2. Run exploratory analysis: python 02_exploratory_analysis.py")
print("  3. Start model training: python 04_model_training.py")
print("\n" + "=" * 80)

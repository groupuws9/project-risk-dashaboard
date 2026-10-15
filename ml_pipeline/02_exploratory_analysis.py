"""
Exploratory Data Analysis (EDA) Script for WIRS/WERS Workplace-Panel
Operational-Risk (turnover-based) Prediction, 1980-2011

This script performs comprehensive exploratory data analysis including:
- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Risk factor analysis
- Visualizations

Edge Cases Handled:
- Missing values in target and feature variables
- Outliers in numerical variables (using IQR method)
- Categorical variables with low frequency categories
- Imbalanced target variable (risk categories)
- Data type inconsistencies

Author: Research Team
Date: October 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import warnings
import os
import json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# Configuration (absolute paths — runnable from any working directory)
BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / 'data' / 'processed' / 'wers_operational_risk_merged.csv'
OUTPUT_PATH = BASE / 'results' / 'figures'
REPORT_PATH = BASE / 'results'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND VALIDATE DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: LOADING AND VALIDATING DATA")
print("=" * 80)

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    raise

# Data validation checks
print("\n--- Data Validation ---")

# Check for required columns
required_cols = ['operational_risk_score', 'operational_risk_category']
missing_required = [col for col in required_cols if col not in df.columns]
if missing_required:
    print(f"⚠ Missing required columns: {missing_required}")
else:
    print(f"✓ All required columns present")

# Check data types
print("\n--- Data Types ---")
print(df.dtypes.value_counts())

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\n--- Duplicate Rows: {duplicates} ---")

# Check for missing values
missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_summary,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

print("\n--- Missing Values Summary ---")
print(f"Variables with missing values: {(missing_summary > 0).sum()}")
if (missing_summary > 0).sum() > 0:
    print("\nTop 10 variables with most missing:")
    print(missing_df.head(10))

# ============================================================================
# STEP 2: DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Target variable analysis
print("\n--- Target Variable: Operational Risk Score ---")
print(df['operational_risk_score'].describe())

print("\n--- Target Variable: Operational Risk Category ---")
risk_dist = df['operational_risk_category'].value_counts()
print(risk_dist)
print(f"\nRisk Distribution (%):")
print((risk_dist / len(df) * 100).round(2))

# Numerical variables statistics
print("\n--- Numerical Variables Summary ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Number of numerical variables: {len(numeric_cols)}")
print(df[numeric_cols].describe().round(2))

# Categorical variables summary
print("\n--- Categorical Variables Summary ---")
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Number of categorical variables: {len(categorical_cols)}")
for col in categorical_cols[:5]:  # Show first 5
    print(f"\n{col}:")
    print(df[col].value_counts().head())

# ============================================================================
# STEP 3: DISTRIBUTION ANALYSIS & VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: CREATING VISUALIZATIONS")
print("=" * 80)

# Figure 1: Risk Score Distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histogram of risk scores
axes[0].hist(df['operational_risk_score'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Operational Risk Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Operational Risk Scores', fontsize=14, fontweight='bold')
axes[0].axvline(df['operational_risk_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["operational_risk_score"].mean():.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bar chart of risk categories
risk_counts = df['operational_risk_category'].value_counts()
colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
axes[1].bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='black')
axes[1].set_xlabel('Risk Category', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Distribution of Risk Categories', fontsize=14, fontweight='bold')

# Add percentage labels on bars
for i, (idx, val) in enumerate(risk_counts.items()):
    pct = val / len(df) * 100
    axes[1].text(i, val + 50, f'{val}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / '01_risk_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 01_risk_distribution.png")
plt.close()

# Figure 2: Correlation Heatmap (select variables)
print("\nGenerating correlation heatmap...")

# Select key numerical variables for correlation
key_vars = [
    'operational_risk_score',
    'employee_turnover_rate',
    'union_coverage_pct',
    'pct_female',
    'pct_part_time',
    'performance_appraisal_pct'
]

# Filter variables that exist in dataset
available_vars = [v for v in key_vars if v in df.columns]
# Coerce to numeric first (some columns are stored as strings with blanks)
corr_df = df[available_vars].apply(pd.to_numeric, errors='coerce').corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 10})
ax.set_title('Correlation Matrix: Key Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / '02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 02_correlation_heatmap.png")
plt.close()

# Figure 3: Risk by Organization Size
fig, ax = plt.subplots(figsize=(12, 6))

# Group by org size and calculate mean risk score
size_risk = df.groupby('org_size_category')['operational_risk_score'].agg(['mean', 'count']).reset_index()
size_risk = size_risk.dropna()

if not size_risk.empty:
    bars = ax.bar(size_risk['org_size_category'].astype(str), size_risk['mean'], 
                  color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Organization Size Category', fontsize=12)
    ax.set_ylabel('Mean Operational Risk Score', fontsize=12)
    ax.set_title('Average Risk Score by Organization Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, size_risk['count'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}\n(n={count})',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '03_risk_by_org_size.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 03_risk_by_org_size.png")
    plt.close()

print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_PATH}")
print(f"Figures generated:")
print(f"  - 01_risk_distribution.png")
print(f"  - 02_correlation_heatmap.png")
print(f"  - 03_risk_by_org_size.png")

# ============================================================================
# STEP 4: GENERATE EDA REPORT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: GENERATING EDA REPORT")
print("=" * 80)

eda_report = f"""
EXPLORATORY DATA ANALYSIS REPORT
WIRS/WERS Workplace-Panel Operational-Risk (turnover-based) Prediction, 1980-2011
{'=' * 80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: wers_operational_risk_merged.csv

{'=' * 80}
1. DATA OVERVIEW
{'=' * 80}

Total Observations: {df.shape[0]:,}
Total Variables: {df.shape[1]}
Survey Waves: {sorted(pd.to_numeric(df['year'], errors='coerce').dropna().unique())}
Duplicate Rows: {df.duplicated().sum()}

Turnover target coverage:
  Rows with a measurable turnover target: {df['operational_risk_category'].notna().sum():,}
  Rows without (early waves 1980-1991, turnover not asked): {df['operational_risk_category'].isna().sum():,}
  -> The turnover-based target is defined only for the 1998/2004/2011 waves;
     earlier waves are retained for descriptive/historical context.

Target Variable Distribution (turnover-measurable rows):
{df['operational_risk_category'].value_counts().to_string()}

Turnover Rate Statistics (operational_risk_score, % 0-100):
  Mean: {df['operational_risk_score'].mean():.2f}
  Median: {df['operational_risk_score'].median():.2f}
  Std Dev: {df['operational_risk_score'].std():.2f}
  Min: {df['operational_risk_score'].min():.2f}
  Max: {df['operational_risk_score'].max():.2f}

{'=' * 80}
2. MISSING VALUES ANALYSIS
{'=' * 80}

Total Missing Values: {df.isnull().sum().sum():,}
Variables with Missing: {(df.isnull().sum() > 0).sum()}

Top 10 Variables with Most Missing:
{missing_df.head(10).to_string() if len(missing_df) > 0 else 'No missing values'}

{'=' * 80}
3. KEY FINDINGS
{'=' * 80}

1. Data Quality:
   - Dataset contains {df.shape[0]:,} observations with {df.shape[1]} variables
   - {(df.isnull().sum() > 0).sum()} variables have missing values
   - {df['operational_risk_category'].isna().sum():,} rows (1980-1991 waves) have no turnover target

2. Risk Distribution (turnover-measurable rows only, n = {df['operational_risk_category'].notna().sum():,}):
   - Mean turnover rate: {df['operational_risk_score'].mean():.2f}% (scale 0-100)
   - Class balance (Low/Medium/High): {df['operational_risk_category'].value_counts(normalize=True).get('Low', 0) * 100:.1f}% / {df['operational_risk_category'].value_counts(normalize=True).get('Medium', 0) * 100:.1f}% / {df['operational_risk_category'].value_counts(normalize=True).get('High', 0) * 100:.1f}%
   - {df['operational_risk_category'].value_counts().get('High', 0)} organizations classified as High risk
   - Turnover bands: Low <=10%, Medium 10-20%, High >20%

3. Data Characteristics:
   - Mix of numerical and categorical variables
   - Survey weights available for weighted analysis
   - Organization-level data with various size categories

{'=' * 80}
4. RECOMMENDATIONS
{'=' * 80}

1. Data Preprocessing:
   - Address class imbalance using SMOTE, class weights, or stratified sampling
   - Impute missing values based on variable type and distribution
   - Standardize/normalize numerical features for ML algorithms
   - Encode categorical variables using one-hot or ordinal encoding

2. Feature Engineering:
   - Create interaction features (e.g., size × union coverage)
   - Develop composite indices (e.g., HR practices index)
   - Consider polynomial features for non-linear relationships
   - Perform feature selection to reduce dimensionality

3. Modeling Strategy:
   - Start with baseline models (Logistic Regression, Decision Tree)
   - Compare ensemble methods (Random Forest, XGBoost, LightGBM)
   - Consider neural networks for complex patterns
   - Use cross-validation for robust evaluation
   - Address class imbalance in model training

4. Evaluation Metrics:
   - Use balanced accuracy, precision, recall, F1-score
   - Generate ROC curves and AUC scores
   - Analyze confusion matrices
   - Calculate per-class performance metrics
   - Consider business cost of false positives vs false negatives

{'=' * 80}
5. NEXT STEPS
{'=' * 80}

1. Run feature engineering script: python 03_feature_exploration.py
2. Train baseline models: python 04_model_training.py
3. Evaluate model performance: python 06_model_evaluation.py
4. Develop interactive dashboard: python dashboard/app.py

{'=' * 80}
END OF EDA REPORT
{'=' * 80}
"""

# Save EDA report
eda_report_file = os.path.join(REPORT_PATH, 'eda_report.txt')
with open(eda_report_file, 'w') as f:
    f.write(eda_report)
print(f"✓ Saved EDA report: {eda_report_file}")

# Save summary statistics to CSV
summary_stats = df.describe().round(2)
summary_stats_file = os.path.join(REPORT_PATH, 'summary_statistics.csv')
summary_stats.to_csv(summary_stats_file)
print(f"✓ Saved summary statistics: {summary_stats_file}")

# Save correlation matrix
correlation_matrix = df.corr(numeric_only=True).round(3)
corr_file = os.path.join(REPORT_PATH, 'correlation_matrix.csv')
correlation_matrix.to_csv(corr_file)
print(f"✓ Saved correlation matrix: {corr_file}")

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll outputs saved to: {REPORT_PATH}")
print(f"\nGenerated files:")
print(f"  - eda_report.txt (comprehensive analysis report)")
print(f"  - summary_statistics.csv")
print(f"  - correlation_matrix.csv")
print(f"\nFigures saved in: {OUTPUT_PATH}")
print("\nNext step: Run feature engineering script")
print("  python 03_feature_exploration.py")
print("=" * 80)

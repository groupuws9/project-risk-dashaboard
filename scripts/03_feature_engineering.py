"""
03_feature_engineering.py — Exploratory feature analysis (RQ1 companion).

Role: leakage-safe candidate-feature exploration — imputation/encoding
experiments, variance & correlation screening, RF-based importance and a
feature-list export. The AUTHORITATIVE modelling pipeline is
scripts/04_model_training.py (single sklearn Pipeline); this script never
trains the deployed model and must agree with it on the leakage rule:

    the raw risk indicators below CONSTRUCT the target and can NEVER be
    candidate features (see models/saved_models/model_metadata.json ->
    leakage_excluded; verified by system test ST3).

Outputs:
    data/processed/wers2011_features_engineered.csv   (exploratory matrix)
    results/features/feature_importance.csv, feature_list.json,
    feature_engineering_report.txt, feature_engineering_artifacts.pkl
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
import json
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration (BASE-anchored like the other pipeline scripts)
BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / 'data' / 'processed' / 'wers2011_operational_risk_merged.csv'
OUTPUT_PATH = BASE / 'data' / 'processed'
FEATURES_PATH = BASE / 'results' / 'features'
os.makedirs(FEATURES_PATH, exist_ok=True)

# Feature engineering parameters
MISSING_THRESHOLD = 0.8  # Drop variables with >80% missing
CORR_THRESHOLD = 0.95  # Drop one of highly correlated pair
OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', or 'winsorize'
SCALING_METHOD = 'standard'  # 'standard', 'minmax', 'robust', or 'none'

print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND INSPECT DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: LOADING AND INSPECTING DATA")
print("=" * 80)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Separate target and features
target_col = 'operational_risk_category'
score_col = 'operational_risk_score'
id_cols = ['year', 'weight']

# Leakage control — identical rule to scripts/04_model_training.py:
# these columns built the target and can never be model candidates.
LEAKAGE_COLS = [
    'employee_turnover_rate', 'employee_turnover_rate_5yr',
    'absenteeism_rate', 'grievance_rate', 'disputes_raised',
    'dispute_risk_level', 'other_industrial_action', 'dismissal_rate',
    'disciplinary_warnings', 'suspensions',
]

y = df[target_col].copy()
y_score = df[score_col].copy()
ids = df[id_cols].copy() if all(col in df.columns for col in id_cols) else pd.DataFrame()

# Candidate features (pre-existing characteristics ONLY). Prefer the exact
# contract the deployed model was trained on so this script can never
# contradict the modelling pipeline.
_meta_path = BASE / 'models' / 'saved_models' / 'model_metadata.json'
if _meta_path.exists():
    feature_cols = json.loads(_meta_path.read_text())['features']
    print(f"✓ Using trained-model feature contract ({len(feature_cols)} features "
          f"from model_metadata.json)")
else:
    feature_cols = [col for col in df.columns
                    if col not in [target_col, score_col] + id_cols + LEAKAGE_COLS]
    print(f"✓ Using leakage-safe candidate features ({len(feature_cols)})")
assert not (set(feature_cols) & set(LEAKAGE_COLS)), "leakage columns leaked in!"
X = df[feature_cols].copy()

print(f"✓ Features: {X.shape[1]} variables")
print(f"✓ Target: {target_col}")
print(f"✓ Classes: {y.nunique()} ({y.unique()})")

# ============================================================================
# STEP 2: HANDLE EDGE CASES - DATA VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: DATA VALIDATION AND EDGE CASE HANDLING")
print("=" * 80)

# Check for infinite values
inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
if inf_count > 0:
    print(f"⚠ Found {inf_count} infinite values - replacing with NaN")
    X = X.replace([np.inf, -np.inf], np.nan)
else:
    print("✓ No infinite values found")

# Check for duplicate columns
duplicate_cols = X.columns[X.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"⚠ Found duplicate columns: {duplicate_cols} - removing duplicates")
    X = X.loc[:, ~X.columns.duplicated()]
else:
    print("✓ No duplicate columns found")

# Check for columns with single unique value (no variance)
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"⚠ Found {len(constant_cols)} constant columns - removing: {constant_cols[:5]}...")
    X = X.drop(columns=constant_cols)
else:
    print("✓ No constant columns found")

# Check for high missing value rates
missing_pct = X.isnull().mean()
high_missing = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
if high_missing:
    print(f"⚠ Found {len(high_missing)} columns with >{MISSING_THRESHOLD*100:.0f}% missing - removing")
    X = X.drop(columns=high_missing)
else:
    print(f"✓ No columns with >{MISSING_THRESHOLD*100:.0f}% missing values")

print(f"\n✓ After validation: {X.shape[1]} feature columns remaining")

# ============================================================================
# STEP 3: MISSING VALUE IMPUTATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: MISSING VALUE IMPUTATION")
print("=" * 80)

# Separate numerical and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Impute numerical features
if numeric_cols:
    # Check missing percentage
    num_missing = X[numeric_cols].isnull().sum()
    num_missing_pct = (num_missing / len(X) * 100).round(2)
    
    print(f"\nNumerical features with missing values: {(num_missing > 0).sum()}")
    
    if (num_missing > 0).sum() > 0:
        # Use median imputation for numerical features (robust to outliers)
        imputer_num = SimpleImputer(strategy='median')
        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])
        print("✓ Imputed numerical features using median strategy")
    else:
        print("✓ No missing values in numerical features")

# Impute categorical features
if categorical_cols:
    cat_missing = X[categorical_cols].isnull().sum()
    
    print(f"\nCategorical features with missing values: {(cat_missing > 0).sum()}")
    
    if (cat_missing > 0).sum() > 0:
        # Use most_frequent imputation for categorical features
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
        print("✓ Imputed categorical features using most_frequent strategy")
    else:
        print("✓ No missing values in categorical features")

print(f"\n✓ Missing value imputation complete")
print(f"  Total missing values remaining: {X.isnull().sum().sum()}")

# ============================================================================
# STEP 4: OUTLIER DETECTION AND TREATMENT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: OUTLIER DETECTION AND TREATMENT")
print("=" * 80)

# Detect outliers using IQR method
def detect_outliers_iqr(data, col):
    """Detect outliers using IQR method"""
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check for outliers in numerical features
outlier_summary = {}
numeric_cols = X.select_dtypes(include=[np.number]).columns

print(f"Checking for outliers in {len(numeric_cols)} numerical features...\n")

for col in numeric_cols:
    outliers, lower, upper = detect_outliers_iqr(X, col)
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(X)) * 100
    
    if outlier_count > 0:
        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': outlier_pct,
            'lower_bound': lower,
            'upper_bound': upper
        }

print(f"Features with outliers: {len(outlier_summary)}")

if outlier_summary:
    print("\nTop 10 features with most outliers:")
    sorted_outliers = sorted(outlier_summary.items(), key=lambda x: x[1]['percentage'], reverse=True)
    for col, info in sorted_outliers[:10]:
        print(f"  {col}: {info['count']} ({info['percentage']:.2f}%)")

# Treat outliers using winsorization (cap at 1st and 99th percentile)
print("\nTreating outliers using winsorization (1st-99th percentile)...")

for col, info in outlier_summary.items():
    lower_percentile = X[col].quantile(0.01)
    upper_percentile = X[col].quantile(0.99)
    
    X[col] = X[col].clip(lower=lower_percentile, upper=upper_percentile)

print("✓ Outlier treatment complete")

# ============================================================================
# STEP 5: FEATURE SCALING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: FEATURE SCALING")
print("=" * 80)

# Separate numerical and categorical features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Scaling {len(numeric_cols)} numerical features using {SCALING_METHOD} scaling")

# Apply scaling
if SCALING_METHOD == 'standard':
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print("✓ Applied StandardScaler (zero mean, unit variance)")
    
elif SCALING_METHOD == 'minmax':
    scaler = MinMaxScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print("✓ Applied MinMaxScaler (range [0, 1])")
    
elif SCALING_METHOD == 'robust':
    scaler = RobustScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print("✓ Applied RobustScaler (median and IQR)")
    
else:
    print("⚠ No scaling applied")

# Save scaler for later use
if SCALING_METHOD in ['standard', 'minmax', 'robust']:
    scaler_file = os.path.join(FEATURES_PATH, f'{SCALING_METHOD}_scaler.pkl')
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: {scaler_file}")

# ============================================================================
# STEP 6: CATEGORICAL ENCODING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: CATEGORICAL ENCODING")
print("=" * 80)

if categorical_cols:
    print(f"Encoding {len(categorical_cols)} categorical variables")
    
    # Use Label Encoding for ordinal, One-Hot for nominal
    # For now, use Label Encoding for simplicity
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  ✓ Encoded {col}: {len(le.classes_)} categories")
    
    # Save label encoders
    encoders_file = os.path.join(FEATURES_PATH, 'label_encoders.pkl')
    with open(encoders_file, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✓ Saved label encoders: {encoders_file}")
    
else:
    print("✓ No categorical variables to encode")

# ============================================================================
# STEP 7: FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: FEATURE SELECTION")
print("=" * 80)

# Remove highly correlated features
print("Checking for highly correlated features (threshold > 0.95)...")

# Calculate correlation matrix
corr_matrix = X.corr().abs()

# Find highly correlated pairs
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESHOLD)]

if to_drop:
    print(f"  Removing {len(to_drop)} highly correlated features:")
    for col in to_drop[:10]:  # Show first 10
        print(f"    - {col}")
    if len(to_drop) > 10:
        print(f"    ... and {len(to_drop) - 10} more")
    
    X_selected = X.drop(columns=to_drop)
    print(f"✓ Features reduced from {X.shape[1]} to {X_selected.shape[1]}")
else:
    print("✓ No highly correlated features found")
    X_selected = X.copy()

# Feature importance using Random Forest
print("\nCalculating feature importance using Random Forest...")

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Train Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_selected, y_encoded)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_selected.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Save feature importance
feature_importance.to_csv(os.path.join(FEATURES_PATH, 'feature_importance.csv'), index=False)
print(f"\n✓ Saved feature importance rankings")

# Select top K features
TOP_K_FEATURES = 30
selected_features = feature_importance.head(TOP_K_FEATURES)['feature'].tolist()
X_final = X_selected[selected_features].copy()

print(f"\n✓ Selected top {TOP_K_FEATURES} features for final model")
print(f"  Final feature matrix shape: {X_final.shape}")

# ============================================================================
# STEP 8: SAVE PROCESSED FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: SAVING PROCESSED FEATURES")
print("=" * 80)

# Create final dataset
final_df = X_final.copy()
final_df[target_col] = y.values
final_df[score_col] = y_score.values
if not ids.empty:
    final_df = pd.concat([ids.reset_index(drop=True), final_df], axis=1)

# Save final feature set
output_file = os.path.join(OUTPUT_PATH, 'wers2011_features_engineered.csv')
final_df.to_csv(output_file, index=False)
print(f"✓ Saved engineered features: {output_file}")
print(f"  Shape: {final_df.shape}")

# Save feature list
feature_list = {
    'target': target_col,
    'features': selected_features,
    'id_columns': id_cols,
    'n_features': len(selected_features),
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

feature_list_file = os.path.join(FEATURES_PATH, 'feature_list.json')
with open(feature_list_file, 'w') as f:
    json.dump(feature_list, f, indent=2)
print(f"✓ Saved feature list: {feature_list_file}")

# Save transformers
artifacts = {
    'label_encoders': label_encoders if 'label_encoders' in locals() else {},
    'scaler': scaler if 'scaler' in locals() else None,
    'label_encoder_target': le_target,
    'selected_features': selected_features
}

artifacts_file = os.path.join(FEATURES_PATH, 'feature_engineering_artifacts.pkl')
with open(artifacts_file, 'wb') as f:
    pickle.dump(artifacts, f)
print(f"✓ Saved feature engineering artifacts: {artifacts_file}")

# ============================================================================
# STEP 9: GENERATE FEATURE ENGINEERING REPORT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: GENERATING FEATURE ENGINEERING REPORT")
print("=" * 80)

feature_eng_report = f"""
FEATURE ENGINEERING REPORT
WERS 2011 Operational Risk Prediction Dataset
{'=' * 80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 80}
1. PROCESSING SUMMARY
{'=' * 80}

Original Dataset:
  - Observations: {df.shape[0]:,}
  - Features: {X.shape[1]}

After Feature Engineering:
  - Observations: {final_df.shape[0]:,}
  - Features: {X_final.shape[1]}
  - Reduction: {((X.shape[1] - X_final.shape[1]) / X.shape[1] * 100):.1f}%

{'=' * 80}
2. PROCESSING STEPS APPLIED
{'=' * 80}

1. Data Validation:
   - Checked for infinite values
   - Removed duplicate columns
   - Removed constant features (zero variance)
   - Identified features with high missing rates (>{MISSING_THRESHOLD*100:.0f}%)

2. Missing Value Imputation:
   - Numerical features: Median imputation
   - Categorical features: Most frequent imputation
   - Result: Zero missing values after imputation

3. Outlier Treatment:
   - Method: Winsorization at 1st and 99th percentiles
   - Applied to all numerical features
   - Preserved data distribution while reducing extreme values

4. Feature Scaling:
   - Method: {SCALING_METHOD.capitalize()} scaling
   - Applied to all numerical features
   - Ensures comparable feature magnitudes

5. Categorical Encoding:
   - Method: Label Encoding
   - All categorical variables converted to numerical
   - Encoders saved for future use

6. Feature Selection:
   - Removed highly correlated features (r > {CORR_THRESHOLD})
   - Applied Random Forest feature importance
   - Selected top {TOP_K_FEATURES} most important features
   - Final feature set optimized for predictive performance

{'=' * 80}
3. FEATURE IMPORTANCE RANKINGS
{'=' * 80}

Top 20 Most Important Features:
{feature_importance.head(20).to_string(index=False)}

{'=' * 80}
4. DATA QUALITY METRICS
{'=' * 80}

Missing Values:
  - Before imputation: {X.isnull().sum().sum():,}
  - After imputation: {X_final.isnull().sum().sum()}

Duplicate Records: {df.duplicated().sum()}

Class Distribution (Target Variable):
{df[target_col].value_counts().to_string()}

Class Balance Ratio: {df[target_col].value_counts().min() / df[target_col].value_counts().max():.3f}

{'=' * 80}
5. OUTPUT FILES GENERATED
{'=' * 80}

1. wers2011_features_engineered.csv
   - Final processed dataset with engineered features
   - Shape: {final_df.shape}
   - Ready for machine learning

2. feature_importance.csv
   - Feature importance rankings from Random Forest
   - {len(feature_importance)} features ranked

3. feature_list.json
   - JSON file with selected features and metadata
   - Includes feature names and processing parameters

4. feature_engineering_artifacts.pkl
   - Pickled transformers (scalers, encoders)
   - For applying same transformations to new data

5. feature_engineering_report.txt
   - This comprehensive report

{'=' * 80}
6. RECOMMENDATIONS
{'=' * 80}

1. Class Imbalance:
   - Current class distribution is imbalanced (Low: {df[target_col].value_counts()['Low']}, 
     Medium: {df[target_col].value_counts()['Medium']}, High: {df[target_col].value_counts()['High']})
   - Recommend using class weights or resampling techniques (SMOTE)
   - Consider using balanced accuracy or F1-macro for evaluation

2. Feature Selection:
   - Top {TOP_K_FEATURES} features selected based on Random Forest importance
   - Consider using other methods (RFE, L1 regularization) for comparison
   - Monitor for overfitting with reduced feature set

3. Model Selection:
   - Start with baseline models (Logistic Regression, Decision Tree)
   - Try ensemble methods (Random Forest, XGBoost, LightGBM)
   - Consider neural networks for complex patterns

4. Validation Strategy:
   - Use stratified k-fold cross-validation due to class imbalance
   - Reserve holdout test set for final evaluation
   - Monitor for overfitting on training data

5. Interpretability:
   - Use SHAP values or LIME for model explainability
   - Generate feature importance plots
   - Create partial dependence plots for key features

{'=' * 80}
END OF FEATURE ENGINEERING REPORT
{'=' * 80}
"""

# Save feature engineering report
report_file = os.path.join(FEATURES_PATH, 'feature_engineering_report.txt')
with open(report_file, 'w') as f:
    f.write(feature_eng_report)
print(f"✓ Saved feature engineering report: {report_file}")

# ============================================================================
# STEP 10: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print(f"\nInput: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Output: {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
print(f"\nFeature reduction: {X.shape[1]} → {X_final.shape[1]} features ({((X.shape[1] - X_final.shape[1]) / X.shape[1] * 100):.1f}% reduction)")
print(f"\nOutput files:")
print(f"  - wers2011_features_engineered.csv")
print(f"  - feature_importance.csv")
print(f"  - feature_list.json")
print(f"  - feature_engineering_artifacts.pkl")
print(f"  - feature_engineering_report.txt")
print(f"\nNext step: Run model training script")
print(f"  python 04_model_training.py")
print("=" * 80)

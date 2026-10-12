# Project Risk Dashboard - Technical Specification

## Dataset Characteristics

### Overview
| Attribute | Value |
|-----------|-------|
| **Dataset** | WERS 2011 Operational Risk Dataset |
| **Source** | UKDA Study 4511 (WERS 2011) |
| **Unit of Analysis** | Organization-level (workplace surveys) |
| **Total Observations** | 6,635 |
| **Total Variables** | 52 |
| **Predictive Features** | 51 |
| **Target Variable** | `operational_risk_category` |

### Class Distribution (Target)
| Category | Count | Percentage |
|----------|-------|------------|
| **Low** | 5,502 | 82.92% |
| **Medium** | 1,105 | 16.65% |
| **High** | 28 | 0.42% |
| **Total** | **6,635** | **100%** |

---

## Target Definition & Construction

### Target Variable
- **Name**: `operational_risk_category`
- **Encoding**: Low=0, Medium=1, High=2 (in model training)
- **Class Labels**: ["Low", "Medium", "High"]

### Risk Score Calculation (`operational_risk_score`)
The operational risk score is a composite index (0-100) calculated from four risk dimensions, each contributing up to 25 points:

#### 1. Employee Turnover (Max 25 points)
| Turnover Rate | Points |
|---------------|--------|
| > 20% | 25 |
| > 15% | 20 |
| > 10% | 15 |
| > 5% | 10 |
| <= 5% | 5 |

#### 2. Absenteeism (Max 25 points)
| Absence Rate | Points |
|--------------|--------|
| > 5% | 25 |
| > 4% | 20 |
| > 3% | 15 |
| > 2% | 10 |
| <= 2% | 5 |

#### 3. Industrial Action (Max 25 points)
| Event | Points |
|-------|--------|
| Dispute risk level = 1 | 25 |
| Other industrial action = 1 | 15 |

#### 4. Grievances (Max 25 points)
| Grievance Rate | Points |
|----------------|--------|
| > 10 | 25 |
| > 5 | 20 |
| > 2 | 15 |
| <= 2 | 5 |

### Risk Category Thresholds
| Category | Score Range | Description |
|----------|-------------|-------------|
| **High** | >= 60 | Critical risk requiring intervention |
| **Medium** | 35-59 | Moderate risk requiring monitoring |
| **Low** | < 35 | Low risk |

---

## Leakage Prevention

### Excluded Target-Defining Predictors
| Category | Excluded Columns |
|----------|------------------|
| **Risk Indicators** | `employee_turnover_rate`, `employee_turnover_rate_5yr`, `absenteeism_rate`, `grievance_rate`, `disputes_raised`, `dispute_risk_level`, `other_industrial_action`, `dismissal_rate`, `disciplinary_warnings`, `suspensions` |
| **Score** | `operational_risk_score` |
| **Identifiers** | `serno`, `year`, `weight` |

**Sources**: `scripts/04_model_training.py:51-55`, `scripts/01_data_preparation.py:74-93`

---

## Implementation Alignment

### 1. Implemented Models (5 total)
| # | Model | Library | Class |
|---|-------|---------|-------|
| 1 | Logistic Regression | sklearn | `LogisticRegression` |
| 2 | Random Forest | sklearn | `RandomForestClassifier` |
| 3 | MLP (Neural Network) | sklearn | `MLPClassifier` |
| 4 | XGBoost | xgboost | `XGBClassifier` |
| 5 | LightGBM | lightgbm | `LGBMClassifier` |

**Sources**: `scripts/04_model_training.py:18-36`

### 2. Imbalance Handling
| Method | Implementation |
|--------|---------------|
| **Class Weights** | `class_weight="balanced"` or `"balanced_subsample"` for all models |
| **Stratified Splits** | `StratifiedKFold` + `cross_validate` |

**Sources**: `scripts/04_model_training.py:27-36, 88-100`

### 3. SHAP Explainers
| Model Type | Explainer | Used For |
|------------|-----------|----------|
| Tree-based (XGB, LGBM, RF) | `shap.TreeExplainer` | `XGBClassifier`, `LGBMClassifier`, `RandomForestClassifier` |
| Linear models | `shap.LinearExplainer` | `LogisticRegression` |

**Sources**: `scripts/06_shap_explanation.py:50-60`

### 4. API Validation Rules
| Error Code | Status | Description |
|------------|--------|-------------|
| `MISSING_FIELD` | 400 | Required field absent or null |
| `INVALID_TYPE` | 400 | Value cannot be coerced to declared type |
| `OUT_OF_RANGE` | 400 | Value outside allowed domain |
| `MALFORMED_JSON` | 400 | Body is not valid JSON |
| `MODEL_ERROR` | 422 | Inference failed (NaN / incomplete vector) |
| `INTERNAL_ERROR` | 500 | Unexpected server failure |
| `RATE_LIMITED` | 429 | Too many requests |
| `SERVICE_UNAVAILABLE` | 503 | Service unreachable |

**Validation Constraints**:
- `greater_than` / `less_than` thresholds (Pydantic Field constraints)
- Percentages: 0-100 range
- Codes: 0-99 range

**Sources**: `dashboard/validation.py:14-45`

## Experiment Results (From Model Training)

### Model Comparison (Test Set)
| Model | Accuracy | Balanced Accuracy | Macro F1 |
|-------|----------|-------------------|----------|
| **LogReg** | 0.650 | 0.548 | 0.410 |
| **LightGBM** | 0.797 | 0.441 | 0.436 |
| **Random Forest** | 0.822 | 0.435 | 0.442 |
| **XGBoost** | 0.829 | 0.404 | 0.417 |
| **MLP** | 0.826 | 0.376 | 0.381 |

*Source: `scripts/04_model_training.py` evaluation output*

---

## Key Observations

1. **Severe Class Imbalance**: 0.42% of organizations are "High" risk - critical for using `class_weight="balanced"`
2. **No Leakage**: Raw risk indicators excluded from features, only pre-existing characteristics used
3. **Ensemble Ready**: Tree-based models handle imbalance naturally; linear models use class weights
4. **Dashboard Ready**: All required files (`shap_importance.csv`, `shap_local.json`) now generated

---

*Document generated from: `scripts/01_data_preparation.py`, `scripts/04_model_training.py`, `scripts/06_shap_explanation.py`, `dashboard/app.py`, `dashboard/validation.py`, and actual data files.*

---

## Feature Categories Used for Modeling

| Category | Variables | Count |
|----------|-----------|-------|
| **Organizational** | org_size_category, org_size_category_3, org_size_category_4, sector, sector_detailed, multinational, org_age_years, workplace_age, trading_status | 9 |
| **HR Practices** | union_coverage_pct, union_coverage_pct_6, union_coverage_pct_7, collective_bargaining, bargaining_level_org, bargaining_level_est, formal_hr_strategy, performance_appraisal_pct, training_budget_exists, training_problems, job_security_policy | 11 |
| **Employee Relations** | jcc_exists, jcc_frequency, jcc_function, jcc_union_chair, information_sharing_score, info_chain_exist, newsletter_exists, investment_plans_shared, consultation_depth, employee_voice_channels, grievance_procedure_formal | 11 |
| **Labor Market** | pct_part_time, pct_part_time_12, pct_female, pct_temporary, skill_shortage_reported | 5 |
| **Total Features** | | **51** |

---

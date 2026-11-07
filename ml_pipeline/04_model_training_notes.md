# ml_pipeline\04_model_training.py - Issues Found and Fixes

## Critical Issues (Fix Before Running)

### 1. f-string Syntax Error
**Location**: Line ~120 in the leakage detection section  
**Current code**: `f"Leakage-prone columns detected: {sorted(leaked_columns}"`  
**Fix**: Change to `f"Leakage-prone columns detected: {sorted(leaked_columns)}"`  
**Why**: The f-string missing closing brace `}` before the closing quote `"` will cause a SyntaxError

### 2. Duplicate `if leaked_columns:` Blocks
**Location**: Lines 116 and 118  
**Issue**: Two `if leaked_columns:` statements in sequence  
**Fix**: Consolidate into a single block. The structure should be:
```python
# Check for leakage-prone columns
if leaked_columns:
    raise ValueError(
        f"Leakage-prone columns detected: {sorted(leaked_columns)}"
    )

# Check for duplicate predictor columns
if len(predictor_columns) != len(set(predictor_columns)):
    raise ValueError("Duplicate predictor columns detected.")
```

### 3. Hardcoded Thresholds for Column Type Detection
**Location**: Lines ~150-173 (numeric/categorical classification)  
**Current thresholds**: `numeric_ratio > 0.85` and `nunique(dropna=True) > 12`  
**Fix**: Make configurable via data_contract.py or function parameters  
**Example**: Add to data_contract.py:
```python
NUMERIC_RATIO_THRESHOLD = 0.85
CARDINALITY_THRESHOLD = 12
```

## Recommended Improvements

### 4. Add Configuration Constants to data_contract.py
```python
# After existing constants
NUMERIC_RATIO_THRESHOLD = 0.85
CARDINALITY_THRESHOLD = 12
RARE_THRESHOLD = 0.01  # already exists
```

### 5. Improve Error Handling in Cross-Validation
Add reporting for failed folds:
```python
# After cross_validate
if any(pd.isna(cv_results["test_accuracy"])):
    print(f"Warning: {sum(pd.isna(cv_results['test_accuracy']))} CV folds had errors")
```

### 6. Add Validation for Empty Predictor Columns
After leakage detection, add:
```python
if not predictor_columns:
    raise ValueError("No predictor columns remaining after leakage exclusion")
```

### 7. Document Model Dependency Requirements
Add a check at the start of the script:
```python
missing_libs = []
if not HAS_XGB:
    missing_libs.append("xgboost")
if not HAS_LGBM:
    missing_libs.append("lightgbm")
if missing_libs:
    print(f"Note: {', '.join(missing_libs)} not installed. Using default models.")
```

### 8. Fix UTF-8 BOM Issue
- Re-save the file without UTF-8 BOM, or
- Ensure all file operations use `encoding='utf-8-sig'`

## Summary of Changes Needed

1. **Fix f-string syntax** - critical for script to run
2. **Consolidate duplicate leakage checks** - critical for logic correctness
3. **Make thresholds configurable** - for maintainability
4. **Add error reporting** - for debugging
5. **Add validation checks** - for robustness
6. **Fix file encoding** - to prevent parsing issues

The core ML pipeline architecture (leakage-safe CV, outer train-test split, proper model selection then evaluation) is well-designed. The identified issues are primarily syntax, configuration, and robustness-related that should be addressed before regular use.
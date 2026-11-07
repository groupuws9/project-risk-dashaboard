#!/usr/bin/env python
# Test the column type classification with MISSING values
import sys
import os
sys.path.insert(0, r"d:\new research topics\project risk dashaboard\ml_pipeline")

import data_contract as dc
import numpy as np
import pandas as pd

# Simulate the column type classification logic from 04_model_training.py
test_series = pd.Series(['5', '10', 'MISSING', '15', '20'])

series = test_series.replace(dc.MISS_TOKENS, np.nan)
coerced = pd.to_numeric(series, errors="coerce")
valid_count = series.notna().sum()
numeric_ratio = (coerced.notna().sum() / valid_count) if valid_count > 0 else 0

print("Test series:", test_series.values)
print("After replacing MISS_TOKENS with NaN:", series.values)
print("Coerced (pd.to_numeric):", coerced.values)
print("valid_count:", valid_count)
print("numeric_ratio:", numeric_ratio)
print("coerced.nunique():", coerced.nunique(dropna=True))
print("nunique > 12:", coerced.nunique(dropna=True) > 12)
print("numeric_ratio > 0.85:", numeric_ratio > 0.85)
print()
print("Column would be classified as numeric:", numeric_ratio > 0.85 and coerced.nunique(dropna=True) > 12)
print()

# Test with more MISSING values
test_series2 = pd.Series(['5', 'MISSING', 'MISSING', '15', '20'])
series2 = test_series2.replace(dc.MISS_TOKENS, np.nan)
coerced2 = pd.to_numeric(series2, errors="coerce")
valid_count2 = series2.notna().sum()
numeric_ratio2 = (coerced2.notna().sum() / valid_count2) if valid_count2 > 0 else 0

print("Test series2:", test_series2.values)
print("After replacing MISS_TOKENS with NaN:", series2.values)
print("Coerced:", coerced2.values)
print("valid_count2:", valid_count2)
print("numeric_ratio2:", numeric_ratio2)
print("nunique2:", coerced2.nunique(dropna=True))
print("Column would be classified as numeric:", numeric_ratio2 > 0.85 and coerced2.nunique(dropna=True) > 12)
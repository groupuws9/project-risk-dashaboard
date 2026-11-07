#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Examine NumericCoercionTransformer and CategoricalCleaningTransformer
"""
import sys
import traceback
sys.path.insert(0, r"d:\new research topics\project risk dashaboard\ml_pipeline")

import data_contract as dc
import numpy as np
import pandas as pd

# Create a test dataframe with MISSING values
test_data = pd.DataFrame({
    'col1': ['5', '10', 'MISSING', '15'],
    'col2': ['a', 'b', 'MISSING', 'd'],
    'col3': ['1', '2', '3', '4']
})

print("Test data:")
print(test_data)
print()

# Test NumericCoercionTransformer
print("Testing NumericCoercionTransformer...")
try:
    transformer = dc.NumericCoercionTransformer(numeric_columns=['col1', 'col3'])
    result = transformer.fit_transform(test_data)
    print("NumericCoercionTransformer result:")
    print(result)
    print("dtypes:", result.dtypes)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print()

# Test CategoricalCleaningTransformer
print("Testing CategoricalCleaningTransformer...")
try:
    transformer = dc.CategoricalCleaningTransformer()
    result = transformer.fit_transform(test_data)
    print("CategoricalCleaningTransformer result:")
    print(result)
    print("dtypes:", result.dtypes)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print()
print("MISS_TOKENS:", dc.MISS_TOKENS)
print("RARE_THRESHOLD:", dc.RARE_THRESHOLD)
print("NUMERIC_RATIO_THRESHOLD:", dc.NUMERIC_RATIO_THRESHOLD)
print("CARDINALITY_THRESHOLD:", dc.CARDINALITY_THRESHOLD)
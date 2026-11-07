#!/usr/bin/env python
# Simple test of NumericCoercionTransformer
import sys
import os
os.chdir(r"d:\new research topics\project risk dashaboard")

# Directly import and test
from ml_pipeline import data_contract as dc
import pandas as pd
import numpy as np

# Create test data with MISSING values
test_df = pd.DataFrame({
    'numeric_with_missing': ['5', '10', 'MISSING', '15', '20'],
    'categorical': ['a', 'b', 'MISSING', 'd', 'e'],
    'pure_numeric': ['1', '2', '3', '4', '5']
})

print("Original DataFrame:")
print(test_df)
print("dtypes:", test_df.dtypes)
print()

# Test NumericCoercionTransformer
print("Testing NumericCoercionTransformer...")
try:
    # Select only the numeric column for testing
    numeric_cols = ['numeric_with_missing']
    transformer = dc.NumericCoercionTransformer(numeric_columns=numeric_cols)
    result = transformer.fit_transform(test_df)
    print("Result:")
    print(result)
    print("dtypes:", result.dtypes)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()
print("MISS_TOKENS:", dc.MISS_TOKENS)
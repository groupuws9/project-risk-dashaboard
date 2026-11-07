#!/usr/bin/env python
import sys
import re

# Read the file
with open(r"d:/new research topics/project risk dashaboard/ml_pipeline/04_model_training.py", "r", encoding="utf-8-sig") as f:
    content = f.read()

print(f"Read file: {len(content)} chars")

# FIX 1: Fix f-string syntax
old_fstring = 'f"Leakage-prone columns detected: {sorted(leaked_columns}"'
new_fstring = 'f"Leakage-prone columns detected: {sorted(leaked_columns)}"'

if old_fstring in content:
    content = content.replace(old_fstring, new_fstring)
    print("FIX 1: f-string syntax fixed")
else:
    print("FIX 1: f-string pattern not found")

# FIX 2: Consolidate leakage check block
# Replace the incomplete block with the complete one
old_block = """leaked_columns = set(predictor_columns) & prohibited_columns

if leaked_columns:



print(f"Verified {len(predictor_columns)} unique leakage-free predictors")"""

new_block = """leaked_columns = set(predictor_columns) & prohibited_columns

# Check for leakage-prone columns
if leaked_columns:
    raise ValueError(
        f"Leakage-prone columns detected: {sorted(leaked_columns)}"
    )

# Check for duplicate predictor columns
if len(predictor_columns) != len(set(predictor_columns)):
    raise ValueError("Duplicate predictor columns detected.")

print(f"Verified {len(predictor_columns)} unique leakage-free predictors")"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print("FIX 2: leakage check consolidated")
else:
    print("FIX 2: old block pattern not found")

# FIX 3: Add MISS_TOKENS replacement after df = pd.read_csv(DATA)
data_pos = content.find('df = pd.read_csv(DATA)')
if data_pos >= 0:
    eol = content.find('\n', data_pos)
    insert_text = '''

# Replace MISS_TOKENS with NaN BEFORE analyzing column types
# This ensures sentinel values like "MISSING" are properly handled
df = df.replace(dc.MISS_TOKENS, np.nan)

'''
    content = content[:eol+1] + insert_text + content[eol+1:]
    print("FIX 3: MISS_TOKENS replacement added")
else:
    print("FIX 3: could not find data position")

# FIX 4: Update thresholds
if 'numeric_ratio > 0.85' in content:
    content = content.replace('numeric_ratio > 0.85', 'numeric_ratio > dc.NUMERIC_RATIO_THRESHOLD')
    print("FIX 4a: numeric ratio threshold updated")

if 'coerced.nunique(dropna=True) > 12' in content:
    content = content.replace('coerced.nunique(dropna=True) > 12', 'coerced.nunique(dropna=True) > dc.CARDINALITY_THRESHOLD')
    print("FIX 4b: cardinality threshold updated")

# Write the file
with open(r"d:/new research topics/project risk dashaboard/ml_pipeline/04_model_training.py", "w", encoding="utf-8-sig") as f:
    f.write(content)

print("File written successfully")
print("All fixes applied")
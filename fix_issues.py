# -*- coding: utf-8 -*-
import re

with open(r"d:/new research topics/project risk dashaboard/ml_pipeline/04_model_training.py", "r", encoding="utf-8-sig") as f:
    content = f.read()

# Fix 1: Fix the f-string syntax - change {sorted(leaked_columns} to {sorted(leaked_columns)}
content = content.replace(
    'f"Leakage-prone columns detected: {sorted(leaked_columns}"',
    'f"Leakage-prone columns detected: {sorted(leaked_columns)}"'
)

# Fix 2: Consolidate the duplicated leakage check
old_pattern = """leaked_columns = set(predictor_columns) & prohibited_columns

if leaked_columns:"""

new_pattern = """leaked_columns = set(predictor_columns) & prohibited_columns

# Check for leakage-prone columns
if leaked_columns:
    raise ValueError(
        f"Leakage-prone columns detected: {sorted(leaked_columns)}"
    )

# Check for duplicate predictor columns
if len(predictor_columns) != len(set(predictor_columns)):
    raise ValueError("Duplicate predictor columns detected.")

print(f"Verified {len(predictor_columns)} unique leakage-free predictors")"""

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print("Replacement successful")
else:
    print("Old pattern not found, trying alternative...")
    # Try simpler replacement
    if 'if leaked_columns:' in content:
        count = content.count('if leaked_columns:')
        print(f"Found {count} occurrences of 'if leaked_columns:'")

with open(r"d:/new research topics/project risk dashaboard/ml_pipeline/04_model_training.py", "w", encoding="utf-8-sig") as f:
    f.write(content)

print("File written successfully")
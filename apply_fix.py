#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply fixes to ml_pipeline/04_model_training.py
"""
import sys
import os

# Change to the correct directory
os.chdir(r"d:\new research topics\project risk dashaboard")

# Read the file using binary to avoid encoding issues
with open(r"ml_pipeline/04_model_training.py", "rb") as f:
    raw = f.read()

# Decode with utf-8-sig to handle BOM
text = raw.decode("utf-8-sig")

print("File size:", len(text), "chars")

# Check for the old pattern
old1 = 'for column in Xtr_raw.columns:'
old2 = 'series = Xtr_raw[column].replace(dc.MISS_TOKENS, np.nan)'
old3 = 'numeric_ratio > 0.85 and coerced.nunique(dropna=True) > 12'

print("old1 found:", old1 in text)
print("old2 found:", old2 in text)
print("old3 found:", old3 in text)

# Check for new thresholds
new1 = 'dc.NUMERIC_RATIO_THRESHOLD'
new2 = 'dc.CARDINALITY_THRESHOLD'

print("new1 found:", new1 in text)
print("new2 found:", new2 in text)

# Check if MISS_TOKENS replacement is before the loop
mid = text.find('df = pd.read_csv(DATA)')
end_loop = text.find('for column in Xtr_raw.columns:')

print("read_csv at:", mid)
print("loop at:", end_loop)
print("loop is after read_csv:", end_loop > mid)

# Now apply the fixes
print("\n--- Applying Fix 1: Add MISS_TOKENS replacement before column type classification ---")

# Find the location after df = pd.read_csv(DATA) and before the column type loop
# We need to insert the MISS_TOKENS replacement there

# First, let's find where to insert
# Look for the pattern after loading data
data_load_pos = text.find('# ---------------------------------------------------------------------------\n# Load data and define leakage-safe predictors')
if data_load_pos >= 0:
    # Find the end of the loading section (before the column type classification)
    print("Found data loading section at position", data_load_pos)
else:
    print("Could not find data loading section")

# Let's find the exact location to insert
# We'll insert after "df = pd.read_csv(DATA)" and before the column classification
insert_pos = text.find('df = pd.read_csv(DATA)')
if insert_pos >= 0:
    # Find the end of that line
    line_end = text.find('\n', insert_pos)
    insert_after = line_end + 1
    print("Insert position after df.read_csv:", insert_after)
    
    # The new code to insert
    miss_token_replacement = '''

# Replace MISS_TOKENS with NaN BEFORE analyzing column types
# This ensures sentinel values like "MISSING" are properly handled
df = df.replace(dc.MISS_TOKENS, np.nan)

'''
    
    # Insert the MISS_TOKENS replacement
    text = text[:insert_after] + miss_token_replacement + text[insert_after:]
    print("Inserted MISS_TOKENS replacement")
    
    # Now we need to update the column type classification section
    # Find the old pattern and replace it
    old_pattern = '''for column in Xtr_raw.columns:\n    series = Xtr_raw[column].replace(dc.MISS_TOKENS, np.nan)\n    coerced = pd.to_numeric(series, errors="coerce")\n\n    valid_count = series.notna().sum()\n    numeric_ratio = (\n        coerced.notna().sum() / valid_count\n        if valid_count > 0\n        else 0\n    )\n\n    if numeric_ratio > 0.85 and coerced.nunique(dropna=True) > 12:\n        numeric_candidates.append(column)\n    else:\n        categorical_candidates.append(column)\n\nprint(f"Numeric candidate columns: {len(numeric_candidates)}")\nprint(f"Categorical candidate columns: {len(categorical_candidates)}")'''
    
    new_pattern = '''# Note: MISS_TOKENS have already been replaced with NaN above\nnumeric_candidates = []\ncategorical_candidates = []\n\nfor column in df.columns:\n    series = df[column]\n    coerced = pd.to_numeric(series, errors="coerce")\n\n    valid_count = series.notna().sum()\n    numeric_ratio = (coerced.notna().sum() / valid_count) if valid_count > 0 else 0\n\n    if numeric_ratio > dc.NUMERIC_RATIO_THRESHOLD and coerced.nunique(dropna=True) > dc.CARDINALITY_THRESHOLD:\n        numeric_candidates.append(column)\n    else:\n        categorical_candidates.append(column)\n\nprint(f"Numeric candidate columns: {len(numeric_candidates)}")\nprint(f"Categorical candidate columns: {len(categorical_candidates)}")'''
    
    print("\n--- Looking for old pattern ---")
    if old_pattern in text:
        print("Found old pattern, replacing...")
        text = text.replace(old_pattern, new_pattern)
        print("Replaced old pattern with new pattern")
    else:
        print("Old pattern not found exactly, searching for partial matches...")
        # Try simpler replacements
        if 'numeric_ratio > 0.85' in text:
            print("Found 'numeric_ratio > 0.85', will replace")
            text = text.replace('numeric_ratio > 0.85', 'numeric_ratio > dc.NUMERIC_RATIO_THRESHOLD')
        if 'coerced.nunique(dropna=True) > 12' in text:
            print("Found 'coerced.nunique(dropna=True) > 12', will replace")
            text = text.replace('coerced.nunique(dropna=True) > 12', 'coerced.nunique(dropna=True) > dc.CARDINALITY_THRESHOLD')
        if 'for column in Xtr_raw.columns:' in text:
            print("Found 'for column in Xtr_raw.columns:', will change to df.columns")
            # This is tricky - need to replace the whole loop
    
    # Also fix the f-string syntax error if present
    print("\n--- Checking f-string syntax ---")
    # Look for the leaked_columns f-string issue
    if 'Leakage-prone columns detected: {sorted(leaked_columns}"' in text:
        print("Found f-string syntax error, fixing...")
        text = text.replace(
            'f"Leakage-prone columns detected: {sorted(leaked_columns}"',
            'f"Leakage-prone columns detected: {sorted(leaked_columns)}"'
        )
        print("Fixed f-string syntax")
    
    # Write the fixed file
    with open(r"ml_pipeline/04_model_training.py", "w", encoding="utf-8-sig") as f:
        f.write(text)
    
    print("\nFile written successfully with fixes applied!")
else:
    print("Could not find insert position")
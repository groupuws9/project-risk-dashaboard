#!/usr/bin/env python
# Fix issues in 04_model_training.py
import re

# Read the file with proper encoding
with open(r"d:\new research topics\project risk dashaboard\ml_pipeline\04_model_training.py", "r", encoding="utf-8-sig") as f:
    content = f.read()

print(f"File size: {len(content)} characters")
print(f"First 200 chars: {content[:200]}")

# Fix 1: Fix the f-string syntax error
# Change {sorted(leaked_columns}" to {sorted(leaked_columns)}
old_fstring = 'f"Leakage-prone columns detected: {sorted(leaked_columns}"'
new_fstring = 'f"Leakage-prone columns detected: {sorted(leaked_columns)}"'

if old_fstring in content:
    content = content.replace(old_fstring, new_fstring)
    print("Fixed f-string syntax error")
else:
    print("f-string pattern not found, checking for variations...")
    # Check what the actual pattern is
    import re
    match = re.search(r'Leakage-prone columns detected:.*?leaked_columns', content)
    if match:
        print(f"Found: {match.group()[:50]}")

# Fix 2: Consolidate the leakage check block
# The current structure has:
# leaked_columns = set(predictor_columns) & prohibited_columns
# if leaked_columns:
# 
# (empty line)
# print(f"Verified {len(predictor_columns)} unique leakage-free predictors")
#
# We need to change it to:
# leaked_columns = set(predictor_columns) & prohibited_columns
# # Check for leakage-prone columns
# if leaked_columns:
#     raise ValueError(
#         f"Leakage-prone columns detected: {sorted(leaked_columns)}"
#     )
# # Check for duplicate predictor columns
# if len(predictor_columns) != len(set(predictor_columns)):
#     raise ValueError("Duplicate predictor columns detected.")
# print(f"Verified {len(predictor_columns)} unique leakage-free predictors")

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
    print("Consolidated leakage check block")
else:
    print("Old block pattern not found exactly, searching for partial matches...")
    # Check for the individual components
    if 'leaked_columns = set(predictor_columns) & prohibited_columns' in content:
        print("  Found: leaked_columns assignment")
    if 'if leaked_columns:' in content:
        print("  Found: if leaked_columns:")
    if 'print(f"Verified {len(predictor_columns)} unique leakage-free predictors")' in content:
        print("  Found: print statement")

# Write the fixed content back
with open(r"d:\new research topics\project risk dashaboard\ml_pipeline\04_model_training.py", "w", encoding="utf-8-sig") as f:
    f.write(content)

print("\nFile written successfully with fixes applied")
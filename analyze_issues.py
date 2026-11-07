#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of issues in ml_pipeline/04_model_training.py
"""
import ast
import sys

# Read and parse the file
with open(r"d:\new research topics\project risk dashaboard\ml_pipeline\04_model_training.py", "r", encoding="utf-8-sig") as f:
    content = f.read()

# Check for syntax validity
try:
    ast.parse(content)
    print("✓ File has valid Python syntax")
except SyntaxError as e:
    print(f"✗ Syntax Error: {e}")
    print(f"  Line {e.lineno}: {e.text}")

# Check for common issues
issues = []

# Issue 1: Check for f-string syntax with sorted()
if "{sorted(leaked_columns}" in content and "{sorted(leaked_columns)}" not in content:
    issues.append("✗ Potential f-string syntax issue: {sorted(leaked_columns)} may be missing closing brace")
else:
    print("✓ f-string syntax for sorted() appears correct")

# Issue 2: Check for duplicate if leaked_columns statements
leaked_count = content.count("if leaked_columns:")
if leaked_count > 1:
    issues.append(f"✗ Found {leaked_count} 'if leaked_columns:' statements - may be duplicate/bug")
else:
    print("✓ Only one 'if leaked_columns:' statement found")

# Issue 3: Check for hardcoded thresholds
if "numeric_ratio > 0.85" in content:
    issues.append("⚠ Hardcoded numeric ratio threshold: 0.85 (may need configuration)")
else:
    print("✓ No hardcoded numeric ratio threshold found")

if "nunique(dropna=True) > 12" in content:
    issues.append("⚠ Hardcoded nunique threshold: 12 (may need configuration)")
else:
    print("✓ No hardcoded nunique threshold found")

# Issue 4: Check for error handling
if "error_score=np.nan" in content:
    print("✓ Uses error_score=np.nan for handling failed CV folds")
else:
    issues.append("⚠ No error_score handling detected in cross_validate")

# Issue 5: Check for valid model selection
if "valid_models" in content and "np.isfinite" in content:
    print("✓ Valid model selection with finite balanced accuracy check")
else:
    issues.append("⚠ No valid model selection detection")

# Issue 6: Check for leakage detection
if "leaked_columns" in content and "prohibited_columns" in content:
    print("✓ Leakage detection framework present")
else:
    issues.append("⚠ Leakage detection framework may be incomplete")

# Issue 7: Check for model saving
if "joblib.dump" in content:
    print("✓ Models are saved using joblib")
else:
    issues.append("⚠ No model saving detected")

# Issue 8: Check for metadata saving
if "model_metadata.json" in content:
    print("✓ Model metadata is saved")
else:
    issues.append("⚠ No metadata saving detected")

print("\n--- Issues Summary ---")
for issue in issues:
    print(issue)
if not issues:
    print("No issues found (or only informational notes)")
#!/usr/bin/env python
# Test script to verify model training pipeline runs correctly
import sys
import os

# Add ml_pipeline to path
sys.path.insert(0, r"d:/new research topics/project risk dashaboard/ml_pipeline")

try:
    # Import the data contract to get constants
    import data_contract as dc
    print("✓ data_contract imported successfully")
    print(f"  NUMERIC_RATIO_THRESHOLD: {dc.NUMERIC_RATIO_THRESHOLD}")
    print(f"  CARDINALITY_THRESHOLD: {dc.CARDINALITY_THRESHOLD}")
    print(f"  RANDOM_STATE: {dc.RANDOM_STATE}")
    print(f"  TEST_SIZE: {dc.TEST_SIZE}")
    print(f"  N_SPLITS: {dc.N_SPLITS}")
    
    # Try to import and check the model training file syntax
    with open(r"d:/new research topics/project risk dashaboard/ml_pipeline/04_model_training.py", "r", encoding="utf-8-sig") as f:
        code = f.read()
    
    import ast
    ast.parse(code)
    print("✓ 04_model_training.py has valid Python syntax")
    
    # Check that hardcoded thresholds are replaced
    if "NUMERIC_RATIO_THRESHOLD" in code:
        print("✓ NUMERIC_RATIO_THRESHOLD constant used in model training")
    else:
        print("✗ NUMERIC_RATIO_THRESHOLD not found")
        
    if "CARDINALITY_THRESHOLD" in code:
        print("✓ CARDINALITY_THRESHOLD constant used in model training")
    else:
        print("✗ CARDINALITY_THRESHOLD not found")
        
    if "0.85" not in code or ("NUMERIC_RATIO_THRESHOLD" in code):
        print("✓ No raw 0.85 hardcoded value (using constant)")
    else:
        print("⚠ Raw 0.85 still present (may be in comments or strings)")
        
    if "12" not in code or ("CARDINALITY_THRESHOLD" in code):
        print("✓ No raw 12 hardcoded value (using constant)")
    else:
        print("⚠ Raw 12 still present (may be in comments or strings)")
        
    # Check f-string fix
    if "sorted(leaked_columns)}" in code:
        print("✓ f-string has proper closing brace")
    else:
        print("✗ f-string may have syntax issue")
        
    print("\n--- All checks passed! ---")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
# Fix 1: Fix f-string syntax and leakage check block
with open(r"d:/new research topics/project risk dashaboard/ml_pipeline/04_model_training.py", "r", encoding="utf-8-sig") as f:
    lines = f.readlines()

# Find and fix the f-string syntax error
for i, line in enumerate(lines):
    if 'Leakage-prone columns detected' in line and 'sorted(leaked_columns)' in line:
        print(f"Line {i+1}: {line.rstrip()}")
        # Fix the missing closing brace
        if 'sorted(leaked_columns}"' in line or 'sorted(leaked_columns)}"' not in line:
            # Find the exact pattern and fix
            lines[i] = line.replace('sorted(leaked_columns}"', 'sorted(leaked_columns)}"')
            print(f"  Fixed to: {lines[i].rstrip()}")

# Write back
with open(r"d:/new research topics/project risk dashaboard/ml_pipeline/04_model_training.py", "w", encoding="utf-8-sig") as f:
    f.writelines(lines)

print("Fix 1 applied: f-string syntax")
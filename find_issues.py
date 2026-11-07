import re
with open(r"d:\new research topics\project risk dashaboard\ml_pipeline\04_model_training.py", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if 'leaked_columns' in line or 'prohibited_columns' in line or 'predictor_columns' in line:
            print(f"{i}: {line.rstrip()}")
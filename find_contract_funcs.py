# -*- coding: utf-8 -*-
with open(r"d:\new research topics\project risk dashaboard\ml_pipeline\data_contract.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if 'def predictor_columns' in line or 'def split_dataset' in line:
            print(f"{i}: {line.rstrip()}")
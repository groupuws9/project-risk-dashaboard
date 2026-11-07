# -*- coding: utf-8 -*-
with open(r"d:\new research topics\project risk dashaboard\ml_pipeline\data_contract.py", "r", encoding="utf-8-sig") as f:
    lines = f.readlines()
    in_class = False
    class_name = ""
    for i, line in enumerate(lines, 1):
        if 'class NumericCoercionTransformer' in line:
            in_class = True
            class_name = 'NumericCoercionTransformer'
        elif 'class CategoricalCleaningTransformer' in line:
            in_class = True
            class_name = 'CategoricalCleaningTransformer'
        elif in_class and line.strip().startswith('class '):
            in_class = False
            print(f"\n--- End of {class_name} ---")
        elif in_class:
            print(f"{i}: {line.rstrip()}")
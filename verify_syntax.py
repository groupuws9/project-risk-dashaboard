# -*- coding: utf-8 -*-
import ast
import sys

try:
    with open(r"d:\new research topics\project risk dashaboard\ml_pipeline\04_model_training.py", "r", encoding="utf-8") as f:
        code = f.read()
    ast.parse(code)
    print("Syntax OK - file is valid Python")
except SyntaxError as e:
    print(f"Syntax Error: {e}")
    print(f"Line {e.lineno}: {e.text}")
except Exception as e:
    print(f"Error: {e}")
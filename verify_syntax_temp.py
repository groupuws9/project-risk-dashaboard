#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
with open(r"d:/new research topics/project risk dashaboard/ml_pipeline/04_model_training.py", "r", encoding="utf-8-sig") as f:
    code = f.read()
ast.parse(code)
print("Syntax OK - file is valid Python")
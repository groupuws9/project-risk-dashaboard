import sys
sys.path.insert(0, 'd:/new research topics/project risk dashaboard')
from dashboard.app import META, predictor, DF

print("META loaded:", bool(META))
print("predictor loaded:", predictor is not None)
print("DF loaded:", DF is not None)
print("META keys:", list(META.keys()) if isinstance(META, dict) else type(META))

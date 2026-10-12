from pathlib import Path
TAB_DIR = Path('d:/new research topics/project risk dashaboard/results/tables')
files = [f.name for f in TAB_DIR.iterdir() if f.is_file()]
required = ['model_comparison.csv', 'feature_importance.csv', 'evaluation_summary.csv', 'shap_importance.csv', 'shap_local.json']
missing = [f for f in required if f not in files]
print('Missing files:', missing if missing else 'None - all files present')

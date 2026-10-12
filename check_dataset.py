import pandas as pd

df = pd.read_csv('d:/new research topics/project risk dashaboard/data/processed/wers2011_operational_risk_merged.csv')
print('Shape:', df.shape)
print('\nTarget distribution:')
print(df['operational_risk_category'].value_counts())
print('\nTarget distribution (%):')
print(df['operational_risk_category'].value_counts(normalize=True)*100)
print('\nFeature count:', df.shape[1] - 1)  # subtract target column
print('\nColumn names:', list(df.columns))

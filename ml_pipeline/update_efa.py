path = "d:/new research topics/project risk dashaboard/ml_pipeline/efa_tools.py"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace ITEMS list
old_items = """# HR / employee-relations practice items expected to co-load on latent
# constructs such as "employee voice" and "formal HR strategy".
ITEMS = ["union_coverage_pct", "collective_bargaining", "bargaining_level_org",
         "bargaining_level_est", "formal_hr_strategy",
         "performance_appraisal_pct", "training_budget_exists",
         "job_security_policy", "jcc_exists", "jcc_frequency",
         "information_sharing_score", "consultation_depth",
         "employee_voice_channels", "grievance_procedure_formal",
         "info_chain_exist", "newsletter_exists"]"""

new_items = """# HR / employee-relations practice items used for exploratory factor analysis.
# grievance_procedure_formal is excluded as a precautionary workforce-instability
# indicator and must not re-enter the model through an EFA factor.
ITEMS = [
    "union_coverage_pct",
    "collective_bargaining",
    "bargaining_level_org",
    "bargaining_level_est",
    "formal_hr_strategy",
    "performance_appraisal_pct",
    "training_budget_exists",
    "job_security_policy",
    "jcc_exists",
    "jcc_frequency",
    "information_sharing_score",
    "consultation_depth",
    "employee_voice_channels",
    "info_chain_exist",
    "newsletter_exists",
]"""

content = content.replace(old_items, new_items)

# 2. Replace fit() docstring
old_docstring = '''def fit(df, items=None):
    """Fit the EFA on the full sample.

    Returns (artifact, diagnostics):
      artifact     — JSON-serialisable scoring contract (items, medians,
                     mean/std standardisation, rotated loading weights)
      diagnostics  — in-memory objects for reporting (loadings, eigenvalues,
                     KMO, Bartlett, parallel-analysis thresholds)
    """'''

new_docstring = '''def fit(df, items=None):
    """Fit EFA on the supplied training data.

    During cross-validation, this function is called within each CV training
    fold. The resulting artifact is used only to score the corresponding
    validation fold. It must not be fitted using validation or test data.
    """'''

content = content.replace(old_docstring, new_docstring)

# 3. Delete ensure() function
old_ensure = '''\ndef ensure(df, artifact_path=None):
    """Return (df_with_factor_columns, artifact).

    Loads the saved artifact when available (exact training-time scoring);
    otherwise fits on this df. Deterministic either way.
    """
    art = None
    if artifact_path and Path(artifact_path).exists():
        art = load(artifact_path)
    if art is None:
        art, _ = fit(df)
    return pd.concat([df, transform(df, art)], axis=1), art'''

content = content.replace(old_ensure, '')

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print("efa_tools.py updated successfully")
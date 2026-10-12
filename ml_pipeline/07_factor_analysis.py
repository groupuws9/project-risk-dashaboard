"""07_factor_analysis.py — Exploratory Factor Analysis of HR practices
(spec 1 §3.7).

Checks sampling adequacy (Bartlett + KMO), extracts factors with principal
axis factoring + varimax rotation (statsmodels), determines the factor count
from eigenvalues/scree, and exports factor scores for downstream modelling.

Outputs:
  results/tables/factor_loadings.csv     rotated loadings per factor
  results/features/factor_scores.csv     per-workplace factor scores
  results/figures/factor_scree.png       eigenvalue scree plot
  results/features/factor_analysis_report.txt
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "wers2011_operational_risk_merged.csv"
TAB = BASE / "results" / "tables"
FEAT = BASE / "results" / "features"
FIG = BASE / "results" / "figures"
for d in (TAB, FEAT, FIG):
    d.mkdir(parents=True, exist_ok=True)

# HR / employee-relations practice items expected to co-load on latent
# constructs such as "employee voice" and "formal HR strategy".
ITEMS = ["union_coverage_pct", "collective_bargaining", "bargaining_level_org",
         "bargaining_level_est", "formal_hr_strategy",
         "performance_appraisal_pct", "training_budget_exists",
         "job_security_policy", "jcc_exists", "jcc_frequency",
         "information_sharing_score", "consultation_depth",
         "employee_voice_channels", "grievance_procedure_formal",
         "info_chain_exist", "newsletter_exists"]

print("=" * 78)
print("EXPLORATORY FACTOR ANALYSIS — HR practice constructs")
print("=" * 78)

df = pd.read_csv(DATA)
items = [c for c in ITEMS if c in df.columns]
print(f"Items available: {len(items)}")

X = df[items].apply(pd.to_numeric, errors="coerce")
X = X.replace({999: np.nan, 99: np.nan, 9: np.nan})   # survey missing codes
X = X.dropna(axis=1, how="all")
items = list(X.columns)
X = X.fillna(X.median())
Z = (X - X.mean()) / X.std(ddof=0)
n, p = Z.shape
R = np.corrcoef(Z.values, rowvar=False)

# ---- Bartlett test of sphericity -----------------------------------------
detR = np.linalg.det(R)
chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(detR)
df_b = p * (p - 1) // 2
pval = 1 - stats.chi2.cdf(chi2, df_b)

# ---- KMO (overall and per item) ------------------------------------------
invR = np.linalg.pinv(R)
partial = np.zeros_like(R)
for i in range(p):
    for j in range(p):
        if i != j:
            partial[i, j] = -invR[i, j] / np.sqrt(abs(invR[i, i] * invR[j, j]))
r2, p2 = R ** 2, partial ** 2
np.fill_diagonal(r2, 0)
np.fill_diagonal(p2, 0)
kmo_item = r2.sum(1) / (r2.sum(1) + p2.sum(1))
kmo_overall = r2.sum() / (r2.sum() + p2.sum())
print(f"Bartlett chi2={chi2:.1f} df={df_b} p={pval:.3e}")
print(f"KMO overall={kmo_overall:.3f} (min item {kmo_item.min():.3f})")

# ---- eigenvalues / scree ---------------------------------------------------
eig = np.sort(np.linalg.eigvalsh(R))[::-1]
n_factors = int((eig > 1).sum()) or 2
n_factors = max(2, min(n_factors, 4, p - 1))
print("Eigenvalues:", np.round(eig, 3).tolist())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))
plt.plot(range(1, p + 1), eig, "o-", color="steelblue")
plt.axhline(1, ls="--", c="grey", lw=1)
plt.xlabel("Factor number")
plt.ylabel("Eigenvalue")
plt.title("Scree plot — HR practice items (WERS 2011)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "factor_scree.png", dpi=130)
plt.close()
print("Saved results/figures/factor_scree.png")

# ---- principal axis factoring + varimax (statsmodels) ---------------------
from statsmodels.multivariate.factor import Factor
from statsmodels.multivariate.factor_rotation import rotate_factors

model = Factor(Z.values, n_factor=n_factors, method="pa")  # principal axis
res = model.fit()
loadings_rot, _rot = rotate_factors(np.asarray(res.loadings), "varimax")
loadings = pd.DataFrame(loadings_rot, index=items,
                        columns=[f"Factor_{i+1}" for i in range(n_factors)])
order = np.argsort(-loadings.abs().values.max(axis=0))
loadings = loadings.iloc[:, order]
print("\nRotated factor loadings (|.| > 0.40):")
print(loadings.round(3).to_string())

# label constructs heuristically by top-loading items
labels = {}
for col in loadings.columns:
    top = loadings[col].abs().sort_values(ascending=False).head(2).index.tolist()
    labels[col] = " / ".join(top)
print("Interpretation hints:", labels)

loadings.insert(0, "item", loadings.index)
loadings.round(4).to_csv(TAB / "factor_loadings.csv", index=False)

# ---- factor scores (regression/Bartlett-style scoring on standardized data)
L = loadings.set_index("item").values
W = np.linalg.pinv(R) @ L                     # regression scoring coefficients
scores = Z.values @ W
score_df = pd.DataFrame(
    scores, columns=[f"factor_{i+1}" for i in range(n_factors)])
if "serno" in df.columns:
    score_df.insert(0, "serno", df.loc[Z.index, "serno"].values)
score_df.to_csv(FEAT / "factor_scores.csv", index=False)
print(f"Saved factor scores: {score_df.shape}")

report = f"""Factor Analysis Report — WERS 2011 HR practices
{'=' * 60}
Items: {len(items)}  |  N: {n}
Bartlett test: chi2={chi2:.2f}, df={df_b}, p={pval:.3e}
KMO overall: {kmo_overall:.3f}
KMO per item: {pd.Series(kmo_item, index=items).round(3).to_dict()}
Eigenvalues: {np.round(eig, 3).tolist()}
Factors retained (Kaiser, 2-4): {n_factors}

Rotated loadings (varimax):
{loadings.round(3).to_string()}

Construct hints: {labels}

Outputs: results/tables/factor_loadings.csv,
         results/features/factor_scores.csv,
         results/figures/factor_scree.png
"""
(FEAT / "factor_analysis_report.txt").write_text(report)
print("Saved results/features/factor_analysis_report.txt")
print("DONE.")


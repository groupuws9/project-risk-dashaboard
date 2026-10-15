"""08_factor_analysis.py — Exploratory Factor Analysis of HR practices
(spec 1 §3.7).

Checks sampling adequacy (Bartlett + KMO), extracts factors with principal
axis factoring + varimax rotation (statsmodels), determines the factor count
from the scree plot + parallel analysis, and saves the factor scores and the
scoring artifact. 04_model_training.py appends the same factor scores to the
feature set, so the ML models consume the constructs identified here (spec:
"factor scores ... for use in subsequent machine learning models").

Outputs:
  results/tables/factor_loadings.csv           rotated loadings per factor
  results/features/factor_scores.csv           per-workplace factor scores
  results/features/factor_scoring.json         scoring artifact (report copy)
  models/saved_models/factor_scoring.json      scoring artifact (inference)
  results/figures/factor_scree.png             scree + parallel analysis
  results/features/factor_analysis_report.txt
"""
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import efa_tools  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"
TAB = BASE / "results" / "tables"
FEAT = BASE / "results" / "features"
FIG = BASE / "results" / "figures"
MODEL_DIR = BASE / "models" / "saved_models"
for d in (TAB, FEAT, FIG, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("EXPLORATORY FACTOR ANALYSIS — HR practice constructs")
print("=" * 78)

df = pd.read_csv(DATA)
artifact, diag = efa_tools.fit(df)
items, eig, loadings = artifact["items"], diag["eigenvalues"], diag["loadings"]
n, p = artifact["n_observations"], len(items)
n_factors = artifact["n_factors"]
pa = artifact["parallel_analysis"]

print(f"Items available: {p}")
print(f"Bartlett chi2={artifact['bartlett_chi2']:.1f} "
      f"df={artifact['bartlett_df']} p={artifact['bartlett_p']:.3e}")
print(f"KMO overall={artifact['kmo_overall']:.3f} "
      f"(min item {min(artifact['kmo_per_item'].values()):.3f})")
print("Eigenvalues:", np.round(eig, 3).tolist())
print(f"Factors — parallel analysis: {pa['n_factors']}, "
      f"Kaiser (>1): {artifact['n_factors_kaiser']}, retained: {n_factors} "
      f"(bounded {efa_tools.MIN_FACTORS}-{efa_tools.MAX_FACTORS})")

# ---- scree plot with the parallel-analysis threshold -----------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))
plt.plot(range(1, p + 1), eig, "o-", color="steelblue", label="observed")
plt.plot(range(1, p + 1), pa["thresholds"], "s--", color="darkorange",
         label=f"parallel analysis (P{pa['percentile']})")
plt.axhline(1, ls=":", c="grey", lw=1)
plt.xlabel("Factor number")
plt.ylabel("Eigenvalue")
plt.title("Scree plot — HR practice items (WIRS/WERS panel 1980-2011)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "factor_scree.png", dpi=130)
plt.close()
print("Saved results/figures/factor_scree.png")

print("\nRotated factor loadings (|.| > 0.40):")
print(loadings.round(3).to_string())
print("Construct hints:", artifact["factor_labels"])

loadings_out = loadings.copy()
loadings_out.insert(0, "item", loadings_out.index)
loadings_out.round(4).to_csv(TAB / "factor_loadings.csv", index=False)

# ---- factor scores (regression scoring on standardized data) ---------------
scores = efa_tools.transform(df, artifact)
scores.to_csv(FEAT / "factor_scores.csv", index=False)
print(f"Saved factor scores: {scores.shape}")

efa_tools.save(artifact, FEAT / "factor_scoring.json")
efa_tools.save(artifact, MODEL_DIR / "factor_scoring.json")

strong = loadings.abs() > 0.40
report = f"""Factor Analysis Report — WIRS/WERS workplace-panel HR practices (1980-2011)
{'=' * 60}
Items: {p}  |  N: {n}
Bartlett test: chi2={artifact['bartlett_chi2']:.2f}, df={artifact['bartlett_df']}, p={artifact['bartlett_p']:.3e}
KMO overall: {artifact['kmo_overall']:.3f}
KMO per item: {artifact['kmo_per_item']}
Eigenvalues: {np.round(eig, 3).tolist()}
Parallel analysis (P{pa['percentile']}, {pa['simulations']} sims, seed {pa['seed']}) retains: {pa['n_factors']}
Kaiser (eigenvalue > 1) retains: {artifact['n_factors_kaiser']}
Factors retained (bounded {efa_tools.MIN_FACTORS}-{efa_tools.MAX_FACTORS}): {n_factors}
Extraction: {artifact['method']}

Rotated loadings (varimax):
{loadings.round(3).to_string()}

Items with |loading| > 0.40 per factor:
{strong.to_string()}

Construct hints: {artifact['factor_labels']}

Outputs: results/tables/factor_loadings.csv,
         results/features/factor_scores.csv,
         results/features/factor_scoring.json,
         models/saved_models/factor_scoring.json (inference scoring),
         results/figures/factor_scree.png
"""
(FEAT / "factor_analysis_report.txt").write_text(report)
print("Saved results/features/factor_analysis_report.txt")
print("DONE.")

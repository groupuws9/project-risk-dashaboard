"""efa_tools.py — shared Exploratory Factor Analysis utilities (spec 1 §3.7).

Fits EFA on the 16 HR / employee-relations practice items (union coverage,
collective bargaining, JCC, information sharing, consultation depth, ...),
checks sampling adequacy (Bartlett + KMO), extracts factors with principal
axis factoring + varimax rotation (statsmodels), and determines the factor
count from the scree plot + parallel analysis.

Used by:
  ml_pipeline/08_factor_analysis.py  — full report (loadings, scree, scores CSV)
  ml_pipeline/04_model_training.py   — appends factor scores as model features
  05 / 06 / 03                  — re-applies the saved scoring artifact
  models/risk_predictor.py           — loads the JSON artifact at inference time

fit() is deterministic (no RNG in extraction; parallel analysis uses a fixed
seed), so every script that fits on the same CSV reproduces identical factor
scores. transform() needs only the saved artifact, so inference never needs
the training data.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# HR / employee-relations practice items expected to co-load on latent
# constructs such as "employee voice" and "formal HR strategy".
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
]

# Survey non-response codes for the coded (1..7 scale) items.
MISS_CODES = (9, 99, 999)
# union_coverage_pct is a genuine 0-100 percentage: only 999 flags "missing"
# (9 % and 99 % are legitimate coverage rates and must be kept).
MISS_OVERRIDE = {"union_coverage_pct": (999,)}

MIN_FACTORS, MAX_FACTORS = 2, 4    # interpretability bounds (spec §3.7)
PA_SIMULATIONS, PA_SEED = 500, 42  # parallel analysis (Horn / Horn-Hunfred)
PA_PERCENTILE = 95


def _numeric_items(df, items=None):
    """Numeric, missing-code-cleaned item frame (all-missing items dropped)."""
    cols = [c for c in (items or ITEMS) if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        X[c] = X[c].replace(list(MISS_OVERRIDE.get(c, MISS_CODES)), np.nan)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] < 2:
        raise ValueError(f"Need >=2 usable EFA items, got {list(X.columns)}")
    return X


def _parallel_analysis(n, p, eig, rng):
    """95th-percentile eigenvalues of random data (parallel analysis)."""
    sims = rng.standard_normal((PA_SIMULATIONS, n, p))
    sims = (sims - sims.mean(axis=1, keepdims=True)) / sims.std(axis=1,
                                                                 keepdims=True)
    C = np.transpose(sims, (0, 2, 1)) @ sims / n   # (sims, p, p) correlations
    sim_eig = np.sort(np.linalg.eigvalsh(C), axis=1)[:, ::-1]  # desc per sim
    thr = np.percentile(sim_eig, PA_PERCENTILE, axis=0)
    return thr, int((eig > thr).sum())


def fit(df, items=None):
    """Fit EFA on the supplied training data.

    During cross-validation, this function is called within each CV training
    fold. The resulting artifact is used only to score the corresponding
    validation fold. It must not be fitted using validation or test data.
    """
    from statsmodels.multivariate.factor import Factor
    from statsmodels.multivariate.factor_rotation import rotate_factors

    X = _numeric_items(df, items)
    used = list(X.columns)
    medians = X.median()
    Xf = X.fillna(medians)
    mean, std = Xf.mean(), Xf.std(ddof=0).replace(0, 1)
    Z = (Xf - mean) / std
    n, p = Z.shape
    R = np.corrcoef(Z.values, rowvar=False)

    # ---- Bartlett test of sphericity ---------------------------------------
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    df_b = p * (p - 1) // 2
    pval = 1 - stats.chi2.cdf(chi2, df_b)

    # ---- KMO (overall and per item) ----------------------------------------
    invR = np.linalg.pinv(R)
    partial = np.zeros_like(R)
    for i in range(p):
        for j in range(p):
            if i != j:
                partial[i, j] = -invR[i, j] / np.sqrt(abs(invR[i, i] * invR[j, j]))
    r2, p2 = R ** 2, partial ** 2
    np.fill_diagonal(r2, 0)
    np.fill_diagonal(p2, 0)
    kmo_item = pd.Series(r2.sum(1) / (r2.sum(1) + p2.sum(1)), index=used)
    kmo_overall = float(r2.sum() / (r2.sum() + p2.sum()))

    # ---- eigenvalues, scree rule, parallel analysis -------------------------
    eig = np.sort(np.linalg.eigvalsh(R))[::-1]
    kaiser = int((eig > 1).sum())
    pa_thr, pa_count = _parallel_analysis(n, p, eig, np.random.default_rng(PA_SEED))
    n_factors = max(MIN_FACTORS, min(pa_count or 1, MAX_FACTORS, p - 1))

    # ---- principal axis factoring + varimax ---------------------------------
    res = Factor(Z.values, n_factor=n_factors, method="pa").fit()
    rot, _ = rotate_factors(np.asarray(res.loadings), "varimax")
    loadings = pd.DataFrame(rot, index=used,
                            columns=[f"factor_{i+1}" for i in range(n_factors)])
    # order factors by their strongest absolute loading, then rename
    loadings = loadings.iloc[:, np.argsort(-loadings.abs().max(axis=0).values)]
    loadings.columns = [f"factor_{i+1}" for i in range(n_factors)]
    W = np.linalg.pinv(R) @ loadings.values          # regression score weights

    labels = {col: " / ".join(loadings[col].abs().sort_values(ascending=False)
                              .head(2).index.tolist())
              for col in loadings.columns}

    artifact = {
        "items": used,
        "miss_codes": {c: list(MISS_OVERRIDE.get(c, MISS_CODES)) for c in used},
        "medians": {c: float(medians[c]) for c in used},
        "mean": {c: float(mean[c]) for c in used},
        "std": {c: float(std[c]) for c in used},
        "weights": np.asarray(W, float).tolist(),     # (n_items, n_factors)
        "n_factors": int(n_factors),
        "factor_names": list(loadings.columns),
        "factor_labels": labels,
        "method": "principal axis factoring + varimax rotation (statsmodels)",
        "n_observations": int(n),
        "bartlett_chi2": float(chi2), "bartlett_df": int(df_b),
        "bartlett_p": float(pval),
        "kmo_overall": kmo_overall,
        "kmo_per_item": {c: float(v) for c, v in kmo_item.items()},
        "eigenvalues": [float(e) for e in eig],
        "n_factors_kaiser": kaiser,
        "parallel_analysis": {
            "simulations": PA_SIMULATIONS, "percentile": PA_PERCENTILE,
            "seed": PA_SEED,
            "thresholds": [float(t) for t in pa_thr],
            "n_factors": pa_count,
        },
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    diag = {"Z": Z, "R": R, "loadings": loadings, "eigenvalues": eig,
            "pa_thresholds": pa_thr, "kaiser": kaiser, "pa_count": pa_count,
            "n_factors": n_factors, "kmo_item": kmo_item,
            "kmo_overall": kmo_overall, "chi2": chi2, "df_b": df_b, "pval": pval}
    return artifact, diag




def transform(df, artifact):
    """Apply a saved scoring artifact -> factor_1..k frame aligned to df.index."""
    used = artifact["items"]
    X = df.reindex(columns=used).apply(pd.to_numeric, errors="coerce")
    for c, codes in artifact.get("miss_codes", {}).items():
        if c in X.columns:
            X[c] = X[c].replace(list(codes), np.nan)
    med = pd.Series({c: float(v) for c, v in artifact["medians"].items()})
    X = X.astype(float).fillna(med)
    Z = (X - pd.Series({c: float(v) for c, v in artifact["mean"].items()})) \
        / pd.Series({c: float(v) for c, v in artifact["std"].items()})
    scores = Z.values @ np.asarray(artifact["weights"], dtype=float)
    return pd.DataFrame(scores, index=df.index,
                        columns=list(artifact["factor_names"]))


def ensure(df, artifact_path):
    """Load the saved scoring artifact (or fit one) and append factor scores.

    Returns ``(df_with_factor_columns, artifact)``. The saved artifact is
    reused when present so this stays deterministic with the deployed model;
    otherwise EFA is fitted on ``df`` and persisted. The input frame is not
    mutated.
    """
    artifact_path = Path(artifact_path)
    if artifact_path.exists():
        artifact = load(artifact_path)
    else:
        artifact, _ = fit(df)
        save(artifact, artifact_path)

    factors = transform(df, artifact)
    out = df.copy()
    for column in factors.columns:
        out[column] = factors[column].to_numpy()
    return out, artifact


def save(artifact, path):
    """Write the scoring artifact as JSON (numpy-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))
    return path


def load(path):
    return json.loads(Path(path).read_text())




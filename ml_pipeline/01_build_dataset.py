"""merge_tools.py — single source of truth for building the WERS/WIRS dataset.

Why this module exists
----------------------
The project uses the UK Data Service *Workplace Employment Relations Study*
time-series (UKDA 4511). Two tab extracts ship in ``data/``:

* ``wirsts10_july2014.tab`` — cross-sectional main survey, 6,635 rows over the
  three modern waves 1998 / 2004 / 2011 (authoritative for those waves).
* ``wirsts25_july2014.tab`` — full panel, 13,284 rows over seven waves
  1980 / 1984 / 1990 / 1991 / 1998 / 2004 / 2011.

The two files share 5,471 ``serno`` values (always the same survey year) and
disagree on only 10 columns (``tempc`` turnover, ``tcovpc`` union coverage,
``tmgrfpc``, ``tverseg``, ``tnummema/b``, ``tudensa/b``, ``tudens6a/b``).
All 5,471 shared rows fall in 1998/2004/2011, so ``wirsts25`` contributes no
*new* information for those waves. Its 7,813 unique rows are exactly the four
early waves (1980+1984+1990+1991 = 7,813).

Authoritative merge rule (no overlap, one coding per wave)
-----------------------------------------------------------
    merged = wirsts10 (all 6,635 rows, waves 1998/2004/2011)
           + wirsts25 rows in the early waves only (7,813 rows)
           = 14,448 rows over 7 waves

Using ``wirsts10`` for the modern waves keeps a single coding of ``tempc``
(turnover) inside the modelling waves instead of silently mixing two codings.

Target construction (turnover-based, sentinel-aware)
----------------------------------------------------
The operational-risk target is the **employee turnover band** derived from the
genuine continuous rate ``tempc``:

    Low    turnover <= 10 %
    Medium 10 %  < turnover <= 20 %
    High    turnover > 20 %

UKDS special missing codes (9 / 99 / 999 / 9999.99) and impossible rates
(negative or > 100) are mapped to NaN so an unknown target is never silently
scored as "Low". Small integers such as 8 or 9 are *real* turnover percentages
and are kept. Rows with an unknown target are dropped from supervised training
but kept (labelled NaN) for EDA.

Note the earlier "0-100 composite over four dimensions" design was retired:
inspection of the raw extract shows absenteeism (tsicklog), industrial action
(tstrike / tothact) and grievances (tprocgrv) are CATEGORICAL {1,2} survey
codes, not rates, so numeric thresholds on them are meaningless (the composite
also produced a 28-row "High" class, i.e. a 196:1 imbalance).

The target is honestly constructable in every modern wave (1998/2004/2011)
where ``tempc`` is observed; the early waves (1980/1984/1990/1991) carry no
turnover and are retained for EDA / feature context only.

Licence: UK Data Service End User Licence (Safeguarded / restricted access).
See DATA_LICENCE.md and the UKDS project referenced there.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_CSV = OUT_DIR / "wers_operational_risk_merged.csv"
OUT_META = OUT_DIR / "dataset_metadata.json"
OUT_SUMMARY = OUT_DIR / "processing_summary.txt"

MAIN_FILE = DATA_DIR / "wirsts10_july2014.tab"      # 1998 / 2004 / 2011
PANEL_FILE = DATA_DIR / "wirsts25_july2014.tab"     # 1980 .. 2011
MODERN_WAVES = [1998, 2004, 2011]
EARLY_WAVES = [1980, 1984, 1990, 1991]

TARGET = "operational_risk_category"
SCORE_COL = "operational_risk_score"

# ---- raw variable mapping (identical to the original preparation script) ----
ID_VARS = ["serno", "year", "weight"]

RISK_VARS = {
    "employee_turnover_rate": "tempc",
    "employee_turnover_rate_5yr": "tempc5",
    "absenteeism_rate": "tsicklog",
    "grievance_rate": "tprocgrv",
    "disputes_raised": "traised",
    "dispute_risk_level": "tstrike",
    "other_industrial_action": "tothact",
    "dismissal_rate": "tdismiss",
    "disciplinary_warnings": "twarning",
    "suspensions": "tsuspend",
}
ORG_VARS = {
    "org_size_category": "torgsize", "org_size_category_3": "torgsiz3",
    "org_size_category_4": "torgsiz4", "sector": "tsector3",
    "sector_detailed": "tsect15", "industry_code": "tsic92",
    "industry_code_80": "tsic80", "multinational": "tmultnat",
    "org_age_years": "tage", "workplace_age": "ttimhere",
    "trading_status": "ttrading",
}
HR_VARS = {
    "union_coverage_pct": "tcovpc", "union_coverage_pct_6": "tcovpc6",
    "union_coverage_pct_7": "tcovpc7", "collective_bargaining": "tnatbarg",
    "bargaining_level_org": "torgbarg", "bargaining_level_est": "testbarg",
    "formal_hr_strategy": "tstaffin", "performance_appraisal_pct": "tapprais",
    "training_budget_exists": "tfinance", "training_problems": "ttrnprob",
    "job_security_policy": "tjobsec",
}
ER_VARS = {
    "jcc_exists": "tjcc", "jcc_frequency": "tjccfreq", "jcc_function": "tjccf",
    "jcc_union_chair": "tjcctu", "information_sharing_score": "tbrief",
    "info_chain_exist": "tchain", "newsletter_exists": "tnews",
    "investment_plans_shared": "tinvplan", "consultation_depth": "tconsall",
    "employee_voice_channels": "tvoice1", "grievance_procedure_formal": "tprocgrv",
}
LABOR_VARS = {
    "pct_part_time": "tptpc", "pct_part_time_12": "tptpc12",
    "pct_female": "tfempc", "pct_temporary": "ttemplyr",
    "skill_shortage_reported": "ttrnprob",
}
ALL_VARIABLES = {}
for _d in (RISK_VARS, ORG_VARS, HR_VARS, ER_VARS, LABOR_VARS):
    ALL_VARIABLES.update(_d)

# Target-constructing columns can NEVER be candidate features (data leakage).
# grievance_procedure_formal maps from tprocgrv (same as grievance_rate) and
# training_problems / skill_shortage_reported both map from ttrnprob -> excluded.
LEAKAGE = [
    "employee_turnover_rate", "employee_turnover_rate_5yr", "absenteeism_rate",
    "grievance_rate", "grievance_procedure_formal", "training_problems",
    "skill_shortage_reported", "disputes_raised", "dispute_risk_level",
    "other_industrial_action", "dismissal_rate", "disciplinary_warnings",
    "suspensions", SCORE_COL,
]

# Special missing-value codes (UKDS). Verified per variable; percentages also
# exclude >100 (impossible rates).
MISSING_CODES = {
    "employee_turnover_rate": [8, 9, 99, 999, 9999, 9999.99, -9, -8],
    "absenteeism_rate": [8, 9, 99, 999, 9999.99, -9, -8],
    "grievance_rate": [8, 9, 99, 999, 9999.99, -9, -8],
    "dispute_risk_level": [8, 9, 99, 999, 9999.99, -9, -8],
    "other_industrial_action": [8, 9, 99, 999, 9999.99, -9, -8],
}

# ----------------------------------------------------------------------------
# Turnover-based operational-risk target
# ----------------------------------------------------------------------------
# The original composite 0-100 score mixed four "dimensions", but the probes in
# run_logs/ prove that absenteeism (tsicklog), industrial action (tstrike /
# tothact) and grievances (tprocgrv) are CATEGORICAL {1,2} survey codes, not
# rates. Applying thresholds such as ">5%" to them is meaningless. Turnover
# (tempc) is the only genuine continuous workplace risk rate in the extract, so
# the target is defined from it alone:
#
#     Low    turnover <= 10 %
#     Medium 10 %  < turnover <= 20 %
#     High    turnover > 20 %
#
# These cut-points sit near the 75th / 90th percentiles of the observed turnover
# distribution and match the conventional HR "healthy / watch / high" bands.
# UKDS missing codes (9/99/999/9999.99) and impossible rates (>100) become NaN
# so an unknown target is never silently scored as "Low".
TURNOVER_COL = "employee_turnover_rate"
# tempc is a genuine continuous percentage (0-100). Only impossible rates
# (>100: the UKDS 999 / 9999.99 skip codes) and negatives are sentinels. Small
# integers such as 8 or 9 are REAL turnover percentages (the distribution's
# median is 1.9 and its 75th percentile is 8.0), so they must NOT be dropped.
TURNOVER_SENTINELS = [999, 9999.99, 99999, -9, -8]
TURNOVER_BANDS = [(10.0, "Low"), (20.0, "Medium")]
HIGH_LABEL = "High"


def predictor_columns():
    """Mapped columns usable as predictors (not IDs, not leakage/target)."""
    return [c for c in ALL_VARIABLES if c not in LEAKAGE and c not in ID_VARS]


def _load_tab(path):
    """Read a UKDA .tab (tab-separated, latin-1) with low_memory off."""
    return pd.read_csv(path, sep="\t", low_memory=False, encoding="latin-1")


def clean_turnover(series):
    """Return turnover as float with UKDS sentinels / impossible rates -> NaN."""
    v = pd.to_numeric(series, errors="coerce")
    v = v.mask(v.isin(TURNOVER_SENTINELS))
    return v.mask((v < 0) | (v > 100))


def turnover_to_category(v):
    """Map a clean turnover rate (0-100) to Low / Medium / High."""
    if pd.isna(v):
        return np.nan
    if v <= TURNOVER_BANDS[0][0]:
        return "Low"
    if v <= TURNOVER_BANDS[1][0]:
        return "Medium"
    return HIGH_LABEL


def build(verbose=True):
    """Build the authoritative merged dataset from the two raw .tab extracts.

    Returns one DataFrame (one row per serno-year) with the mapped HR / ER /
    organisational / labour columns (renamed), a sentinel-cleaned
    ``employee_turnover_rate`` and the derived ``operational_risk_category``.

    Merge rule (no serno-year overlap, one coding per wave):
        modern waves (1998/2004/2011) <- wirsts10   (authoritative)
        early  waves (1980/1984/1990/1991) <- wirsts25
    """
    if not MAIN_FILE.exists() or not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing raw extract(s): {MAIN_FILE.name} / {PANEL_FILE.name} "
            f"in {DATA_DIR}")

    main = _load_tab(MAIN_FILE)
    panel = _load_tab(PANEL_FILE)
    main_mod = main[main["year"].isin(MODERN_WAVES)].copy()
    panel_early = panel[panel["year"].isin(EARLY_WAVES)].copy()

    # One row per RAW column: tprocgrv feeds grievance_rate +
    # grievance_procedure_formal, and ttrnprob feeds training_problems +
    # skill_shortage_reported. Selecting a raw twice yields pandas '.1' dup
    # columns, so dedupe (order-preserving) and materialise twins after rename.
    cols = list(dict.fromkeys(ID_VARS + list(ALL_VARIABLES.values())))

    def _select(frame):
        return frame[[c for c in cols if c in frame.columns]]

    merged = pd.concat([_select(main_mod), _select(panel_early)],
                       ignore_index=True)
    merged = merged.rename(columns={v: k for k, v in ALL_VARIABLES.items()})
    # After an injective rename only one friendly name survives per shared raw
    # source; copy it to the twin (both twins are leakage columns regardless).
    for _a, _b in (("grievance_rate", "grievance_procedure_formal"),
                   ("grievance_procedure_formal", "grievance_rate"),
                   ("training_problems", "skill_shortage_reported"),
                   ("skill_shortage_reported", "training_problems")):
        if _a not in merged.columns and _b in merged.columns:
            merged[_a] = merged[_b]

    merged = merged.drop_duplicates(subset=["serno", "year"], keep="first")
    merged = merged.sort_values(["year", "serno"]).reset_index(drop=True)

    merged[TURNOVER_COL] = clean_turnover(merged[TURNOVER_COL])
    merged[SCORE_COL] = merged[TURNOVER_COL]     # retained alias
    merged[TARGET] = merged[TURNOVER_COL].map(turnover_to_category)

    if verbose:
        waves = merged["year"].value_counts().sort_index()
        print(f"merged: {len(merged):,} rows x {merged.shape[1]} cols")
        print("  waves:", {int(k): int(v) for k, v in waves.items()})
        print("  target available:", int(merged[TARGET].notna().sum()), "rows")
        print(merged[TARGET].value_counts().to_string())
    return merged


def write(merged, verbose=True):
    """Persist the merged frame + corrected metadata to data/processed/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)

    counts = merged[TARGET].value_counts()
    n_valid = int(merged[TARGET].notna().sum())
    meta = {
        "dataset_name": "WERS/WIRS Operational Risk Dataset (turnover-based)",
        "source": "UKDA Study 4511 - Workplace Employment Relations Study "
                  "time-series (wirsts10 + wirsts25)",
        "licence": "UK Data Service End User Licence - Safeguarded Data",
        "waves": {int(y): int((merged["year"] == y).sum())
                  for y in sorted(merged["year"].unique())},
        "n_observations": int(len(merged)),
        "n_variables": int(merged.shape[1]),
        "target_variable": TARGET,
        "target_basis": "employee turnover rate (tempc), sentinel-cleaned",
        "target_bands": {"Low": "<=10%", "Medium": "10-20%", "High": ">20%"},
        "target_available_rows": n_valid,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_pct": {k: round(v / max(n_valid, 1) * 100, 2)
                      for k, v in counts.items()},
        "predictor_columns": predictor_columns(),
        "leakage_excluded": LEAKAGE,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    # Human-readable companion summary (kept in sync with dataset_metadata.json
    # so no stale "6,635 rows / 52 variables / composite score" copy survives).
    counts = merged[TARGET].value_counts()
    lines = [
        "",
        "WERS/WIRS OPERATIONAL RISK DATASET - PROCESSING SUMMARY",
        "=" * 80,
        "",
        "Dataset Information:",
        "- Source: UKDA Study 4511 - WERS time-series (wirsts10 + wirsts25)",
        "- Licence: UK Data Service End User Licence (Safeguarded data)",
        f"- Output File: {OUT_CSV.name}",
        "",
        "Merge rule (ml_pipeline/merge_tools.py):",
        f"- {MAIN_FILE.name}: main workplace survey, waves 1998/2004/2011",
        f"- {PANEL_FILE.name}: panel file, waves 1980-2011",
        "- merged = all main-survey rows + early-wave panel rows (no serno-year overlap)",
        "",
        "Dimensions:",
        f"- Total Observations: {len(merged):,}",
        f"- Total Variables: {merged.shape[1]}",
        f"- Survey waves: {len(meta['waves'])} ({min(meta['waves'])}-{max(meta['waves'])})",
        "",
        "Target Variable:",
        "- operational_risk_category (employee turnover band)",
        "- Basis: employee_turnover_rate (tempc), sentinel-cleaned",
        "- Bands: Low <=10%, Medium 10-20%, High >20%",
        f"- Rows with an observed target: {n_valid:,} of {len(merged):,}",
        "",
        "Target Variable Distribution:",
    ]
    lines += [f"{k:<10}{v:,}" for k, v in counts.items()]
    lines += [
        "",
        "Turnover Rate Statistics (%, cleaned):",
        f"- Mean: {merged[TURNOVER_COL].mean():.2f}",
        f"- Median: {merged[TURNOVER_COL].median():.2f}",
        f"- Min: {merged[TURNOVER_COL].min():.2f}",
        f"- Max: {merged[TURNOVER_COL].max():.2f}",
        "",
        "Feature Contract:",
        f"- Mapped leakage-safe predictors: {len(predictor_columns())}",
        "- 04_model_training.py drops 2 high-cardinality SIC codes "
        "(industry_code, industry_code_80) -> 33 raw inputs",
        "- Plus 4 EFA factor scores (07_factor_analysis.py) = 37 model inputs",
        f"- Leakage-excluded columns: {len(LEAKAGE)}",
        "- Identifiers excluded: serno, year, weight",
        "",
        "Data Quality:",
        f"- Duplicate serno-year rows: {int(merged.duplicated(subset=['serno', 'year']).sum())}",
        f"- Predictors with missing values: "
        f"{int(merged[predictor_columns()].isna().any().sum())}",
        "",
        "Files Created:",
        f"1. {OUT_CSV.name} - analysis-ready dataset",
        f"2. {OUT_META.name} - machine-readable metadata + feature contract",
        f"3. {OUT_SUMMARY.name} - this summary report",
        "",
        "Next Steps:",
        "1. Exploratory analysis: python ml_pipeline/02_exploratory_analysis.py",
        "2. Feature engineering: python ml_pipeline/03_feature_engineering.py",
        "3. Model training: python ml_pipeline/04_model_training.py",
        "4. Dashboard: python dashboard/app.py",
        "",
        "=" * 80,
    ]
    OUT_SUMMARY.write_text("\n".join(lines))

    if verbose:
        print(f"wrote {OUT_CSV.relative_to(BASE)}")
        print(f"wrote {OUT_META.relative_to(BASE)}")
        print(f"wrote {OUT_SUMMARY.relative_to(BASE)}")
    return meta


if __name__ == "__main__":
    write(build())

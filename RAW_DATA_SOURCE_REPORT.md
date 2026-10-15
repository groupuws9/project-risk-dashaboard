# RAW WERS/WIRS DATA SOURCE REPORT

This report documents the **raw** survey files, the authoritative merge rule and
the target construction. Everything below is generated from the executed
pipeline (`ml_pipeline/01_build_dataset.py`) — no figure is hand-typed.

---

## 1. Original data files

Two raw extracts from the **UK Data Service, Study 4511** (Workplace Employment
Relations Study time-series), obtained under the **End User Licence
(Safeguarded data)**. Both shipped as tab-delimited `.tab` files and were
converted verbatim to CSV for pandas (no transformation, no aggregation).

### 1.1 `wirsts10_july2014.tab` — main workplace survey

| Attribute | Value |
|---|---|
| Original format | Tab-delimited (`.tab`), latin-1 |
| Converted file | `wirsts10_july2014.csv` (4.81 MB) |
| Observations | 6,635 |
| Variables | 320 |
| Waves | 3 (1998 / 2004 / 2011) — authoritative for the modern waves |

| Year | Count | Percentage |
|------|------:|-----------:|
| 1998 | 2,191 | 33.03 % |
| 2004 | 2,062 | 31.08 % |
| 2011 | 2,382 | 35.89 % |
| **Total** | **6,635** | **100 %** |

### 1.2 `wirsts25_july2014.tab` — panel file

| Attribute | Value |
|---|---|
| Original format | Tab-delimited (`.tab`), latin-1 |
| Converted file | `wirsts25_july2014.csv` (9.54 MB) |
| Observations | 13,284 |
| Variables | 320 |
| Waves | 7 (1980 / 1984 / 1990 / 1991 / 1998 / 2004 / 2011) |

| Year | Count | Percentage |
|------|------:|-----------:|
| 1980 | 2,040 | 15.36 % |
| 1984 | 2,019 | 15.20 % |
| 1990 | 2,061 | 15.53 % |
| 1991 | 1,693 | 12.75 % |
| 1998 | 1,929 | 14.52 % |
| 2004 | 1,648 | 12.41 % |
| 2011 | 1,894 | 14.26 % |
| **Total** | **13,284** | **100 %** |

**Overlap.** The two files share 5,471 `serno` values (always the same survey
year) and disagree on only 10 columns (`tempc` turnover, `tcovpc` union
coverage, `tmgrfpc`, `tverseg`, `tnummema/b`, `tudensa/b`, `tudens6a/b`). All
5,471 shared rows fall in 1998/2004/2011, so `wirsts25` contributes **no new
information** for the modern waves.


---

## 2. Merge rule (`ml_pipeline/01_build_dataset.py`)

The merge is a **union with no `serno`-year overlap and one coding per wave**:

```
merged = wirsts10 (all 6,635 rows, waves 1998/2004/2011)   # authoritative
       + wirsts25 rows in the early waves only (7,813 rows)  # 1980+1984+1990+1991
       = 14,448 rows over 7 waves
```

Using `wirsts10` for the modern waves keeps a **single coding of `tempc`**
(turnover) inside the modelling waves instead of silently mixing two codings.
`wirsts25`'s 7,813 unique rows are exactly the four early waves.

Resulting dimensions: **14,448 rows × 53 columns**, waves
`{1980: 2,040, 1984: 2,019, 1990: 2,061, 1991: 1,693, 1998: 2,191, 2004: 2,062, 2011: 2,382}`.

Note this is **not** "the 1998/2004/2011 waves only": the early waves are
retained for EDA and feature context. `year` is excluded from the *feature* set
so multi-wave data is not treated as a single-year cross-section.

---

## 3. Target construction (turnover-based)

The target `operational_risk_category` is the **employee turnover band** derived
from the genuine continuous rate `tempc`:

| Band | Rule |
|---|---|
| Low | turnover ≤ 10 % |
| Medium | 10 % < turnover ≤ 20 % |
| High | turnover > 20 % |

* UKDS special missing codes (9 / 99 / 999 / 9999.99) and impossible rates
  (negative or > 100) map to **NaN**, so an unknown target is never silently
  scored "Low".
* Small integers such as 8 or 9 are **real** turnover percentages (median 1.9,
  75th percentile 8.0) and are kept.
* Rows with an unknown target are dropped from supervised training but kept
  (labelled NaN) for EDA: **6,385 of 14,448 rows** carry a target.

**Why the composite score was retired.** The original design built a 0–100
composite over four "dimensions" (turnover, absenteeism, grievances, industrial
action) with 25 points each. Inspection of the raw extract shows `tsicklog`
(absenteeism), `tstrike` / `tothact` (industrial action) and `tprocgrv`
(grievances) are **categorical {1, 2} survey codes, not rates** — numeric
thresholds such as ">5 %" applied to them are meaningless, and the resulting
"High" class had 28 members (196 : 1 imbalance). Turnover is the only genuine
continuous workplace risk rate in the extract, so the target is defined from it
alone. The `operational_risk_score` column retained in the CSV is only an alias
of the cleaned turnover rate and is **excluded as leakage**.

---

## 4. Feature contract

| Item | Count |
|---|---|
| Mapped columns (identifiers, risk, org, HR, ER, labour) | 51 |
| Leakage-excluded columns | 14 |
| Mapped leakage-safe predictors | 35 |
| Dropped for high cardinality (`industry_code`, `industry_code_80`) | 2 |
| **Raw model inputs** | **33** |
| EFA factor scores added by `08_factor_analysis.py` | 4 |
| **Total model inputs** | **37** (9 numeric + 28 categorical) |

Identifiers excluded from features: `serno`, `year`, `weight`.

---

## 5. Output artefacts

| File | Purpose |
|---|---|
| `data/processed/wers_operational_risk_merged.csv` | analysis-ready dataset (14,448 × 53) |
| `data/processed/dataset_metadata.json` | machine-readable metadata + predictor/leakage contract |
| `data/processed/processing_summary.txt` | human-readable summary (regenerated on every run) |
| `data/processed/wers_panel_features_engineered.csv` | exploratory feature matrix (step 03) |

Both the JSON and the text summary are **written by the pipeline**, so they can
never drift from the executed numbers.

---

## 6. Data access and licensing

**Source:** UK Data Archive, Study 4511 — Workplace Employment Relations Study
time-series.
**Access:** registration + acceptance of the End User Licence.
**Restrictions:**

* Raw `.tab` / `.csv` extracts are **not** in this repository (licence).
* `data/processed/*.csv` derived files are also untracked.
* Do not redistribute the raw or processed survey data.
* Cite the UK Data Service in any publication.

**Acknowledgement:** WERS was funded by the UK Department for Business,
Innovation and Skills and conducted by the Institute for Employment Studies, the
National Institute of Economic and Social Research, and the Policy Studies
Institute.

---

## 7. Dissertation-facing summary (Chapter 3 §3.4)

**Table 3.2: Verified dataset characteristics**

| Attribute | Value |
|---|---|
| Dataset | WERS/WIRS Operational Risk Dataset (turnover-based) |
| Source | UKDA Study 4511 (WERS time-series: `wirsts10` + `wirsts25`) |
| Licence | UK Data Service End User Licence (Safeguarded data) |
| Original format | Tab-delimited (`.tab`), converted to CSV |
| Unit of analysis | Workplace (organization) |
| Total observations | 14,448 |
| Total variables | 53 |
| Modelled rows (target available) | 6,385 (turnover observed in 1998/2004/2011) |
| Model inputs | 37 (33 raw + 4 EFA factors; 9 numeric + 28 categorical) |
| High-cardinality excluded | `industry_code`, `industry_code_80` |
| Target variable | `operational_risk_category` |
| Target basis | employee turnover rate (`tempc`), sentinel-cleaned |
| Target bands | Low ≤ 10 %, Medium 10–20 %, High > 20 % |
| Class distribution (Low) | 5,077 (79.51 %) |
| Class distribution (Medium) | 621 (9.73 %) |
| Class distribution (High) | 687 (10.76 %) |
| Imbalance ratio (Low : High) | 7.4 : 1 |
| Missing values (predictors) | 0 |
| Turnover rate range (%) | 0.0 – 100.0 |
| Survey waves | 7 (1980–2011); target defined for 3 (1998/2004/2011) |
| Train / test split | stratified 80 : 20, `random_state=42` → 5,108 / 1,277 |
| Cross-validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |

**Suggested wording (§3.4.1):**

> The raw WERS time-series data were obtained from the UK Data Service
> (Study 4511) under its End User Licence. The original files were provided in
> tab-delimited format: `wirsts10_july2014.tab` (main workplace survey, 6,635
> observations, 320 variables, waves 1998/2004/2011) and
> `wirsts25_july2014.tab` (panel data, 13,284 observations, 320 variables,
> waves 1980–2011), converted to CSV without transformation. The merge unions
> all 6,635 main-survey rows with the 7,813 early-wave panel rows, yielding
> 14,448 workplace-level observations across seven waves. The turnover-based
> target is defined for the 6,385 rows of the 1998/2004/2011 waves.

**Do not use** (superseded values):

* 6,635 rows / 52 variables as the *analysis* dataset
* the 0–100 composite score and its High ≥ 60 / Medium 35–59 bands
* class counts 5,502 / 1,105 / 28 and the 196 : 1 imbalance ratio
* 36 (or 51) model inputs, and three-fold cross-validation

See `EVIDENCE_INTEGRATION_SUMMARY.md` for the full corrections log.


"""09_export_model_dataset.py — export the final ML model matrix.

Writes the exact modelling dataset the trained pipeline consumes: the 33
leakage-safe raw predictors (canonically cleaned + rare-grouped) plus the 4
EFA factor scores = **37 model variables**, together with the target label
and a train/test split flag, for the 6,385 rows with an observed target.

Output:
    data/processed/wers_ml_model_dataset.csv

The frame is rebuilt from the shared contract
(``data_contract.build_model_frames``) so it can never drift from training.
Factor scores use the persisted EFA scoring artifact
(``models/saved_models/factor_scoring.json``), matching deployment; during
cross-validation EFA is re-fitted inside each fold, so per-fold scores differ
very slightly from this static snapshot.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ml_pipeline"))

import data_contract as dc  # noqa: E402

OUT = BASE / "data" / "processed" / "wers_ml_model_dataset.csv"

LABEL_BY_CODE = {index: label for index, label in enumerate(dc.LABELS)}


def main() -> None:
    frames = dc.build_model_frames()

    Xtr, Xte = frames["Xtr"], frames["Xte"]
    ytr, yte = frames["ytr"], frames["yte"]

    factor_cols = [c for c in Xtr.columns if c.startswith("factor_")]
    model_vars = list(frames["features"]) + factor_cols

    df, _ = dc.load_dataset()
    labels = df[dc.TARGET].astype(str).str.strip()
    modelled = df.loc[labels.isin(dc.LABELS)].reset_index(drop=True)

    def assemble(X: pd.DataFrame, y, split: str) -> pd.DataFrame:
        block = X.copy()
        block.insert(0, "serno", modelled.loc[X.index, "serno"].to_numpy())
        block.insert(1, "year", modelled.loc[X.index, "year"].to_numpy())
        block.insert(2, "split", split)
        block[dc.TARGET] = pd.Series(y).map(LABEL_BY_CODE).to_numpy()
        return block

    out = pd.concat(
        [assemble(Xtr, ytr, "train"), assemble(Xte, yte, "test")],
        axis=0,
        ignore_index=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Wrote {OUT}")
    print(f"  rows                : {len(out):,}")
    print(f"  model variables     : {len(model_vars)} "
          f"({len(frames['features'])} raw predictors + {len(factor_cols)} EFA factors)")
    print(f"  columns             : 3 admin (serno, year, split) "
          f"+ {len(model_vars)} variables + 1 target")
    print(f"  train / test rows   : {int((out['split'] == 'train').sum()):,}"
          f" / {int((out['split'] == 'test').sum()):,}")


if __name__ == "__main__":
    main()

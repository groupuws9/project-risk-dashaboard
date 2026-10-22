"""OperationalRiskPredictor — single/batch inference wrapper (README usage)."""
from pathlib import Path
import sys
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "saved_models"
sys.path.insert(0, str(BASE / "ml_pipeline"))
import data_contract as dc  # noqa: E402

_FACTOR_CACHE = {}


def _factor_artifact():
    """Load the EFA scoring artifact once (factor_scoring.json, written by
    ml_pipeline/04_model_training.py / 08_factor_analysis.py)."""
    if "artifact" not in _FACTOR_CACHE:
        import json
        path = MODEL_DIR / "factor_scoring.json"
        _FACTOR_CACHE["artifact"] = (json.loads(path.read_text())
                                     if path.exists() else None)
    return _FACTOR_CACHE["artifact"]


class OperationalRiskPredictor:
    """Load a saved pipeline and predict Low/Medium/High risk."""

    def __init__(self, pipe, features, labels=("Low", "Medium", "High"),
                 numeric_features=None, rare_category_map=None,
                 categorical_features=None):
        self.pipe = pipe
        self.features = list(features)
        self.labels = list(labels)
        self.numeric = set(numeric_features or [])
        # TRAIN-derived rare categories -> "OTHER_RARE". Persisted by
        # 04_model_training.py so inference folds exactly the same categories
        # the model was fitted on (train/serve parity).
        self.rare_map = dict(rare_category_map or {})
        # categorical set: explicit if given, else the complement of the
        # numeric features among the model features (minus the EFA factors,
        # which are numeric).
        if categorical_features is not None:
            self.categorical = set(categorical_features)
        else:
            self.categorical = {c for c in self.features
                                if c not in self.numeric
                                and not c.startswith("factor_")}
        self.numeric_list = [c for c in self.features if c in self.numeric]
        self.categorical_list = [c for c in self.features if c in self.categorical]

    @classmethod
    def load(cls, path=None):
        """Load best model (default) or a given .pkl path."""
        import json
        if path is None:
            meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
            best = meta["best_model"]
            path = MODEL_DIR / f"{best}.pkl"
        else:
            path = Path(path)
            try:
                meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
            except Exception:
                meta = {"labels": ["Low", "Medium", "High"], "features": None}
        labels = meta.get("labels", ["Low", "Medium", "High"])
        features = meta.get("features")
        numeric = meta.get("numeric_features", [])
        rare_map = meta.get("rare_category_map", {})
        categorical = meta.get("categorical_features")
        pipe = joblib.load(path)
        feats = features or getattr(pipe, "feature_names_in_", None)
        return cls(pipe, feats, labels, numeric, rare_map, categorical)

    def _frame(self, data):
        """Coerce input into the exact dtypes the pipeline was trained on.

        The cleaning itself is delegated to ``data_contract.apply_serve_transform``
        so the serving path is literally the SAME code as the training path:
        numeric coercion, sentinel handling, categorical folding and the
        TRAIN-derived rare-category grouping.  Any divergence here would mean
        the deployed model scores rows it was never fitted on.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame([dict(data)])
        for c in self.features:
            if c not in df.columns:
                df[c] = pd.NA
        # factor_* scores are derived from the raw HR items (train/serve parity)
        self._derive_factors(df)
        df = dc.apply_serve_transform(df, self.numeric_list, self.categorical_list,
                                      self.rare_map)
        return df[self.features]

    def _derive_factors(self, df):
        """Compute factor_* scores from the raw HR items (train/serve parity).

        The model was trained on EFA factor scores derived from items such as
        union coverage, collective bargaining, JCC, information sharing and
        consultation depth; the same scoring coefficients stored in
        models/saved_models/factor_scoring.json are re-applied here so a raw
        API payload can never be scored differently from training.
        """
        if not any(c.startswith("factor_") for c in self.features):
            return
        art = _factor_artifact()
        if not art:
            return
        import numpy as np
        X = df.reindex(columns=art["items"]).apply(pd.to_numeric,
                                                   errors="coerce")
        for c, codes in art.get("miss_codes", {}).items():
            if c in X.columns:
                X[c] = X[c].replace(list(codes), np.nan)
        med = pd.Series({c: float(v) for c, v in art["medians"].items()})
        X = X.astype(float).fillna(med)
        Z = (X - pd.Series({c: float(v) for c, v in art["mean"].items()})) \
            / pd.Series({c: float(v) for c, v in art["std"].items()})
        scores = Z.values @ np.asarray(art["weights"], dtype=float)
        for i, name in enumerate(art["factor_names"]):
            if name in df.columns:
                df[name] = scores[:, i]

    def predict(self, org_data):
        """Return (risk_level, max_probability)."""
        df = self._frame(org_data)
        proba = self.pipe.predict_proba(df)[0]
        idx = int(proba.argmax())
        return self.labels[idx], float(proba[idx])

    def predict_proba_dict(self, org_data):
        """Return {label: probability}."""
        df = self._frame(org_data)
        proba = self.pipe.predict_proba(df)[0]
        return {lab: float(p) for lab, p in zip(self.labels, proba)}

    def predict_batch(self, df):
        """Return list of predicted labels for a dataframe."""
        preds = self.pipe.predict(self._frame(df))
        return [self.labels[int(i)] for i in preds]

    # ------------------------------------------------------------------
    # Live local SHAP explanation (dashboard "Why?" chart)
    # ------------------------------------------------------------------
    def _orig_map(self):
        """Transformed feature name -> original feature name (same rule as
        ml_pipeline/07_shap_explanation.py so dashboards stay consistent)."""
        if getattr(self, "_mapping", None) is not None:
            return self._mapping
        t_names = list(self.pipe.named_steps["prep"].get_feature_names_out())
        feats = sorted(self.features, key=len, reverse=True)
        mapping = []
        for t in t_names:
            raw = t.split("__", 1)[-1]
            orig = raw
            for f in feats:
                if raw == f or raw.startswith(f + "_") or raw.startswith(f):
                    orig = f
                    break
            mapping.append(orig)
        self._mapping = mapping
        return mapping

    def _ensure_background(self):
        """Lazy: mean of the transformed real dataset (cached on the instance).

        Used as the reference point for exact linear SHAP values
        (contribution_j = coef_j * (x_j - E[x_j])), mirroring what
        shap.LinearExplainer computes for linear models.
        """
        if getattr(self, "_bg_mean", None) is not None:
            return
        import numpy as np
        import pandas as pd
        csv = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"
        df = pd.read_csv(csv, usecols=lambda c: c in set(self.features))
        frame = self._frame(df)
        t = self.pipe.named_steps["prep"].transform(frame)
        if hasattr(t, "toarray"):
            t = t.toarray()
        self._bg_mean = np.asarray(t, dtype=float).mean(axis=0)

    def explain(self, org_data, top_k=12):
        """Local explanation for ONE input, aggregated to original features.

        Explains the predicted class. Returns None on any failure — the
        prediction path must never depend on explainability.
        """
        try:
            import numpy as np
            frame = self._frame(org_data)
            clf = self.pipe.named_steps["clf"]
            prep = self.pipe.named_steps["prep"]
            proba = self.pipe.predict_proba(frame)[0]
            k = int(proba.argmax())
            t = prep.transform(frame)
            if hasattr(t, "toarray"):
                t = t.toarray()
            x = np.asarray(t, dtype=float)[0]

            if hasattr(clf, "coef_"):          # exact linear SHAP (logreg)
                self._ensure_background()
                coef = np.asarray(clf.coef_, dtype=float)
                inter = np.asarray(clf.intercept_, dtype=float).reshape(-1)
                if coef.shape[0] == 1 and len(self.labels) == 2:
                    row_coef = coef[0] if k == 1 else -coef[0]
                    base = (float(inter[0]) if k == 1 else -float(inter[0])) \
                        + float(row_coef @ self._bg_mean)
                else:
                    row_coef = coef[k]
                    base = float(inter[k]) + float(row_coef @ self._bg_mean)
                contrib = row_coef * (x - self._bg_mean)
            else:                              # tree/other: best-effort via shap
                import shap
                expl = shap.TreeExplainer(clf)
                sv = expl.shap_values(t)
                if isinstance(sv, list):
                    arr = np.stack(sv, axis=-1)[0]
                else:
                    arr = np.asarray(sv, dtype=float)
                    arr = arr[0] if arr.ndim == 2 else arr[0]
                contrib = arr[:, k] if arr.ndim == 2 else arr
                ev = np.asarray(getattr(expl, "expected_value", 0.0)).reshape(-1)
                base = float(ev[k]) if ev.size > k else float(ev[0])

            agg = {}
            for orig, v in zip(self._orig_map(), np.asarray(contrib, dtype=float)):
                agg[orig] = agg.get(orig, 0.0) + float(v)
            row_vals = frame.iloc[0]
            items = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
            return {
                "model": type(clf).__name__,
                "class_explained": self.labels[k],
                "base_value": round(base, 5),
                "output": round(base + float(np.sum(contrib)), 5),
                "contributions": [
                    {"feature": f, "shap": round(v, 5),
                     "value": str(row_vals.get(f, ""))} for f, v in items],
            }
        except Exception:
            return None

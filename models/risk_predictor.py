"""
OperationalRiskPredictor — single and batch inference wrapper.

This wrapper loads a saved full pipeline and applies it to raw organisation
records. It does not manually repeat preprocessing.

The saved pipeline contains the complete inference workflow:

    Categorical cleaning
    -> numeric coercion
    -> rare-category grouping
    -> EFA
    -> imputation, scaling and encoding
    -> classifier

Therefore, raw API payloads are converted to a raw one-row DataFrame and
passed directly to the saved pipeline.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "saved_models"

sys.path.insert(0, str(BASE / "ml_pipeline"))

import data_contract as dc  # noqa: E402


class OperationalRiskPredictor:
    """
    Load and apply a saved full pipeline to raw organisation records.

    The predictor does not preprocess records manually. It constructs a raw
    one-row DataFrame from the API payload and passes it directly to the
    saved pipeline.
    """

    def __init__(
        self,
        pipeline,
        predictor_columns,
        labels=None,
    ):
        self.pipeline = pipeline
        self.predictor_columns = list(predictor_columns)
        self.labels = list(labels or dc.LABELS)

    @classmethod
    def load(cls, model_path=None):
        """
        Load the best saved full pipeline and its feature contract.

        Args:
            model_path: Optional path to a specific saved pipeline. If omitted,
                the pipeline recorded as `best_model` in model_metadata.json
                is loaded.
        """
        metadata_path = MODEL_DIR / "model_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {metadata_path}. "
                "Run 04_model_training.py before loading the predictor."
            )

        metadata = json.loads(metadata_path.read_text())

        predictor_columns = metadata["features"]
        labels = metadata.get(
            "labels",
            dc.LABELS,
        )

        if model_path is None:
            best_model_name = metadata["best_model"]
            model_path = MODEL_DIR / f"{best_model_name}.pkl"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model artefact not found: {model_path}"
            )

        pipeline = joblib.load(model_path)

        return cls(
            pipeline=pipeline,
            predictor_columns=predictor_columns,
            labels=labels,
        )

    def _to_raw_frame(self, record):
        """
        Convert a payload dictionary or DataFrame to a raw prediction frame.

        The returned frame contains the same raw predictor columns expected by
        the saved full pipeline. Missing columns are filled with missing
        values, which the pipeline is designed to handle.
        """
        if isinstance(record, pd.DataFrame):
            raw_frame = record.copy()
        else:
            raw_frame = pd.DataFrame([dict(record)])

        missing_columns = [
            column
            for column in self.predictor_columns
            if column not in raw_frame.columns
        ]

        for column in missing_columns:
            raw_frame[column] = np.nan

        return raw_frame[self.predictor_columns]

    def predict(self, record):
        """
        Return the predicted class label and its probability.

        Args:
            record: A dictionary representing one organisation record.

        Returns:
            A tuple containing the predicted label and its probability.
        """
        raw_frame = self._to_raw_frame(record)

        probabilities = self.pipeline.predict_proba(raw_frame)[0]
        predicted_class_index = int(np.argmax(probabilities))

        predicted_label = self.pipeline.classes_[predicted_class_index]
        predicted_probability = float(
            probabilities[predicted_class_index]
        )

        return predicted_label, predicted_probability

    def predict_proba_dict(self, record):
        """
        Return class probabilities as a dictionary.

        Args:
            record: A dictionary representing one organisation record.

        Returns:
            A dictionary mapping each class label to its predicted probability.
        """
        raw_frame = self._to_raw_frame(record)

        probabilities = self.pipeline.predict_proba(raw_frame)[0]
        class_labels = self.pipeline.classes_

        return {
            str(label): float(probability)
            for label, probability in zip(
                class_labels,
                probabilities,
            )
        }

    def predict_batch(self, records):
        """
        Return predicted class labels for a batch of organisation records.

        Args:
            records: A pandas DataFrame containing raw predictor columns.

        Returns:
            A list of predicted class labels.
        """
        raw_frame = self._to_raw_frame(records)

        predictions = self.pipeline.predict(raw_frame)

        return [str(prediction) for prediction in predictions]

    def explain(self, record, top_k=12):
        """
        Return a local SHAP explanation for one organisation record.

        The explanation uses the saved pipeline's preprocessing steps and the
        fitted classifier. It explains the predicted class and aggregates
        transformed-feature contributions to original predictor columns.

        The method returns None if explanation fails. Prediction must never
        depend on explanation.
        """
        try:
            raw_frame = self._to_raw_frame(record)

            preprocessing_pipeline = self.pipeline[:-1]
            transformed_record = preprocessing_pipeline.transform(raw_frame)

            classifier = self.pipeline.named_steps["classifier"]
            preprocessing_step = self.pipeline.named_steps["preprocessing"]

            transformed_feature_names = list(
                preprocessing_step.get_feature_names_out()
            )

            probabilities = self.pipeline.predict_proba(raw_frame)[0]
            predicted_class_index = int(np.argmax(probabilities))

            import shap

            if (
                hasattr(classifier, "estimators_")
                or type(classifier).__name__ in (
                    "XGBClassifier",
                    "LGBMClassifier",
                )
            ):
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(transformed_record)

            elif hasattr(classifier, "coef_"):
                explainer = shap.LinearExplainer(
                    classifier,
                    transformed_record,
                )
                shap_values = explainer.shap_values(transformed_record)

            else:
                explainer = shap.Explainer(
                    classifier.predict_proba,
                    transformed_record,
                )
                shap_values = explainer(transformed_record).values

            shap_array = np.asarray(shap_values, dtype=float)

            if shap_array.ndim == 2:
                shap_array = shap_array[:, :, np.newaxis]

            if shap_array.ndim == 3:
                class_contributions = shap_array[
                    0,
                    :,
                    predicted_class_index,
                ]
            else:
                class_contributions = shap_array[0]

            expected_value = np.asarray(
                getattr(explainer, "expected_value", 0.0)
            ).reshape(-1)

            if expected_value.size > predicted_class_index:
                base_value = float(
                    expected_value[predicted_class_index]
                )
            else:
                base_value = float(expected_value[0])

            original_feature_mapping = [
                self._map_transformed_feature_to_original(
                    transformed_feature_name
                )
                for transformed_feature_name in transformed_feature_names
            ]

            contributions_by_original_feature = {}

            for original_feature, contribution in zip(
                original_feature_mapping,
                class_contributions,
            ):
                contributions_by_original_feature[original_feature] = (
                    contributions_by_original_feature.get(
                        original_feature,
                        0.0,
                    )
                    + float(contribution)
                )

            top_contributions = sorted(
                contributions_by_original_feature.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:top_k]

            example_record = raw_frame.iloc[0].to_dict()

            return {
                "model": type(classifier).__name__,
                "class_explained": self.pipeline.classes_[
                    predicted_class_index
                ],
                "base_value": round(base_value, 5),
                "output": round(
                    base_value + float(np.sum(class_contributions)),
                    5,
                ),
                "contributions": [
                    {
                        "feature": feature,
                        "shap": round(contribution, 5),
                        "value": str(
                            example_record.get(feature, "")
                        ),
                    }
                    for feature, contribution in top_contributions
                ],
            }

        except Exception:
            return None

    def _map_transformed_feature_to_original(
        self,
        transformed_feature_name,
    ):
        """
        Map a transformed feature name to its original predictor column.

        Numeric transformed features retain their original names. One-hot
        encoded categorical features use the form `column_value`.
        """
        if transformed_feature_name in self.predictor_columns:
            return transformed_feature_name

        categorical_columns = [
            column
            for column in self.predictor_columns
            if column not in self.pipeline.named_steps[
                "preprocessing"
            ].transformers_[0][2]
        ]

        for column in sorted(
            categorical_columns,
            key=len,
            reverse=True,
        ):
            if transformed_feature_name == column:
                return column

            if transformed_feature_name.startswith(f"{column}_"):
                return column

        return transformed_feature_name
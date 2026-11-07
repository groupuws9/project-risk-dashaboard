"""
06_model_evaluation.py — Deep evaluation of trained models.

Evaluates the saved full pipelines on the canonical held-out test partition.

The saved model artefacts contain the complete inference pipeline:

    Categorical cleaning -> numeric coercion -> rare-category grouping
    -> EFA -> preprocessing -> classifier

Therefore, this script passes the raw held-out predictor matrix directly to
each saved pipeline. It does not preprocess the test partition before
prediction.

The held-out test partition is used once, after model selection.
"""

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    RocCurveDisplay,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_contract as dc  # noqa: E402


BASE = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE / "models" / "saved_models"
TAB_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"
REP_DIR = BASE / "results" / "evaluation_reports"
DATA = BASE / "data" / "processed" / "wers_operational_risk_merged.csv"

TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


RANDOM_STATE = dc.RANDOM_STATE


print("=" * 78)
print("MODEL EVALUATION")
print("=" * 78)


# ---------------------------------------------------------------------------
# Load metadata and reconstruct the canonical held-out partition
# ---------------------------------------------------------------------------

metadata_path = MODEL_DIR / "model_metadata.json"

if not metadata_path.exists():
    raise FileNotFoundError(
        f"Model metadata not found: {metadata_path}. "
        "Run 04_model_training.py before running evaluation."
    )

metadata = json.loads(metadata_path.read_text())

TARGET = metadata["target"]
LABELS = metadata["labels"]
METADATA_FEATURES = metadata["features"]

df = pd.read_csv(DATA)

Xtr_raw, Xte_raw, ytr, yte = dc.split_dataset(
    df,
    METADATA_FEATURES,
    target_column=TARGET,
)

Xtr_raw = Xtr_raw.copy()
Xte_raw = Xte_raw.copy()

print(f"Outer training partition: {Xtr_raw.shape}")
print(f"Outer held-out test partition: {Xte_raw.shape}")
print("Training class distribution:", ytr.value_counts().to_dict())
print("Test class distribution:", yte.value_counts().to_dict())

if list(Xte_raw.columns) != list(METADATA_FEATURES):
    raise ValueError(
        "Feature columns in the reconstructed test partition do not match "
        "the features recorded in model_metadata.json."
    )


# ---------------------------------------------------------------------------
# Evaluate saved full pipelines
# ---------------------------------------------------------------------------

summary = []

model_files = sorted(
    path
    for path in MODEL_DIR.glob("*.pkl")
    if path.name != "best_model.pkl"
)

for model_path in model_files:
    model_name = model_path.stem

    print(f"\n--- Evaluating {model_name} ---")

    try:
        pipeline = joblib.load(model_path)

        predictions = pipeline.predict(Xte_raw)

        try:
            probabilities = pipeline.predict_proba(Xte_raw)
        except Exception:
            probabilities = None

        accuracy = accuracy_score(yte, predictions)
        balanced_accuracy = balanced_accuracy_score(yte, predictions)

        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                yte,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        roc_auc = np.nan

        if probabilities is not None:
            try:
                roc_auc = roc_auc_score(
                    yte,
                    probabilities,
                    multi_class="ovr",
                )
            except Exception as error:
                print(f"ROC-AUC skipped: {error}")

        confusion = confusion_matrix(yte, predictions)

        summary.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "roc_auc_ovr": roc_auc,
            }
        )

        print(
            f"{model_name:15s} "
            f"accuracy={accuracy:.3f} "
            f"balanced accuracy={balanced_accuracy:.3f} "
            f"macro-F1={macro_f1:.3f} "
            f"ROC-AUC={roc_auc:.3f}"
        )

        confusion_frame = pd.DataFrame(
            confusion,
            index=LABELS,
            columns=LABELS,
        )

        confusion_frame.to_csv(
            TAB_DIR / f"confusion_matrix_{model_name}.csv",
            index=True,
        )

        if HAS_MATPLOTLIB:
            figure, axis = plt.subplots(figsize=(5, 4.2))

            axis.imshow(confusion, cmap="Blues")
            axis.set_xticks(range(len(LABELS)))
            axis.set_xticklabels(LABELS)
            axis.set_yticks(range(len(LABELS)))
            axis.set_yticklabels(LABELS)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class")
            axis.set_title(f"Confusion matrix — {model_name}")

            for row_index in range(len(LABELS)):
                for column_index in range(len(LABELS)):
                    axis.text(
                        column_index,
                        row_index,
                        confusion[row_index, column_index],
                        ha="center",
                        va="center",
                    )

            figure.tight_layout()
            figure.savefig(
                FIG_DIR / f"confusion_matrix_{model_name}.png",
                dpi=130,
            )
            plt.close(figure)

            if probabilities is not None:
                try:
                    roc_figure, roc_axis = plt.subplots(figsize=(6, 5))

                    for class_index, class_label in enumerate(LABELS):
                        RocCurveDisplay.from_predictions(
                            (yte == class_label).astype(int),
                            probabilities[:, class_index],
                            name=class_label,
                            ax=roc_axis,
                        )

                    roc_axis.plot([0, 1], [0, 1], "k--", linewidth=1)
                    roc_axis.set_title(
                        f"One-vs-rest ROC curves — {model_name}"
                    )

                    roc_figure.tight_layout()
                    roc_figure.savefig(
                        FIG_DIR / f"roc_curves_{model_name}.png",
                        dpi=130,
                    )
                    plt.close(roc_figure)

                except Exception as error:
                    print(f"ROC plotting skipped: {error}")

        try:
            sample_size = min(400, len(Xte_raw))
            sample_indices = Xte_raw.sample(
                n=sample_size,
                random_state=RANDOM_STATE,
            ).index

            X_sample = Xte_raw.loc[sample_indices]
            y_sample = yte.loc[sample_indices]

            permutation_result = permutation_importance(
                pipeline,
                X_sample,
                y_sample,
                n_repeats=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            permutation_frame = pd.DataFrame(
                {
                    "feature": Xte_raw.columns,
                    "importance": permutation_result.importances_mean,
                    "standard_deviation": permutation_result.importances_std,
                }
            ).sort_values(
                by="importance",
                ascending=False,
            )

            permutation_frame.to_csv(
                TAB_DIR / f"permutation_importance_{model_name}.csv",
                index=False,
            )

            if HAS_MATPLOTLIB:
                top_features = permutation_frame.head(15).iloc[::-1]

                importance_figure, importance_axis = plt.subplots(
                    figsize=(9, 6)
                )

                importance_axis.barh(
                    top_features["feature"].astype(str),
                    top_features["importance"],
                    xerr=top_features["standard_deviation"],
                )

                importance_axis.set_title(
                    f"Permutation importance — {model_name}"
                )

                importance_figure.tight_layout()
                importance_figure.savefig(
                    FIG_DIR / f"permutation_importance_{model_name}.png",
                    dpi=130,
                )
                plt.close(importance_figure)

        except Exception as error:
            print(f"Permutation importance skipped: {error}")

    except Exception as error:
        print(f"{model_name}: evaluation failed ({error})")


# ---------------------------------------------------------------------------
# Save evaluation outputs
# ---------------------------------------------------------------------------

if summary:
    evaluation_summary = pd.DataFrame(summary).sort_values(
        by="balanced_accuracy",
        ascending=False,
    )
else:
    evaluation_summary = pd.DataFrame(
        columns=[
            "model",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "roc_auc_ovr",
        ]
    )

evaluation_summary.to_csv(
    TAB_DIR / "evaluation_summary.csv",
    index=False,
)

evaluation_summary.to_csv(
    REP_DIR / "evaluation_summary.csv",
    index=False,
)

report_lines = [
    "# Model Evaluation Report",
    "",
    "## Workplace operational-risk classification",
    "",
    f"Best model selected using training cross-validated balanced accuracy: "
    f"**{metadata['best_model']}**",
    "",
    "## Held-out test metrics",
    "",
    "The following metrics were calculated on the held-out 20% test partition.",
    "",
]

if not evaluation_summary.empty:
    try:
        report_lines.append(
            evaluation_summary.round(3).to_markdown(index=False)
        )
    except Exception:
        report_lines.append(
            evaluation_summary.round(3).to_string(index=False)
        )
else:
    report_lines.append(
        "No models were evaluated successfully. See console output."
    )

report_lines.extend(
    [
        "",
        "## Methodological notes",
        "",
        "- The saved model artefacts contain the complete inference pipeline.",
        "- Raw held-out predictor data were passed directly to each pipeline.",
        "- Rare-category grouping, EFA, imputation, scaling and encoding were "
        "applied using the transformations fitted during final refitting.",
        "- The held-out test partition was not used for model selection or "
        "hyperparameter tuning.",
        "- Balanced accuracy and macro-F1 are reported alongside accuracy "
        "because the target classes are imbalanced.",
        "",
        f"Target classes: {LABELS}",
    ]
)

(REP_DIR / "evaluation_report.md").write_text(
    "\n".join(report_lines)
)

print("\nSaved evaluation summary, report, confusion matrices and figures.")
print("DONE.")
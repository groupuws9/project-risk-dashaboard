# Model Evaluation Report

## Workplace operational-risk classification

Best model selected using training cross-validated balanced accuracy: **logreg**

## Held-out test metrics

The following metrics were calculated on the held-out 20% test partition.

| model            |   accuracy |   balanced_accuracy |   macro_precision |   macro_recall |   macro_f1 |   roc_auc_ovr |
|:-----------------|-----------:|--------------------:|------------------:|---------------:|-----------:|--------------:|
| rf_smoteenn      |      0.423 |               0.421 |             0.385 |          0.421 |      0.33  |         0.594 |
| xgboost_tuned    |      0.358 |               0.41  |             0.384 |          0.41  |      0.296 |         0.596 |
| logreg_smote     |      0.442 |               0.387 |             0.37  |          0.387 |      0.324 |         0.582 |
| logreg           |      0.442 |               0.385 |             0.373 |          0.385 |      0.323 |         0.59  |
| logreg_smoteenn  |      0.392 |               0.377 |             0.37  |          0.377 |      0.301 |         0.578 |
| stacked_ensemble |      0.642 |               0.376 |             0.369 |          0.376 |      0.37  |         0.585 |

## Methodological notes

- The saved model artefacts contain the complete inference pipeline.
- Raw held-out predictor data were passed directly to each pipeline.
- Rare-category grouping, EFA, imputation, scaling and encoding were applied using the transformations fitted during final refitting.
- The held-out test partition was not used for model selection or hyperparameter tuning.
- Balanced accuracy and macro-F1 are reported alongside accuracy because the target classes are imbalanced.

Target classes: ['Low', 'Medium', 'High']
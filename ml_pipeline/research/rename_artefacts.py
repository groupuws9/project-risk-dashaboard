"""Rename tables and figures to follow logical numbering."""
from pathlib import Path
import shutil

TABLES_DIR = Path(__file__).resolve().parents[2] / "research" / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "research" / "figures"

# Table renames to fix conflicts
table_renames = {
    # Fix table_4_2 conflict (excluded_features vs reproducibility)
    "table_4_2_reproducibility_configuration.csv": "table_4_6_reproducibility_configuration.csv",
    
    # Fix table_5_1 conflict (model_comparison vs dataset_verification)
    "table_5_1_dataset_verification.csv": "table_5_10_dataset_verification.csv",
    
    # Fix table_5_3 conflict (logreg_metrics vs fold_level_performance)
    "table_5_3_fold_level_performance.csv": "table_5_11_fold_level_performance.csv",
    
    # Fix table_5_4 conflict (random_forest_metrics vs class_specific_performance)
    "table_5_4_class_specific_performance.csv": "table_5_12_class_specific_performance.csv",
    
    # Fix table_5_5 conflict (xgboost_metrics vs shap_importance)
    # Keep existing table_5_5_xgboost_metrics.csv as is
    
    # Fix table_5_8 conflict (confusion_matrix_logreg vs prediction_consistency)
    "table_5_8_prediction_consistency.csv": "table_5_14_prediction_consistency.csv",
}

# Figure renames to fix conflicts
figure_renames = {
    # Fix fig_4_1 conflict (turnover_distribution vs training_pipeline)
    "fig_4_1_training_pipeline.png": "fig_4_5_training_pipeline.png",
    
    # Fix fig_4_3 conflict (correlation_heatmap vs api_integration)
    "fig_4_3_api_integration.png": "fig_4_6_api_integration.png",
    
    # Fix fig_5_5 conflict (runtime_validation vs shap_importance)
    "fig_5_5_runtime_validation.png": "fig_5_8_runtime_validation.png",
}

print("Renaming tables...")
for old, new in table_renames.items():
    old_path = TABLES_DIR / old
    new_path = TABLES_DIR / new
    if old_path.exists():
        if new_path.exists():
            print(f"  Skipping {old} -> {new} (target exists)")
        else:
            shutil.move(old_path, new_path)
            print(f"  Renamed: {old} -> {new}")
    else:
        print(f"  Not found: {old}")

print("\nRenaming figures...")
for old, new in figure_renames.items():
    old_path = FIGURES_DIR / old
    new_path = FIGURES_DIR / new
    if old_path.exists():
        if new_path.exists():
            print(f"  Skipping {old} -> {new} (target exists)")
        else:
            shutil.move(old_path, new_path)
            print(f"  Renamed: {old} -> {new}")
    else:
        print(f"  Not found: {old}")

print("\nDone!")

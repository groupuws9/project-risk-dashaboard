"""Fix remaining table numbering conflicts."""
from pathlib import Path
import shutil

TABLES_DIR = Path(__file__).resolve().parents[2] / "research" / "tables"

# Remaining table renames
table_renames = {
    # Fix table_5_10 conflict (dataset_verification vs feature_importance_top20)
    "table_5_10_feature_importance_top20.csv": "table_5_15_feature_importance_top20.csv",
    
    # Fix table_5_11 conflict (fold_level_performance vs shap_importance_top20)
    "table_5_11_shap_importance_top20.csv": "table_5_16_shap_importance_top20.csv",
    
    # Fix table_5_12 conflict (class_specific_performance vs imbalance_strategies)
    "table_5_12_imbalance_strategies.csv": "table_5_17_imbalance_strategies.csv",
}

print("Fixing remaining table conflicts...")
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

print("\nDone!")

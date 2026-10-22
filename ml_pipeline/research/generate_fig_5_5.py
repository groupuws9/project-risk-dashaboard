"""Generate Figure 5.5: Runtime-validation test results visualization."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
TAB_DIR = BASE / "research" / "tables"
FIG_DIR = BASE / "research" / "figures"

# Read system test results
sys_tests = pd.read_csv(TAB_DIR / "table_5_13_system_tests.csv")

# Count results by status
status_counts = sys_tests["Result"].value_counts()

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart of test results
colors = {"PASS": "#2ecc71", "FAIL": "#e74c3c", "NOT RUN": "#f39c12"}
pie_colors = [colors.get(s, "#95a5a6") for s in status_counts.index]
ax1.pie(status_counts.values, labels=status_counts.index, colors=pie_colors,
        autopct='%1.1f%%', startangle=90)
ax1.set_title("System Test Results (ST1-ST19)", fontsize=12, fontweight="bold")

# Bar chart by category
category_order = ["Dataset Verification", "Leakage Control", "Model Loading",
                  "Prediction Contract", "Input Validation", "Error Handling",
                  "Rate Limiting", "API Contract", "Feature Testing", "Data Integrity"]
sys_tests["Category"] = sys_tests["Description"].map({
    "Dataset loads with expected dimensions (14,448 x 53) and target present": "Dataset Verification",
    "Target class counts match turnover spec (Low 5,077 / Med 621 / High 687)": "Dataset Verification",
    "Leakage features excluded from model input (14 excluded columns; turnover not a feature)": "Leakage Control",
    "Best model loads from disk with its full feature vector": "Model Loading",
    "Valid input returns 200 with risk level and probabilities summing to 1": "Prediction Contract",
    "Missing required field returns 400 MISSING_FIELD": "Input Validation",
    "Empty payload lists all missing fields (400)": "Input Validation",
    "Non-numeric field type returns 400 INVALID_TYPE": "Input Validation",
    "Out-of-range value returns 400 OUT_OF_RANGE": "Input Validation",
    "Extra unknown fields are ignored; prediction still returned": "Input Validation",
    "Null optional field is median-imputed; prediction returned": "Input Validation",
    "Null required field treated as missing (400)": "Input Validation",
    "Malformed JSON body returns 400 MALFORMED_JSON": "Input Validation",
    "Model inference failure returns 422 MODEL_ERROR": "Error Handling",
    "Unexpected exception returns 500 INTERNAL_ERROR": "Error Handling",
    "Per-IP rate limit returns 429 RATE_LIMITED with Retry-After": "Rate Limiting",
    "Spec endpoint aliases functional (/predict, /data, /simulate)": "API Contract",
    "What-if lab returns baseline and scenario predictions": "Feature Testing",
    "Insights and data endpoints return valid strict JSON (NaN-free)": "Data Integrity",
})

cat_counts = sys_tests["Category"].value_counts()
cat_counts = cat_counts.reindex(category_order, fill_value=0)
cat_colors = [colors["PASS"] if c > 0 else "#95a5a6" for c in cat_counts]
bars = ax2.barh(cat_counts.index, cat_counts.values, color=cat_colors)
ax2.set_xlabel("Number of Tests", fontsize=10)
ax2.set_title("Tests by Category", fontsize=12, fontweight="bold")
ax2.set_xlim(0, max(cat_counts.values) + 1)
for i, (cat, count) in enumerate(zip(cat_counts.index, cat_counts.values)):
    ax2.text(count + 0.1, i, str(count), va="center", fontsize=9)

plt.tight_layout()
output_path = FIG_DIR / "fig_5_5_runtime_validation.png"
plt.savefig(output_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}")
print(f"Total tests: {len(sys_tests)}, Passed: {status_counts.get('PASS', 0)}, Failed: {status_counts.get('FAIL', 0)}")

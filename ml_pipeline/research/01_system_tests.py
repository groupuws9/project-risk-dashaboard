"""Execute the dissertation system-test plan (ST1-ST19) and emit results.

Runs the pytest suite programmatically plus data/metadata checks, then
writes research/tables/table_5_13_system_tests.csv and
research/system_test_report.md
Run:  python ml_pipeline/research/01_system_tests.py
"""
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
TAB = BASE / "research" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

ST_PLAN = [
    ("ST1",  "Dataset loads with expected dimensions (6,635 x 52) and target present",
             "special:dataset"),
    ("ST2",  "Target class counts match specification (Low 5,502 / Med 1,105 / High 28)",
             "special:target"),
    ("ST3",  "Leakage features excluded from model input (11 excluded columns)",
             "special:leakage"),
    ("ST4",  "Best model loads from disk (logreg, 36 features)",
             "test_load_best_model"),
    ("ST5",  "Valid input returns 200 with risk level and probabilities summing to 1",
             "test_valid_input_returns_prediction"),
    ("ST6",  "Missing required field returns 400 MISSING_FIELD",
             "test_missing_required_field"),
    ("ST7",  "Empty payload lists all missing fields (400)",
             "test_empty_payload_lists_all_missing_fields"),
    ("ST8",  "Non-numeric field type returns 400 INVALID_TYPE",
             "test_wrong_type_rejected"),
    ("ST9",  "Out-of-range value returns 400 OUT_OF_RANGE",
             "test_out_of_range_rejected"),
    ("ST10", "Extra unknown fields are ignored; prediction still returned",
             "test_extra_field_ignored_but_prediction_returned"),
    ("ST11", "Null optional field is median-imputed; prediction returned",
             "test_null_optional_field_is_imputed"),
    ("ST12", "Null required field treated as missing (400)",
             "test_null_required_field_is_missing"),
    ("ST13", "Malformed JSON body returns 400 MALFORMED_JSON",
             "test_malformed_json"),
    ("ST14", "Model inference failure returns 422 MODEL_ERROR",
             "test_model_error_returns_422"),
    ("ST15", "Unexpected exception returns 500 INTERNAL_ERROR",
             "test_internal_error_returns_500"),
    ("ST16", "Per-IP rate limit returns 429 RATE_LIMITED with Retry-After",
             "test_rate_limit_returns_429"),
    ("ST17", "Spec endpoint aliases functional (/predict, /data, /simulate)",
             "test_spec_endpoint_aliases"),
    ("ST18", "What-if lab returns baseline and scenario predictions",
             "test_whatif_missing_scenario_block"),
    ("ST19", "Insights and data endpoints return valid strict JSON (NaN-free)",
             "test_features_endpoint test_data_endpoints"),
]



def run_pytest():
    """Run full suite once, collect per-test outcomes."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=line"],
        capture_output=True, text=True, cwd=BASE)
    outcomes = {}
    for line in proc.stdout.splitlines():
        if "::" in line and any(s in line for s in ("PASSED", "FAILED", "ERROR")):
            test_name = line.split("::")[1].split()[0]
            status = ("PASSED" if "PASSED" in line else
                      "FAILED" if "FAILED" in line else "ERROR")
            outcomes[test_name] = status
    return outcomes, proc.returncode


def special_check(kind):
    """Data/metadata checks for ST1-ST3."""
    import pandas as pd
    df = pd.read_csv(BASE / "data" / "processed" /
                     "wers2011_operational_risk_merged.csv")
    if kind == "dataset":
        ok = df.shape == (6635, 52)
        return ("PASS" if ok else "FAIL", f"shape={df.shape}")
    if kind == "target":
        c = df["operational_risk_category"].value_counts()
        ok = (c.get("Low", 0) == 5502 and c.get("Medium", 0) == 1105
              and c.get("High", 0) == 28)
        return ("PASS" if ok else "FAIL",
                f"Low={c.get('Low')} Med={c.get('Medium')} High={c.get('High')}")
    if kind == "leakage":
        meta = json.loads((BASE / "models" / "saved_models" /
                           "model_metadata.json").read_text())
        leak = meta.get("leakage_excluded", [])
        ok = len(leak) == 11 and "operational_risk_score" in leak
        return ("PASS" if ok else "FAIL",
                f"{len(leak)} leakage features excluded")
    return "FAIL", "unknown special check"


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Running system test plan at {ts} ...")
    outcomes, rc = run_pytest()
    rows, n_pass = [], 0
    for st_id, desc, selector in ST_PLAN:
        if selector.startswith("special:"):
            status, evidence = special_check(selector.split(":")[1])
        else:
            names = selector.split()
            statuses = [outcomes.get(n, "NOT RUN") for n in names]
            if all(s == "PASSED" for s in statuses):
                status = "PASS"
            elif any(s in ("FAILED", "ERROR") for s in statuses):
                status = "FAIL"
            else:
                status = "NOT RUN"
            evidence = "pytest: " + ", ".join(names)
        n_pass += status == "PASS"
        rows.append({"Test ID": st_id, "Description": desc,
                     "Method": evidence, "Result": status, "Executed": ts})
        print(f"  {st_id}: {status}  ({desc[:58]})")

    out = TAB / "table_5_13_system_tests.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Test ID", "Description", "Method",
                                          "Result", "Executed"])
        w.writeheader()
        w.writerows(rows)

    md = BASE / "research" / "system_test_report.md"
    md.write_text(
        "# System Test Report (ST1-ST19)\n\n"
        f"Executed: {ts}  \n"
        f"Result: **{n_pass}/{len(rows)} passed** (pytest exit code {rc})\n\n"
        "| Test ID | Description | Result |\n|---|---|---|\n" +
        "\n".join(f"| {r['Test ID']} | {r['Description']} | {r['Result']} |"
                  for r in rows) + "\n",
        encoding="utf-8")
    print(f"\n{n_pass}/{len(rows)} system tests PASSED")
    print("Saved:", out.relative_to(BASE))
    print("Saved:", md.relative_to(BASE))


if __name__ == "__main__":
    main()

# System Test Report (ST1-ST19)

Executed: 2026-10-11 21:49:12  
Result: **19/19 passed** (pytest exit code 0)

| Test ID | Description | Result |
|---|---|---|
| ST1 | Dataset loads with expected dimensions (6,635 x 52) and target present | PASS |
| ST2 | Target class counts match specification (Low 5,502 / Med 1,105 / High 28) | PASS |
| ST3 | Leakage features excluded from model input (11 excluded columns) | PASS |
| ST4 | Best model loads from disk (logreg, 36 features) | PASS |
| ST5 | Valid input returns 200 with risk level and probabilities summing to 1 | PASS |
| ST6 | Missing required field returns 400 MISSING_FIELD | PASS |
| ST7 | Empty payload lists all missing fields (400) | PASS |
| ST8 | Non-numeric field type returns 400 INVALID_TYPE | PASS |
| ST9 | Out-of-range value returns 400 OUT_OF_RANGE | PASS |
| ST10 | Extra unknown fields are ignored; prediction still returned | PASS |
| ST11 | Null optional field is median-imputed; prediction returned | PASS |
| ST12 | Null required field treated as missing (400) | PASS |
| ST13 | Malformed JSON body returns 400 MALFORMED_JSON | PASS |
| ST14 | Model inference failure returns 422 MODEL_ERROR | PASS |
| ST15 | Unexpected exception returns 500 INTERNAL_ERROR | PASS |
| ST16 | Per-IP rate limit returns 429 RATE_LIMITED with Retry-After | PASS |
| ST17 | Spec endpoint aliases functional (/predict, /data, /simulate) | PASS |
| ST18 | What-if lab returns baseline and scenario predictions | PASS |
| ST19 | Insights and data endpoints return valid strict JSON (NaN-free) | PASS |

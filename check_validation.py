import re

# Check API validation rules
with open('d:/new research topics/project risk dashaboard/dashboard/validation.py') as f:
    content = f.read()

# Error types defined in the validation module
error_codes = [
    "MISSING_FIELD", "INVALID_TYPE", "OUT_OF_RANGE", 
    "MALFORMED_JSON", "MODEL_ERROR", "INTERNAL_ERROR", 
    "RATE_LIMITED", "SERVICE_UNAVAILABLE"
]

# Check validation constraints
print("API Error Codes:")
for code in error_codes:
    if code in content:
        print(f"  - {code}")

print("\nValidation Constraints:")
if 'greater_than' in content:
    print("  - greater_than threshold (OUT_OF_RANGE)")
if 'less_than' in content:
    print("  - less_than threshold (OUT_OF_RANGE)")
if 'Field' in content or 'field' in content:
    print("  - Pydantic Field constraints")

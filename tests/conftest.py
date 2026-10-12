"""Shared fixtures: project root on sys.path, Flask test client, sample payloads."""
from pathlib import Path
import sys

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from dashboard.app import app  # noqa: E402

VALID_PAYLOAD = {
    "sector": "1",
    "org_size_category_3": "2",
    "multinational": "2",
    "union_coverage_pct": "30",
    "collective_bargaining": "2",
    "performance_appraisal_pct": "3",
    "jcc_exists": "2",
    "information_sharing_score": "3",
    "pct_part_time": "20",
    "pct_temporary": "5",
}


@pytest.fixture()
def client():
    app.config.update(TESTING=False, DEBUG=False, PROPAGATE_EXCEPTIONS=False)
    with app.test_client() as c:
        yield c


@pytest.fixture()
def valid_payload():
    return dict(VALID_PAYLOAD)

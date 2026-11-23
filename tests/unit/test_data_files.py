import csv
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.mark.parametrize("path", sorted(DATA.glob("*.csv")), ids=lambda p: p.name)
def test_csv_rows_match_header(path):
    """An unquoted comma in a text cell shifts every later column (e.g. '50,000 hours')."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    bad = [i + 1 for i, r in enumerate(rows) if r and len(r) != len(rows[0])]
    assert bad == [], f"{path.name}: lines {bad[:5]} have the wrong number of fields"

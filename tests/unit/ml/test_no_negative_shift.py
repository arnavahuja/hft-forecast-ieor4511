"""No feature may shift backwards. Grepping for it is cheap and catches it at review time."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FEATURES = Path(__file__).parents[3] / "backend" / "hft_forecast" / "ml" / "features"
NEGATIVE_SHIFT = re.compile(r"\.shift\(\s*-")
MODULES = sorted(FEATURES.glob("*.py"))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_negative_shift_in_feature_code(path):
    offending = [
        f"{path.name}:{n}"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if NEGATIVE_SHIFT.search(line)
    ]
    assert offending == []


def test_the_pattern_would_catch_one():
    assert NEGATIVE_SHIFT.search("pl.col('mid').shift(-1)")
    assert NEGATIVE_SHIFT.search("x.shift( -steps )")
    assert not NEGATIVE_SHIFT.search("pl.col('mid').shift(1)")

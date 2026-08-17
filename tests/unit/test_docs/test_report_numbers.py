"""Test: the LaTeX report's generated inputs still match the repository.

Reports/morphology_gate_racing.tex quotes no numbers of its own. The results
table and every inline figure come from Reports/make_figures.py, and both are
tracked, so this pins them the same way the markdown tables are pinned.

The report previously carried its inline numbers as literals and one drifted:
the raw torque range was quoted over t = 24-66 deg, an earlier sweep's span,
rather than over the feasible family the report defines. That is the failure
this guards.
"""

# Standard library
import sys
from pathlib import Path

# Third-party libraries
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Reports"))

import make_figures as mf  # noqa: E402

TEX = REPO / "Reports" / "morphology_gate_racing.tex"


def _angles() -> np.ndarray:
    lo, hi = mf.feasible_range()
    return np.linspace(lo, hi, 7)


def test_inline_numbers_regenerate_unchanged() -> None:
    """numbers.tex still matches what the repository produces."""
    committed = (REPO / "Reports" / "numbers.tex").read_text()
    assert mf.numbers_tex(_angles()) == committed, (
        "rerun Reports/make_figures.py and rebuild the report")


def test_results_table_regenerates_unchanged() -> None:
    """The max-speed table still matches the CSVs in docs/data/."""
    committed = (REPO / "Reports" / "results_table.tex").read_text()
    assert mf.results_table() == committed, "rerun Reports/make_figures.py"


def test_report_quotes_no_literal_numbers_for_generated_quantities() -> None:
    """The prose uses the macros, so the values cannot be edited out of sync."""
    tex = TEX.read_text()
    assert r"\input{numbers}" in tex
    generated = {"20.44", "69.56", "313.3", "121.3", "2.58", "120.8",
                 "1.489", "3.996", "1.735", "3.896"}
    stale = sorted(v for v in generated if v in tex)
    assert not stale, f"literal values in the prose, use the macros instead: {stale}"

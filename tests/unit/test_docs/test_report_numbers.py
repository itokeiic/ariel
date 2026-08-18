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
                 "1.489", "3.996", "1.735", "3.896",
                 # speeds, current and superseded -- both come from numbers.tex
                 "10.164", "8.367", "6.648", "12.172", "8.578", "7.078"}
    stale = sorted(v for v in generated if v in tex)
    assert not stale, f"literal values in the prose, use the macros instead: {stale}"


def test_every_symbol_in_the_report_is_defined_in_its_notation_section() -> None:
    """No mathematical symbol appears in the body without a definition.

    Cheap to satisfy and easy to lose: a symbol introduced in a later revision
    reads fine to whoever added it and is opaque to everyone else.
    """
    import re  # noqa: PLC0415

    tex = TEX.read_text()
    body = tex[tex.index(r"\begin{document}"):]
    i, j = body.index(r"\section*{Notation}"), body.index(r"\section{Airframes}")
    notation, rest = body[i:j], body[:i] + body[j:]

    units = {"circ", "per", "metre", "second", "percent", "degree", "kilo", "gram",
             "newton", "radian", "squared", "frac", "tfrac", "qquad", "cos", "sin",
             "times", "in", "approx", "leq", "neq", "pm", "max", "mathrm", "text",
             "emph", "textbf", "num", "SI", "SIrange", "S", "ref", "label", "texttt"}

    def symbols(txt: str) -> set[str]:
        segs = re.findall(r"\$([^$]+)\$", txt) + re.findall(
            r"\\begin\{equation\}(.*?)\\end\{equation\}", txt, re.S)
        out: set[str] = set()
        for seg in segs:
            seg = re.sub(r"\\(SI|SIrange|num|text|mathrm|emph|textbf)\{[^{}]*\}", " ", seg)
            for tok in re.findall(
                    r"\\[a-zA-Z]+|[A-Za-z]_\{[^{}]+\}|[A-Za-z]_\\[a-zA-Z]+|[A-Za-z]", seg):
                if tok.lstrip("\\") not in units:
                    out.add(tok)
        return out

    # Generated value macros carry their own meaning; they are not notation.
    generated = set(re.findall(r"\\newcommand\{\\(\w+)\}", (REPO / "Reports" / "numbers.tex").read_text()))
    used = {t for t in symbols(rest) if t.lstrip("\\") not in generated}
    undefined = sorted(used - symbols(notation))
    assert not undefined, f"symbols used but never defined: {undefined}"

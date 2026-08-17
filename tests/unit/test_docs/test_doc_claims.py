"""Test: the documentation checker, and the documents it covers.

docs/tools/check_doc_claims.py verifies the mechanically checkable parts of a
markdown document against the repository: cited paths and line numbers, CLI
flags, identifiers, and markdown that renders as something other than it looks.

Tracking the checker is not enough on its own -- a tool nobody runs protects
nothing -- so it runs here.

Scope is deliberately split. The rendering lint makes no assumptions about the
repository and is applied to every document under docs/. The claim checks assume
a document written *about* this repository; the older Sphinx pages predate them
and use bare filenames as prose ("edit conf.py"), so they are opted in per
document via CHECKED below. Add to it when a document is written to that
standard.
"""

# Standard library
import sys
from pathlib import Path

# Third-party libraries
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "tools"))

import check_doc_claims as cdc  # noqa: E402

CHECKED = (REPO / "docs" / "drone_morphology_gate_racing.md",)
ALL_DOCS = sorted((REPO / "docs").rglob("*.md"))


def test_there_are_documents_to_check() -> None:
    """Guard against the globs silently matching nothing."""
    assert ALL_DOCS, "no markdown found under docs/"
    assert all(d.is_file() for d in CHECKED)


@pytest.mark.parametrize("doc", ALL_DOCS, ids=lambda p: p.name)
def test_markdown_renders_as_written(doc: Path) -> None:
    """No table that GFM would render as literal text or detach from its list."""
    assert cdc.check_rendering(doc) == []


@pytest.mark.parametrize("doc", CHECKED, ids=lambda p: p.name)
def test_claims_match_the_repository(doc: Path) -> None:
    """Cited paths, line numbers, flags and identifiers still resolve."""
    assert cdc.check(doc, quiet=True) == []


# --- the lint's own behaviour, on synthetic input ---------------------------

TABLE = "| a | b |\n|---|---|\n| 1 | 2 |\n"


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "t.md"
    p.write_text(text)
    return p


def test_lint_catches_table_indented_under_a_list_item(tmp_path: Path) -> None:
    """GFM renders this as literal pipe-delimited text."""
    body = "x\n\n1. item\n\n" + "".join(f"   {ln}\n" for ln in TABLE.splitlines())
    assert cdc.check_rendering(_write(tmp_path, body)) != []


def test_lint_catches_prose_detached_by_a_table(tmp_path: Path) -> None:
    """A table at column 0 ends the list, so indented prose after it detaches."""
    assert cdc.check_rendering(
        _write(tmp_path, "x\n\n1. item\n\n" + TABLE + "\n   continuation\n")) != []


def test_lint_accepts_a_table_that_properly_follows_a_list(tmp_path: Path) -> None:
    """The blank line closes the list; the table is its own block. Not a defect."""
    assert cdc.check_rendering(
        _write(tmp_path, "x\n\n- item\n\n" + TABLE + "\nback at column 0\n")) == []

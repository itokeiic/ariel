#!/usr/bin/env python3
"""Verify a markdown document's claims against the repository.

Prose drifts from code silently: a renamed function, a shifted line number, a
citation to a file that never existed. None of that is visible from inside the
document, and re-reading does not catch it -- the same reasoning that wrote the
sentence finds it plausible. This checks the parts that are mechanically
checkable.

Checks
  paths   `path/to/file.py` mentioned in backticks -> exists
  cites   file.py:123 or file.py:12-20              -> exists, line in range
  flags   --some-flag                               -> defined by some script
                                                       the document references
  names   `identifier` or `func()` that look like code -> found in the repo

Checks are split by how safe they are to apply broadly. `--rendering-only` runs
just the markdown-rendering lint, which has no repository-specific assumptions
and is clean across all of docs/. The claim checks (paths, cites, flags, names)
assume a document written about this repository, and the older Sphinx pages
predate them, so they are applied per-document.

Usage
  uv run --no-sync python docs/tools/check_doc_claims.py docs/whatever.md
  uv run --no-sync python docs/tools/check_doc_claims.py --quiet docs/*.md
  uv run --no-sync python docs/tools/check_doc_claims.py --rendering-only docs/**/*.md

Exit code 1 if any hard check fails, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CODE_EXT = (".py", ".md", ".toml", ".yaml", ".yml", ".json", ".usda", ".xacro", ".sh")
CITE_RE = re.compile(r"([\w./-]+\.(?:py|md|toml|yaml|yml|json|usda|xacro|sh)):(\d+)(?:-(\d+))?")
TICK_RE = re.compile(r"`([^`\n]+)`")
FLAG_RE = re.compile(r"(?<![\w-])--([a-z][a-z0-9_-]{2,})")
ARGPARSE_RE = re.compile(r"add_argument\(\s*[\"']--([a-z][a-z0-9_-]*)")
# Flags that come from tools rather than from the scripts a doc references.
EXTERNAL_FLAGS = {
    "no-sync", "help", "quiet", "version", "active", "no-video", "out-dir",
    "python", "directory",
    # Defined by the SPEAR consortium's Isaac scripts, which are supplied by
    # the consortium and not tracked here -- see EXTERNAL_PREFIXES.
    "base_body_name",
}
# Referenced from documents but not part of this repository, so not checkable.
# A path under one of these is skipped rather than reported missing.
EXTERNAL_PREFIXES = ("airevolve/", "examples/spear_vua_upb/", "/", "~", "http")
IDENT_RE = re.compile(r"^[A-Za-z_][\w.]*(?:\(\))?$")


def tracked_files() -> set[str]:
    """Files a fresh clone would have.

    The working tree is the wrong authority: it contains untracked scratch and
    consortium-supplied files, so checking against it passes locally and fails
    for everyone else -- exactly the drift this tool exists to catch.
    """
    if not hasattr(tracked_files, "_cache"):
        r = subprocess.run(["git", "ls-files"], cwd=REPO,
                           capture_output=True, text=True, timeout=30)
        tracked_files._cache = set(r.stdout.split())
    return tracked_files._cache


def repo_has(needle: str) -> bool:
    """Is this identifier anywhere in the tracked source?"""
    try:
        r = subprocess.run(
            ["git", "grep", "-qF", "--", needle.rstrip("()")],
            cwd=REPO, capture_output=True, timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False


LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def check_rendering(doc: Path) -> list[str]:
    """Markdown that parses as something other than it looks.

    Two failure modes, both seen in this repo: a table indented under a list
    item renders as literal pipe-delimited text, and a table at column 0 inside
    a list silently terminates the list, detaching the indented prose after it.
    """
    lines = doc.read_text().split("\n")
    fails = []
    for n, line in enumerate(lines, 1):
        if not (line.lstrip().startswith("|") and line.rstrip().endswith("|")):
            continue
        if line[:1].isspace():
            fails.append(f"{doc.name}:{n}  render -> table indented under a list "
                         f"item; GFM will not render it")
            continue
        # A table at column 0 after a list is fine -- the blank line closes the
        # list and the table is its own block. What breaks is INDENTED prose
        # after the table: it was written as a list continuation, but the list
        # already ended, so it detaches.
        rest = lines[n:]
        if not rest or (rest and rest[0].lstrip().startswith("|")):
            continue                       # still inside the table
        nxt = next((l for l in rest if l.strip()), "")
        if nxt[:1].isspace() and not LIST_RE.match(nxt):
            fails.append(f"{doc.name}:{n}  render -> indented prose follows this "
                         f"table; the table ended the list, so it detaches")
    return fails


def check(doc: Path, quiet: bool) -> list[str]:
    text = doc.read_text()
    lines = text.split("\n")
    fails: list[str] = []
    ok = {"paths": 0, "cites": 0, "flags": 0, "names": 0}

    # --- scripts this document tells the reader to run; their real flags ---
    referenced_scripts = {
        m for m in re.findall(r"[\w./-]+\.py", text) if m in tracked_files()
    }
    known_flags = set(EXTERNAL_FLAGS)
    for sc in referenced_scripts:
        known_flags |= set(ARGPARSE_RE.findall((REPO / sc).read_text()))

    absent_markers = ("absent", "does not exist", "not present", "never existed",
                      "never been in its history", "unverifiable", "missing",
                      "no safety net")
    # Absence is asserted across a sentence, not always on one line, so the
    # marker is looked for in the surrounding paragraph.
    paras, cur = [], []
    para_of = {}
    for i, ln in enumerate(lines, 1):
        if ln.strip():
            cur.append(i)
        else:
            for j in cur:
                para_of[j] = len(paras)
            paras.append(" ".join(lines[j - 1] for j in cur).lower())
            cur = []
    for j in cur:
        para_of[j] = len(paras)
    paras.append(" ".join(lines[j - 1] for j in cur).lower())

    for n, line in enumerate(lines, 1):
        para = paras[para_of.get(n, 0)] if paras else ""
        if any(m in para for m in absent_markers):
            continue                          # the doc is asserting absence
        # --- path:line citations ---
        for path, start, end in CITE_RE.findall(line):
            if path.startswith(EXTERNAL_PREFIXES):
                continue                      # outside this repo
            if path not in tracked_files():
                fails.append(f"{doc.name}:{n}  cite -> missing file: {path}")
                continue
            total = len((REPO / path).read_text().split("\n"))
            hi = int(end or start)
            if hi > total:
                fails.append(
                    f"{doc.name}:{n}  cite -> {path}:{hi} beyond EOF ({total} lines)")
            else:
                ok["cites"] += 1

        # --- backticked paths and identifiers ---
        for tick in TICK_RE.findall(line):
            t = tick.strip()
            if CITE_RE.fullmatch(t):
                continue                      # already handled
            # Maths in backticks contains slashes too: `km/kf`, `sqrt(K_rot/I)`,
            # `v^2/R`. Anything with operator or bracket characters is an
            # expression, not a path.
            if any(c in t for c in "()[]^=<>+*·−,;$\\"):
                continue
            # A path has a file extension. Without this, maths like `km/kf`
            # and `1/omega_n` reads as a path because it contains a slash.
            looks_pathy = t.endswith(CODE_EXT) or (
                "/" in t and " " not in t and t.split(":")[0].endswith(CODE_EXT))
            if looks_pathy:
                # `uv run python docs/tools/x.py` is a command, not a path: the
                # first token is the runner. Take the token that looks like a
                # file, not the first one.
                toks = [w for w in t.split() if w.split(":")[0].endswith(CODE_EXT)]
                base = (toks[-1] if toks else t).split(":")[0]
                if any(ch in base for ch in "*?"):
                    continue                  # a glob, not a claim
                if base.startswith(EXTERNAL_PREFIXES):
                    continue                  # outside this repo, not checkable
                if base in tracked_files() or (
                        REPO / base).is_dir() and base in {
                        f.rsplit("/", 1)[0] for f in tracked_files() if "/" in f}:
                    ok["paths"] += 1
                else:
                    fails.append(f"{doc.name}:{n}  path -> not found: {base}")
            elif IDENT_RE.fullmatch(t) and len(t) > 4 and "_" in t or t.endswith("()"):
                if repo_has(t):
                    ok["names"] += 1
                else:
                    fails.append(f"{doc.name}:{n}  name -> not in repo: {t}")

        # --- CLI flags ---
        for flag in FLAG_RE.findall(line):
            if flag in known_flags:
                ok["flags"] += 1
            else:
                fails.append(f"{doc.name}:{n}  flag -> --{flag} not defined by "
                             f"any script this doc references")

    if not quiet:
        print(f"{doc}: {ok['cites']} citations, {ok['paths']} paths, "
              f"{ok['names']} identifiers, {ok['flags']} flags verified")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("docs", nargs="+", type=Path)
    ap.add_argument("--quiet", action="store_true", help="only report failures")
    ap.add_argument("--rendering-only", action="store_true",
                    help="run only the markdown-rendering lint, which makes no "
                         "assumptions about the repository")
    args = ap.parse_args()

    all_fails: list[str] = []
    for d in args.docs:
        all_fails += check_rendering(d)
        if not args.rendering_only:
            all_fails += check(d, args.quiet)

    if all_fails:
        print(f"\n{len(all_fails)} unverified claim(s):", file=sys.stderr)
        for f in all_fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("all mechanically checkable claims verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

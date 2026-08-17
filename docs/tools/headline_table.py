#!/usr/bin/env python3
"""Emit the headline max-speed table under the strict gate criterion.

    uv run --no-sync python docs/tools/headline_table.py

Writes docs/data/headline_strict_table.md and prints it. Section 2 of
docs/drone_morphology_gate_racing.md embeds the output verbatim, and
tests/unit/test_docs/ asserts the embedded copy still matches, so the headline
result cannot drift from the data behind it -- which is exactly what happened
to the pre-correction version.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "docs" / "data" / "tuned_strict_completion.csv"
OUT = REPO / "docs" / "data" / "headline_strict_table.md"
TURNS = (60.0, 90.0, 120.0)
SPEED_TOL = 0.0625      # bisection tolerance; anything closer is a tie


def table() -> str:
    data: dict[float, dict[float, float]] = {}
    with SRC.open() as fh:
        for r in csv.DictReader(fh):
            data.setdefault(float(r["turn_deg"]), {})[
                round(float(r["half_angle_deg"]), 2)] = float(r["max_speed"])
    best = {t: max(col.values()) for t, col in data.items()}
    angles = sorted(data[TURNS[1]])

    lines = ["| half-angle | 60° | 90° | 120° |", "|---|---|---|---|"]
    for a in angles:
        cells = []
        for t in TURNS:
            v = data[t][a]
            # Bold every value inside the tolerance of the best: naming one
            # winner out of a tie is a claim the resolution does not support.
            cells.append(f"**{v:.3f}**" if best[t] - v < SPEED_TOL else f"{v:.3f}")
        lines.append(f"| {a:.2f}° | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    md = table()
    OUT.write_text(md)
    print(md)
    print(f"wrote {OUT.relative_to(REPO)}")

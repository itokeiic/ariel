#!/usr/bin/env python3
"""Draw the canonical single-arm perturbations, annotated with flight results.

    uv run --no-sync python docs/tools/perturbation_figure.py [perturbation.csv]

Morphology comes from ariel.simulation.drone.reference_morphologies -- the same
definition the sweep flies -- so the drawing cannot disagree with the numbers
printed beneath it. Speeds come from a --perturbation-sweep CSV.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ariel.simulation.drone.reference_morphologies import reference_morphologies

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "data" / "perturbation_morphologies.png"
PROP_R, CORE_R = 0.0635, 0.05
INK, ACCENT, GOOD, BAD, MUTED = "#1a202c", "#2b6cb0", "#2f855a", "#c53030", "#a0aec0"


def load(path: Path) -> dict[tuple[str, float], dict]:
    with path.open() as fh:
        return {(r["key"], float(r["turn_deg"])): r for r in csv.DictReader(fh)}


def draw(ax, m, speeds: dict[float, str]) -> None:
    g = m.genome
    lengths, az, el = g[:, 0], g[:, 1], g[:, 2]
    for k in range(len(g)):
        # Project the arm onto the horizontal plane; elevation shortens it.
        r = lengths[k] * np.cos(el[k])
        x, y = r * np.cos(az[k]), r * np.sin(az[k])
        lifted = abs(el[k]) > 1e-9
        ax.plot([0, x], [0, y], "-", color=BAD if lifted else INK,
                lw=2.4 if lifted else 1.5, zorder=3)
        ax.add_patch(plt.Circle((x, y), PROP_R, color=BAD if lifted else GOOD,
                                alpha=0.5 if lifted else 0.38, zorder=2))
        if lifted:
            ax.annotate(f"+{np.degrees(el[k]):.0f}° out of plane",
                        xy=(x, y), xytext=(x * 1.25, y * 1.55), fontsize=7.5,
                        color=BAD, ha="center",
                        arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    ax.add_patch(plt.Circle((0, 0), CORE_R, color=ACCENT, alpha=0.55, zorder=4))
    ax.arrow(0, 0, 0.10, 0, head_width=0.022, head_length=0.022, color=INK,
             lw=1.1, zorder=5, length_includes_head=True)
    lim = 0.30
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cbd5e0")
    lines = "\n".join(f"{t:.0f}°: {v}" for t, v in sorted(speeds.items()))
    ax.set_title(f"{m.label}\n{lines}", fontsize=9, pad=8)


def markdown(data, fam, turns) -> str:
    head = "| perturbation | " + " | ".join(f"{t:.0f}° max speed" for t in turns) + " | plant |"
    lines = [head, "|---" * (len(turns) + 2) + "|"]
    for m in fam.values():
        cells = []
        for t in turns:
            r = data.get((m.key, t))
            v = float(r["max_speed"]) if r else float("nan")
            cells.append("did not fly" if v != v else f"{v:.3f} m/s")
        ok = "axial" if m.axial_thrust else "**not simulated**"
        lines.append(f"| {m.label} | " + " | ".join(cells) + f" | {ok} |")
    return "\n".join(lines) + "\n"


def main(csv_path: Path) -> None:
    data = load(csv_path)
    fam = reference_morphologies()
    turns = sorted({t for _, t in data})
    fig, axes = plt.subplots(1, len(fam), figsize=(3.6 * len(fam), 4.8))
    for ax, m in zip(np.atleast_1d(axes).ravel(), fam.values()):
        speeds = {}
        for t in turns:
            r = data.get((m.key, t))
            if r is None:
                continue
            tag = "" if int(r["axial_thrust"]) else "  (invalid)"
            speeds[t] = f"{float(r['max_speed']):.3f} m/s{tag}"
        draw(ax, m, speeds)
        if not m.axial_thrust:
            ax.text(0.5, -0.16, "thrust axis tilts — reduced plant\ndoes not "
                    "simulate this; speed not comparable",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color=BAD)
    fig.suptitle("Single-arm perturbations of the X quad: geometry and max "
                 "completing speed", fontsize=12, y=1.04)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    md = markdown(data, fam, turns)
    (REPO / "docs" / "data" / "perturbation_table.md").write_text(md)
    print(md)
    print(f"wrote {OUT.relative_to(REPO)} and docs/data/perturbation_table.md")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1
         else REPO / "docs" / "data" / "perturbation.csv")

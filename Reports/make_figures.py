#!/usr/bin/env python3
"""Generate the figures and the results table for morphology_gate_racing.tex.

Everything is derived from the repository -- geometry from the decoder and the
plant interface, speeds from the CSVs in docs/data/ -- so the report cannot
drift from the code or the measurements.

    uv run --no-sync python Reports/make_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.plant import RotorGeometry, moment_allocation
from ariel.simulation.tasks.slalom_course import slalom_gates

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ARM_L, PROP_R, CORE_R, PROP_SIZE = 0.20, 0.0635, 0.05, 5
PAYLOAD, N_ARMS, TOL = 0.667, 4, 0.1
INK, ACCENT, GOOD, BAD = "#1a202c", "#2b6cb0", "#2f855a", "#c53030"


def genome(half_angle: float, arm_length: float = ARM_L) -> np.ndarray:
    az = np.array([half_angle, np.pi - half_angle, np.pi + half_angle, -half_angle])
    g = np.zeros((N_ARMS, 6))
    g[:, 0] = arm_length
    g[:, 1] = az
    g[:, 3] = np.pi
    g[:, 5] = np.arange(N_ARMS) % 2
    return g


def agility(half_angle: float) -> tuple[float, float]:
    """(alpha_roll, alpha_pitch) in rad/s^2 per newton."""
    g = genome(half_angle)
    props = blueprint_to_propellers(
        spherical_angular_to_blueprint(g, propsize=PROP_SIZE), convention="z_up")
    cfg = DroneConfiguration(props, payload_mass=PAYLOAD)
    kf, km = cfg.propellers[0]["constants"]
    geom = RotorGeometry(
        positions=np.array([p["loc"] for p in props], dtype=float),
        axes=np.array([p["dir"][:3] for p in props], dtype=float),
        spins=np.array([1.0 if p["dir"][3] == "ccw" else -1.0 for p in props]))
    A = moment_allocation(geom, torque_to_thrust_ratio=km / kf, inertia=cfg.inertia_matrix)
    return float(np.linalg.norm(A[0])), float(np.linalg.norm(A[1]))


def feasible_range(arm_length: float = ARM_L) -> tuple[float, float]:
    c = (1.0 + TOL) * PROP_R / arm_length
    return float(np.arcsin(c)), float(np.pi / 2 - np.arcsin(c))


# ---------------------------------------------------------------- figure 1
def figure_morphologies(angles: np.ndarray) -> None:
    fig, axes = plt.subplots(1, len(angles), figsize=(13.6, 2.5))
    for ax, t in zip(axes, angles):
        az = np.array([t, np.pi - t, np.pi + t, -t])
        for k, a in enumerate(az):
            x, y = ARM_L * np.cos(a), ARM_L * np.sin(a)
            ax.plot([0, x], [0, y], "-", color=INK, lw=1.4, zorder=3)
            ax.add_patch(plt.Circle((x, y), PROP_R, color=GOOD, alpha=0.40, zorder=2))
        ax.add_patch(plt.Circle((0, 0), CORE_R, color=ACCENT, alpha=0.55, zorder=4))
        ax.arrow(0, 0, 0.10, 0, head_width=0.022, head_length=0.022,
                 color=INK, lw=1.1, zorder=5, length_includes_head=True)
        ar, ap = agility(float(t))
        lim = ARM_L + PROP_R * 1.25
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cbd5e0")
        ax.set_title(rf"$t={np.degrees(t):.1f}^\circ$" "\n"
                     rf"$\alpha_\phi={ar:.0f}$, $\alpha_\theta={ap:.0f}$", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "morphologies.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def figure_courses(turns=(60.0, 90.0, 120.0)) -> None:
    fig, axes = plt.subplots(len(turns), 1, figsize=(7.0, 5.4), sharex=True)
    for ax, turn in zip(axes, turns):
        c = slalom_gates(turn, n_gates=15)
        ax.plot(c.path_pos[:, 0], c.path_pos[:, 1], "-", color=ACCENT, lw=1.1, zorder=1)
        for k, (p, yaw) in enumerate(zip(c.path_pos, c.path_yaw)):
            n = np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2)]) * c.gate_size / 2
            scored = k < len(c.gate_pos)
            ax.plot([p[0] - n[0], p[0] + n[0]], [p[1] - n[1], p[1] + n[1]], "-",
                    color=BAD if scored else "#a0aec0", lw=2.4,
                    solid_capstyle="butt", zorder=2)
        ax.plot(c.starting_pos[0], c.starting_pos[1], "o", color=GOOD, ms=5, zorder=3)
        ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.4)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(rf"$\theta={turn:.0f}^\circ$:  $R={c.radius:.2f}$ m,  "
                     rf"spacing ${c.spacing:.2f}$ m,  amplitude $\pm{c.amplitude:.2f}$ m",
                     fontsize=8.5, loc="left")
    axes[-1].set_xlabel("x (m)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "courses.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- table
def results_table() -> str:
    files = {60.0: "tuned_60deg.csv", 90.0: "tuned_90deg.csv", 120.0: "tuned_60_120deg.csv"}
    data: dict[float, dict[float, dict]] = {}
    for turn, fn in files.items():
        rows = list(csv.DictReader(open(REPO / "docs" / "data" / fn)))
        rows = [r for r in rows if abs(float(r["turn_deg"]) - turn) < 1e-9]
        data[turn] = {round(float(r["half_angle_deg"]), 2): r for r in rows}

    angles = sorted(data[90.0])
    best = {t: max(float(r["max_speed"]) for r in data[t].values()) for t in files}
    lines = []
    for a in angles:
        ar, _ = agility(np.radians(a))
        cells = []
        for t in (60.0, 90.0, 120.0):
            v = float(data[t][a]["max_speed"])
            cells.append(rf"\textbf{{{v:.3f}}}" if abs(v - best[t]) < 1e-9 else f"{v:.3f}")
        lines.append(f"    {a:.2f} & {ar:.1f} & " + " & ".join(cells) + r" \\")
    # Emit the complete tabular, not just the rows. Splicing rows in with
    # \input lands mid-alignment, where \bottomrule's \noalign is not at a row
    # boundary and TeX raises "Misplaced \noalign".
    head = [r"\begin{tabular}{r r ccc}", r"  \toprule",
            r"  & & \multicolumn{3}{c}{turn angle $\theta$} \\",
            r"  \cmidrule(lr){3-5}",
            r"  $t$ (\si{\degree}) & $\alpha_\phi$ & $60^\circ$ & $90^\circ$ & $120^\circ$ \\",
            r"  \midrule"]
    tail = [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(head + lines + tail) + "\n"


if __name__ == "__main__":
    lo, hi = feasible_range()
    angles = np.linspace(lo, hi, 7)
    figure_morphologies(angles)
    figure_courses()
    (OUT.parent / "results_table.tex").write_text(results_table())
    print(f"feasible half-angle range: {np.degrees(lo):.2f}-{np.degrees(hi):.2f} deg")
    print("wrote", OUT / "morphologies.pdf", OUT / "courses.pdf",
          OUT.parent / "results_table.tex", sep="\n      ")

#!/usr/bin/env python3
"""Generate the figures and the results table for morphology_gate_racing.tex.

Everything is derived from the repository -- geometry from the decoder and the
plant interface, speeds from the CSVs in docs/data/ -- so the report cannot
drift from the code or the measurements.

    uv run --no-sync python Reports/make_figures.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.plant import RotorGeometry, moment_allocation
from ariel.simulation.drone.gate_metrics import crossing_report, n_passed
from ariel.simulation.drone.controllers.trajectory_generation.bspline_gate_trajectory import (
    BSplineGateTrajectory,
)
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


def roll_torque(half_angle: float) -> float:
    """Raw roll torque authority at full thrust, N m -- *not* inertia-normalised."""
    g = genome(half_angle)
    props = blueprint_to_propellers(
        spherical_angular_to_blueprint(g, propsize=PROP_SIZE), convention="z_up")
    cfg = DroneConfiguration(props, payload_mass=PAYLOAD)
    kf, km = cfg.propellers[0]["constants"]
    geom = RotorGeometry(
        positions=np.array([p["loc"] for p in props], dtype=float),
        axes=np.array([p["dir"][:3] for p in props], dtype=float),
        spins=np.array([1.0 if p["dir"][3] == "ccw" else -1.0 for p in props]))
    B_M = moment_allocation(geom, torque_to_thrust_ratio=km / kf)
    return float(np.linalg.norm(B_M[0]) * kf * props[0]["wmax"] ** 2)


def feasible_range(arm_length: float = ARM_L) -> tuple[float, float]:
    c = (1.0 + TOL) * PROP_R / arm_length
    return float(np.arcsin(c)), float(np.pi / 2 - np.arcsin(c))


# ---------------------------------------------------------------- figure 1
def figure_morphologies(angles: np.ndarray, ncols: int = 3) -> None:
    """One panel per airframe, at most `ncols` per row so the titles stay legible."""
    nrows = int(np.ceil(len(angles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(angles):]:
        ax.axis("off")
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
        ax.set_xlim(-lim, lim)
        # Body frame is NED as well: y right, z down, so the top-down view puts
        # y downward on the page. Drawn the other way the airframes would be
        # mirrored relative to the courses they fly.
        ax.set_ylim(lim, -lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cbd5e0")
        ax.set_title(rf"$t={np.degrees(t):.2f}^\circ$" "\n"
                     rf"$\alpha_\phi={ar:.1f}$,  $\alpha_\theta={ap:.1f}$", fontsize=11)
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
        # NED: z is down, so a true top-down view (looking along +z) has y
        # running DOWN the page when x runs right. Plotting y upward would
        # be the view from below, and would mirror the weave.
        ax.invert_yaxis()
        ax.set_ylabel("y (m)", fontsize=8)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.01)
        # R (the three-gate circle radius) was dropped from this title on
        # 2026-09-24: the drone does not fly that circle.
        ax.set_title(rf"$\theta={turn:.0f}^\circ$:  "
                     rf"spacing ${c.spacing:.2f}$ m,  amplitude $\pm{c.amplitude:.2f}$ m",
                     fontsize=8.5, loc="left")
    axes[-1].set_xlabel("x (m)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "courses.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2b
def _u_label(u: float) -> str:
    """Waypoint parameter as an exact label.

    The interior waypoints land on integer knots; the second and second-to-last
    land on thirds (verified to 1e-10), so a decimal reads as an approximation
    of something that is in fact exact.
    """
    f = Fraction(u).limit_denominator(3)
    if f.denominator == 1:
        return f"{f.numerator}"
    whole, rem = divmod(f.numerator, f.denominator)
    frac = rf"\frac{{{rem}}}{{{f.denominator}}}"
    return f"${frac}$" if whole == 0 else f"${whole}{frac}$"


DEMAND_SPEED = 6.0   # nominal speed at which the Courses section quotes demand


def course_demand_vals(turns=(60.0, 90.0, 120.0)) -> dict[str, str]:
    """The Courses section's length and demand figures, as macros.

    Demand is measured on the reference (``SlalomCourse.reference_demand``),
    not taken as ``v**2`` over the three-gate circle radius, which is about
    1.5x too low at a typical gate (corrected 2026-09-24).
    """
    names = {60.0: "Sixty", 90.0: "Ninety", 120.0: "OneTwenty"}
    courses = {t: slalom_gates(t, n_gates=15) for t in turns}
    dem = {t: c.reference_demand(DEMAND_SPEED) for t, c in courses.items()}
    last_excess = [100.0 * (d.a_lat_peak / d.a_lat_median - 1.0) for d in dem.values()]
    arcs = [d.arc_length for d in dem.values()]
    # The same turn angles at fixed longitudinal spacing, for the length
    # confound the fixed leg avoids. 2.0 m is the spacing the original
    # comparison used.
    spaced = [slalom_gates(t, n_gates=15, leg=2.0 / np.cos(np.radians(t) / 2.0)).path_length
              for t in (turns[0], turns[-1])]
    return {
        "DemandSpeed": f"{DEMAND_SPEED:.0f}",
        **{f"Dem{names[t]}": f"{d.a_lat_median:.0f}" for t, d in dem.items()},
        "DemLastLo": f"{min(last_excess):.0f}", "DemLastHi": f"{max(last_excess):.0f}",
        "CoursePoly": f"{courses[turns[0]].path_length:.1f}",
        "CourseArcLo": f"{min(arcs):.1f}", "CourseArcHi": f"{max(arcs):.1f}",
        "SpacingPolyLo": f"{spaced[0]:.1f}", "SpacingPolyHi": f"{spaced[1]:.1f}",
    }


def spline_gearing(turn: float = 90.0) -> dict[str, float]:
    """Metres of path per unit of `u`, per span. No plotting.

    Split out of `figure_spline_parameter` so `numbers_tex` can quote these
    without rendering a figure -- the numbers test regenerates numbers.tex and
    should not have to draw.
    """
    co = slalom_gates(turn, n_gates=15)
    tr = BSplineGateTrajectory(co.trajectory_config())
    tr.fit_offsets_to_gates()
    s = tr.spline
    u = np.linspace(s.u_min, s.u_max, 40001)
    P = np.array([s.position(x) for x in u])
    g = np.linalg.norm(np.gradient(P, u, axis=0), axis=1)
    first = float(g[u <= 1].mean())
    interior = float(g[(u > 1) & (u < 13)].mean())
    last = float(g[u >= 13].mean())
    return {"first": first, "interior": interior, "last": last,
            "ratio": first / interior}


def figure_spline_parameter(turn: float = 90.0) -> dict[str, float]:
    """Why the spline parameter is not arc length, and what it costs at the gates.

    The clamped knot vector repeats the end knots degree+1 times, so the first
    and last spans each carry TWO waypoints while every interior span carries
    one. At a constant du/dt that doubles the reference speed over those two
    spans and starves the rest. Returns the measured per-span gearing so the
    caption cannot drift from the picture.
    """
    co = slalom_gates(turn, n_gates=15)
    tr = BSplineGateTrajectory(co.trajectory_config())
    tr.fit_offsets_to_gates()
    s = tr.spline

    u = np.linspace(s.u_min, s.u_max, 40001)
    P = np.array([s.position(x) for x in u])
    g = np.linalg.norm(np.gradient(P, u, axis=0), axis=1)
    wp_u = np.array([u[np.argmin(np.linalg.norm(P - w, axis=1))] for w in co.path_pos])

    fig = plt.figure(figsize=(7.0, 4.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.16)

    for col, (ua, ub, tag, tone) in enumerate((
            (0.0, 2.0, "first two spans", BAD),
            (6.0, 8.0, "two interior spans", GOOD))):
        ax = fig.add_subplot(grid[0, col])
        m = (u >= ua) & (u <= ub)
        arc = float(np.linalg.norm(np.diff(P[m], axis=0), axis=1).sum())
        ax.plot(P[m, 0], P[m, 1], "-", color=INK, lw=1.0, zorder=2)
        # Twelve dots per span, not ten: the second waypoint sits at u=1/3
        # exactly, so a step of 0.1 straddles it and the marker looks
        # misaligned. Twelfths land on every waypoint in both panels.
        ud = np.arange(ua, ub + 1e-9, 1.0 / 12.0)
        D = np.array([s.position(x) for x in ud])
        ax.plot(D[:, 0], D[:, 1], "o", color=ACCENT, ms=2.6, zorder=3)
        sel = (wp_u >= ua - 1e-9) & (wp_u <= ub + 1e-9)
        W = co.path_pos[sel]
        ax.plot(W[:, 0], W[:, 1], "s", mfc="none", mec=tone, mew=1.2, ms=7, zorder=4)
        for wu, w in zip(wp_u[sel], W):
            # wp_u comes from a nearest-sample search; the knots are exact.
            # y is inverted, so a waypoint at y~0 sits at the TOP of the weave
            # on screen and its label must go up; one at the amplitude goes
            # down. A fixed offset lands on the curve at every other waypoint.
            dy = 9 if w[1] < 0.5 * co.amplitude * 2 else -13
            ax.annotate(f"$u$={_u_label(wu)}", (w[0], w[1]), fontsize=6.5,
                        color=tone, textcoords="offset points",
                        xytext=(0, dy), ha="center")
        x0 = P[m, 0][0]
        ax.set_xlim(x0 - 1.5, x0 - 1.5 + 8.6)
        ax.set_ylim(2.75, -1.05)          # NED: y runs down the page, as in Fig. 2
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x (m)", fontsize=8)
        if col == 0:
            ax.set_ylabel("y (m)", fontsize=8)
        # Kept short: two titles share one line at \textwidth and collide if
        # the descriptive tag goes here. The tag lives in the caption instead.
        ax.set_title(rf"({'ab'[col]})  $u\in[{ua:g},{ub:g}]$:  {arc:.2f} m,  "
                     rf"{len(W)} waypoints",
                     fontsize=8, loc="left", color=tone)

    ax = fig.add_subplot(grid[1, :])
    ax.axvspan(0, 1, color=BAD, alpha=0.10, zorder=0)
    ax.axvspan(13, 14, color=BAD, alpha=0.10, zorder=0)
    for k in range(15):
        ax.axvline(k, color="0.87", lw=0.5, zorder=0)
    ax.plot(u, g, "-", color=ACCENT, lw=1.2, zorder=3)
    ax.plot(wp_u, np.full_like(wp_u, 0.3), "^", ms=3.4, color=GOOD, zorder=4)
    ax.plot(wp_u[[1, 15]], [0.3, 0.3], "^", ms=3.4, color=BAD, zorder=5)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.0)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("spline parameter $u$   (triangles: waypoints; grey: knots)", fontsize=8)
    ax.set_ylabel(r"$|dP/du|$ (m)", fontsize=8)

    first = float(g[u <= 1].mean())
    interior = float(g[(u > 1) & (u < 13)].mean())
    last = float(g[u >= 13].mean())
    ax.set_title(rf"(c)  metres of path per unit of $u$:  {first:.2f} in each end "
                 rf"span against {interior:.2f} inside", fontsize=8, loc="left")

    fig.savefig(OUT / "spline_parameter.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"first": first, "interior": interior, "last": last,
            "ratio": first / interior}


# ---------------------------------------------------------------- figure 3
# Two result sets (2026-09-17). The report's Results section presents the REDUCED
# model -- the plant without the gyroscopic term, with hover-tangent yaw drag and a
# 75 rad/s controller motor floor -- restored from commit 8d3174b. Its last
# subsection presents the CORRECTED model, whose data keep the canonical names.
LOGS = REPO / "docs" / "data" / "flight_logs"          # corrected model
LOGS_REDUCED = LOGS / "reduced_model"
STARTUP_TIME = 3.0


def _cross_track(pos: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from each flown point to the nearest point on the reference.

    Nearest-point, not same-time: the interest is how far the drone strayed from
    the *path*, independent of whether it was early or late along it.
    """
    return np.array([np.min(np.linalg.norm(ref - p, axis=1)) for p in pos])


def flight_stats(log_dir: Path = LOGS_REDUCED) -> list[dict]:
    out = []
    for turn in (60, 90, 120):
        d = np.load(log_dir / f"slalom_{turn}deg.npz")
        pos, ref = d["pos"].astype(float), d["ref"].astype(float)
        rep = crossing_report(pos, d["gate_pos"].astype(float),
                              d["gate_yaw"].astype(float), float(d["gate_size"]))
        # Exclude the startup ramp. The drone spawns start_offset = 1.0 m behind
        # gate 0 while the spline begins AT gate 0, so t=0 contributes a 1.00 m
        # "error" that is the initial condition, not tracking -- it swamped the
        # maximum and read identically on all three courses.
        cruise = d["t"].astype(float) > STARTUP_TIME
        xt = _cross_track(pos[cruise], ref)
        out.append({
            "turn": turn, "d": d, "rep": rep, "pos": pos, "ref": ref,
            "passed": n_passed(rep), "n_gates": len(rep),
            "xt_mean": float(xt.mean()), "xt_max": float(xt.max()),
            "offset_max": max(c.offset for c in rep if c.crossed),
            "half_angle": float(d["half_angle_deg"]), "speed": float(d["speed"]),
        })
    return out


def figure_flights(stats: list[dict], name: str = "flights.pdf") -> None:
    """Reference vs flown, one row per course -- the discrepancy the tables hide."""
    # Not sharex: the courses differ in length (a 60 deg slalom runs further in
    # x than a 120 deg one), and sharing the axis squeezes the sharpest course
    # into half the row.
    fig, axes = plt.subplots(len(stats), 1, figsize=(7.2, 7.4))
    for ax, st in zip(np.atleast_1d(axes).ravel(), stats):
        d, pos, ref = st["d"], st["pos"], st["ref"]
        half = float(d["gate_size"]) / 2.0
        for c, (g, yaw) in zip(st["rep"], zip(d["gate_pos"].astype(float),
                                              d["gate_yaw"].astype(float))):
            n = np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2)]) * half
            ax.plot([g[0] - n[0], g[0] + n[0]], [g[1] - n[1], g[1] + n[1]], "-",
                    color=(GOOD if c.passed else BAD), lw=2.6,
                    solid_capstyle="butt", zorder=2)
        ax.plot(ref[:, 0], ref[:, 1], "-", color="#a0aec0", lw=1.6, zorder=1,
                label="B-spline reference")
        ax.plot(pos[:, 0], pos[:, 1], "-", color=ACCENT, lw=1.4, zorder=3,
                label="flown")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, lw=0.4)
        # NED: z is down, so a true top-down view (looking along +z) has y
        # running DOWN the page when x runs right. Plotting y upward would
        # be the view from below, and would mirror the weave.
        ax.invert_yaxis()
        ax.set_ylabel("y (m)", fontsize=8)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.01)
        ax.set_title(
            rf"$\theta={st['turn']}^\circ$, $t={st['half_angle']:.2f}^\circ$, "
            rf"$v={st['speed']:.3f}$ m/s  —  {st['passed']}/{st['n_gates']} gates; "
            rf"cross-track mean {st['xt_mean']:.2f} m, max {st['xt_max']:.2f} m",
            fontsize=8.5, loc="left")
    axes[0].legend(fontsize=7, loc="upper center", ncol=2, framealpha=0.95,
                   bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- table
SPEED_TOL = 0.0625   # the bisection tolerance; differences below it are ties
TURNS = ((60.0, "Sixty"), (90.0, "Ninety"), (120.0, "OneTwenty"))
DATA = REPO / "docs" / "data"
TABLE_REDUCED = DATA / "tuned_strict_completion_reduced_model.csv"
TABLE_CORRECTED = DATA / "tuned_strict_completion.csv"
RANKINGS_REDUCED = DATA / "speed_semantics_rankings_reduced_model.csv"
RANKINGS_CORRECTED = DATA / "speed_semantics_rankings.csv"
COUNTERFACTUAL = DATA / "gyroscopic_counterfactual"


def load_speeds(path: Path = TABLE_REDUCED) -> dict[float, dict[float, dict]]:
    """Max completing speeds under the STRICT gate criterion.

    Earlier versions of this report read the sequential-criterion CSVs. That
    test checks lateral error in the horizontal plane only, so a drone below
    the gates still scored passes -- see docs/drone_morphology_gate_racing.md
    section 3.8. Those numbers were upper bounds, by a sixth at 60 degrees.
    """
    rows = list(csv.DictReader(open(path)))
    data: dict[float, dict[float, dict]] = {}
    for r in rows:
        data.setdefault(float(r["turn_deg"]), {})[
            round(float(r["half_angle_deg"]), 2)] = r
    return data


def results_table(path: Path = TABLE_REDUCED) -> str:
    data = load_speeds(path)

    angles = sorted(data[90.0])
    best = {t: max(float(r["max_speed"]) for r in data[t].values()) for t in data}
    lines = []
    for a in angles:
        ar, _ = agility(np.radians(a))
        cells = []
        for t in (60.0, 90.0, 120.0):
            v = float(data[t][a]["max_speed"])
            # Bold every value within the bisection tolerance of the best:
            # marking one of a tie as the winner is a claim the data does not
            # support.
            cells.append(rf"\textbf{{{v:.3f}}}" if best[t] - v < SPEED_TOL
                         else f"{v:.3f}")
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


def _speed_semantics_module():
    """Import docs/tools/speed_semantics.py, which is a script, not a package."""
    path = REPO / "docs" / "tools" / "speed_semantics.py"
    spec = importlib.util.spec_from_file_location("speed_semantics", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["speed_semantics"] = mod
    spec.loader.exec_module(mod)
    return mod


def speed_semantics_vals(best: dict[str, str]) -> dict[str, str]:
    """The nominal-vs-gate-speed decomposition, as macros.

    Driven by the per-course best speeds already read from the CSV, so the
    operating points these are measured at cannot drift from Table 1. The two
    factors are asserted to reproduce the ratio: if they ever stop multiplying
    out, the decomposition in the report is wrong and the build should fail
    rather than emit a plausible-looking number.
    """
    measure = _speed_semantics_module().measure
    vals: dict[str, str] = {}
    arcs = []
    for turn, name in ((60.0, "Sixty"), (90.0, "Ninety"), (120.0, "OneTwenty")):
        r = measure(turn, float(best[f"Best{name}"]))
        assert abs(r["startup_factor"] * r["gearing_factor"] - r["ratio"]) < 5e-3, (
            f"{turn} deg: startup x gearing does not reproduce the ratio")
        arcs.append(r["arc"])
        vals |= {
            f"GateSpeed{name}": f"{r['gate_med']:.2f}",
            f"GateLo{name}": f"{r['gate_lo']:.2f}",
            f"GateHi{name}": f"{r['gate_hi']:.2f}",
            f"MeanSpeed{name}": f"{r['mean']:.2f}",
            f"StartupFac{name}": f"{r['startup_factor']:.2f}",
            f"GearingFac{name}": f"{r['gearing_factor']:.2f}",
            f"SpeedRatio{name}": f"{r['ratio']:.2f}",
        }
        vals["Polyline"] = f"{r['polyline']:.1f}"
        vals["PeakRef"] = f"{max(float(vals.get('PeakRef', 0)), r['peak']):.0f}"
    vals |= {"ArcLo": f"{min(arcs):.1f}", "ArcHi": f"{max(arcs):.1f}"}
    return vals


def gate_speed_spread_vals(rankings: Path = RANKINGS_REDUCED,
                           table: Path = TABLE_REDUCED, prefix: str = "") -> dict[str, str]:
    """Table 1's within-column spreads, re-expressed as reference gate speed.

    Read from docs/data/speed_semantics_rankings.csv, written by
    docs/tools/speed_semantics.py. It takes about six minutes of spline sampling,
    too slow to redo on every build. The CSV's nominal column is checked against
    Table 1 so the two cannot drift apart. The report's claim about it -- the
    ranking is unchanged -- is asserted rather than assumed. (It also asserted
    that the spread grows with sharpness, until 2026-09-16; see below.)
    """
    path = rankings
    rows = list(csv.DictReader(open(path)))
    data = load_speeds(table)

    def order(xs: list[float]) -> list[int]:
        return sorted(range(len(xs)), key=lambda i: (xs[i], i))

    vals: dict[str, str] = {}
    spreads = []
    for t, name in ((60.0, "Sixty"), (90.0, "Ninety"), (120.0, "OneTwenty")):
        ours = sorted((float(r["half_angle_deg"]), float(r["max_speed"]),
                       float(r["gate_speed_median"]))
                      for r in rows if float(r["turn_deg"]) == t)
        table1 = [float(r["max_speed"]) for _, r in sorted(data[t].items())]
        nominal = [v for _, v, _ in ours]
        gate = [g for _, _, g in ours]
        assert nominal == table1, (
            f"{t} deg: {path.name} is stale against Table 1 -- rerun speed_semantics.py")
        assert order(nominal) == order(gate), f"{t} deg: gate speed reorders Table 1"
        spreads.append(100 * (max(gate) - min(gate)) / max(gate))
        vals[f"{prefix}GateSpread{name}"] = f"{spreads[-1]:.1f}"
    # SUPERSEDED 2026-09-16: this asserted the spread grows with sharpness, a
    # claim of the report's item-9 draft sentence. On the plant with the
    # gyroscopic term it does not (nominal spreads 4.1 / 32.7 / 25.8%), so the
    # claim is withdrawn, not the check relaxed to keep it passing.
    return vals


def column_macros(speeds: dict[float, dict[float, float]], prefix: str = "") -> dict[str, str]:
    """Best, argmax (ties within SPEED_TOL), tie count, spread and end costs per course.

    `speeds` is {turn: {half-angle rounded to 0.01: max speed}}.
    """
    vals: dict[str, str] = {}
    for t, name in TURNS:
        col = speeds[t]
        b = max(col.values())
        ties = sorted(a for a, v in col.items() if b - v < SPEED_TOL)
        ang = sorted(col)
        # \text{--} rather than a bare "--": the report cites these inside math
        # mode ($\ArgmaxSixty^\circ$), where the en-dash ligature does not
        # apply and "--" sets as two minus signs.
        # A range claims every body between its ends ties too. When the tied
        # bodies are not adjacent, list them instead (corrected model, 90 deg:
        # 45.00 and 69.56 tie, 53.19 and 61.37 do not). Fixed 2026-09-17.
        idx = [ang.index(a) for a in ties]
        if len(ties) == 1:
            rng = f"{ties[0]:.2f}"
        elif idx == list(range(idx[0], idx[-1] + 1)):
            rng = rf"{ties[0]:.2f}\text{{--}}{ties[-1]:.2f}"
        else:
            rng = r"\text{, }".join(f"{a:.2f}" for a in ties)
        vals |= {
            f"{prefix}Best{name}": f"{b:.3f}",
            f"{prefix}Argmax{name}": rng,
            f"{prefix}NTies{name}": f"{len(ties)}",
            f"{prefix}Spread{name}": f"{100 * (b - min(col.values())) / b:.1f}",
            f"{prefix}Narrow{name}": f"{100 * (col[ang[0]] - b) / b:.1f}",
            f"{prefix}Wide{name}": f"{100 * (col[ang[-1]] - b) / b:.1f}",
        }
    return vals


def table_speeds(path: Path) -> dict[float, dict[float, float]]:
    return {t: {a: float(r["max_speed"]) for a, r in col.items()}
            for t, col in load_speeds(path).items()}


def counterfactual_speeds(variant: str) -> dict[float, dict[float, float]]:
    """{turn: {half-angle: max speed}} from docs/data/gyroscopic_counterfactual/<variant>.json."""
    import json  # noqa: PLC0415
    cells = json.loads((COUNTERFACTUAL / f"{variant}.json").read_text())
    out: dict[float, dict[float, float]] = {}
    for key, cell in cells.items():
        if key.startswith("_"):
            continue
        turn, half = key.split("@")
        out.setdefault(float(turn), {})[round(float(half), 2)] = float(cell["max_speed"])
    return out


def flight_macros(stats: list[dict], prefix: str = "") -> dict[str, str]:
    return {
        **{f"{prefix}XTrack{n}": f"{st['xt_max']:.2f}"
           for (_, n), st in zip(TURNS, stats)},
        f"{prefix}XTrackMeanLo": f"{min(st['xt_mean'] for st in stats):.2f}",
        f"{prefix}XTrackMeanHi": f"{max(st['xt_mean'] for st in stats):.2f}",
        f"{prefix}XTrackMax": f"{max(st['xt_max'] for st in stats):.2f}",
    }


def _tool_functions(name: str) -> dict:
    """The definitions in docs/tools/<name>.py, without running its simulation.

    The investigation tools are scripts: after their `summarise` branch they bind
    the sweep's CLI and start simulating. Only the part before that branch -- the
    imports, constants and the pre-registered `score` function -- is executed, so
    the report states exactly the verdicts the tool computes.
    """
    src = (REPO / "docs" / "tools" / f"{name}.py").read_text()
    head = src[:src.index('if __name__ == "__main__" and len(sys.argv) > 1')]
    ns: dict = {"__file__": str(REPO / "docs" / "tools" / f"{name}.py"), "__name__": name}
    exec(compile(head, ns["__file__"], "exec"), ns)
    return ns


def gyroscopic_mechanism_vals() -> dict[str, str]:
    """Why the corrected model reverses the ranking (2026-09-17), as macros.

    Geometry: the Euler-equation gyroscopic coefficients across the family.
    Tests: docs/tools/gyroscopic_axis_attribution.py and gyroscopic_yaw_path.py,
    both pre-registered. Every verdict the report states is asserted here, so the
    prose cannot outlive the data behind it.
    """
    lo, hi = feasible_range()
    coeffs = []
    for t in np.linspace(lo, hi, 7):
        props = blueprint_to_propellers(
            spherical_angular_to_blueprint(genome(float(t)), propsize=PROP_SIZE), convention="ned")
        Ixx, Iyy, Izz = np.diag(DroneConfiguration(props, payload_mass=PAYLOAD).inertia_matrix)
        coeffs.append(((Iyy - Izz) / Ixx, (Izz - Ixx) / Iyy, (Ixx - Iyy) / Izz, Ixx, Iyy, Izz))
    c = np.array(coeffs)
    izz = c[:, 5]
    assert izz.max() - izz.min() < 1e-6 * izz.max(), "Izz is supposed to be the same for every frame"
    assert np.all(np.abs(izz - (c[:, 3] + c[:, 4])) < 0.02 * izz), "Izz ~ Ixx + Iyy (planar body)"
    rp = np.abs(c[:, :2])

    ax_tool = _tool_functions("gyroscopic_axis_attribution")
    runs = {k: json.loads((DATA / "gyroscopic_axis_attribution" / f"{k}.json").read_text())
            for k in ax_tool["CONDITIONS"]}
    ax = ax_tool["score"](runs)
    assert (ax["H-yaw"], ax["H-pitch"], ax["H-interaction"]) == ("SUPPORTED", "REFUTED", "REFUTED"), ax
    assert max(l["fraction"] for l in ax["axes"]["x"]["loss"]) <= 0.05, "roll reproduces none of the loss"
    # SUPERSEDED 2026-09-17: this asserted "narrow frame fastest at 90 deg without
    # coupling" by a strict max, but 8.406 vs 8.367 is inside SPEED_TOL -- a tie --
    # and at 120 deg the X quad is ahead. What the report now states is asserted.
    none = {(turn, h): runs["none"][f"{turn}@{h:.2f}"]["max_speed"]
            for turn in (90, 120) for h in (20.44, 45.00, 69.56)}
    assert abs(none[(90, 20.44)] - none[(90, 45.00)]) < SPEED_TOL, "narrow and X quad tie at 90 deg"
    assert none[(120, 45.00)] - none[(120, 20.44)] >= SPEED_TOL, "X quad ahead of narrow at 120 deg"
    x_gain = next(g for g in ax["axes"]["x"]["gain"] if g["turn"] == 120.0)["fraction"]
    assert 0.3 < x_gain < 0.6, "roll alone reproduces part of the wide frame's gain"

    yaw_tool = _tool_functions("gyroscopic_yaw_path")
    yp = yaw_tool["score"](DATA / "gyroscopic_yaw_path")
    assert yp["A"] == "A-plant", yp["A"]
    lim = yp["limits"]
    for turn in ("90", "120"):   # controller-only has the mirror effect
        assert lim["Zc"][f"{turn}@20.44"]["max_speed"] > lim["N"][f"{turn}@20.44"]["max_speed"]
        assert lim["Zc"][f"{turn}@69.56"]["max_speed"] < lim["N"][f"{turn}@69.56"]["max_speed"]
    assert yp["B-heading"]["verdict"] == "REFUTED" and yp["B-crab"]["verdict"] == "REFUTED"
    # Exploratory (logged, not pre-registered): 3-D sideslip at the gates, N vs Z.
    beta = {(k, c): yp["paths"][k][c]["beta_gates_deg"]
            for k in ("90@20.44", "120@20.44", "90@69.56", "120@69.56") for c in ("N", "Z")}
    for turn in ("90", "120"):   # the narrow frame's grows more than the wide frame's
        assert (beta[(f"{turn}@20.44", "Z")] / beta[(f"{turn}@20.44", "N")]
                > 2 * beta[(f"{turn}@69.56", "Z")] / beta[(f"{turn}@69.56", "N")])

    z, y = ax["axes"]["z"], ax["axes"]["y"]
    pct = lambda f: f"{100 * f:.0f}"  # noqa: E731
    return {
        "EulerRPMin": f"{rp.min():.2f}", "EulerRPMax": f"{rp.max():.2f}",
        "EulerYawNarrow": f"{c[0, 2]:.2f}", "EulerYawWide": f"{c[-1, 2]:.2f}",
        "GyAxisZLossNinety": pct(z["loss"][0]["fraction"]),
        "GyAxisZLossOneTwenty": pct(z["loss"][1]["fraction"]),
        "GyAxisZGainOneTwenty": pct(next(g for g in z["gain"] if g["turn"] == 120.0)["fraction"]),
        "GyAxisYLossMax": pct(max(l["fraction"] for l in y["loss"])),
        "GyPlantLossNinety": pct(yp["a"]["Zp"]["coverage"][0]["fraction"]),
        "GyPlantLossOneTwenty": pct(yp["a"]["Zp"]["coverage"][1]["fraction"]),
        "GyHeadingNarrowNinety": f"{yp['B-heading']['ratios']['90@20.44']:.2f}",
        "GyHeadingNarrowOneTwenty": f"{yp['B-heading']['ratios']['120@20.44']:.2f}",
        "GyCrabNarrowNinety": f"{yp['B-crab']['ratios']['90@20.44']:.2f}",
        "GyCrabNarrowOneTwenty": f"{yp['B-crab']['ratios']['120@20.44']:.2f}",
        "GyNoneNarrowNinety": f"{none[(90, 20.44)]:.3f}", "GyNoneXNinety": f"{none[(90, 45.00)]:.3f}",
        "GyNoneNarrowOneTwenty": f"{none[(120, 20.44)]:.3f}",
        "GyNoneXOneTwenty": f"{none[(120, 45.00)]:.3f}",
        "GyAxisXGainOneTwenty": pct(x_gain),
        **{f"GyBeta{'Narrow' if k.endswith('20.44') else 'Wide'}"
           f"{'Ninety' if k.startswith('90') else 'OneTwenty'}{c}": f"{v:.1f}"
           for (k, c), v in beta.items()},
    }


def numbers_tex(angles: np.ndarray, gearing: dict[str, float] | None = None) -> str:
    """Every number quoted inline in the report, as macros.

    The prose used to carry these as literals, and one of them drifted: the raw
    torque range was quoted as 1.735-3.896 N m, which is the span over
    t = 24-66 deg -- an earlier sweep's range -- not over the feasible family
    the report actually defines. Generating them removes the failure mode.
    """
    if gearing is None:
        gearing = spline_gearing()
    lo, hi = float(np.degrees(angles[0])), float(np.degrees(angles[-1]))
    roll = np.array([agility(float(t))[0] for t in angles])
    pitch = np.array([agility(float(t))[1] for t in angles])
    tor = np.array([roll_torque(float(t)) for t in angles])
    # Unprefixed macros describe the reduced model (the Results section as
    # restored); Corr* the corrected model (its last subsection).
    speed_vals = column_macros(table_speeds(TABLE_REDUCED))
    vals = {
        "FeasLo": f"{lo:.2f}", "FeasHi": f"{hi:.2f}",
        "NPoints": f"{len(angles)}",
        "RollNarrow": f"{roll[0]:.1f}", "RollWide": f"{roll[-1]:.1f}",
        "RollRatio": f"{roll[0] / roll[-1]:.2f}",
        "PitchNarrow": f"{pitch[0]:.1f}",
        "TorqueLo": f"{tor.min():.3f}", "TorqueHi": f"{tor.max():.3f}",
        # Figure 3's cross-track, maximum per course and the mean range. The mean
        # was a literal (0.28-0.33 m) until 2026-09-16.
        **flight_macros(flight_stats(LOGS_REDUCED)),
        "PubBestSixty": "12.172", "PubBestNinety": "8.578",
        "PubBestOneTwenty": "7.078",
        # SUPERSEDED 2026-09-17: PrevBest* (10.164 / 8.367 / 6.648) held the
        # reduced-model bests while Best* held the corrected ones. With Best*
        # bound to the reduced model again they were identical, so they are gone.
        **speed_vals,
        **speed_semantics_vals(speed_vals),
        **gate_speed_spread_vals(RANKINGS_REDUCED, TABLE_REDUCED),
        # Corrected model (gyroscopic term, k_m W^2 rotor drag, motor floor = w_min).
        **column_macros(table_speeds(TABLE_CORRECTED), "Corr"),
        **gate_speed_spread_vals(RANKINGS_CORRECTED, TABLE_CORRECTED, "Corr"),
        **flight_macros(flight_stats(LOGS), "Corr"),
        # Counterfactual attribution (docs/data/gyroscopic_counterfactual.md):
        # quadratic yaw drag alone, and the corrected plant with a yaw-last mixer.
        **column_macros(counterfactual_speeds("current_quadyaw"), "CfQuadYaw"),
        **column_macros(counterfactual_speeds("physical_quadyaw_yawlast"), "CfYawLast"),
        # Why the corrected model reverses the ranking (pre-registered tests).
        **gyroscopic_mechanism_vals(),
        "GearEnd": f"{gearing['first']:.2f}",
        "GearInterior": f"{gearing['interior']:.2f}",
        "GearRatio": f"{gearing['ratio']:.2f}",
        **course_demand_vals(),
        # Hub to rotor tip: arm length plus propeller radius. The gate test
        # scores the centre of mass only, so this much airframe goes unchecked.
        "RotorTipReach": f"{ARM_L + PROP_R:.2f}",
    }
    body = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items())
    return "% Generated by Reports/make_figures.py -- do not edit.\n" + body


if __name__ == "__main__":
    lo, hi = feasible_range()
    angles = np.linspace(lo, hi, 7)
    figure_morphologies(angles)
    figure_courses()
    gearing = figure_spline_parameter()
    figure_flights(flight_stats(LOGS_REDUCED), "flights.pdf")
    figure_flights(flight_stats(LOGS), "flights_corrected.pdf")
    (OUT.parent / "results_table.tex").write_text(results_table(TABLE_REDUCED))
    (OUT.parent / "results_table_corrected.tex").write_text(results_table(TABLE_CORRECTED))
    (OUT.parent / "numbers.tex").write_text(numbers_tex(angles, gearing))
    print(f"feasible half-angle range: {np.degrees(lo):.2f}-{np.degrees(hi):.2f} deg")
    print("wrote", OUT / "morphologies.pdf", OUT / "courses.pdf",
          OUT / "spline_parameter.pdf", OUT / "flights.pdf", OUT / "flights_corrected.pdf",
          OUT.parent / "results_table.tex", OUT.parent / "results_table_corrected.tex",
          OUT.parent / "numbers.tex", sep="\n      ")

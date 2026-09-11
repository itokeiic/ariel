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
        ax.set_title(rf"$\theta={turn:.0f}^\circ$:  $R={c.radius:.2f}$ m,  "
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
LOGS = REPO / "docs" / "data" / "flight_logs"
STARTUP_TIME = 3.0


def _cross_track(pos: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from each flown point to the nearest point on the reference.

    Nearest-point, not same-time: the interest is how far the drone strayed from
    the *path*, independent of whether it was early or late along it.
    """
    return np.array([np.min(np.linalg.norm(ref - p, axis=1)) for p in pos])


def flight_stats() -> list[dict]:
    out = []
    for turn in (60, 90, 120):
        d = np.load(LOGS / f"slalom_{turn}deg.npz")
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


def figure_flights(stats: list[dict]) -> None:
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
    fig.savefig(OUT / "flights.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- table
SPEED_TOL = 0.0625   # the bisection tolerance; differences below it are ties


def load_speeds() -> dict[float, dict[float, dict]]:
    """Max completing speeds under the STRICT gate criterion.

    Earlier versions of this report read the sequential-criterion CSVs. That
    test checks lateral error in the horizontal plane only, so a drone below
    the gates still scored passes -- see docs/drone_morphology_gate_racing.md
    section 3.8. Those numbers were upper bounds, by a sixth at 60 degrees.
    """
    rows = list(csv.DictReader(
        open(REPO / "docs" / "data" / "tuned_strict_completion.csv")))
    data: dict[float, dict[float, dict]] = {}
    for r in rows:
        data.setdefault(float(r["turn_deg"]), {})[
            round(float(r["half_angle_deg"]), 2)] = r
    return data


def results_table() -> str:
    data = load_speeds()

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
            f"MeanSpeed{name}": f"{r['mean']:.2f}",
            f"StartupFac{name}": f"{r['startup_factor']:.2f}",
            f"GearingFac{name}": f"{r['gearing_factor']:.2f}",
            f"SpeedRatio{name}": f"{r['ratio']:.2f}",
        }
        vals["Polyline"] = f"{r['polyline']:.1f}"
        vals["PeakRef"] = f"{max(float(vals.get('PeakRef', 0)), r['peak']):.0f}"
    vals |= {"ArcLo": f"{min(arcs):.1f}", "ArcHi": f"{max(arcs):.1f}"}
    return vals


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
    data = load_speeds()
    def _col(t):
        col = {a: float(r["max_speed"]) for a, r in data[t].items()}
        b = max(col.values())
        ties = sorted(a for a, v in col.items() if b - v < SPEED_TOL)
        ang = sorted(col)
        return col, b, ties, ang

    speed_vals = {}
    for t, name in ((60.0, "Sixty"), (90.0, "Ninety"), (120.0, "OneTwenty")):
        col, b, ties, ang = _col(t)
        # \text{--} rather than a bare "--": the report cites these inside math
        # mode ($\ArgmaxSixty^\circ$), where the en-dash ligature does not
        # apply and "--" sets as two minus signs.
        rng = (f"{ties[0]:.2f}" if len(ties) == 1
               else rf"{ties[0]:.2f}\text{{--}}{ties[-1]:.2f}")
        speed_vals |= {
            f"Best{name}": f"{b:.3f}",
            f"Argmax{name}": rng,
            f"NTies{name}": f"{len(ties)}",
            f"Spread{name}": f"{100 * (b - min(col.values())) / b:.1f}",
            f"Narrow{name}": f"{100 * (col[ang[0]] - b) / b:.1f}",
            f"Wide{name}": f"{100 * (col[ang[-1]] - b) / b:.1f}",
        }

    vals = {
        "FeasLo": f"{lo:.2f}", "FeasHi": f"{hi:.2f}",
        "NPoints": f"{len(angles)}",
        "RollNarrow": f"{roll[0]:.1f}", "RollWide": f"{roll[-1]:.1f}",
        "RollRatio": f"{roll[0] / roll[-1]:.2f}",
        "PitchNarrow": f"{pitch[0]:.1f}",
        "TorqueLo": f"{tor.min():.3f}", "TorqueHi": f"{tor.max():.3f}",
        # Superseded sequential-criterion figures, quoted only in the note that
        # explains why they changed.
        **{f"XTrack{n}": f"{st['xt_max']:.2f}"
           for n, st in zip(("Sixty", "Ninety", "OneTwenty"), flight_stats())},
        "PubBestSixty": "12.172", "PubBestNinety": "8.578",
        "PubBestOneTwenty": "7.078",
        **speed_vals,
        **speed_semantics_vals(speed_vals),
        "GearEnd": f"{gearing['first']:.2f}",
        "GearInterior": f"{gearing['interior']:.2f}",
        "GearRatio": f"{gearing['ratio']:.2f}",
    }
    body = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items())
    return "% Generated by Reports/make_figures.py -- do not edit.\n" + body


if __name__ == "__main__":
    lo, hi = feasible_range()
    angles = np.linspace(lo, hi, 7)
    figure_morphologies(angles)
    figure_courses()
    gearing = figure_spline_parameter()
    figure_flights(flight_stats())
    (OUT.parent / "results_table.tex").write_text(results_table())
    (OUT.parent / "numbers.tex").write_text(numbers_tex(angles, gearing))
    print(f"feasible half-angle range: {np.degrees(lo):.2f}-{np.degrees(hi):.2f} deg")
    print("wrote", OUT / "morphologies.pdf", OUT / "courses.pdf",
          OUT / "spline_parameter.pdf", OUT / "flights.pdf",
          OUT.parent / "results_table.tex", OUT.parent / "numbers.tex",
          sep="\n      ")

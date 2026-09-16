#!/usr/bin/env python3
"""What the `--speed` knob actually means, measured on the reference itself.

    uv run --no-sync python docs/tools/speed_semantics.py

`--speed` sets `total_time = startup + path_length / speed`, where
`path_length` is the POLYLINE through the waypoints. The speed a gate is
actually crossed at is lower, by two independent factors that multiply:

1. **Startup accounting** -- dominant at gentle turns. The spline is traversed
   over the WHOLE `total_time`, ramp included, not over the `path/speed` cruise
   window the formula suggests. Mean speed along the path is therefore
   `arc / total_time`, short of `speed` by `(1 + startup*speed/path)*(path/arc)`.
   The second term is the interpolating spline bowing outside the polyline
   (arc is slightly LONGER, not shorter), which is small and conservative.
2. **Non-uniform parameterisation** -- dominant at sharp turns. The spline is
   parameterised uniformly in `u`, not by arc length. World speed is
   `|dP/du| * du/dt`, and `|dP/du|` runs ~1.5x its mean near the clamped ends
   against ~0.85x through the interior, so at a constant `du/dt` the gates are
   geared low while the final span -- which lies past the last gate -- sprints.

SUPERSEDED (2026-08-19): this file previously attributed the whole shortfall to
factor 2 and did not account for factor 1, which is the larger of the two at 60
and 90 degrees (1.75 of the 1.84x total at 60 degrees, where factor 2 is only
1.05). The measured gate speeds were correct; the explanation was not. The
`startup_factor` / `gearing_factor` columns below decompose it, and their
product reproduces the reported ratio.

Everything here is a property of the reference trajectory alone -- no rollout,
so it is cheap and cannot drift with the plant or the controller.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ariel.simulation.drone.controllers.trajectory_generation.bspline_gate_trajectory import (
    BSplineGateTrajectory,
)
from ariel.simulation.drone.gate_metrics import crossing_report
from ariel.simulation.tasks.slalom_course import slalom_gates

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "data" / "speed_semantics.md"
STARTUP = 3.0
def cases() -> tuple[tuple[float, float], ...]:
    """The strict-criterion best operating point per course, read from Table 1.

    SUPERSEDED 2026-09-16: these were hard-coded as ((60, 10.164), (90, 8.367),
    (120, 6.648)), which went stale when the plant gained the gyroscopic term
    and Table 1 was regenerated. Read from the CSV so they cannot drift again.
    """
    rows = list(csv.DictReader(open(TABLE1)))
    return tuple((turn, max(float(r["max_speed"]) for r in rows
                            if float(r["turn_deg"]) == turn))
                 for turn in sorted({float(r["turn_deg"]) for r in rows}))
N_SAMPLES = 20001


def measure(turn: float, nominal: float) -> dict:
    course = slalom_gates(turn, n_gates=15)
    traj = BSplineGateTrajectory(course.trajectory_config())
    traj.fit_offsets_to_gates()
    traj.startup_time = STARTUP
    traj.total_time = course.traversal_time(nominal, startup_time=STARTUP)

    u = np.linspace(traj.spline.u_min, traj.spline.u_max, N_SAMPLES)
    pts = np.array([traj.spline.position(x) for x in u])
    arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    # Reference speed at the instant the reference crosses each gate plane --
    # the same detector the flights are scored with, so the two are comparable.
    # Matching by nearest point instead picks t~0 for gate 0, where the ramp has
    # not started and the speed is 0.
    ts = np.linspace(0.0, traj.total_time, N_SAMPLES)
    traj_pts = np.array([traj.evaluate(t)[0] for t in ts])
    speeds = np.array([np.linalg.norm(traj.evaluate(t)[1]) for t in ts])
    rep = crossing_report(traj_pts, course.gate_pos,
                          course.path_yaw[:len(course.gate_pos)], course.gate_size)
    # Gate 0 is excluded: the spline's first control point IS gate 0's centre,
    # so the reference starts exactly on that plane and never "crosses" it.
    # Flights spawn behind it and are unaffected -- see crossing_report.
    scored = rep[1:]
    at_gates = np.array([speeds[c.step] for c in scored if c.crossed])
    n_missed = sum(1 for c in scored if not c.passed)

    cruise = ts > STARTUP
    # The shortfall factors into two independent causes that multiply. `mean` is
    # the speed the path is actually covered at; `startup_factor` is how far the
    # nominal figure sits above it (the ramp is inside total_time, and the arc is
    # longer than the polyline the nominal divides); `gearing_factor` is how far
    # the gates sit below that mean (uniform-in-u parameterisation).
    mean = arc / traj.total_time
    gate_med = float(np.median(at_gates))
    return {
        "turn": turn,
        "nominal": nominal,
        "polyline": float(course.path_length),
        "arc": arc,
        "total_time": float(traj.total_time),
        "mean": mean,
        "gate_lo": float(at_gates.min()),
        "gate_med": gate_med,
        "gate_hi": float(at_gates.max()),
        "peak": float(speeds[cruise].max()),
        "startup_factor": float(nominal / mean),
        "gearing_factor": float(mean / gate_med),
        "ratio": float(nominal / gate_med),
        "ref_missed": n_missed,
    }


def table(rows: list[dict]) -> str:
    """Markdown summary. Asserts the reference passes every gate it defines."""
    lines = [
        "| course | `--speed` | polyline | spline arc | ref speed at gates (min/median/max) | peak ref speed | nominal / median |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['turn']:.0f}° | {r['nominal']:.3f} m/s | {r['polyline']:.2f} m | "
            f"{r['arc']:.2f} m | {r['gate_lo']:.2f} / {r['gate_med']:.2f} / "
            f"{r['gate_hi']:.2f} m/s | {r['peak']:.1f} m/s | "
            f"{r['ratio']:.2f}x |")
    lines += [
        "",
        "Decomposition of that last column into its two independent causes:",
        "",
        "| course | total_time | mean = arc/total_time | startup accounting | parameterisation | product | reported |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['turn']:.0f}° | {r['total_time']:.3f} s | {r['mean']:.3f} m/s | "
            f"{r['startup_factor']:.2f}x | {r['gearing_factor']:.2f}x | "
            f"{r['startup_factor'] * r['gearing_factor']:.2f}x | {r['ratio']:.2f}x |")
    assert all(r["ref_missed"] == 0 for r in rows), (
        "the reference itself misses a gate -- the interpolation fit is broken")
    assert all(abs(r["startup_factor"] * r["gearing_factor"] - r["ratio"]) < 5e-3
               for r in rows), "the two factors must reproduce the reported ratio"
    return "\n".join(lines) + "\n"


TABLE1 = REPO / "docs" / "data" / "tuned_strict_completion.csv"
RANKINGS_OUT = REPO / "docs" / "data" / "speed_semantics_rankings.md"
# Read by Reports/make_figures.py, which checks its nominal column against TABLE1
# rather than re-running the ~6 minutes of spline sampling on every build.
RANKINGS_CSV = REPO / "docs" / "data" / "speed_semantics_rankings.csv"


def gate_speed_rankings(csv_path: Path = TABLE1) -> dict[float, dict]:
    """Does quoting reference gate speed instead of nominal speed change Table 1?

    For every body at its limiting nominal speed, the reference's median speed at
    the gates. Gate speed is not a fixed multiple of nominal speed -- the startup
    term (1 + startup*speed/path) grows with speed -- so rankings could in
    principle reorder, and spreads do change. Returns, per turn angle, the rows,
    whether the two rankings agree, and the spread in each unit.

    This is the reference's speed at the gates, not the drone's: the drone lags
    its reference.
    """
    rows = list(csv.DictReader(open(csv_path)))
    cache: dict[tuple[float, float], float] = {}
    out: dict[float, dict] = {}
    for turn in sorted({float(r["turn_deg"]) for r in rows}):
        col = sorted((float(r["half_angle_deg"]), float(r["max_speed"]))
                     for r in rows if float(r["turn_deg"]) == turn)
        body = []
        for t, v in col:
            if (turn, v) not in cache:
                cache[(turn, v)] = measure(turn, v)["gate_med"]
            body.append((t, v, cache[(turn, v)]))
        nominal = [v for _, v, _ in body]
        gate = [g for _, _, g in body]

        def order(xs: list[float]) -> list[int]:
            return sorted(range(len(xs)), key=lambda i: (xs[i], i))

        def spread(xs: list[float]) -> float:
            return 100.0 * (max(xs) - min(xs)) / max(xs)

        out[turn] = {"rows": body, "same_order": order(nominal) == order(gate),
                     "spread_nominal": spread(nominal), "spread_gate": spread(gate)}
    return out


def rankings_table(res: dict[float, dict]) -> str:
    lines = ["| turn | ranking by nominal = ranking by gate speed | spread, nominal | spread, gate speed |",
             "|---|---|---|---|"]
    for turn, r in res.items():
        lines.append(f"| {turn:.0f}° | {'yes' if r['same_order'] else '**no**'} | "
                     f"{r['spread_nominal']:.2f}% | {r['spread_gate']:.2f}% |")
    lines += ["", "Per body, at its limiting nominal speed (reference speed at the gates, not the drone's):", "",
              "| turn | t (deg) | nominal | gate median | ratio |", "|---|---|---|---|---|"]
    for turn, r in res.items():
        for t, v, g in r["rows"]:
            lines.append(f"| {turn:.0f}° | {t:.2f} | {v:.4f} | {g:.4f} | {v / g:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rows = [measure(t, v) for t, v in cases()]
    md = table(rows)
    OUT.write_text(md)
    print(md)
    print(f"wrote {OUT.relative_to(REPO)}")
    res = gate_speed_rankings()
    rk = rankings_table(res)
    RANKINGS_OUT.write_text(rk)
    print(rk)
    print(f"wrote {RANKINGS_OUT.relative_to(REPO)}")
    with open(RANKINGS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_deg", "half_angle_deg", "max_speed", "gate_speed_median"])
        for turn, r in res.items():
            for t, v, g in r["rows"]:
                w.writerow([turn, t, v, g])
    print(f"wrote {RANKINGS_CSV.relative_to(REPO)}")

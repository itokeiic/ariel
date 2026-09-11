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
# The strict-criterion best operating point per course (docs/data/tuned_strict_completion.csv).
CASES = ((60.0, 10.164), (90.0, 8.367), (120.0, 6.648))
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


if __name__ == "__main__":
    rows = [measure(t, v) for t, v in CASES]
    md = table(rows)
    OUT.write_text(md)
    print(md)
    print(f"wrote {OUT.relative_to(REPO)}")

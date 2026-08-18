#!/usr/bin/env python3
"""What the `--speed` knob actually means, measured on the reference itself.

    uv run --no-sync python docs/tools/speed_semantics.py

`--speed` sets `total_time = startup + path_length / speed`, where
`path_length` is the POLYLINE through the waypoints. Two things then separate
that number from the speed the drone flies through a gate:

1. the interpolating spline does not follow the polyline -- it must pass
   through every waypoint and bows outside the straight legs between them, so
   its arc length is slightly LONGER, not shorter; and
2. far more importantly, it is parameterised uniformly in `u`, not by arc
   length. World speed is `|dP/du| * du/dt`, and on a clamped spline `|dP/du|`
   is much larger near the ends than in the middle, so a constant `du/dt`
   starves the middle of the course -- where every gate is -- and sprints the
   final span.

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
    return {
        "turn": turn,
        "nominal": nominal,
        "polyline": float(course.path_length),
        "arc": arc,
        "gate_lo": float(at_gates.min()),
        "gate_med": float(np.median(at_gates)),
        "gate_hi": float(at_gates.max()),
        "peak": float(speeds[cruise].max()),
        "ratio": float(nominal / np.median(at_gates)),
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
    assert all(r["ref_missed"] == 0 for r in rows), (
        "the reference itself misses a gate -- the interpolation fit is broken")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rows = [measure(t, v) for t, v in CASES]
    md = table(rows)
    OUT.write_text(md)
    print(md)
    print(f"wrote {OUT.relative_to(REPO)}")

"""Slalom gate course with difficulty as a single explicit parameter.

The quintic-planner circuits used by the drone-morphology examples turn very
little — a median of 0.4 degrees between consecutive legs — so they ask almost
nothing of the airframe and every morphology scores the same. This module
builds a course whose difficulty is one interpretable number: the turn angle
demanded at each gate.

Parameterised by turn angle at **fixed leg length**, which matters. Holding the
longitudinal spacing fixed instead makes the knob non-monotonic: a sharper turn
also lengthens the legs, so the corner ends up *wider* (radius 4.00 / 2.31 /
2.00 / 2.31 m for 30/60/90/120 degrees). With the leg fixed, corner radius falls
monotonically with turn angle, which is what "harder" should mean.

Geometry, for a leg length ``s`` and turn angle ``theta``:

    x-spacing  d = s * cos(theta/2)
    amplitude  A = (s/2) * sin(theta/2)
    corner radius R = s / (2 * sin(theta/2))

Gates alternate to ``+A`` and ``-A``; the bisector at every gate is +x, so every
gate plane faces down-course and ``gate_yaw`` is zero throughout. That is what
the pass detector needs — ``GateChecker`` builds the gate normal as
``(cos yaw, sin yaw, 0)`` and counts a pass only when the drone crosses that
plane. (The ``SlalomGates`` preset in ``controllers/utils/gate_configs.py`` uses
a ``[1,0,-1,0]*pi/2`` yaw pattern that does not follow the path.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass
class SlalomCourse:
    """A slalom course plus the difficulty it actually produces.

    Attributes:
        gate_pos: gate centres, shape (n_gates, 3).
        gate_yaw: gate plane headings, shape (n_gates,) — all zero here.
        starting_pos: spawn, ``offset`` metres behind gate 0 along the course.
        gate_size: gate opening, m.
        turn_deg: requested turn angle between consecutive legs.
        leg: leg length, m.
        spacing: resulting longitudinal gate spacing, m.
        amplitude: resulting lateral offset, m.
        radius: corner radius, m — the difficulty that matters, since demanded
            lateral acceleration is ``speed**2 / radius``.
        path_length: total path length along the gate polyline, m.
        periodic: always False; a slalom is flown once, not looped.
    """

    gate_pos: Array
    gate_yaw: Array
    starting_pos: Array
    gate_size: float
    turn_deg: float
    leg: float
    spacing: float
    amplitude: float
    radius: float
    path_length: float
    periodic: bool = field(default=False)

    def measured_turn_angles(self) -> Array:
        """Actual angles between consecutive legs, degrees. For verification."""
        v = np.diff(self.gate_pos[:, :2], axis=0)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip(np.sum(v[:-1] * v[1:], axis=1), -1.0, 1.0)))

    def lateral_acceleration(self, speed: float) -> float:
        """Lateral acceleration this course demands at ``speed``, m/s^2."""
        return float(speed) ** 2 / self.radius

    def traversal_time(self, speed: float, startup_time: float = 3.0) -> float:
        """``total_time`` for BSplineGateTrajectory at a fixed nominal speed.

        The spline's loop phase covers the course in ``total_time -
        startup_time``; when that is <= 0 the implementation silently falls back
        to ``loop_speed = 1.0`` and produces a near-static reference the drone
        tracks beautifully while passing almost no gates. Callers get a hard
        error instead of that.
        """
        if speed <= 0.0:
            raise ValueError(f"speed must be > 0, got {speed}")
        total = startup_time + self.path_length / float(speed)
        if total <= startup_time:
            raise ValueError("degenerate traversal time")
        return total


def slalom_gates(
    turn_deg: float,
    *,
    leg: float = 2.4,
    n_gates: int = 15,
    gate_size: float = 1.0,
    z: float = -1.5,
    start_offset: float = 1.0,
) -> SlalomCourse:
    """Build a slalom course demanding ``turn_deg`` at every gate.

    Args:
        turn_deg: angle between consecutive legs, degrees. 0 is a straight line.
        leg: distance between consecutive gates, m. Held fixed so that corner
            radius falls monotonically with ``turn_deg``.
        n_gates: number of gates.
        gate_size: gate opening, m.
        z: course altitude (NED, so negative is up).
        start_offset: spawn distance behind gate 0, m.

    Raises:
        ValueError: if the turn angle is out of range, or if the resulting
            longitudinal spacing would not exceed ``gate_size`` — gate planes
            would then overlap and "passing a gate" stops being well defined.
    """
    if not 0.0 <= turn_deg < 180.0:
        raise ValueError(f"turn_deg must be in [0, 180), got {turn_deg}")
    if n_gates < 3:
        raise ValueError(f"need at least 3 gates to define a turn, got {n_gates}")

    th = np.radians(float(turn_deg))
    spacing = leg * np.cos(th / 2.0)
    amplitude = (leg / 2.0) * np.sin(th / 2.0)

    if spacing <= gate_size:
        raise ValueError(
            f"turn_deg={turn_deg} with leg={leg} gives gate spacing {spacing:.3f} m, "
            f"which does not exceed gate_size={gate_size} m: gate planes would "
            f"overlap and a pass would be ambiguous. Lengthen the leg or shrink "
            f"the gate."
        )

    x = np.arange(n_gates, dtype=float) * spacing
    y = amplitude * ((-1.0) ** np.arange(n_gates))
    gate_pos = np.column_stack([x, y, np.full(n_gates, float(z))])
    gate_yaw = np.zeros(n_gates, dtype=float)

    radius = leg / (2.0 * np.sin(th / 2.0)) if th > 1e-9 else float("inf")
    path_length = float(leg * (n_gates - 1))

    # Spawn behind gate 0 along the first leg, matching the fly-in convention
    # the gate examples use.
    first_leg = gate_pos[1, :2] - gate_pos[0, :2]
    first_leg = first_leg / np.linalg.norm(first_leg)
    start = gate_pos[0].copy()
    start[:2] -= start_offset * first_leg

    return SlalomCourse(
        gate_pos=gate_pos,
        gate_yaw=gate_yaw,
        starting_pos=start,
        gate_size=float(gate_size),
        turn_deg=float(turn_deg),
        leg=float(leg),
        spacing=float(spacing),
        amplitude=float(amplitude),
        radius=float(radius),
        path_length=path_length,
    )

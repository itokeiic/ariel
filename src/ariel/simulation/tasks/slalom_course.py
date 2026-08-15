"""Slalom gate course with difficulty as a single explicit parameter.

The quintic-planner circuits used by the drone-morphology examples turn very
little -- a median of 0.4 degrees between consecutive legs -- so they ask almost
nothing of the airframe and every morphology scores the same. This module
builds a course whose difficulty is one interpretable number: the turn angle
demanded at each gate.

Parameterised by turn angle at **fixed leg length**, which matters. Holding the
longitudinal spacing fixed instead makes the knob non-monotonic: a sharper turn
also lengthens the legs, so the corner ends up *wider* (radius 4.00 / 2.31 /
2.00 / 2.31 m for 30/60/90/120 degrees). With the leg fixed, corner radius falls
monotonically with turn angle, which is what "harder" should mean.

For a leg length ``s`` and turn angle ``theta``, consecutive legs run at
``+/- theta/2`` to the course axis, so:

    x-spacing  d = s * cos(theta/2)
    amplitude  A = (s/2) * sin(theta/2)
    corner radius R = s / (2 * sin(theta/2))

``gate_yaw`` is the bisector of the two adjacent legs, which is what the pass
detector needs: ``GateChecker`` builds the gate normal as
``(cos yaw, sin yaw, 0)`` and counts a pass only when the drone crosses that
plane. (The ``SlalomGates`` preset in ``controllers/utils/gate_configs.py`` uses
a ``[1,0,-1,0]*pi/2`` yaw pattern that does not follow the path.)

Two details exist because of measured artefacts:

* **Lead-out.** The trajectory clamps at its final waypoint, so a drone
  arriving at speed sails past a course that ends on the last scored gate --
  measured 2.1-2.7 m of overshoot, making the final gate unreachable for every
  morphology and capping the score for all of them. The spline therefore
  continues ``lead_out`` legs beyond the last scored gate, so the drone flies
  *through* the finish and decelerates afterwards.
* **Course variation.** Optimising against one fixed layout selects the body
  that suits that layout. ``seed`` jitters the per-corner turn angle so a set of
  courses shares a difficulty but not a shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass
class SlalomCourse:
    """A slalom course plus the difficulty it actually produces.

    Attributes:
        gate_pos: scored gate centres, shape (n_gates, 3).
        gate_yaw: scored gate plane headings, shape (n_gates,).
        path_pos: spline waypoints, shape (n_gates + lead_out, 3). Equal to
            ``gate_pos`` plus the lead-out; hand this to the trajectory and
            ``gate_pos`` to the pass detector.
        path_yaw: headings for ``path_pos``.
        starting_pos: spawn, ``start_offset`` metres behind gate 0.
        gate_size: gate opening, m.
        turn_deg: requested (mean) turn angle between consecutive legs.
        leg: leg length, m.
        spacing: mean longitudinal gate spacing, m.
        amplitude: mean lateral offset, m.
        radius: corner radius, m -- the difficulty that matters, since demanded
            lateral acceleration is ``speed**2 / radius``.
        path_length: length of the full flown path including lead-out, m.
        periodic: always False; a slalom is flown once, not looped.
    """

    gate_pos: Array
    gate_yaw: Array
    path_pos: Array
    path_yaw: Array
    starting_pos: Array
    gate_size: float
    turn_deg: float
    leg: float
    spacing: float
    amplitude: float
    radius: float
    path_length: float
    periodic: bool = field(default=False)

    def trajectory_config(self) -> SimpleNamespace:
        """Gate-config view over the *path*, for BSplineGateTrajectory.

        The trajectory must fly the lead-out; only the detector should see the
        scored gates.
        """
        return SimpleNamespace(
            gate_pos=self.path_pos, gate_yaw=self.path_yaw,
            gate_size=self.gate_size, starting_pos=self.starting_pos,
            periodic=self.periodic,
        )

    def measured_turn_angles(self) -> Array:
        """Actual angles between consecutive legs of the scored course, degrees."""
        v = np.diff(self.gate_pos[:, :2], axis=0)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip(np.sum(v[:-1] * v[1:], axis=1), -1.0, 1.0)))

    def lateral_acceleration(self, speed: float) -> float:
        """Lateral acceleration this course demands at ``speed``, m/s^2."""
        return float(speed) ** 2 / self.radius

    def traversal_time(self, speed: float, startup_time: float = 3.0) -> float:
        """``total_time`` for BSplineGateTrajectory at a fixed nominal speed.

        Covers the whole path, lead-out included.
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
    lead_out: int = 2,
    seed: int | None = None,
    jitter: float = 0.25,
) -> SlalomCourse:
    """Build a slalom course demanding ``turn_deg`` at every gate.

    Args:
        turn_deg: mean angle between consecutive legs, degrees.
        leg: distance between consecutive waypoints, m. Held fixed so corner
            radius falls monotonically with ``turn_deg``.
        n_gates: number of *scored* gates.
        gate_size: gate opening, m.
        z: course altitude (NED, so negative is up).
        start_offset: spawn distance behind gate 0, m.
        lead_out: waypoints appended past the last scored gate so the drone can
            fly through the finish rather than stop on it.
        seed: when given, per-corner turn angles are jittered, producing a
            different layout of the same difficulty. ``None`` gives the uniform
            course.
        jitter: fractional spread of the per-corner turn angle when seeded.

    Raises:
        ValueError: if the turn angle is out of range, or the resulting
            longitudinal spacing would not exceed ``gate_size`` -- gate planes
            would then overlap and "passing a gate" stops being well defined.
    """
    if not 0.0 <= turn_deg < 180.0:
        raise ValueError(f"turn_deg must be in [0, 180), got {turn_deg}")
    if n_gates < 3:
        raise ValueError(f"need at least 3 gates to define a turn, got {n_gates}")
    if lead_out < 0:
        raise ValueError(f"lead_out must be >= 0, got {lead_out}")

    n_way = n_gates + lead_out
    n_legs = n_way - 1

    rng = np.random.default_rng(seed) if seed is not None else None
    if rng is None:
        thetas = np.full(n_legs, np.radians(turn_deg))
    else:
        lo, hi = 1.0 - jitter, 1.0 + jitter
        thetas = np.radians(turn_deg) * rng.uniform(lo, hi, size=n_legs)

    # Legs alternate +/- theta/2 about the course axis, so the turn between
    # consecutive legs is (theta_i + theta_i+1)/2 -- exactly turn_deg when
    # uniform.
    signs = (-1.0) ** np.arange(n_legs)
    if rng is not None:
        # Zero the net heading change. Leg directions alternate +/- theta_i/2,
        # so with random theta the alternating deviations do not cancel and the
        # course curves away from its axis -- measured up to 1.85 m over 16
        # legs. That is a second, uncontrolled difficulty axis (a sustained
        # turn) on top of the corner sharpness this parameter is meant to vary.
        # Removing the residual shifts each angle by a fraction of a degree and
        # leaves the corner range identical.
        # Lateral displacement is sum(leg * sin(a_i)), not sum(a_i), so zeroing
        # the angles only approximates zero drift (1.85 m -> 0.29 m). Shifting
        # every theta_i by signs_i*delta shifts every leg direction by the same
        # -delta/2, i.e. rotates the course, so one Newton step per iteration on
        # d(sum sin a)/d(delta) = -sum(cos a)/2 converges in two or three.
        for _ in range(6):
            a = signs * thetas / 2.0
            drift = float(np.sum(np.sin(a)))
            if abs(drift) < 1e-13:
                break
            denom = 0.5 * float(np.sum(np.cos(a)))
            if abs(denom) < 1e-12:
                break
            thetas = thetas - signs * (drift / denom)
    leg_dir = signs * thetas / 2.0

    spacing = float(np.mean(leg * np.cos(thetas / 2.0)))
    if spacing <= gate_size:
        raise ValueError(
            f"turn_deg={turn_deg} with leg={leg} gives gate spacing {spacing:.3f} m, "
            f"which does not exceed gate_size={gate_size} m: gate planes would "
            f"overlap and a pass would be ambiguous. Lengthen the leg or shrink "
            f"the gate."
        )

    pos = np.zeros((n_way, 3), dtype=float)
    pos[:, 2] = float(z)
    for i, a in enumerate(leg_dir):
        pos[i + 1, :2] = pos[i, :2] + leg * np.array([np.cos(a), np.sin(a)])

    # Gate plane faces the bisector of its adjacent legs; the end waypoints have
    # only one neighbour.
    yaw = np.zeros(n_way, dtype=float)
    yaw[0] = leg_dir[0]
    yaw[-1] = leg_dir[-1]
    yaw[1:-1] = 0.5 * (leg_dir[:-1] + leg_dir[1:])

    mean_theta = float(np.mean(thetas))
    radius = leg / (2.0 * np.sin(mean_theta / 2.0)) if mean_theta > 1e-9 else float("inf")

    first = pos[1, :2] - pos[0, :2]
    first = first / np.linalg.norm(first)
    start = pos[0].copy()
    start[:2] -= start_offset * first

    return SlalomCourse(
        gate_pos=pos[:n_gates].copy(),
        gate_yaw=yaw[:n_gates].copy(),
        path_pos=pos,
        path_yaw=yaw,
        starting_pos=start,
        gate_size=float(gate_size),
        turn_deg=float(turn_deg),
        leg=float(leg),
        spacing=spacing,
        amplitude=float(np.mean((leg / 2.0) * np.sin(thetas / 2.0))),
        radius=float(radius),
        path_length=float(leg * n_legs),
    )

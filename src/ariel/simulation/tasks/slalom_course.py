"""Slalom gate course with difficulty as a single explicit parameter.

The quintic-planner circuits used by the drone-morphology examples turn very
little -- a median of 0.4 degrees between consecutive legs -- so they ask almost
nothing of the airframe and every morphology scores the same. This module
builds a course whose difficulty is one interpretable number: the turn angle
demanded at each gate.

Parameterised by turn angle at **fixed leg length**, which matters. With the
leg fixed, the course is the same length at every turn angle (polyline
``leg * n_legs``; the flown spline varies by about 3%), so the turn angle is the
only thing that changes. Holding the longitudinal spacing fixed instead makes
the legs longer as the turn sharpens: at 2.0 m spacing the polyline grows from
33 m at 30 degrees to 64 m at 120. Difficulty would then be confounded with
course length, and anything that accumulates over a flight -- altitude sag,
drift -- would grow with it.

Difficulty is measured on the reference the drone is asked to fly, not on the
gate layout. The B-spline through the gates is a smooth weave that bends
hardest at each gate, and the demand is its lateral acceleration there
(``SlalomCourse.reference_demand``). At fixed leg length both the typical
(median) and the worst gate's demand rise monotonically with turn angle. The
worst gate is the last scored one, 24-44% above the typical gate: nearing the
spline's clamped end, the reference both speeds up (5-14% at the last gate) and
bends more tightly (3-10%), measured over 60-120 degrees.

*Corrected 2026-09-24.* This docstring previously justified fixed leg length
by a "corner radius" ``s / (2 sin(theta/2))``, the circle through three
consecutive gates, and took ``speed**2 / radius`` as the demanded lateral
acceleration. It argued that at fixed spacing that radius is non-monotonic
(4.00 / 2.31 / 2.00 / 2.31 m for 30/60/90/120 degrees), so the corner ends up
*wider*. The drone does not fly that circle. On the reference, the tightest
radius is 4-14x smaller, and the lateral acceleration is 1.5x
``speed**2 / radius`` at a typical gate and up to 2.2x at the last one. At fixed
spacing the reference's tightest radius falls monotonically after all. The circle's radius is kept as
``gate_circle_radius``, which describes the layout and nothing more.

For a leg length ``s`` and turn angle ``theta``, consecutive legs run at
``+/- theta/2`` to the course axis, so:

    x-spacing  d = s * cos(theta/2)
    amplitude  A = (s/2) * sin(theta/2)
    gate-circle radius  s / (2 * sin(theta/2))   (layout only; see above)

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


@dataclass(frozen=True)
class ReferenceDemand:
    """What the reference trajectory asks of the vehicle, over the scored gates.

    Attributes:
        a_lat_per_gate: peak lateral acceleration, ``|v x a| / |v|``, near each
            scored gate (each reference sample is assigned to its nearest
            gate), m/s^2.
        a_lat_peak: the largest of those -- the corner that decides completion.
            On a uniform course this is the *last* scored gate, not a typical
            one: nearing the spline's clamped end, the reference both speeds
            up and bends more tightly.
        a_lat_median: median over gates -- the typical corner. Robust to the
            startup ramp (low) and the finish (high).
        min_radius: tightest radius of curvature of the reference over the
            scored gates, m. Geometric, so independent of timing. Not a demand
            on its own: ``speed**2 / min_radius`` overstates it badly, because
            the reference is slow where it is tightest.
        arc_length: length of the whole reference, lead-out included, m.
    """

    a_lat_per_gate: tuple[float, ...]
    a_lat_peak: float
    a_lat_median: float
    min_radius: float
    arc_length: float


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
        gate_circle_radius: radius of the circle through three consecutive
            gates, m. Describes the layout only: the reference bends far more
            tightly than this, so ``speed**2 / gate_circle_radius`` understates
            the demand about twofold. Use ``reference_demand`` for that.
        path_length: length of the full polyline including lead-out, m.
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
    gate_circle_radius: float
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

    def reference_demand(self, speed: float, startup_time: float = 3.0,
                         dt: float = 0.002) -> ReferenceDemand:
        """What the reference trajectory demands at nominal ``speed``.

        Builds the reference exactly as the sweep does (``BSplineGateTrajectory``
        over the path, offsets fitted so it interpolates the gates, timed by
        ``traversal_time``) and measures it from the start until it crosses the
        last scored gate's plane. The lead-out is flown but not scored, so it is
        excluded.

        The spline is timed uniformly in its parameter, not in arc length, so
        the reference slows where it bends hardest. The demand is therefore
        ``|v x a| / |v|`` measured along the timed reference. ``speed**2`` times
        the peak curvature would overstate it by ~7x at 120 degrees.
        """
        # Imported here: the course is plain geometry, and only this method
        # needs the trajectory stack.
        from ariel.simulation.drone.controllers.trajectory_generation.bspline_gate_trajectory import (  # noqa: E501
            BSplineGateTrajectory,
        )

        tr = BSplineGateTrajectory(self.trajectory_config())
        tr.fit_offsets_to_gates()
        tr.total_time = self.traversal_time(speed, startup_time=startup_time)
        tr.startup_time = startup_time
        last_normal = np.array([np.cos(self.gate_yaw[-1]), np.sin(self.gate_yaw[-1]), 0.0])

        def scored(pos: Array) -> int:
            """Samples up to and including the last scored gate's plane."""
            past = np.flatnonzero((pos - self.gate_pos[-1]) @ last_normal >= 0.0)
            return int(past[0]) + 1 if len(past) else len(pos)

        # Demand: along the timed reference, as the controller receives it.
        ts = np.arange(0.0, tr.total_time + dt / 2, dt)
        pva = [tr.evaluate(float(t)) for t in ts]
        pos = np.array([p for p, _, _ in pva])
        end = scored(pos)
        vel = np.array([v for _, v, _ in pva[:end]])
        acc = np.array([a for _, _, a in pva[:end]])
        spd = np.linalg.norm(vel, axis=1)
        a_lat = np.linalg.norm(np.cross(vel, acc), axis=1) / np.maximum(spd, 1e-9)
        nearest = np.argmin(
            np.linalg.norm(pos[:end, None, :] - self.gate_pos[None], axis=2), axis=1)
        per_gate = tuple(float(a_lat[nearest == k].max()) if np.any(nearest == k) else 0.0
                         for k in range(len(self.gate_pos)))

        # Curvature: along the spline parameter, where it is well defined even
        # at the start. Measured through time it is 0/0 while the ramp is at
        # rest, which returned radii of 0.01-0.04 m.
        sp = tr.spline
        assert sp is not None, "fit_offsets_to_gates leaves a spline"
        u = np.linspace(sp.u_min, sp.u_max, 20001)
        p_u = np.array([sp.position(x) for x in u])
        d1 = np.array([sp.velocity(x, 1.0) for x in u])
        d2 = np.array([sp.acceleration(x, 1.0, 0.0) for x in u])
        u_end = scored(p_u)
        kappa = (np.linalg.norm(np.cross(d1[:u_end], d2[:u_end]), axis=1)
                 / np.linalg.norm(d1[:u_end], axis=1) ** 3)

        return ReferenceDemand(
            a_lat_per_gate=per_gate,
            a_lat_peak=max(per_gate),
            a_lat_median=float(np.median(per_gate)),
            min_radius=float(1.0 / kappa.max()) if kappa.max() > 0 else float("inf"),
            arc_length=float(np.sum(np.linalg.norm(np.diff(p_u, axis=0), axis=1))),
        )

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
    circle = leg / (2.0 * np.sin(mean_theta / 2.0)) if mean_theta > 1e-9 else float("inf")

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
        gate_circle_radius=float(circle),
        path_length=float(leg * n_legs),
    )

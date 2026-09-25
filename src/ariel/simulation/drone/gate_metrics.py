"""Did the drone actually fly through the gate?

Three criteria are in use in this repository and they disagree badly, so they
are named and implemented here rather than reinvented per call site.

``sequential`` (``GateChecker``)
    Crossing of the gate plane, with lateral error measured from ``pos[:2]``.
    Altitude is never checked, so a drone below the gate still passes.

``proximity``
    Minimum distance to the gate centre over the whole trajectory. Ignores
    *when* that minimum occurred, so a drone that sails past a gate without
    crossing its plane still passes -- and on a slalom, where consecutive gates
    are closer together than the gate is wide, that is easy to do by accident.

``strict`` (this module's default)
    At the instant the drone crosses the gate plane, its offset from the gate
    centre *within that plane* -- lateral and vertical together -- must be
    within the half-opening. This is what a physical gate means.

    The opening is a circle of radius ``gate_size / 2``, not a square: decided
    2026-09-11 (docs/drone_morphology_gate_racing.md, section 8 item 8). A
    square of the same half-width admits 27% more area. The sweep repeats this
    test inline in ``rollout``; tests/unit/test_simulation/test_gate_criterion.py
    pins the two against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GateCrossing", "crossing_report", "gate_normal", "in_plane_offset",
           "n_passed", "passes"]


def gate_normal(gate_yaw: np.ndarray | float) -> np.ndarray:
    """Unit normal of a gate plane, ``(cos yaw, sin yaw, 0)``; shape (..., 3).

    Gates are vertical: the normal is horizontal, and a drone passes a gate by
    crossing this plane in the direction of the normal.
    """
    yaw = np.asarray(gate_yaw, dtype=float)
    return np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=-1)


def in_plane_offset(pos: np.ndarray, gate_pos: np.ndarray, gate_yaw: np.ndarray | float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where ``pos`` sits relative to a gate: the single definition of the test.

    Broadcasts over leading dimensions, so one call scores many drones against
    their own gates. Returns ``(signed, lateral, vertical, offset)``:

    * ``signed`` -- distance along the gate normal; negative before the gate.
    * ``lateral`` -- horizontal in-plane distance from the centre (unsigned).
    * ``vertical`` -- in-plane vertical offset (signed: + is below in NED).
    * ``offset`` -- in-plane distance from the centre, ``hypot(lateral, vertical)``.

    The along-normal part is removed before measuring, so a sample taken just
    past the plane is not penalised for how far past it landed.
    """
    normal = gate_normal(gate_yaw)
    rel = np.asarray(pos, dtype=float) - np.asarray(gate_pos, dtype=float)
    signed = np.sum(rel * normal, axis=-1)
    in_plane = rel - signed[..., None] * normal
    lateral = np.linalg.norm(in_plane[..., :2], axis=-1)
    vertical = in_plane[..., 2]
    return signed, lateral, vertical, np.hypot(lateral, vertical)


def passes(offset: np.ndarray | float, gate_size: float,
           clearance: np.ndarray | float = 0.0) -> np.ndarray:
    """Whether an in-plane offset clears a circular opening of diameter ``gate_size``.

    ``clearance`` is how far the airframe extends from its centre within the
    gate plane. At 0 the drone is scored as a point, which is how every result
    in the report was scored. A positive value asks the whole airframe to fit.
    """
    return np.asarray(offset) + np.asarray(clearance) <= gate_size / 2.0


@dataclass(frozen=True)
class GateCrossing:
    """Where the drone was when it crossed one gate's plane."""

    index: int
    crossed: bool
    step: int
    lateral: float      # in-plane, horizontal
    vertical: float     # in-plane, vertical (signed: + is below in NED)
    offset: float       # in-plane distance from the centre
    passed: bool


def crossing_report(positions: np.ndarray, gate_pos: np.ndarray,
                    gate_yaw: np.ndarray, gate_size: float) -> list[GateCrossing]:
    """One entry per gate, describing the drone's first crossing of its plane.

    A crossing is a negative-to-non-negative transition of the signed distance
    along the gate normal, so a trajectory that *begins* on or past a gate plane
    never registers one and that gate reports ``crossed=False``. Flights spawn
    behind the first gate (``start_offset``), so this does not arise in scoring;
    it does arise when analysing the reference trajectory itself, whose first
    control point is gate 0's centre.
    """
    positions = np.asarray(positions, dtype=float)
    out: list[GateCrossing] = []
    for k, (g, yaw) in enumerate(zip(np.asarray(gate_pos, dtype=float),
                                     np.asarray(gate_yaw, dtype=float))):
        signed, lateral, vertical, offset = in_plane_offset(positions, g, yaw)
        sign_change = np.flatnonzero((signed[:-1] < 0.0) & (signed[1:] >= 0.0))
        if not len(sign_change):
            out.append(GateCrossing(k, False, -1, np.inf, np.inf, np.inf, False))
            continue
        i = int(sign_change[0]) + 1
        out.append(GateCrossing(k, True, i, float(lateral[i]), float(vertical[i]),
                                float(offset[i]), bool(passes(offset[i], gate_size))))
    return out


def n_passed(report: list[GateCrossing]) -> int:
    return sum(1 for c in report if c.passed)

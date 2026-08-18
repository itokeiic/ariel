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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GateCrossing", "crossing_report", "n_passed"]


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
    half = gate_size / 2.0
    out: list[GateCrossing] = []
    for k, (g, yaw) in enumerate(zip(np.asarray(gate_pos, dtype=float),
                                     np.asarray(gate_yaw, dtype=float))):
        normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        signed = (positions - g) @ normal
        sign_change = np.flatnonzero((signed[:-1] < 0.0) & (signed[1:] >= 0.0))
        if not len(sign_change):
            out.append(GateCrossing(k, False, -1, np.inf, np.inf, np.inf, False))
            continue
        i = int(sign_change[0]) + 1
        # Project the offset into the gate plane: remove the along-normal part.
        rel = positions[i] - g
        in_plane = rel - float(rel @ normal) * normal
        lateral = float(np.linalg.norm(in_plane[:2]))
        vertical = float(in_plane[2])
        offset = float(np.hypot(lateral, vertical))
        out.append(GateCrossing(k, True, i, lateral, vertical, offset,
                                bool(offset <= half)))
    return out


def n_passed(report: list[GateCrossing]) -> int:
    return sum(1 for c in report if c.passed)

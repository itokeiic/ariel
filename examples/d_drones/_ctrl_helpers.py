"""Shared helpers for drone simulation and controller-tuning examples.

Shared helpers for examples 4 (simulate) and 5 (tune).
"""

from __future__ import annotations

import numpy as np

from ariel.simulation.drone.drone_interface import DroneInterface

# 2-inch quadrotor constants (60 mm arm, 2-inch propellers, ~0.08 kg total)
ARM_LENGTH: float = 0.06   # metres
PROP_SIZE: int = 2         # inches


def create_2inch_quad() -> DroneInterface:
    """Instantiate a standard 2-inch X-configuration quadrotor."""
    propellers = [
        {"loc": [ ARM_LENGTH,  ARM_LENGTH, 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
        {"loc": [-ARM_LENGTH,  ARM_LENGTH, 0], "dir": [0, 0, -1, "cw"],  "propsize": PROP_SIZE},
        {"loc": [-ARM_LENGTH, -ARM_LENGTH, 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
        {"loc": [ ARM_LENGTH, -ARM_LENGTH, 0], "dir": [0, 0, -1, "cw"],  "propsize": PROP_SIZE},
    ]
    return DroneInterface(0, propellers=propellers)


class GateChecker:
    """Detect when the drone flies through a sequence of gates (in order).

    Frame-intersection detector: a gate is counted when the trajectory segment
    between the previous and current sample intersects the gate plane *inside*
    the ±gate_size/2 opening (lateral AND vertical), regardless of crossing
    direction. This is step-size robust and, unlike a strict one-directional
    (- -> +) crossing test, does not lock on densely spaced courses where the
    drone is already on the far side of gate i's plane by the time gate i
    becomes the active target (the old test would then never see another - -> +
    crossing and freeze every later gate). A catch-up loop advances through
    every gate a single segment clears. Counting stays strictly sequential
    (gates must be flown through in index order).

    Notes vs the previous implementation (changes the gate count):
      - normal is oriented along the local course (gate i -> i+1), not raw yaw;
      - a vertical-offset check is added (was lateral-only);
      - crossing is direction-agnostic and uses the interpolated plane crossing.
    `eps` widens the grazing band; `proximity_radius` (default 0, off) re-enables
    a centre-radius fallback for pathological hairpins.
    """

    def __init__(
        self,
        gate_pos: np.ndarray,
        gate_yaw: np.ndarray,
        gate_size: float = 1.0,
        max_gate_distance: float = 10.0,
        eps: float = 0.02,
        proximity_radius: float = 0.0,
    ) -> None:
        self.gate_pos = np.asarray(gate_pos, dtype=np.float64)
        self.gate_yaw = np.asarray(gate_yaw, dtype=np.float64)
        self.gate_size = gate_size
        self.max_gate_distance = max_gate_distance
        self.eps = eps
        self.proximity_radius = proximity_radius
        self.num_gates = len(gate_pos)
        self.reset()

    def reset(self) -> None:
        self.gates_passed = 0
        self._next_gate = 0
        self._prev_pos: np.ndarray | None = None
        self._prev_signed_dist: float | None = None  # kept for external readers

    def _gate_normal(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Gate centre + plane normal oriented along the local course direction
        (gate i -> i+1); leaves the opening geometry unchanged."""
        gate_p = self.gate_pos[i]
        gate_y = float(self.gate_yaw[i])
        normal = np.array([np.cos(gate_y), np.sin(gate_y), 0.0])
        if i < self.num_gates - 1:
            course = self.gate_pos[i + 1] - gate_p
        else:
            course = gate_p - self.gate_pos[i - 1]
        if float(np.dot(normal[:2], course[:2])) < 0.0:
            normal = -normal
        return gate_p, normal

    def _frame_errs(self, pt: np.ndarray, gate_p: np.ndarray, normal: np.ndarray):
        """Signed plane distance + in-opening lateral / vertical offsets."""
        s = float(np.dot(pt - gate_p, normal))
        lat = float(np.linalg.norm(pt[:2] - gate_p[:2] - s * normal[:2]))
        vert = float(abs(pt[2] - gate_p[2]))
        return s, lat, vert

    def check_gate_passing(self, pos: np.ndarray, return_info: bool = False):
        """Update with a new sample; return True if >=1 gate was just passed.

        With ``return_info=True`` returns ``(passed, info_dict)`` for debug
        logging. ``orientation_err_deg`` is always None here (this method only
        sees position, not heading); callers needing it must compute it.
        """
        pos = np.asarray(pos, dtype=np.float64)
        prev_pos = self._prev_pos
        half = self.gate_size / 2.0
        passed_any = False
        entry_gate = int(self._next_gate)
        info_signed: float | None = None
        info_lat: float | None = None
        info_vert: float | None = None
        info_crossed = False

        # Walk the current segment [prev_pos, pos] against the active gate,
        # advancing through every in-order gate whose opening it passes through.
        while self._next_gate < self.num_gates:
            gate_p, normal = self._gate_normal(self._next_gate)
            s_cur = float(np.dot(pos - gate_p, normal))
            self._prev_signed_dist = s_cur
            if info_signed is None:
                info_signed = s_cur
            if prev_pos is None:
                break  # need a previous sample to form a segment

            s_prev = float(np.dot(prev_pos - gate_p, normal))
            hit = False

            # Primary: does the segment straddle the plane (any direction)?
            if (s_prev <= 0.0 <= s_cur) or (s_cur <= 0.0 <= s_prev):
                info_crossed = True
                denom = s_prev - s_cur
                f = min(max((s_prev / denom) if denom != 0.0 else 0.0, 0.0), 1.0)
                cross_pt = prev_pos + f * (pos - prev_pos)
                _, lat, vert = self._frame_errs(cross_pt, gate_p, normal)
                if lat <= half and vert <= half:
                    hit = True
                    info_lat, info_vert = lat, vert

            # Grazing fallback: a sample inside the opening, on the plane.
            if not hit:
                _, lat_now, vert_now = self._frame_errs(pos, gate_p, normal)
                if abs(s_cur) <= self.eps and lat_now <= half and vert_now <= half:
                    hit = True
                    info_lat, info_vert = lat_now, vert_now

            # Optional centre-radius proximity fallback (off when radius is 0).
            if not hit and self.proximity_radius > 0.0:
                if float(np.linalg.norm(pos - gate_p)) <= self.proximity_radius:
                    hit = True
                    _, info_lat, info_vert = self._frame_errs(pos, gate_p, normal)

            if not hit:
                break

            self.gates_passed += 1
            self._next_gate = (self._next_gate + 1) % self.num_gates
            passed_any = True
            if self._next_gate == 0:  # wrapped past the last gate
                break

        self._prev_pos = pos
        if return_info:
            return passed_any, {
                "gate_index_checked": entry_gate,
                "prev_signed_dist": None,
                "signed_dist": info_signed,
                "crossed_plane": bool(info_crossed),
                "lateral_err": info_lat,
                "vertical_err": info_vert,
                "orientation_err_deg": None,
                "passed": bool(passed_any),
                "next_gate_after": int(self._next_gate),
            }
        return passed_any

    def get_normalized_distance_to_next_gate(self, pos: np.ndarray) -> float:
        """Return a [0,1] proximity bonus to the next gate (1 = at gate, 0 = far)."""
        if self._next_gate >= self.num_gates:
            return 0.0
        distance = float(np.linalg.norm(pos - self.gate_pos[self._next_gate]))
        return max(0.0, min(1.0, 1.0 - distance / self.max_gate_distance))

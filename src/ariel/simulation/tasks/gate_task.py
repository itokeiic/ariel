"""A gate-passing task that does not know how the drone is flown.

What the task *is* -- which gates, in which order, what counts as passing one,
when an episode ends and how it scored -- separated from what flies it. The Lee
controller tracking a B-spline reference is one way to fly it; an RL policy is
another. Both see the same gates and are judged by the same rules.

Deliberately **not** here:

* **reward** -- a training choice, not part of the task. Build it from the
  events ``update`` returns (a pass, a miss) and the gate poses it exposes.
* **observation frames** -- ``upcoming`` gives gate poses in the world frame;
  expressing them in the body or gate frame belongs to the environment.
* **dynamics, controller, reference trajectory.**

Batched over ``num_envs`` with array operations only, no per-environment Python
loop, so a torch port for Isaac Lab is a change of array library, not of logic.
One environment (``num_envs=1``) is the Lee baseline.

The pass test is ``gate_metrics.in_plane_offset`` and ``gate_metrics.passes``,
the same functions ``crossing_report`` uses, so a flight scored here and by the
report's criterion gets the same per-gate verdicts. A gate is a circular
opening of diameter ``gate_size`` in a vertical plane; the drone passes when it
crosses the plane along the normal with its in-plane offset, plus an optional
airframe ``clearance``, inside the radius. The gates have no physical presence:
nothing here stops a drone that misses.

Gates are taken in order. Only the current target's plane is watched, which is
racing semantics, and matches ``crossing_report`` whenever the drone crosses
the gate planes in course order -- true of every flight checked in
tests/unit/test_simulation/test_gate_task.py.

Example, one drone (``num_envs=1``)::

    course = slalom_gates(90.0)
    task = GateTask(course, max_time=30.0,
                    bounds=course_bounds(course, course.starting_pos))
    task.reset(course.starting_pos, t=0.0)
    while task.running.any():
        pos, t = step_simulation(...)
        events = task.update(pos, t)
    task.completed, task.finish_time
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from ariel.simulation.drone.gate_metrics import gate_normal, in_plane_offset, passes

Array = npt.NDArray[np.float64]


class GateLayout(Protocol):
    """Anything with gates. ``SlalomCourse`` qualifies; so does a plain namespace.

    Attributes:
        gate_pos: gate centres, shape (n_gates, 3), NED.
        gate_yaw: gate plane headings, shape (n_gates,); the pass direction is
            ``(cos yaw, sin yaw, 0)``.
        gate_size: diameter of the circular opening, m.
    """

    gate_pos: Array
    gate_yaw: Array
    gate_size: float


class Status(IntEnum):
    """Episode state, one per environment."""

    RUNNING = 0
    FINISHED = 1        # crossed the last gate's plane (see ``completed``)
    MISSED = 2          # crossed a gate plane outside the opening
    OUT_OF_BOUNDS = 3
    TIMEOUT = 4


@dataclass(frozen=True)
class StepEvents:
    """What happened in one ``update``, per environment; all shape (num_envs,).

    Attributes:
        passed: a gate was passed this step.
        missed: the target gate's plane was crossed outside the opening.
        gate: the gate index those refer to, -1 where neither happened.
        done: the episode ended this step, for any reason.
    """

    passed: npt.NDArray[np.bool_]
    missed: npt.NDArray[np.bool_]
    gate: npt.NDArray[np.int64]
    done: npt.NDArray[np.bool_]


def course_bounds(layout: GateLayout, *extra_points: Array, margin: float = 2.0
                  ) -> tuple[Array, Array]:
    """Axis-aligned box around the gates (and e.g. the spawn), padded by ``margin``."""
    pts = np.vstack([np.asarray(layout.gate_pos, dtype=float),
                     *(np.atleast_2d(np.asarray(p, dtype=float)) for p in extra_points)])
    return pts.min(axis=0) - margin, pts.max(axis=0) + margin


class GateTask:
    """Gate order, pass test and episode rules for ``num_envs`` drones at once.

    Args:
        layout: the gates (``SlalomCourse`` or any ``GateLayout``).
        num_envs: drones tracked in parallel.
        on_miss: ``"terminate"`` ends the episode at a missed gate -- racing
            semantics, and what a physical frame would do. ``"continue"``
            records the miss and moves on to the next gate, which is how the
            report's criterion scores a flight (every gate judged separately).
        bounds: ``(lo, hi)`` corners of the allowed box, or None for no box.
        max_time: episode time limit in the units of ``t``, or None.
    """

    def __init__(self, layout: GateLayout, num_envs: int = 1, *,
                 on_miss: Literal["terminate", "continue"] = "terminate",
                 bounds: tuple[Array, Array] | None = None,
                 max_time: float | None = None) -> None:
        if on_miss not in ("terminate", "continue"):
            raise ValueError(f"on_miss must be 'terminate' or 'continue', got {on_miss!r}")
        self.gate_pos = np.asarray(layout.gate_pos, dtype=float)
        self.gate_yaw = np.asarray(layout.gate_yaw, dtype=float)
        self.gate_size = float(layout.gate_size)
        self.n_gates = len(self.gate_pos)
        if self.gate_pos.shape != (self.n_gates, 3) or self.gate_yaw.shape != (self.n_gates,):
            raise ValueError("gate_pos must be (n_gates, 3) and gate_yaw (n_gates,)")
        self.num_envs = int(num_envs)
        self.on_miss = on_miss
        self.bounds = (None if bounds is None else
                       (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)))
        self.max_time = max_time
        self._normals = gate_normal(self.gate_yaw)

        n, g = self.num_envs, self.n_gates
        self.status = np.full(n, Status.RUNNING, dtype=np.int64)
        self.target = np.zeros(n, dtype=np.int64)
        self.gates_passed = np.zeros(n, dtype=np.int64)
        self.start_time = np.zeros(n)
        self.finish_time = np.full(n, np.nan)
        # Per gate, at the step its plane was crossed; NaN / False if never.
        self.crossing_offset = np.full((n, g), np.nan)
        self.crossing_time = np.full((n, g), np.nan)
        self.passed = np.zeros((n, g), dtype=bool)
        self._prev_signed = np.full(n, np.nan)

    # ── episode control ─────────────────────────────────────────────────────

    def reset(self, pos: Array, *, mask: npt.NDArray[np.bool_] | None = None,
              start_gate: int | npt.NDArray[np.int64] = 0, t: float | Array = 0.0) -> None:
        """Start episodes for the environments in ``mask`` (all by default).

        ``pos`` is where each drone starts, shape (num_envs, 3) or (3,); it sets
        the side of the first target plane, so a crossing on the very first
        step counts. ``start_gate`` lets RL training start mid-course.
        """
        m = np.ones(self.num_envs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        pos = self._as_pos(pos)
        start = np.broadcast_to(np.asarray(start_gate, dtype=np.int64), (self.num_envs,))
        if np.any((start[m] < 0) | (start[m] >= self.n_gates)):
            raise ValueError(f"start_gate must be in [0, {self.n_gates})")
        self.status[m] = Status.RUNNING
        self.target[m] = start[m]
        self.gates_passed[m] = 0
        self.start_time[m] = np.broadcast_to(np.asarray(t, dtype=float), (self.num_envs,))[m]
        self.finish_time[m] = np.nan
        self.crossing_offset[m] = np.nan
        self.crossing_time[m] = np.nan
        self.passed[m] = False
        self._prev_signed[m] = self._signed_to_target(pos)[m]

    def update(self, pos: Array, t: float | Array,
               clearance: float | Array = 0.0) -> StepEvents:
        """Advance the task after one simulation step.

        Args:
            pos: drone positions now, (num_envs, 3) or (3,).
            t: simulation time now, scalar or (num_envs,).
            clearance: airframe extent in the gate plane, scalar or
                (num_envs,). 0 scores the drone as a point, as the report does.
                A morphing drone can pass its current extent every step.
        """
        pos = self._as_pos(pos)
        n = self.num_envs
        t = np.broadcast_to(np.asarray(t, dtype=float), (n,))
        clearance = np.broadcast_to(np.asarray(clearance, dtype=float), (n,))
        running = self.status == Status.RUNNING
        rows = np.arange(n)

        k = np.minimum(self.target, self.n_gates - 1)
        signed, _, _, offset = in_plane_offset(pos, self.gate_pos[k], self.gate_yaw[k])
        # Same crossing rule as crossing_report: negative before, >= 0 now.
        crossed = running & (self._prev_signed < 0.0) & (signed >= 0.0)
        ok = crossed & passes(offset, self.gate_size, clearance)
        miss = crossed & ~ok

        c = rows[crossed]
        self.crossing_offset[c, k[c]] = offset[c]
        self.crossing_time[c, k[c]] = t[c]
        self.passed[c, k[c]] = ok[c]
        self.gates_passed += ok

        advance = ok | (miss if self.on_miss == "continue" else np.zeros(n, dtype=bool))
        self.target[advance] += 1
        finished = advance & (self.target >= self.n_gates)
        self.status[finished] = Status.FINISHED
        self.finish_time[finished] = t[finished]
        if self.on_miss == "terminate":
            self.status[miss] = Status.MISSED

        still = self.status == Status.RUNNING
        if self.bounds is not None:
            lo, hi = self.bounds
            out = still & np.any((pos < lo) | (pos > hi), axis=1)
            self.status[out] = Status.OUT_OF_BOUNDS
            still &= ~out
        if self.max_time is not None:
            late = still & (t - self.start_time >= self.max_time)
            self.status[late] = Status.TIMEOUT

        # The new target's plane, seen from where the drone is now.
        self._prev_signed = np.where(advance, self._signed_to_target(pos), signed)
        return StepEvents(passed=ok, missed=miss, gate=np.where(crossed, k, -1),
                          done=running & (self.status != Status.RUNNING))

    # ── what a policy or a score needs ──────────────────────────────────────

    @property
    def running(self) -> npt.NDArray[np.bool_]:
        return self.status == Status.RUNNING

    @property
    def completed(self) -> npt.NDArray[np.bool_]:
        """Finished with every gate passed -- the report's completion test."""
        return (self.status == Status.FINISHED) & (self.gates_passed == self.n_gates)

    @property
    def course_time(self) -> Array:
        """Start to last gate, for completed episodes; inf otherwise.

        The natural score once the flyer chooses its own speed, as a policy
        does. (The Lee baseline instead reports the fastest nominal speed at
        which it still completes.)
        """
        return np.where(self.completed, self.finish_time - self.start_time, np.inf)

    def upcoming(self, n: int = 1) -> tuple[Array, Array, npt.NDArray[np.bool_]]:
        """The next ``n`` gates for each environment, target first, world frame.

        Returns ``(pos (num_envs, n, 3), yaw (num_envs, n), valid (num_envs, n))``.
        Past the last gate, entries repeat the last gate and ``valid`` is False.
        """
        idx = self.target[:, None] + np.arange(n)[None, :]
        valid = idx < self.n_gates
        idx = np.minimum(idx, self.n_gates - 1)
        return self.gate_pos[idx], self.gate_yaw[idx], valid

    # ── internals ───────────────────────────────────────────────────────────

    def _as_pos(self, pos: Array) -> Array:
        pos = np.asarray(pos, dtype=float)
        if pos.shape == (3,):
            pos = np.broadcast_to(pos, (self.num_envs, 3))
        if pos.shape != (self.num_envs, 3):
            raise ValueError(f"pos must be ({self.num_envs}, 3) or (3,), got {pos.shape}")
        return pos

    def _signed_to_target(self, pos: Array) -> Array:
        k = np.minimum(self.target, self.n_gates - 1)
        return np.sum((pos - self.gate_pos[k]) * self._normals[k], axis=1)

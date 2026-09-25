"""The flyer-agnostic gate task agrees with the report's criterion.

``GateTask`` (src/ariel/simulation/tasks/gate_task.py) is meant to be the task
an RL policy and the Lee baseline share. Its pass test must be the report's:
``crossing_report`` in src/ariel/simulation/drone/gate_metrics.py. Pinned on the
committed flight logs, as flown and shifted sideways so that some gates are
missed, plus the episode rules the report never needed: order, termination,
bounds, time limit, clearance and batching.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ariel.simulation.drone.gate_metrics import crossing_report
from ariel.simulation.tasks.gate_task import GateTask, Status, course_bounds
from ariel.simulation.tasks.slalom_course import slalom_gates

REPO = Path(__file__).resolve().parents[3]
LOGS = sorted((REPO / "docs" / "data" / "flight_logs").rglob("*.npz"))


def _layout(log) -> SimpleNamespace:
    return SimpleNamespace(gate_pos=log["gate_pos"], gate_yaw=log["gate_yaw"],
                           gate_size=float(log["gate_size"]))


def _fly(layout, pos, t, **kwargs) -> GateTask:
    task = GateTask(layout, **kwargs)
    task.reset(pos[0], t=float(t[0]))
    for p, ti in zip(pos[1:], t[1:]):
        task.update(p, float(ti))
    return task


def test_logs_exist() -> None:
    assert len(LOGS) == 6, LOGS


@pytest.mark.parametrize("shift", [0.0, 0.25, 0.35])
@pytest.mark.parametrize("path", LOGS, ids=lambda p: str(p.relative_to(p.parents[1])))
def test_matches_crossing_report(path: Path, shift: float) -> None:
    """Same per-gate verdicts and offsets as the report's criterion.

    ``on_miss="continue"`` judges every gate, as ``crossing_report`` does. The
    shifted flights miss 5-6 of 15 gates, so agreement is not trivially 15/15.
    """
    log = np.load(path)
    pos = log["pos"] + np.array([0.0, shift, 0.0])
    rep = crossing_report(pos, log["gate_pos"], log["gate_yaw"], float(log["gate_size"]))
    task = _fly(_layout(log), pos, log["t"], on_miss="continue")

    expected = np.array([r.passed for r in rep])
    if shift > 0:
        assert 0 < expected.sum() < len(expected), "shift no longer produces misses"
    np.testing.assert_array_equal(task.passed[0], expected)
    np.testing.assert_allclose(
        task.crossing_offset[0], [r.offset if r.crossed else np.nan for r in rep])
    assert bool(task.completed[0]) == bool(expected.all())


def test_terminate_stops_at_first_miss() -> None:
    log = np.load(LOGS[0])
    pos = log["pos"] + np.array([0.0, 0.35, 0.0])
    judged = _fly(_layout(log), pos, log["t"], on_miss="continue")
    first_miss = int(np.argmin(judged.passed[0]))

    task = _fly(_layout(log), pos, log["t"], on_miss="terminate")
    assert task.status[0] == Status.MISSED
    assert task.target[0] == first_miss
    assert task.gates_passed[0] == first_miss
    assert np.isinf(task.course_time[0])


def test_batch_equals_one_at_a_time() -> None:
    """Three flights in one batch score as they do alone.

    A batch shares one course, so the three are one log shifted sideways by
    different amounts: one clean, two with misses at different gates.
    """
    log = np.load(LOGS[0])
    layout = _layout(log)
    shifts = np.array([0.0, 0.25, 0.35])
    single = [_fly(layout, log["pos"] + [0.0, s, 0.0], log["t"]) for s in shifts]

    batch = GateTask(layout, num_envs=3)
    pos0 = log["pos"][0] + np.outer(shifts, [0.0, 1.0, 0.0])
    batch.reset(pos0, t=float(log["t"][0]))
    for p, ti in zip(log["pos"][1:], log["t"][1:]):
        batch.update(p + np.outer(shifts, [0.0, 1.0, 0.0]), float(ti))
    for i, s in enumerate(single):
        assert batch.status[i] == s.status[0]
        assert batch.gates_passed[i] == s.gates_passed[0]
        np.testing.assert_array_equal(batch.passed[i], s.passed[0])


def _straight_through(offset: float, clearance: float) -> GateTask:
    gate = SimpleNamespace(gate_pos=np.array([[1.0, 0.0, -1.5]]), gate_yaw=np.array([0.0]),
                           gate_size=1.0)
    task = GateTask(gate)
    task.reset(np.array([0.0, offset, -1.5]))
    task.update(np.array([1.1, offset, -1.5]), 0.1, clearance=clearance)
    return task


@pytest.mark.parametrize(("clearance", "passed"), [(0.0, True), (0.19, True), (0.21, False)])
def test_clearance_asks_the_airframe_to_fit(clearance: float, passed: bool) -> None:
    """A centre 0.3 m off-axis passes as a point, and with up to 0.2 m of airframe."""
    task = _straight_through(0.3, clearance)
    assert bool(task.passed[0, 0]) is passed
    assert task.status[0] == (Status.FINISHED if passed else Status.MISSED)


def test_timeout_bounds_and_mid_course_start() -> None:
    course = slalom_gates(90.0)
    task = GateTask(course, num_envs=3, max_time=1.0,
                    bounds=course_bounds(course, course.starting_pos))
    start = np.vstack([course.starting_pos] * 3)
    task.reset(start, start_gate=np.array([0, 0, 7]))
    assert task.target.tolist() == [0, 0, 7]
    pos, yaw, valid = task.upcoming(3)
    np.testing.assert_allclose(pos[2, 0], course.gate_pos[7])
    assert valid.all()

    far = start.copy()
    far[1] += [0.0, 0.0, 100.0]           # env 1 leaves the box
    events = task.update(far, 0.5)
    assert task.status.tolist() == [Status.RUNNING, Status.OUT_OF_BOUNDS, Status.RUNNING]
    assert events.done.tolist() == [False, True, False]
    task.update(far, 1.0)                 # time limit for the rest
    assert task.status.tolist() == [Status.TIMEOUT, Status.OUT_OF_BOUNDS, Status.TIMEOUT]

    task.reset(start, start_gate=13)
    _, _, valid = task.upcoming(3)
    assert valid[0].tolist() == [True, True, False]

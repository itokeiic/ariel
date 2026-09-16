"""Test: what counts as flying through a gate.

A gate is a circular opening of radius `gate_size / 2`: at the instant the drone
crosses the gate plane, its offset from the centre *within that plane* --
lateral and vertical together -- must not exceed the radius. Decided 2026-09-11;
see docs/drone_morphology_gate_racing.md, section 3.8 and section 8 item 8.

Every speed in the report is scored this way, but the criterion exists twice:

* `crossing_report` in src/ariel/simulation/drone/gate_metrics.py, used by the
  report's figures, the flight videos and docs/tools/speed_semantics.py;
* inline in `rollout` in examples/spear/19_morphology_design_sweep.py, which
  decides completion for Table 1.

Pinned here:

* **Altitude counts.** The horizontal-only `GateChecker` in
  examples/d_drones/_ctrl_helpers.py passed a drone flying below the gates.
* **Circle, not square.** The evaluator's `GateChecker` tests a square, which
  admits a corner this criterion rejects.
* **The offset is projected into the plane.** The crossing is taken at the first
  sample past the plane, not interpolated, so a coarse step leaves an
  along-normal component that must not count against the drone.
* **The two implementations agree** on a real flight that both passes and misses
  gates.
"""

# Standard library
import importlib.util
import sys
from pathlib import Path

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.simulation.drone.gate_metrics import crossing_report, n_passed

REPO = Path(__file__).resolve().parents[3]
SWEEP = REPO / "examples" / "spear" / "19_morphology_design_sweep.py"
GATE = np.array([2.0, -1.0, -1.5])   # NED: 1.5 m above ground
SIZE = 1.0                            # radius 0.5 m


def _pass_through(yaw: float, lateral: float, vertical: float,
                  along: tuple[float, ...] = tuple(np.linspace(-1.0, 1.0, 41))):
    """A straight flight along the gate normal, offset within the gate plane.

    `lateral` is along the plane's horizontal axis, `vertical` along +z (down in
    NED). Returns the single-gate crossing report.
    """
    normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, 1.0])
    pos = np.array([GATE + s * normal + lateral * right + vertical * down
                    for s in along])
    (rep,) = crossing_report(pos, GATE[None, :], np.array([yaw]), SIZE)
    return rep


@pytest.mark.parametrize(("vertical", "passed"), [(0.45, True), (0.55, False),
                                                  (-0.55, False)])
def test_altitude_counts(vertical: float, passed: bool) -> None:
    """Centred laterally, too high or too low is a miss."""
    rep = _pass_through(0.0, 0.0, vertical)
    assert rep.crossed
    assert rep.lateral == pytest.approx(0.0, abs=1e-12)
    assert rep.vertical == pytest.approx(vertical)
    assert rep.passed is passed


@pytest.mark.parametrize(("offset", "passed"), [(0.35, True), (0.36, False)])
def test_opening_is_a_circle_not_a_square(offset: float, passed: bool) -> None:
    """On the diagonal the radius is reached at 0.3536 m per axis.

    A square of half-width 0.5 m would pass both cases.
    """
    rep = _pass_through(0.0, offset, offset)
    assert rep.offset == pytest.approx(np.hypot(offset, offset))
    assert rep.passed is passed


@pytest.mark.parametrize("yaw", np.radians([0.0, 60.0, -135.0, 180.0]))
@pytest.mark.parametrize(("lateral", "passed"), [(0.45, True), (-0.45, True),
                                                 (0.55, False)])
def test_offset_is_projected_into_a_yawed_gate(yaw: float, lateral: float,
                                               passed: bool) -> None:
    """Coarse samples land 0.3 m past the plane; that distance must not count.

    Unprojected, the 0.45 m case would sit 0.54 m from the centre and fail.
    """
    rep = _pass_through(yaw, lateral, 0.0, along=(-0.9, -0.3, 0.3, 0.9))
    assert rep.step == 2
    assert rep.offset == pytest.approx(abs(lateral))
    assert rep.passed is passed


@pytest.fixture(scope="module")
def sweep():
    """The sweep script, loaded with the flags that produced Table 1.

    It parses its arguments at import, so argv is set for the duration of the
    load (docs/drone_morphology_gate_racing.md, section 7.9).
    """
    argv = sys.argv
    sys.argv = [str(SWEEP), "--att-omega-n", "24", "--pos-omega-n", "2.0",
                "--pos-zeta", "1.0", "--feedforward", "--max-accel", "40"]
    try:
        spec = importlib.util.spec_from_file_location("morphology_design_sweep", SWEEP)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def test_sweep_inline_criterion_matches_crossing_report(sweep) -> None:
    """A real 90-degree flight, above the limit, scored both ways.

    The body is t = 36.81 deg, the 90-degree Table 1 winner until 2026-09-16; on
    the current plant it completes at 6.961 m/s. At 8.6 m/s it passes some gates
    and misses others, so a count of 0 or 15 from either implementation cannot
    hide a disagreement -- the guard below fails if that stops being true.
    `rollout` exposes counts, not per-gate verdicts, so counts are compared.
    """
    from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
    from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
    from ariel.simulation.tasks.slalom_course import slalom_gates

    genome = sweep.make_genome(np.radians(36.81))
    props = blueprint_to_propellers(
        spherical_angular_to_blueprint(genome, propsize=sweep.PROP_SIZE),
        convention="ned")
    course = slalom_gates(90.0, leg=sweep.args.leg, n_gates=sweep.args.n_gates,
                          gate_size=sweep.args.gate_size)
    out = sweep.rollout(props, course, 8.6, record=True)

    rep = crossing_report(out["log"]["pos"], course.gate_pos,
                          course.path_yaw[:len(course.gate_pos)], course.gate_size)
    n = n_passed(rep)
    assert 0 < n < len(course.gate_pos), (
        f"flight scored {n}/{len(course.gate_pos)}: no longer discriminates, "
        "pick a speed with both passes and misses")
    assert out["gates_strict"] == n
    assert out["completed_strict"] is False

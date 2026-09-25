"""The slalom's difficulty knob, measured on the reference the drone flies.

Pins the 2026-09-24 correction: difficulty used to be read from the circle
through three consecutive gates, which the drone does not fly. See
docs/drone_morphology_gate_racing.md section 5.
"""
from __future__ import annotations

import numpy as np
import pytest

from ariel.simulation.tasks.slalom_course import slalom_gates

TURNS = (60.0, 90.0, 120.0)


@pytest.fixture(scope="module")
def demands():
    return {t: slalom_gates(t).reference_demand(6.0) for t in TURNS}


def test_fixed_leg_keeps_course_length() -> None:
    """The reason for fixing the leg: length does not co-vary with difficulty."""
    lengths = {slalom_gates(t).path_length for t in (30.0, *TURNS)}
    assert len(lengths) == 1


def test_demand_rises_with_turn_angle(demands) -> None:
    median = [demands[t].a_lat_median for t in TURNS]
    peak = [demands[t].a_lat_peak for t in TURNS]
    assert np.all(np.diff(median) > 0), median
    assert np.all(np.diff(peak) > 0), peak


def test_gate_circle_understates_demand(demands) -> None:
    """v^2 over the three-gate circle is well below what the reference asks."""
    for t, d in demands.items():
        circle = 6.0 ** 2 / slalom_gates(t).gate_circle_radius
        assert d.a_lat_median > 1.3 * circle, (t, d.a_lat_median, circle)
        assert d.min_radius < slalom_gates(t).gate_circle_radius / 3.0


def test_last_gate_is_hardest(demands) -> None:
    """The uniform course is not one corner repeated: its finish is harder."""
    for t, d in demands.items():
        assert int(np.argmax(d.a_lat_per_gate)) == len(d.a_lat_per_gate) - 1, t
        assert d.a_lat_peak > 1.15 * d.a_lat_median, t

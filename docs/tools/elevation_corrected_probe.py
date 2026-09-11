"""Fly the `elevation` reference body with the thrust direction its genome asks for.

SUPERSEDED 2026-09-11: the decoder is fixed, so this probe's correction now equals
the decoder's own output. Kept only to reproduce the 2026-09-10 table in
docs/decoder_thrust_composition.md section 5.

The decoder compensates a motor's pose by subtracting the arm's Euler angles
(decoders.py:67). That cancels exactly for arm azimuth, which is a pure
z-rotation, but not for arm pitch, which introduces a non-commuting y-rotation.
The `elevation` body therefore decodes with its arm-0 thrust axis tilted 11.37
deg even though its genome sets motor_pitch = 0 for every rotor.

This reruns that body with the composition done correctly -- thrust taken
straight from (motor_az, motor_pitch), which is what the encoding documents --
so the case actually tests what verification item 3 meant to test: an
out-of-plane rotor POSITION, with axial thrust.

Everything else (course, tuning, completion criterion) matches the 2026-09-10
perturbation run.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/keiichi-ito/Documents/sandbox/ariel")

# The sweep parses args at import time, so bind the tuned operating point first.
sys.argv = [
    "19_morphology_design_sweep.py",
    "--completion", "strict",
    "--speed-lo", "2", "--speed-hi", "12", "--speed-cap", "25",
    "--speed-tol", "0.0625",
    "--att-omega-n", "24", "--pos-omega-n", "2.0", "--pos-zeta", "1.0",
    "--feedforward", "--max-accel", "40",
]
sys.path.insert(0, str(REPO / "examples" / "spear"))
spec = importlib.util.spec_from_file_location(
    "sweep", REPO / "examples" / "spear" / "19_morphology_design_sweep.py")
sweep = importlib.util.module_from_spec(spec)
sys.modules["sweep"] = sweep
spec.loader.exec_module(sweep)

from ariel.body_phenotypes.drone.backends import _rpy_to_R  # noqa: E402
from ariel.simulation.drone.reference_morphologies import (  # noqa: E402
    COL_MOTOR_PITCH,
    reference_morphologies,
)
from ariel.simulation.tasks.slalom_course import slalom_gates  # noqa: E402

COL_MOTOR_AZ = 3
_orig = sweep.blueprint_to_propellers
_genome_for_fix: list = []


def _corrected(bp, *, convention="z_up"):
    """As the real decoder, but with each thrust axis rebuilt from the genome.

    Direction comes from (motor_az, motor_pitch) alone -- absolute in body
    frame, independent of the arm -- then the same z-flip the backend applies
    for the NED convention.
    """
    props = _orig(bp, convention=convention)
    g = _genome_for_fix[0]
    rows = g[~np.isnan(g[:, 0])]
    z_sign = -1.0 if convention == "ned" else 1.0
    for row, p in zip(rows, props, strict=True):
        w = _rpy_to_R(0.0, float(row[COL_MOTOR_PITCH]), float(row[COL_MOTOR_AZ])) @ np.array([0, 0, 1.0])
        p["dir"] = [float(w[0]), float(w[1]), float(z_sign * w[2]), p["dir"][3]]
    return props


fam = reference_morphologies(arm_length=sweep.ARM_LENGTH)
m = fam["elevation"]
_genome_for_fix.append(m.genome)

# Confirm the correction does what it claims before spending the rollouts.
sweep.blueprint_to_propellers = _corrected
props = _corrected(
    sweep.spherical_angular_to_blueprint(m.genome, propsize=sweep.PROP_SIZE),
    convention="ned")
tilts = [np.degrees(np.arccos(np.clip(
    np.dot(np.asarray(p["dir"][:3], float) / np.linalg.norm(p["dir"][:3]),
           [0, 0, -1.0]), -1, 1))) for p in props]
print(f"corrected thrust tilts: {[f'{t:.2f}' for t in tilts]} deg")
print(f"arm-0 rotor position  : {np.round(props[0]['loc'], 4)}  (still out of plane)")
assert max(tilts) < 1e-6, "correction failed: thrust is still tilted"
print()

for turn in (60.0, 90.0, 120.0):
    course = slalom_gates(turn, leg=sweep.args.leg, n_gates=sweep.args.n_gates,
                          gate_size=sweep.args.gate_size)
    r = sweep.max_completing_speed(m.genome, course)
    print(f"{turn:5.0f} deg   max speed {r['max_speed']:7.3f} m/s   "
          f"trk {r['tracking_err']:6.3f}   clip_lo {100 * r['saturation_lo']:5.1f}%   "
          f"rollouts {r['n_rollouts']}")

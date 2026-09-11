"""Canonical airframes used to probe how morphology affects control.

One definition, shared by the sweep in ``examples/spear/19_morphology_design_sweep.py``
and by the tables in ``docs/drone_morphology_gate_racing.md``. They were
duplicated once and the copies disagreed, which is how an unreproducible row
reached the document.

The family is the bilaterally symmetric X quad plus three **single-arm**
perturbations of it, one degree of freedom each. Single-arm perturbations are
the point: a symmetric perturbation (alternating arm elevation, say) leaves the
products of inertia at exactly zero, so it fails to exercise the thing these
bodies exist to exercise.

.. note::
   ``elevation`` lifts rotor 0 about 5 cm out of plane with every thrust axis
   still along body z. Until the 2026-09-11 decoder fix it decoded with an
   11.37 deg tilt leaked from its arm pitch (docs/decoder_thrust_composition.md),
   which is why earlier flight tables report it as not flying.

   SUPERSEDED (2026-09-11): this warning read "``elevation`` decodes with its
   rotor's thrust axis tilted 11.37 deg out of the body z axis, and its flight
   results are therefore not physically meaningful. That tilt is a decoder
   defect, not a property of the perturbation". The decoder is fixed; the body
   decodes axial and its flight results mean what they say.
   SUPERSEDED (2026-09-10): before that it blamed the plant, fixed that day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ReferenceMorphology", "x_quad_genome", "reference_morphologies"]

X_QUAD_HALF_ANGLE = np.pi / 4
DEFAULT_ARM_LENGTH = 0.20
N_ARMS = 4

# Genome columns, matching spherical_angular_to_blueprint:
#   [arm length, arm azimuth, arm elevation, motor azimuth, motor pitch, spin]
COL_LENGTH, COL_ARM_AZ, COL_ARM_EL, COL_MOTOR_AZ, COL_MOTOR_PITCH, COL_SPIN = range(6)


@dataclass(frozen=True)
class ReferenceMorphology:
    """One airframe, with what it perturbs and whether its flight result means what it says."""

    key: str
    label: str
    genome: np.ndarray
    axial_thrust: bool
    note: str

    @property
    def flight_is_trustworthy(self) -> bool:
        """False when this body decodes with tilted thrust.

        Since the 2026-09-10 plant fix the plant simulates tilt correctly, so
        this no longer means the plant cannot fly the body. Since the
        2026-09-11 decoder fix every body in this family decodes axial; the
        ``elevation`` body's earlier tilt was leaked by the decoder
        (docs/decoder_thrust_composition.md).

        SUPERSEDED 2026-09-11: previously "False when the reduced plant would
        mis-simulate this body."
        """
        return self.axial_thrust


def x_quad_genome(half_angle: float = X_QUAD_HALF_ANGLE,
                  arm_length: float = DEFAULT_ARM_LENGTH) -> np.ndarray:
    """Bilaterally symmetric quad: arms at +/-t and 180 +/- t.

    Both mirror planes survive, so the products of inertia vanish identically
    and the body axes are principal axes.
    """
    az = np.array([half_angle, np.pi - half_angle,
                   np.pi + half_angle, -half_angle], dtype=float)
    g = np.zeros((N_ARMS, 6), dtype=float)
    g[:, COL_LENGTH] = arm_length
    g[:, COL_ARM_AZ] = az
    g[:, COL_MOTOR_AZ] = np.pi
    g[:, COL_SPIN] = np.arange(N_ARMS) % 2
    return g


def reference_morphologies(
    half_angle: float = X_QUAD_HALF_ANGLE,
    arm_length: float = DEFAULT_ARM_LENGTH,
    *,
    azimuth_deg: float = 30.0,
    lengthened: float = 0.26,
    elevation_deg: float = 15.0,
) -> dict[str, ReferenceMorphology]:
    """The X quad and three single-arm perturbations, keyed by short name."""
    base = x_quad_genome(half_angle, arm_length)

    azimuth = base.copy()
    azimuth[0, COL_ARM_AZ] += np.radians(azimuth_deg)

    length = base.copy()
    length[0, COL_LENGTH] = lengthened

    elevation = base.copy()
    elevation[0, COL_ARM_EL] = np.radians(elevation_deg)

    out = [
        ReferenceMorphology(
            "symmetric", "symmetric X quad (swept family)", base, True,
            "both mirror planes intact; products of inertia identically zero"),
        ReferenceMorphology(
            "azimuth", f"arm 0 azimuth +{azimuth_deg:g} deg", azimuth, True,
            "breaks the mirror plane containing z; rotors stay coplanar"),
        ReferenceMorphology(
            "length", f"arm 0 lengthened to {lengthened:g} m", length, True,
            "an arm at 45 deg loads Ixx and Iyy equally, so it grows Ixy "
            "while leaving Ixx - Iyy unchanged"),
        ReferenceMorphology(
            "elevation", f"arm 0 elevated {elevation_deg:g} deg (out of plane)",
            elevation, True,
            "lifts the rotor out of plane with every thrust axis still along "
            "body z (motor_pitch=0). Before the 2026-09-11 decoder fix it decoded "
            "with an 11.37 deg spurious tilt; see docs/decoder_thrust_composition.md"),
    ]
    return {m.key: m for m in out}

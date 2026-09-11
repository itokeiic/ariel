"""Test: the spherical decoder gives each motor the thrust direction its genome asks for.

``motor_az`` / ``motor_pitch`` are documented as ABSOLUTE -- a thrust direction
in the body frame, independent of the arm. The decoder used to cancel the arm
by subtracting its Euler angles, which is exact only for arm azimuth: an
elevated arm leaked into the thrust direction. The MJCF backend added angles
back and was wrong a different way, by twice the arm pitch. See
docs/decoder_thrust_composition.md.

Pinned here:

* planar genomes (arm pitch 0) decode to the legacy motor pose bit for bit --
  every published result lives there;
* for any arm pitch, the thrust axis equals ``R(0, motor_pitch, motor_az) @ +Z``
  in the propeller list and in the compiled MJCF model;
* the legacy composition fails that check, so the test discriminates;
* ``_R_to_rpy`` inverts ``_rpy_to_R``, including at the gimbal singularity;
* the ``elevation`` reference body decodes axial, rotor still out of plane.
"""

# Standard library
import math

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.body_phenotypes.drone.backends import (
    _R_to_rpy,
    _rpy_to_R,
    blueprint_to_mjspec,
    blueprint_to_propellers,
)
from ariel.body_phenotypes.drone.blueprint import ArmNode, MotorNode
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.reference_morphologies import reference_morphologies

EZ = np.array([0.0, 0.0, 1.0])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    # atan2(|a x b|, a . b), not acos(a . b): acos loses precision near 0 deg,
    # where its floating-point noise (~1e-6 deg) is the size of the threshold.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))


def _intended(row: np.ndarray) -> np.ndarray:
    return _rpy_to_R(0.0, float(row[4]), float(row[3])) @ EZ


def _legacy_axis(row: np.ndarray) -> np.ndarray:
    """Thrust axis under the pre-fix composition (angles subtracted)."""
    _, arm_az, arm_pitch, motor_az, motor_pitch, _ = (float(x) for x in row)
    r_arm = _rpy_to_R(0.0, -arm_pitch, arm_az)
    r_local = _rpy_to_R(0.0, motor_pitch - arm_pitch, motor_az - arm_az)
    return r_arm @ r_local @ EZ


def _random_elevated(rng: np.random.Generator, n: int = 4) -> np.ndarray:
    g = np.zeros((n, 6))
    g[:, 0] = rng.uniform(0.12, 0.25, n)
    g[:, 1] = rng.uniform(-math.pi, math.pi, n)
    g[:, 2] = rng.uniform(-math.radians(85), math.radians(85), n)
    g[:, 3] = rng.uniform(-math.pi, math.pi, n)
    g[:, 4] = rng.uniform(-math.radians(150), math.radians(150), n)
    g[:, 5] = rng.integers(0, 2, n)
    return g


def _motor_poses(bp) -> list:
    out = []
    for arm_id in bp.children(bp.root_id):
        if not isinstance(bp.payload(arm_id), ArmNode):
            continue
        for child in bp.children(arm_id):
            node = bp.payload(child)
            if isinstance(node, MotorNode):
                out.append(node.pose)
    return out


def test_r_to_rpy_inverts_rpy_to_r() -> None:
    """Round trip to machine precision, the gimbal singularity included."""
    rng = np.random.default_rng(0)
    cases = [tuple(rng.uniform(-math.pi, math.pi, 3)) for _ in range(500)]
    cases += [(0.3, math.pi / 2, -1.1), (-0.7, -math.pi / 2, 2.0)]
    for rpy in cases:
        r = _rpy_to_R(*rpy)
        np.testing.assert_allclose(_rpy_to_R(*_R_to_rpy(r)), r, atol=1e-12)


def test_planar_genomes_decode_bit_identically_to_legacy() -> None:
    """Arm pitch 0 is where the old subtraction was exact; nothing may move."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        g = np.zeros((6, 6))
        g[:, 0] = 0.2
        g[:, 1] = rng.uniform(-math.pi, math.pi, 6)
        g[:, 3] = rng.uniform(-math.pi, math.pi, 6)
        g[:, 4] = rng.uniform(-1.5, 1.5, 6)
        poses = _motor_poses(spherical_angular_to_blueprint(g))
        for pose, row in zip(poses, g, strict=True):
            legacy = (0.0, float(row[4]) - float(row[2]), float(row[3]) - float(row[1]))
            assert tuple(pose.rpy) == legacy


def test_propeller_thrust_axis_matches_genome_for_any_arm_pitch() -> None:
    rng = np.random.default_rng(2)
    worst_new = worst_legacy = 0.0
    for _ in range(200):
        g = _random_elevated(rng)
        props = blueprint_to_propellers(spherical_angular_to_blueprint(g), convention="z_up")
        for prop, row in zip(props, g, strict=True):
            worst_new = max(worst_new, _angle_deg(np.array(prop["dir"][:3]), _intended(row)))
            worst_legacy = max(worst_legacy, _angle_deg(_legacy_axis(row), _intended(row)))
    assert worst_new < 1e-6, f"decoded thrust off by {worst_new:.3g} deg"
    # Instrument check: the same assertion must fail on the old composition.
    assert worst_legacy > 1.0, "legacy composition passed; the test cannot discriminate"


def test_mjcf_thrust_axis_matches_genome_for_any_arm_pitch() -> None:
    mujoco = pytest.importorskip("mujoco")  # noqa: F841

    def axis(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(50):
        g = _random_elevated(rng)
        model = blueprint_to_mjspec(spherical_angular_to_blueprint(g)).compile()
        quats = [model.body_quat[i] for i in range(model.nbody) if "_motor_" in model.body(i).name]
        for q, row in zip(quats, g, strict=True):
            worst = max(worst, _angle_deg(axis(q), _intended(row)))
    assert worst < 1e-6, f"MJCF thrust axis off by {worst:.3g} deg"


def test_elevation_reference_body_decodes_axial() -> None:
    """Its genome asks for axial thrust; it used to decode 11.37 deg tilted."""
    m = reference_morphologies()["elevation"]
    assert m.axial_thrust
    props = blueprint_to_propellers(spherical_angular_to_blueprint(m.genome), convention="ned")
    for prop in props:
        np.testing.assert_allclose(prop["dir"][:3], [0.0, 0.0, -1.0], atol=1e-12)
    assert abs(props[0]["loc"][2]) > 0.05, "rotor 0 should still sit out of plane"

#!/usr/bin/env python3
"""Self-test for genome_to_usd.py. No pytest dependency -- run with `python3 test_genome_to_usd.py`.

Only exercises the pure-Python genome -> DroneSpec -> xacro path; xacro/IsaacLab
subprocess steps are not available in every environment and aren't covered here.
"""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

from genome_to_usd import (
    build_drone_xacro,
    decode_motor_spec,
    decode_rotor_position,
    decode_thrust_direction,
    genome_to_drone_spec,
)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def random_genome(rng: np.random.Generator, narms: int, max_narms: int) -> np.ndarray:
    genome = np.full((max_narms, 6), np.nan)
    for i in range(narms):
        magnitude = rng.uniform(0.055, 0.17)
        arm_rotation = rng.uniform(-np.pi, np.pi)
        arm_pitch = rng.uniform(-np.pi / 2, np.pi / 2)
        motor_rotation = rng.uniform(-np.pi, np.pi)
        motor_pitch = rng.uniform(-np.pi, np.pi)
        direction = rng.integers(0, 2)
        genome[i] = [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
    return genome


def test_round_trip_geometry(n_trials=500):
    rng = np.random.default_rng(42)
    max_err_pos = 0.0
    max_err_dir = 0.0
    for _ in range(n_trials):
        magnitude = rng.uniform(0.055, 0.17)
        arm_rotation = rng.uniform(-np.pi, np.pi)
        arm_pitch = rng.uniform(-np.pi / 2, np.pi / 2)
        motor_rotation = rng.uniform(-np.pi, np.pi)
        motor_pitch = rng.uniform(-np.pi, np.pi)
        direction = rng.integers(0, 2)
        row = np.array([magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction])

        motor = decode_motor_spec("arm_1", row)

        target_pos = np.array(decode_rotor_position(magnitude, arm_rotation, arm_pitch))
        r_arm = _rot_z(arm_rotation) @ _rot_y(-arm_pitch)
        got_pos = r_arm @ np.array(motor.motor_pose.xyz)
        max_err_pos = max(max_err_pos, float(np.max(np.abs(got_pos - target_pos))))

        target_dir = np.array(decode_thrust_direction(motor_rotation, motor_pitch))
        rp_roll, rp_pitch, rp_yaw = motor.rotor_pose.rpy
        got_dir = r_arm @ _rot_z(rp_yaw) @ _rot_y(rp_pitch) @ np.array([0.0, 0.0, 1.0])
        max_err_dir = max(max_err_dir, float(np.max(np.abs(got_dir - target_dir))))

    assert max_err_pos < 1e-9, f"position round-trip error too large: {max_err_pos}"
    assert max_err_dir < 1e-9, f"thrust-direction round-trip error too large: {max_err_dir}"
    print(f"  position max err: {max_err_pos:.2e}, direction max err: {max_err_dir:.2e}")


def test_edge_cases():
    cases = [
        (0.1, 0.0, math.pi / 2, 0.0, 0.0, 0),      # straight up
        (0.1, 0.0, -math.pi / 2, 0.0, 0.0, 1),     # straight down
        (0.1, math.pi, math.pi / 2, math.pi, math.pi, 0),   # boundary azimuth/pitch
        (0.1, -math.pi, -math.pi / 2, -math.pi, -math.pi, 1),
    ]
    for row in cases:
        motor = decode_motor_spec("arm_1", np.array(row))
        assert all(math.isfinite(v) for v in motor.arm_pose.rpy)
        assert all(math.isfinite(v) for v in motor.rotor_pose.rpy)
    print(f"  {len(cases)} edge cases decoded without error")


def test_xacro_structure_and_naming(n_arms=6, max_narms=8):
    rng = np.random.default_rng(7)
    genome = random_genome(rng, n_arms, max_narms)
    spec = genome_to_drone_spec(genome, "test_drone")
    assert len(spec.motors) == n_arms

    xacro_text = build_drone_xacro(spec)
    root = ET.fromstring(xacro_text)  # raises if malformed

    joint_names = [j.get("name") for j in root.findall("joint")]
    assert len(joint_names) == len(set(joint_names)), "duplicate joint names"

    arm_pattern = re.compile(r"base_to_arm_\d+_arm_joint")
    motor_pattern = re.compile(r"arm_\d+_arm_to_motor_joint")
    rotor_pattern = re.compile(r"arm_\d+_motor_to_rotor_joint")

    n_arm_joints = sum(1 for n in joint_names if arm_pattern.fullmatch(n))
    n_motor_joints = sum(1 for n in joint_names if motor_pattern.fullmatch(n))
    n_rotor_joints = sum(1 for n in joint_names if rotor_pattern.fullmatch(n))

    assert n_arm_joints == n_arms, f"expected {n_arms} arm joints, got {n_arm_joints}"
    assert n_motor_joints == n_arms, f"expected {n_arms} motor joints, got {n_motor_joints}"
    assert n_rotor_joints == n_arms, f"expected {n_arms} rotor joints, got {n_rotor_joints}"

    expected_names = {f"arm_{i + 1}" for i in range(n_arms)}
    actual_names = {m.name for m in spec.motors}
    assert actual_names == expected_names, f"contiguous naming broken: {actual_names}"

    print(f"  {n_arms} arms -> {len(joint_names)} joints, all names regex-valid and contiguous")


def test_nan_padding_contiguous_naming():
    genome = np.full((8, 6), np.nan)
    genome[1] = [0.1, 0.0, 0.0, 0.0, 0.0, 0]
    genome[3] = [0.12, 1.0, 0.2, 0.5, 0.3, 1]
    genome[6] = [0.09, -1.0, -0.2, -0.5, -0.3, 0]

    spec = genome_to_drone_spec(genome, "sparse_drone")
    names = [m.name for m in spec.motors]
    assert names == ["arm_1", "arm_2", "arm_3"], f"expected contiguous naming, got {names}"
    print(f"  sparse genome (rows 1,3,6 valid) -> contiguous names {names}")


if __name__ == "__main__":
    tests = [
        test_round_trip_geometry,
        test_edge_cases,
        test_xacro_structure_and_naming,
        test_nan_padding_contiguous_naming,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
    print("All tests passed.")

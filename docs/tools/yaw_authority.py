#!/usr/bin/env python3
"""Angular-acceleration authority per axis across the swept family.

    uv run --no-sync python docs/tools/yaw_authority.py

Writes docs/data/yaw_authority.md. Everything here is geometry -- no rollout --
so it is cheap and cannot drift with the plant or controller.

The point of the table is the yaw column. Sweeping arm azimuth rotates the
rotors in the plane, which changes thrust lever arms but leaves the sum of rotor
drag torques untouched, so yaw authority is *identical* across the family while
roll varies 2.6x. That is what makes the yaw-tracking lag common-mode: every
airframe suffers it equally, so it cannot bias the comparison between them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.plant import RotorGeometry, moment_allocation
from ariel.simulation.tasks.slalom_course import slalom_gates
from ariel.simulation.drone.controllers.trajectory_generation.bspline_gate_trajectory import (
    BSplineGateTrajectory,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "data" / "yaw_authority.md"
ARM_L, PROP_SIZE, PAYLOAD, N_ARMS, TOL, PROP_R = 0.20, 5, 0.667, 4, 0.1, 0.0635


def genome(half_angle: float) -> np.ndarray:
    az = np.array([half_angle, np.pi - half_angle, np.pi + half_angle, -half_angle])
    g = np.zeros((N_ARMS, 6))
    g[:, 0] = ARM_L
    g[:, 1] = az
    g[:, 3] = np.pi
    g[:, 5] = np.arange(N_ARMS) % 2
    return g


def authority(half_angle: float) -> tuple[float, float, float]:
    """(roll, pitch, yaw) row norms of I^-1 B_M, rad/s^2 per newton."""
    props = blueprint_to_propellers(
        spherical_angular_to_blueprint(genome(half_angle), propsize=PROP_SIZE),
        convention="z_up")
    cfg = DroneConfiguration(props, payload_mass=PAYLOAD)
    kf, km = cfg.propellers[0]["constants"]
    geom = RotorGeometry(
        positions=np.array([p["loc"] for p in props], dtype=float),
        axes=np.array([p["dir"][:3] for p in props], dtype=float),
        spins=np.array([1.0 if p["dir"][3] == "ccw" else -1.0 for p in props]))
    A = moment_allocation(geom, torque_to_thrust_ratio=km / kf,
                          inertia=cfg.inertia_matrix)
    return tuple(float(np.linalg.norm(A[i])) for i in range(3))


def commanded_yaw(turn: float, speed: float) -> tuple[float, float]:
    """(amplitude in deg, peak rate in deg/s) of the yaw the trajectory demands."""
    course = slalom_gates(turn, n_gates=15)
    traj = BSplineGateTrajectory(course.trajectory_config())
    traj.fit_offsets_to_gates()
    traj.startup_time = 3.0
    traj.total_time = course.traversal_time(speed, startup_time=3.0)
    ts = np.linspace(3.0, traj.total_time, 4001)
    vel = np.array([traj.evaluate(t)[1] for t in ts])
    psi = np.arctan2(vel[:, 1], vel[:, 0])
    rate = np.degrees(np.gradient(np.unwrap(psi), ts))
    return float(np.degrees(np.abs(psi)).max()), float(np.abs(rate).max())


def rows() -> list[dict]:
    c = (1.0 + TOL) * PROP_R / ARM_L
    lo, hi = np.arcsin(c), np.pi / 2 - np.arcsin(c)
    out = []
    for t in np.linspace(lo, hi, 7):
        r, p, y = authority(float(t))
        out.append({"t_deg": float(np.degrees(t)), "roll": r, "pitch": p, "yaw": y})
    return out


def table(rs: list[dict]) -> str:
    lines = ["| $t$ | $\\alpha_\\phi$ (roll) | $\\alpha_\\theta$ (pitch) | $\\alpha_\\psi$ (yaw) | roll / yaw |",
             "|---|---|---|---|---|"]
    for r in rs:
        lines.append(f"| {r['t_deg']:.2f}° | {r['roll']:.1f} | {r['pitch']:.1f} | "
                     f"{r['yaw']:.2f} | {r['roll'] / r['yaw']:.0f}x |")
    yaws = [r["yaw"] for r in rs]
    assert max(yaws) - min(yaws) < 1e-9, (
        "yaw authority is supposed to be constant across an in-plane azimuth "
        f"sweep; got {min(yaws)}..{max(yaws)}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rs = rows()
    md = table(rs)
    OUT.write_text(md)
    print(md)
    for turn, speed in ((60.0, 10.164), (90.0, 8.367), (120.0, 6.648)):
        amp, rate = commanded_yaw(turn, speed)
        print(f"  {turn:.0f}°: commanded yaw amplitude +/-{amp:.0f}°, peak rate {rate:.0f}°/s")
    print(f"\nwrote {OUT.relative_to(REPO)}")

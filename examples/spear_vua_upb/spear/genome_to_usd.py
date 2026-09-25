#!/usr/bin/env python3
"""Convert an airevolve Spherical Angular genome matrix into a spear USD drone.

Genome format (airevolve, SphericalAngularDroneGenomeHandler): a (narms, 6)
numpy array, one row per rotor:
    [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
Unused rows (narms < max_narms) are NaN-padded.

Pipeline: genome -> DroneSpec (reusing spear's dataclasses) -> xacro text
-> xacro CLI -> URDF -> IsaacLab convert_urdf.py -> USD.
"""
import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np

_gen_spec = importlib.util.spec_from_file_location(
    "gen_novel_morphology",
    Path(__file__).resolve().parent / "02_generate_novel_morphology.py",
)
_gen = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(_gen)

Pose = _gen.Pose
JointSpec = _gen.JointSpec
MotorSpec = _gen.MotorSpec
Defaults = _gen.Defaults
DroneSpec = _gen.DroneSpec
build_drone_xacro = _gen.build_drone_xacro
convert_xacro_to_urdf = _gen.convert_xacro_to_urdf
convert_urdf_to_usd = _gen.convert_urdf_to_usd
find_isaaclab_root = _gen.find_isaaclab_root
clear_directory = _gen.clear_directory


def load_genome(path: Path) -> np.ndarray:
    return np.load(path)


def valid_arm_mask(genome: np.ndarray) -> np.ndarray:
    return ~np.isnan(genome[:, 0])


def decode_rotor_position(magnitude: float, arm_rotation: float, arm_pitch: float) -> tuple[float, float, float]:
    """Elevation-convention spherical -> Cartesian (0 = horizontal, +pi/2 = straight up).

    Deliberately NOT airevolve's arm_conversions.spherical_to_cartesian, which uses the
    polar-from-Z convention and disagrees with this formula by up to ~0.09m over the
    genome's documented ranges -- that helper is inconsistent with how arm_pitch is used
    everywhere else in airevolve (drone_gate_env.py, inspection_tools/utils.py).
    """
    x = magnitude * math.cos(arm_pitch) * math.cos(arm_rotation)
    y = magnitude * math.cos(arm_pitch) * math.sin(arm_rotation)
    z = magnitude * math.sin(arm_pitch)
    return (x, y, z)


def decode_thrust_direction(motor_rotation: float, motor_pitch: float) -> tuple[float, float, float]:
    """Unit thrust-axis direction in base_link frame from the genome's motor angles."""
    x = math.cos(motor_rotation) * math.sin(motor_pitch)
    y = math.sin(motor_rotation) * math.sin(motor_pitch)
    z = math.cos(motor_pitch)
    return (x, y, z)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def decode_motor_spec(
    name: str,
    row: np.ndarray,
    *,
    arm_tilt_limit: float = 0.35,
    motor_tilt_limit: float = 0.30,
    arm_tilt_effort: float = 8.0,
    arm_tilt_velocity: float = 3.0,
    motor_tilt_effort: float = 6.0,
    motor_tilt_velocity: float = 3.0,
    rotor_effort: float = 2.5,
    rotor_velocity: float = 400.0,
    rotor_mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.016),
    rotor_radius: float = 0.0635,
    propsize: float = 2.0,
    pitch: float = 0.045,
    blades: int = 2,
) -> MotorSpec:
    magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction = row

    # Position: arm tilts about local Y (yaw baked into static rpy.z, elevation into rpy.y),
    # then the motor sits at the tip via a local +X offset of length `magnitude`.
    arm_pose = Pose(xyz=(0.0, 0.0, 0.0), rpy=(0.0, -arm_pitch, arm_rotation))
    motor_pose_xyz = (magnitude, 0.0, 0.0)

    # Orientation: arm_joint (Y) and motor_joint (X) hinge axes can't reach an arbitrary
    # target direction on their own, so the residual is folded into rotor_pose.rpy (a
    # static-only knob -- spin phase about the resolved Z axis is physically irrelevant).
    # Do not "fix" this by moving the tilt onto motor_pose.rpy; see plan notes.
    r_arm = _rot_z(arm_rotation) @ _rot_y(-arm_pitch)
    target_dir = np.array(decode_thrust_direction(motor_rotation, motor_pitch))
    local_target = r_arm.T @ target_dir
    lx, ly, lz = local_target
    rotor_pitch = math.acos(max(-1.0, min(1.0, lz)))
    rotor_yaw = math.atan2(ly, lx)

    arm_joint = JointSpec(
        type="revolute", axis=(0.0, 1.0, 0.0),
        lower=-arm_tilt_limit, upper=arm_tilt_limit,
        effort=arm_tilt_effort, velocity=arm_tilt_velocity, damping=0.08,
    )
    motor_joint = JointSpec(
        type="revolute", axis=(1.0, 0.0, 0.0),
        lower=-motor_tilt_limit, upper=motor_tilt_limit,
        effort=motor_tilt_effort, velocity=motor_tilt_velocity, damping=0.05,
    )
    rotor_joint = JointSpec(
        type="continuous", axis=(0.0, 0.0, 1.0),
        effort=rotor_effort, velocity=rotor_velocity, damping=0.001,
    )

    spin = "cw" if direction >= 0.5 else "ccw"

    return MotorSpec(
        name=name,
        arm_length=float(magnitude),
        arm_pose=arm_pose,
        motor_pose=Pose(xyz=motor_pose_xyz, rpy=(0.0, 0.0, 0.0)),
        rotor_pose=Pose(xyz=rotor_mount_xyz, rpy=(0.0, rotor_pitch, rotor_yaw)),
        spin=spin,
        propsize=propsize,
        rotor_radius=rotor_radius,
        pitch=pitch,
        blades=blades,
        arm_joint=arm_joint,
        motor_joint=motor_joint,
        rotor_joint=rotor_joint,
    )


def genome_to_drone_spec(
    genome: np.ndarray,
    name: str,
    *,
    core_mass: float = 0.4,
    core_radius: float = 0.05,
    core_thickness: float = 0.01,
    defaults: Defaults | None = None,
    **motor_kwargs,
) -> DroneSpec:
    mask = valid_arm_mask(genome)
    valid_rows = genome[mask]
    if len(valid_rows) == 0:
        raise ValueError("Genome has no valid (non-NaN) rotor rows")

    motors = [
        decode_motor_spec(f"arm_{i + 1}", row, **motor_kwargs)
        for i, row in enumerate(valid_rows)
    ]

    return DroneSpec(
        name=name,
        core_mass=core_mass,
        core_radius=core_radius,
        core_thickness=core_thickness,
        motors=motors,
        defaults=defaults or Defaults(),
    )


def genome_to_usd(
    genome_path: Path,
    output_name: str,
    output_dir: Path,
    isaaclab_root: Path | None = None,
    skip_urdf_and_usd: bool = False,
    **drone_spec_kwargs,
) -> dict[str, Path]:
    genome = load_genome(genome_path)
    spec = genome_to_drone_spec(genome, output_name, **drone_spec_kwargs)

    xacro_dir = output_dir / "xacro"
    urdf_dir = output_dir / "urdf"
    usd_dir = output_dir / "usd"
    xacro_dir.mkdir(parents=True, exist_ok=True)

    xacro_path = xacro_dir / f"{output_name}.xacro"
    xacro_path.write_text(build_drone_xacro(spec), encoding="utf-8")

    result = {"xacro": xacro_path}
    if skip_urdf_and_usd:
        return result

    urdf_path = urdf_dir / f"{output_name}.urdf"
    usd_path = usd_dir / f"{output_name}.usd"
    convert_xacro_to_urdf(xacro_path, urdf_path)
    result["urdf"] = urdf_path

    root = isaaclab_root or find_isaaclab_root(Path(__file__).resolve().parent)
    convert_urdf_to_usd(urdf_path, usd_path, root)
    result["usd"] = usd_path

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome-npy", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "conversion")
    parser.add_argument("--arm-tilt-limit", type=float, default=0.35)
    parser.add_argument("--motor-tilt-limit", type=float, default=0.30)
    parser.add_argument("--rotor-radius", type=float, default=0.0635)
    parser.add_argument("--propsize", type=float, default=2.0)
    parser.add_argument("--pitch", type=float, default=0.045)
    parser.add_argument("--blades", type=int, default=2)
    parser.add_argument("--core-mass", type=float, default=0.4)
    parser.add_argument("--core-radius", type=float, default=0.05)
    parser.add_argument("--core-thickness", type=float, default=0.01)
    parser.add_argument("--isaaclab-root", type=Path, default=None)
    parser.add_argument("--skip-urdf-and-usd", action="store_true",
                         help="Only write the .xacro; skip xacro/IsaacLab subprocess steps.")
    args = parser.parse_args()

    result = genome_to_usd(
        genome_path=args.genome_npy,
        output_name=args.output_name,
        output_dir=args.output_dir,
        isaaclab_root=args.isaaclab_root,
        skip_urdf_and_usd=args.skip_urdf_and_usd,
        core_mass=args.core_mass,
        core_radius=args.core_radius,
        core_thickness=args.core_thickness,
        arm_tilt_limit=args.arm_tilt_limit,
        motor_tilt_limit=args.motor_tilt_limit,
        rotor_radius=args.rotor_radius,
        propsize=args.propsize,
        pitch=args.pitch,
        blades=args.blades,
    )

    for kind, path in result.items():
        print(f"Wrote {kind}: {path}")


if __name__ == "__main__":
    main()

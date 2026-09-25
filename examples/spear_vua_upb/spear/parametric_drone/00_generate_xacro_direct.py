#!/usr/bin/env python3
from dataclasses import dataclass, field
from pathlib import Path
import math
import random
import shutil
import subprocess


@dataclass
class Pose:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass
class MotorSpec:
    name: str
    arm_length: float
    arm_pose: Pose
    motor_pose: Pose
    rotor_pose: Pose
    spin: str
    propsize: float
    rotor_radius: float
    pitch: float
    blades: int


@dataclass
class Defaults:
    arm_radius: float = 0.008
    arm_mass_per_m: float = 0.18
    motor_radius: float = 0.014
    motor_height: float = 0.02
    motor_mass: float = 0.06
    rotor_hub_radius: float = 0.01
    rotor_hub_height: float = 0.006
    rotor_hub_mass: float = 0.01
    rotor_visual_thickness: float = 0.002
    rotor_visual_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotor_mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.016)
    rotor_mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DroneSpec:
    name: str
    core_mass: float
    core_radius: float
    core_thickness: float
    motors: list[MotorSpec] = field(default_factory=list)
    defaults: Defaults = field(default_factory=Defaults)


def fmt_vec(v):
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


def clear_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def build_drone_xacro(spec: DroneSpec) -> str:
    d = spec.defaults
    parts = []
    parts.append('<?xml version="1.0"?>')
    parts.append(f'<robot name="{spec.name}" xmlns:xacro="http://www.ros.org/wiki/xacro">')
    parts.append('')
    parts.append('  <material name="mat_core"><color rgba="0.15 0.15 0.18 1.0"/></material>')
    parts.append('  <material name="mat_arm"><color rgba="0.22 0.22 0.24 1.0"/></material>')
    parts.append('  <material name="mat_motor"><color rgba="0.70 0.70 0.72 1.0"/></material>')
    parts.append('  <material name="mat_rotor_ccw"><color rgba="0.10 0.10 0.10 0.85"/></material>')
    parts.append('  <material name="mat_rotor_cw"><color rgba="0.05 0.25 0.55 0.85"/></material>')
    parts.append('')
    parts.append('  <xacro:macro name="inertial_box" params="mass x y z">')
    parts.append('    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="${mass}"/><inertia ixx="${mass*(y*y + z*z)/12.0}" ixy="0.0" ixz="0.0" iyy="${mass*(x*x + z*z)/12.0}" iyz="0.0" izz="${mass*(x*x + y*y)/12.0}"/></inertial>')
    parts.append('  </xacro:macro>')
    parts.append('  <xacro:macro name="inertial_cylinder_z" params="mass radius length">')
    parts.append('    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="${mass}"/><inertia ixx="${mass*(3*radius*radius + length*length)/12.0}" ixy="0.0" ixz="0.0" iyy="${mass*(3*radius*radius + length*length)/12.0}" iyz="0.0" izz="${0.5*mass*radius*radius}"/></inertial>')
    parts.append('  </xacro:macro>')
    parts.append('')
    parts.append('  <link name="base_link">')
    parts.append(f'    <visual><geometry><cylinder radius="{spec.core_radius:.6f}" length="{spec.core_thickness:.6f}"/></geometry><material name="mat_core"/></visual>')
    parts.append(f'    <collision><geometry><cylinder radius="{spec.core_radius:.6f}" length="{spec.core_thickness:.6f}"/></geometry></collision>')
    parts.append(f'    <xacro:inertial_cylinder_z mass="{spec.core_mass:.6f}" radius="{spec.core_radius:.6f}" length="{spec.core_thickness:.6f}"/>')
    parts.append('  </link>')
    parts.append('')
    parts.append('  <link name="imu_link">')
    parts.append('    <visual><origin xyz="0 0 0.012" rpy="0 0 0"/><geometry><box size="0.012 0.012 0.004"/></geometry><material name="mat_motor"/></visual>')
    parts.append('    <collision><origin xyz="0 0 0.012" rpy="0 0 0"/><geometry><box size="0.012 0.012 0.004"/></geometry></collision>')
    parts.append('    <xacro:inertial_box mass="0.005" x="0.012" y="0.012" z="0.004"/>')
    parts.append('  </link>')
    parts.append('  <joint name="imu_joint" type="fixed"><parent link="base_link"/><child link="imu_link"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>')
    parts.append('')

    for m in spec.motors:
        rotor_mat = 'mat_rotor_ccw' if m.spin.lower() == 'ccw' else 'mat_rotor_cw'
        arm_mass = m.arm_length * d.arm_mass_per_m

        parts.append(f'  <!-- {m.name}: spin={m.spin}, propsize={m.propsize}, pitch={m.pitch}, blades={m.blades}, arm_rpy={m.arm_pose.rpy}, motor_xyz={m.motor_pose.xyz}, rotor_rpy={m.rotor_pose.rpy} -->')
        parts.append(f'  <link name="{m.name}_arm_link">')
        parts.append(f'    <visual><origin xyz="{m.arm_length/2:.6f} 0 0" rpy="0 1.57079632679 0"/><geometry><cylinder radius="{d.arm_radius:.6f}" length="{m.arm_length:.6f}"/></geometry><material name="mat_arm"/></visual>')
        parts.append(f'    <collision><origin xyz="{m.arm_length/2:.6f} 0 0" rpy="0 1.57079632679 0"/><geometry><cylinder radius="{d.arm_radius:.6f}" length="{m.arm_length:.6f}"/></geometry></collision>')
        parts.append(f'    <xacro:inertial_cylinder_z mass="{arm_mass:.6f}" radius="{d.arm_radius:.6f}" length="{m.arm_length:.6f}"/>')
        parts.append('  </link>')
        parts.append(f'  <joint name="base_to_{m.name}_arm_joint" type="fixed"><parent link="base_link"/><child link="{m.name}_arm_link"/><origin xyz="{fmt_vec(m.arm_pose.xyz)}" rpy="{fmt_vec(m.arm_pose.rpy)}"/></joint>')
        parts.append('')
        parts.append(f'  <link name="{m.name}_motor_link">')
        parts.append(f'    <visual><geometry><cylinder radius="{d.motor_radius:.6f}" length="{d.motor_height:.6f}"/></geometry><material name="mat_motor"/></visual>')
        parts.append(f'    <collision><geometry><cylinder radius="{d.motor_radius:.6f}" length="{d.motor_height:.6f}"/></geometry></collision>')
        parts.append(f'    <xacro:inertial_cylinder_z mass="{d.motor_mass:.6f}" radius="{d.motor_radius:.6f}" length="{d.motor_height:.6f}"/>')
        parts.append('  </link>')
        parts.append(f'  <joint name="{m.name}_arm_to_motor_joint" type="fixed"><parent link="{m.name}_arm_link"/><child link="{m.name}_motor_link"/><origin xyz="{fmt_vec(m.motor_pose.xyz)}" rpy="{fmt_vec(m.motor_pose.rpy)}"/></joint>')
        parts.append('')
        parts.append(f'  <link name="{m.name}_rotor_link">')
        parts.append(f'    <visual><origin xyz="0 0 0" rpy="{fmt_vec(d.rotor_visual_rpy)}"/><geometry><cylinder radius="{m.rotor_radius:.6f}" length="{d.rotor_visual_thickness:.6f}"/></geometry><material name="{rotor_mat}"/></visual>')
        parts.append(f'    <collision><origin xyz="0 0 0" rpy="{fmt_vec(d.rotor_visual_rpy)}"/><geometry><cylinder radius="{m.rotor_radius:.6f}" length="{d.rotor_visual_thickness:.6f}"/></geometry></collision>')
        parts.append(f'    <xacro:inertial_cylinder_z mass="{d.rotor_hub_mass:.6f}" radius="{d.rotor_hub_radius:.6f}" length="{d.rotor_hub_height:.6f}"/>')
        parts.append('  </link>')
        parts.append(f'  <joint name="{m.name}_motor_to_rotor_joint" type="fixed"><parent link="{m.name}_motor_link"/><child link="{m.name}_rotor_link"/><origin xyz="{fmt_vec(m.rotor_pose.xyz)}" rpy="{fmt_vec(m.rotor_pose.rpy)}"/></joint>')
        parts.append('')

    parts.append('</robot>')
    return '\n'.join(parts)


def clamp_tilt_components(roll: float, pitch: float, max_tilt: float) -> tuple[float, float]:
    tilt_mag = math.sqrt(roll * roll + pitch * pitch)
    if tilt_mag <= max_tilt or tilt_mag == 0.0:
        return roll, pitch
    scale = max_tilt / tilt_mag
    return roll * scale, pitch * scale


def sample_rotor_rpy(
    randomize_rotor_tilt: bool,
    rotor_tilt_magnitude: float,
    rotor_tilt_yaw_range: float,
    max_rotor_tilt: float,
    default_rpy: tuple[float, float, float],
) -> tuple[float, float, float]:
    if randomize_rotor_tilt and rotor_tilt_magnitude > 0.0:
        tilt_dir = random.uniform(0.0, 2.0 * math.pi)
        raw_roll = rotor_tilt_magnitude * math.cos(tilt_dir)
        raw_pitch = rotor_tilt_magnitude * math.sin(tilt_dir)
        rotor_roll, rotor_pitch = clamp_tilt_components(raw_roll, raw_pitch, max_rotor_tilt)
        rotor_yaw = random.uniform(-rotor_tilt_yaw_range, rotor_tilt_yaw_range)
        return (rotor_roll, rotor_pitch, rotor_yaw)
    return default_rpy


def make_symmetric_drone(
    name: str,
    num_motors: int,
    arm_length: float,
    rotor_radius: float,
    core_mass: float = 0.4,
    core_radius: float = 0.05,
    core_thickness: float = 0.01,
    pitch: float = 0.045,
    blades: int = 2,
    propsize: float = 2.0,
    first_spin: str = 'ccw',
    alternate_spin: bool = True,
    arm_z_offset: float = 0.0,
    arm_roll_tilt: float = 0.0,
    arm_pitch_tilt: float = 0.0,
    motor_roll: float = 0.0,
    motor_pitch: float = 0.0,
    motor_yaw_offset: float = 0.0,
    motor_z_offset: float = 0.02,
    rotor_mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.016),
    rotor_mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    randomize_rotor_tilt: bool = False,
    rotor_tilt_magnitude: float = 0.0,
    rotor_tilt_yaw_range: float = 0.4,
    max_rotor_tilt: float = 0.20,
    random_seed: int | None = None,
    defaults: Defaults | None = None,
) -> DroneSpec:
    if not (3 <= num_motors <= 8):
        raise ValueError("num_motors must be between 3 and 8")
    if random_seed is not None:
        random.seed(random_seed)

    defaults = defaults or Defaults(
        rotor_mount_xyz=rotor_mount_xyz,
        rotor_mount_rpy=rotor_mount_rpy,
        rotor_visual_rpy=(0.0, 0.0, 0.0),
    )

    motors = []
    for i in range(num_motors):
        yaw = 2.0 * math.pi * i / num_motors
        spin = first_spin if (not alternate_spin or i % 2 == 0) else ('cw' if first_spin == 'ccw' else 'ccw')
        rotor_rpy = sample_rotor_rpy(
            randomize_rotor_tilt=randomize_rotor_tilt,
            rotor_tilt_magnitude=rotor_tilt_magnitude,
            rotor_tilt_yaw_range=rotor_tilt_yaw_range,
            max_rotor_tilt=max_rotor_tilt,
            default_rpy=defaults.rotor_mount_rpy,
        )
        motors.append(
            MotorSpec(
                name=f'arm_{i+1}',
                arm_length=arm_length,
                arm_pose=Pose(
                    xyz=(0.0, 0.0, arm_z_offset),
                    rpy=(arm_roll_tilt, arm_pitch_tilt, yaw),
                ),
                motor_pose=Pose(
                    xyz=(arm_length, 0.0, motor_z_offset),
                    rpy=(motor_roll, motor_pitch, motor_yaw_offset),
                ),
                rotor_pose=Pose(
                    xyz=defaults.rotor_mount_xyz,
                    rpy=rotor_rpy,
                ),
                spin=spin,
                propsize=propsize,
                rotor_radius=rotor_radius,
                pitch=pitch,
                blades=blades,
            )
        )

    return DroneSpec(
        name=name,
        core_mass=core_mass,
        core_radius=core_radius,
        core_thickness=core_thickness,
        motors=motors,
        defaults=defaults,
    )


def make_asymmetric_drone(
    name: str,
    num_motors: int,
    core_mass: float = 0.4,
    core_radius: float = 0.05,
    core_thickness: float = 0.01,
    pitch: float = 0.045,
    blades: int = 2,
    propsize: float = 2.0,
    first_spin: str = 'ccw',
    alternate_spin: bool = True,
    min_arm_length: float = 0.14,
    max_arm_length: float = 0.24,
    arm_z_offset_range: tuple[float, float] = (-0.01, 0.01),
    arm_roll_tilt_range: tuple[float, float] = (-0.10, 0.10),
    arm_pitch_tilt_range: tuple[float, float] = (-0.10, 0.10),
    motor_roll_range: tuple[float, float] = (-0.08, 0.08),
    motor_pitch_range: tuple[float, float] = (-0.08, 0.08),
    motor_yaw_offset_range: tuple[float, float] = (-0.20, 0.20),
    motor_z_offset_range: tuple[float, float] = (0.015, 0.03),
    rotor_radius_range: tuple[float, float] = (0.055, 0.07),
    rotor_mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.016),
    rotor_mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    randomize_rotor_tilt: bool = True,
    rotor_tilt_magnitude: float = 0.18,
    rotor_tilt_yaw_range: float = 0.35,
    max_rotor_tilt: float = 0.20,
    min_angular_separation: float = 0.55,
    random_seed: int | None = None,
    defaults: Defaults | None = None,
) -> DroneSpec:
    if not (3 <= num_motors <= 8):
        raise ValueError("num_motors must be between 3 and 8")
    if random_seed is not None:
        random.seed(random_seed)

    defaults = defaults or Defaults(
        rotor_mount_xyz=rotor_mount_xyz,
        rotor_mount_rpy=rotor_mount_rpy,
        rotor_visual_rpy=(0.0, 0.0, 0.0),
    )

    if min_angular_separation * num_motors >= 2.0 * math.pi:
        raise ValueError("min_angular_separation is too large for the requested num_motors")

    yaws = []
    for _ in range(num_motors):
        for _attempt in range(1000):
            candidate = random.uniform(0.0, 2.0 * math.pi)
            if all(abs((candidate - y + math.pi) % (2.0 * math.pi) - math.pi) >= min_angular_separation for y in yaws):
                yaws.append(candidate)
                break
        else:
            raise RuntimeError("Failed to sample asymmetric yaw layout with requested separation")
    yaws.sort()

    motors = []
    for i, yaw in enumerate(yaws):
        spin = first_spin if (not alternate_spin or i % 2 == 0) else ('cw' if first_spin == 'ccw' else 'ccw')
        arm_length = random.uniform(min_arm_length, max_arm_length)
        arm_z_offset = random.uniform(*arm_z_offset_range)
        arm_roll_tilt = random.uniform(*arm_roll_tilt_range)
        arm_pitch_tilt = random.uniform(*arm_pitch_tilt_range)
        motor_roll = random.uniform(*motor_roll_range)
        motor_pitch = random.uniform(*motor_pitch_range)
        motor_yaw_offset = random.uniform(*motor_yaw_offset_range)
        motor_z_offset = random.uniform(*motor_z_offset_range)
        rotor_radius = random.uniform(*rotor_radius_range)
        rotor_rpy = sample_rotor_rpy(
            randomize_rotor_tilt=randomize_rotor_tilt,
            rotor_tilt_magnitude=rotor_tilt_magnitude,
            rotor_tilt_yaw_range=rotor_tilt_yaw_range,
            max_rotor_tilt=max_rotor_tilt,
            default_rpy=defaults.rotor_mount_rpy,
        )

        motors.append(
            MotorSpec(
                name=f'arm_{i+1}',
                arm_length=arm_length,
                arm_pose=Pose(
                    xyz=(0.0, 0.0, arm_z_offset),
                    rpy=(arm_roll_tilt, arm_pitch_tilt, yaw),
                ),
                motor_pose=Pose(
                    xyz=(arm_length, 0.0, motor_z_offset),
                    rpy=(motor_roll, motor_pitch, motor_yaw_offset),
                ),
                rotor_pose=Pose(
                    xyz=defaults.rotor_mount_xyz,
                    rpy=rotor_rpy,
                ),
                spin=spin,
                propsize=propsize,
                rotor_radius=rotor_radius,
                pitch=pitch,
                blades=blades,
            )
        )

    return DroneSpec(
        name=name,
        core_mass=core_mass,
        core_radius=core_radius,
        core_thickness=core_thickness,
        motors=motors,
        defaults=defaults,
    )


def convert_xacro_to_urdf(xacro_path: Path, urdf_path: Path):
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    with urdf_path.open('w', encoding='utf-8') as f:
        subprocess.run(
            ['xacro', str(xacro_path)],
            check=True,
            stdout=f,
            text=True,
        )


def convert_urdf_to_usd(urdf_path: Path, usd_path: Path, isaaclab_root: Path):
    usd_path.parent.mkdir(parents=True, exist_ok=True)

    converter_script = isaaclab_root / "scripts" / "tools" / "convert_urdf.py"
    isaaclab_sh = isaaclab_root / "isaaclab.sh"

    subprocess.run(
        [
            str(isaaclab_sh),
            "-p",
            str(converter_script),
            str(urdf_path),
            str(usd_path),
            "--merge-joints",
            "--joint-stiffness", "0.0",
            "--joint-damping", "0.0",
            "--joint-target-type", "none",
        ],
        check=True,
        text=True,
    )


def find_isaaclab_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "isaaclab.sh").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find IsaacLab root from {start}. "
        f"Expected a parent directory containing 'isaaclab.sh'."
    )


def main():
    base_dir = Path(__file__).resolve().parent
    isaaclab_root = find_isaaclab_root(base_dir)

    conversion_dir = base_dir / "conversion"
    urdf_dir = conversion_dir / "urdf"
    usd_dir = conversion_dir / "usd"

    clear_directory(urdf_dir)
    clear_directory(usd_dir)

    layout_mode = "asymmetric"  # choose: "symmetric" or "asymmetric"
    num_motors = 7

    if layout_mode == "symmetric":
        spec = make_symmetric_drone(
            name=f"symmetric_{num_motors}rotor_drone",
            num_motors=num_motors,
            arm_length=0.20,
            rotor_radius=0.0635,
            alternate_spin=True,
            arm_pitch_tilt=0.0,
            motor_roll=0.0,
            motor_pitch=0.0,
            motor_yaw_offset=0.0,
            motor_z_offset=0.02,
            rotor_mount_xyz=(0.0, 0.0, 0.016),
            rotor_mount_rpy=(0.0, 0.0, 0.0),
            randomize_rotor_tilt=True,
            rotor_tilt_magnitude=0.18,
            rotor_tilt_yaw_range=0.35,
            max_rotor_tilt=0.20,
            random_seed=42,
        )
    elif layout_mode == "asymmetric":
        spec = make_asymmetric_drone(
            name=f"asymmetric_{num_motors}rotor_drone",
            num_motors=num_motors,
            min_arm_length=0.15,
            max_arm_length=0.24,
            motor_z_offset_range=(0.016, 0.03),
            rotor_radius_range=(0.055, 0.07),
            randomize_rotor_tilt=True,
            rotor_tilt_magnitude=0.16,
            rotor_tilt_yaw_range=0.30,
            max_rotor_tilt=0.18,
            min_angular_separation=0.75,
            random_seed=42,
        )
    else:
        raise ValueError(f"Unknown layout_mode: {layout_mode}")

    xacro_path = base_dir / f"{spec.name}.xacro"
    urdf_path = urdf_dir / f"{spec.name}.urdf"
    usd_path = usd_dir / f"{spec.name}.usd"

    xacro_text = build_drone_xacro(spec)
    xacro_path.write_text(xacro_text, encoding="utf-8")

    print(f"Using IsaacLab root: {isaaclab_root}")
    print(f"Using launcher: {isaaclab_root / 'isaaclab.sh'}")
    print(f"Using converter: {isaaclab_root / 'scripts' / 'tools' / 'convert_urdf.py'}")
    print(f"Layout mode: {layout_mode}")

    for m in spec.motors:
        print(
            f"{m.name}: arm_length={m.arm_length:.3f}, arm_rpy={m.arm_pose.rpy}, "
            f"motor_xyz={m.motor_pose.xyz}, rotor_radius={m.rotor_radius:.3f}, rotor_rpy={m.rotor_pose.rpy}"
        )

    convert_xacro_to_urdf(xacro_path, urdf_path)
    convert_urdf_to_usd(urdf_path, usd_path, isaaclab_root)

    print(f"Wrote Xacro to: {xacro_path}")
    print(f"Wrote URDF to:  {urdf_path}")
    print(f"Wrote USD to:   {usd_path}")
    print(f"Motor count: {len(spec.motors)}")
    print(f"Robot name: {spec.name}")


if __name__ == "__main__":
    main()
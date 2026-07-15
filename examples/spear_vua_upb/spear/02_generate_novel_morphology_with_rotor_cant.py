
#!/usr/bin/env python3
from dataclasses import dataclass, field
from pathlib import Path
import math
import json
import os
import random
import shutil
import subprocess

import numpy as np


@dataclass
class Pose:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass
class JointSpec:
    type: str = "fixed"          # "fixed", "revolute", "continuous"
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float = -0.4
    upper: float = 0.4
    effort: float = 5.0
    velocity: float = 4.0
    damping: float = 0.05
    friction: float = 0.0


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
    arm_joint: JointSpec = field(default_factory=lambda: JointSpec(type="fixed"))
    motor_joint: JointSpec = field(default_factory=lambda: JointSpec(type="fixed"))
    rotor_joint: JointSpec = field(default_factory=lambda: JointSpec(type="continuous", axis=(0, 0, 1)))


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


def joint_tag(name, jtype, parent, child, xyz, rpy, jspec: JointSpec):
    lines = []
    lines.append(f'  <joint name="{name}" type="{jtype}">')
    lines.append(f'    <parent link="{parent}"/>')
    lines.append(f'    <child link="{child}"/>')
    lines.append(f'    <origin xyz="{fmt_vec(xyz)}" rpy="{fmt_vec(rpy)}"/>')
    if jtype in ("revolute", "continuous", "prismatic"):
        lines.append(f'    <axis xyz="{fmt_vec(jspec.axis)}"/>')
        lines.append(f'    <dynamics damping="{jspec.damping:.4f}" friction="{jspec.friction:.4f}"/>')
    if jtype in ("revolute", "prismatic"):
        lines.append(f'    <limit lower="{jspec.lower:.6f}" upper="{jspec.upper:.6f}" effort="{jspec.effort:.4f}" velocity="{jspec.velocity:.4f}"/>')
    elif jtype == "continuous":
        lines.append(f'    <limit effort="{jspec.effort:.4f}" velocity="{jspec.velocity:.4f}"/>')
    lines.append('  </joint>')
    return "\n".join(lines)


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

        parts.append(f'  <!-- {m.name}: spin={m.spin}, propsize={m.propsize}, pitch={m.pitch}, blades={m.blades} -->')

        parts.append(f'  <link name="{m.name}_arm_link">')
        parts.append(f'    <visual><origin xyz="{m.arm_length/2:.6f} 0 0" rpy="0 1.57079632679 0"/><geometry><cylinder radius="{d.arm_radius:.6f}" length="{m.arm_length:.6f}"/></geometry><material name="mat_arm"/></visual>')
        parts.append(f'    <collision><origin xyz="{m.arm_length/2:.6f} 0 0" rpy="0 1.57079632679 0"/><geometry><cylinder radius="{d.arm_radius:.6f}" length="{m.arm_length:.6f}"/></geometry></collision>')
        parts.append(f'    <xacro:inertial_cylinder_z mass="{arm_mass:.6f}" radius="{d.arm_radius:.6f}" length="{m.arm_length:.6f}"/>')
        parts.append('  </link>')
        parts.append(joint_tag(f'base_to_{m.name}_arm_joint', m.arm_joint.type, 'base_link', f'{m.name}_arm_link',
                                m.arm_pose.xyz, m.arm_pose.rpy, m.arm_joint))
        parts.append('')

        parts.append(f'  <link name="{m.name}_motor_link">')
        parts.append(f'    <visual><geometry><cylinder radius="{d.motor_radius:.6f}" length="{d.motor_height:.6f}"/></geometry><material name="mat_motor"/></visual>')
        parts.append(f'    <collision><geometry><cylinder radius="{d.motor_radius:.6f}" length="{d.motor_height:.6f}"/></geometry></collision>')
        parts.append(f'    <xacro:inertial_cylinder_z mass="{d.motor_mass:.6f}" radius="{d.motor_radius:.6f}" length="{d.motor_height:.6f}"/>')
        parts.append('  </link>')
        parts.append(joint_tag(f'{m.name}_arm_to_motor_joint', m.motor_joint.type, f'{m.name}_arm_link', f'{m.name}_motor_link',
                                m.motor_pose.xyz, m.motor_pose.rpy, m.motor_joint))
        parts.append('')

        parts.append(f'  <link name="{m.name}_rotor_link">')
        parts.append(f'    <visual><origin xyz="0 0 0" rpy="{fmt_vec(d.rotor_visual_rpy)}"/><geometry><cylinder radius="{m.rotor_radius:.6f}" length="{d.rotor_visual_thickness:.6f}"/></geometry><material name="{rotor_mat}"/></visual>')
        parts.append(f'    <collision><origin xyz="0 0 0" rpy="{fmt_vec(d.rotor_visual_rpy)}"/><geometry><cylinder radius="{m.rotor_radius:.6f}" length="{d.rotor_visual_thickness:.6f}"/></geometry></collision>')
        parts.append(f'    <xacro:inertial_cylinder_z mass="{d.rotor_hub_mass:.6f}" radius="{d.rotor_hub_radius:.6f}" length="{d.rotor_hub_height:.6f}"/>')
        parts.append('  </link>')
        parts.append(joint_tag(f'{m.name}_motor_to_rotor_joint', m.rotor_joint.type, f'{m.name}_motor_link', f'{m.name}_rotor_link',
                                m.rotor_pose.xyz, m.rotor_pose.rpy, m.rotor_joint))
        parts.append('')

    parts.append('</robot>')
    return '\n'.join(parts)


def clamp_tilt_components(roll, pitch, max_tilt):
    tilt_mag = math.sqrt(roll * roll + pitch * pitch)
    if tilt_mag <= max_tilt or tilt_mag == 0.0:
        return roll, pitch
    scale = max_tilt / tilt_mag
    return roll * scale, pitch * scale


def sample_rotor_rpy(randomize_rotor_tilt, rotor_tilt_magnitude, rotor_tilt_yaw_range, max_rotor_tilt, default_rpy):
    if randomize_rotor_tilt and rotor_tilt_magnitude > 0.0:
        tilt_dir = random.uniform(0.0, 2.0 * math.pi)
        raw_roll = rotor_tilt_magnitude * math.cos(tilt_dir)
        raw_pitch = rotor_tilt_magnitude * math.sin(tilt_dir)
        rotor_roll, rotor_pitch = clamp_tilt_components(raw_roll, raw_pitch, max_rotor_tilt)
        rotor_yaw = random.uniform(-rotor_tilt_yaw_range, rotor_tilt_yaw_range)
        return (rotor_roll, rotor_pitch, rotor_yaw)
    return default_rpy


def make_adaptive_drone(
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
    motor_z_offset: float = 0.02,
    rotor_mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.016),
    enable_arm_tilt: bool = True,
    arm_tilt_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    arm_tilt_limit: float = 0.35,
    arm_tilt_effort: float = 8.0,
    arm_tilt_velocity: float = 3.0,
    enable_motor_tilt: bool = True,
    motor_tilt_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    motor_tilt_limit: float = 0.30,
    motor_tilt_effort: float = 6.0,
    motor_tilt_velocity: float = 3.0,
    rotor_effort: float = 2.5,
    rotor_velocity: float = 400.0,
    rotor_cant_deg: float = 0.0,
    cant_alternate: bool = True,
    random_seed: int | None = None,
    defaults: Defaults | None = None,
) -> DroneSpec:
    if not (3 <= num_motors <= 8):
        raise ValueError("num_motors must be between 3 and 8")
    if random_seed is not None:
        random.seed(random_seed)

    defaults = defaults or Defaults(rotor_mount_xyz=rotor_mount_xyz)

    motors = []
    for i in range(num_motors):
        yaw = 2.0 * math.pi * i / num_motors
        spin = first_spin if (not alternate_spin or i % 2 == 0) else ('cw' if first_spin == 'ccw' else 'ccw')

        # Fixed thrust-vector cant: roll each rotor about its arm's radial (x) axis so the
        # thrust tilts tangentially. Alternating the sign keeps net horizontal force and net
        # yaw torque at zero for equal thrust (clean hover) while giving true yaw authority.
        cant_sign = (1.0 if i % 2 == 0 else -1.0) if cant_alternate else 1.0
        rotor_cant = math.radians(rotor_cant_deg) * cant_sign

        arm_joint = JointSpec(
            type="revolute" if enable_arm_tilt else "fixed",
            axis=arm_tilt_axis, lower=-arm_tilt_limit, upper=arm_tilt_limit,
            effort=arm_tilt_effort, velocity=arm_tilt_velocity, damping=0.08,
        )
        motor_joint = JointSpec(
            type="revolute" if enable_motor_tilt else "fixed",
            axis=motor_tilt_axis, lower=-motor_tilt_limit, upper=motor_tilt_limit,
            effort=motor_tilt_effort, velocity=motor_tilt_velocity, damping=0.05,
        )
        rotor_joint = JointSpec(
            type="continuous", axis=(0.0, 0.0, 1.0),
            effort=rotor_effort, velocity=rotor_velocity, damping=0.001,
        )

        motors.append(
            MotorSpec(
                name=f'arm_{i+1}',
                arm_length=arm_length,
                arm_pose=Pose(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, yaw)),
                motor_pose=Pose(xyz=(arm_length, 0.0, motor_z_offset), rpy=(0.0, 0.0, 0.0)),
                rotor_pose=Pose(xyz=defaults.rotor_mount_xyz, rpy=(rotor_cant, 0.0, 0.0)),
                spin=spin, propsize=propsize, rotor_radius=rotor_radius, pitch=pitch, blades=blades,
                arm_joint=arm_joint, motor_joint=motor_joint, rotor_joint=rotor_joint,
            )
        )

    return DroneSpec(name=name, core_mass=core_mass, core_radius=core_radius,
                      core_thickness=core_thickness, motors=motors, defaults=defaults)


def _rpy_to_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis rpy -> rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def estimate_total_mass(spec: DroneSpec) -> float:
    """Sum the link masses the xacro assigns (core + imu + per-arm arm/motor/rotor-hub)."""
    d = spec.defaults
    total = spec.core_mass + 0.005  # + imu_link
    for mo in spec.motors:
        total += mo.arm_length * d.arm_mass_per_m + d.motor_mass + d.rotor_hub_mass
    return total


def compute_allocation(spec: DroneSpec, thrust_coeff: float = 1.0, moment_coeff: float = 0.05) -> dict:
    """Compute hover allocation metadata (rotor geometry + B_rpyt) from the same MotorSpec
    poses the xacro bakes into the USD, so the controller's model matches the geometry.

    B_rpyt maps per-rotor thrust magnitudes -> [Fz, tau_x, tau_y, tau_z] (body frame),
    matching the wrench reconstruction in 01_load_single_usd_apply_wrench_hover.py:
        Fz           = sum_i f_i * d_i_z
        tau_offset   = sum_i r_i x (f_i * d_i)
        tau_reaction = sum_i spin_i * moment_coeff * f_i     (about body z only)
    """
    rotor_names, positions, directions = [], [], []
    spin_signs = {}
    for mo in spec.motors:
        R_arm = _rpy_to_matrix(mo.arm_pose.rpy)
        R_motor = _rpy_to_matrix(mo.motor_pose.rpy)
        R_rotor = _rpy_to_matrix(mo.rotor_pose.rpy)
        R = R_arm @ R_motor @ R_rotor
        direction = R @ np.array([0.0, 0.0, 1.0])           # canted thrust axis (body frame)
        # forward kinematics of the rotor origin (thrust point) in the base_link frame
        p_arm = np.array(mo.arm_pose.xyz, dtype=float)
        p_motor = p_arm + R_arm @ np.array(mo.motor_pose.xyz, dtype=float)
        p_rotor = p_motor + R_arm @ R_motor @ np.array(mo.rotor_pose.xyz, dtype=float)
        rotor_names.append(mo.name)
        positions.append(p_rotor)
        directions.append(direction)
        spin_signs[mo.name] = 1.0 if mo.spin.lower() == 'ccw' else -1.0

    positions = np.array(positions)
    directions = np.array(directions)
    n = len(rotor_names)

    B = np.zeros((4, n))
    for i in range(n):
        r = positions[i]
        d = directions[i]
        tau = np.cross(r, d)                                 # thrust-offset torque per unit thrust
        B[0, i] = d[2]                                        # vertical force per unit thrust
        B[1, i] = tau[0]
        B[2, i] = tau[1]
        B[3, i] = tau[2] + spin_signs[rotor_names[i]] * moment_coeff

    return {
        "morphology_id": spec.name,
        "rotor_names": rotor_names,
        "rotor_positions": positions.tolist(),
        "rotor_directions": directions.tolist(),
        "spin_signs": spin_signs,
        "thrust_coeff": float(thrust_coeff),
        "moment_coeff": float(moment_coeff),
        "mass_kg": round(estimate_total_mass(spec), 4),
        "rank_rpyt": int(np.linalg.matrix_rank(B)),
        "cond_rpyt": float(np.linalg.cond(B)),
        "allocation_matrix_rpyt": B.tolist(),
        "B_rpyt": B.tolist(),
    }


def convert_xacro_to_urdf(xacro_path: Path, urdf_path: Path):
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    with urdf_path.open('w', encoding='utf-8') as f:
        subprocess.run(['xacro', str(xacro_path)], check=True, stdout=f, text=True)


def convert_urdf_to_usd(urdf_path: Path, usd_path: Path, isaaclab_root: Path):
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    converter_script = isaaclab_root / "scripts" / "tools" / "convert_urdf.py"
    isaaclab_sh = isaaclab_root / "isaaclab.sh"
    subprocess.run(
        [str(isaaclab_sh), "-p", str(converter_script), str(urdf_path), str(usd_path),
         "--joint-stiffness", "0.0", "--joint-damping", "0.0", "--joint-target-type", "none",
         "--headless"],
        check=True, text=True,
    )


def find_isaaclab_root(start: Path) -> Path:
    env = os.environ.get("ISAACLAB_PATH")
    if env and (Path(env) / "isaaclab.sh").exists():
        return Path(env)
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "isaaclab.sh").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find IsaacLab root from {start}. "
        "Set the ISAACLAB_PATH env var to your IsaacLab repo."
    )


def main():
    base_dir = Path(__file__).resolve().parent
    isaaclab_root = find_isaaclab_root(base_dir)

    conversion_dir = base_dir / "conversion"
    urdf_dir = conversion_dir / "urdf"
    usd_dir = conversion_dir / "usd"
    allocation_dir = conversion_dir / "allocation"

    clear_directory(urdf_dir)
    clear_directory(usd_dir)
    allocation_dir.mkdir(parents=True, exist_ok=True)

    num_motors = 6
    rotor_cant_deg = 15.0

    spec = make_adaptive_drone(
        name=f"hex_cant{int(rotor_cant_deg)}_{num_motors}rotor_drone",
        num_motors=num_motors,
        arm_length=0.20,
        rotor_radius=0.0635,
        rotor_cant_deg=rotor_cant_deg,
        cant_alternate=True,
        enable_arm_tilt=False,   # rigid frame: the only tilt is the fixed rotor cant
        enable_motor_tilt=False,
        random_seed=42,
    )

    xacro_path = base_dir / f"{spec.name}.xacro"
    urdf_path = urdf_dir / f"{spec.name}.urdf"
    usd_path = usd_dir / f"{spec.name}.usd"
    allocation_path = allocation_dir / f"{spec.name}.json"

    xacro_text = build_drone_xacro(spec)
    xacro_path.write_text(xacro_text, encoding="utf-8")

    # Allocation JSON derived from the same MotorSpec poses -> directly flyable by
    # 01_load_single_usd_apply_wrench_hover.py / 01_hover_record.py.
    allocation = compute_allocation(spec, thrust_coeff=1.0, moment_coeff=0.05)
    allocation_path.write_text(json.dumps(allocation, indent=2), encoding="utf-8")

    print(f"Using IsaacLab root: {isaaclab_root}")
    print(f"Robot name: {spec.name}  motors: {len(spec.motors)}  rotor_cant: {rotor_cant_deg} deg")
    print(f"Estimated mass: {allocation['mass_kg']} kg | "
          f"B_rpyt rank {allocation['rank_rpyt']}/4  cond {allocation['cond_rpyt']:.1f}")

    convert_xacro_to_urdf(xacro_path, urdf_path)
    convert_urdf_to_usd(urdf_path, usd_path, isaaclab_root)

    print(f"Wrote Xacro to:      {xacro_path}")
    print(f"Wrote URDF to:       {urdf_path}")
    print(f"Wrote USD to:        {usd_path}")
    print(f"Wrote allocation to: {allocation_path}")
    print()
    print("Fly it with:")
    print(f"  python 01_hover_record.py \\")
    print(f"    --usd {usd_path} \\")
    print(f"    --allocation_json {allocation_path} \\")
    print(f"    --video_path hover_{spec.name}.mp4 \\")
    print(f"    --max_steps 800 --record_every 2 --headless --enable_cameras")


if __name__ == "__main__":
    main()
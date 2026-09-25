#!/usr/bin/env python3
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load one articulated drone USD and run PD hover with x-y position hold using allocation metadata.")
parser.add_argument("--usd", type=str, required=True, help="Path to a single drone USD file.")
parser.add_argument("--allocation_json", type=str, required=True, help="Path to allocation JSON for this morphology.")
parser.add_argument("--x", type=float, default=0.0, help="Spawn x position.")
parser.add_argument("--y", type=float, default=0.0, help="Spawn y position.")
parser.add_argument("--z", type=float, default=1.5, help="Spawn z position.")
parser.add_argument("--hover_x", type=float, default=0.0, help="Target x position in world frame.")
parser.add_argument("--hover_y", type=float, default=0.0, help="Target y position in world frame.")
parser.add_argument("--hover_z", type=float, default=1.5, help="Target z position in world frame.")
parser.add_argument("--yaw_deg", type=float, default=0.0, help="Target yaw in degrees.")
parser.add_argument("--max_rotor_thrust", type=float, default=15.0, help="Maximum thrust per rotor.")
parser.add_argument("--mass_kg", type=float, default=None, help="Override vehicle mass in kg.")
parser.add_argument("--base_body_name", type=str, default="base_link", help="Body name to which the net wrench is applied.")
parser.add_argument("--kp_x", type=float, default=0.8, help="X position proportional gain.")
parser.add_argument("--kd_x", type=float, default=1.2, help="X velocity derivative gain.")
parser.add_argument("--kp_y", type=float, default=0.8, help="Y position proportional gain.")
parser.add_argument("--kd_y", type=float, default=1.2, help="Y velocity derivative gain.")
parser.add_argument("--kp_z", type=float, default=8.0, help="Altitude proportional gain.")
parser.add_argument("--kd_z", type=float, default=5.0, help="Altitude derivative gain.")
parser.add_argument("--kp_roll", type=float, default=8.0, help="Roll proportional gain.")
parser.add_argument("--kd_roll", type=float, default=2.0, help="Roll derivative gain.")
parser.add_argument("--kp_pitch", type=float, default=8.0, help="Pitch proportional gain.")
parser.add_argument("--kd_pitch", type=float, default=2.0, help="Pitch derivative gain.")
parser.add_argument("--kp_yaw", type=float, default=2.0, help="Yaw proportional gain.")
parser.add_argument("--kd_yaw", type=float, default=0.6, help="Yaw derivative gain.")
parser.add_argument("--max_tilt_deg", type=float, default=12.0, help="Maximum commanded roll/pitch angle in degrees.")
parser.add_argument("--debug_every", type=int, default=120, help="Print controller state every N simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import json
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg

sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 1.0])

ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/GroundPlane", ground_cfg)

light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.95, 0.95, 0.95))
light_cfg.func("/World/Light", light_cfg)

usd_path = Path(args_cli.usd).resolve()
allocation_json_path = Path(args_cli.allocation_json).resolve()
if not usd_path.exists():
    raise FileNotFoundError(f"USD file does not exist: {usd_path}")
if not allocation_json_path.exists():
    raise FileNotFoundError(f"Allocation JSON does not exist: {allocation_json_path}")

with open(allocation_json_path, "r", encoding="utf-8") as f:
    allocation = json.load(f)

rotor_names = allocation["rotor_names"]
num_rotors = len(rotor_names)
max_rotor_thrust = float(args_cli.max_rotor_thrust)

B_rpyt_data = allocation.get("allocation_matrix_rpyt")
if B_rpyt_data is None:
    B_rpyt_data = allocation.get("B_rpyt")
if B_rpyt_data is None:
    raise KeyError("Could not find hover allocation matrix. Expected 'allocation_matrix_rpyt' or 'B_rpyt'.")
A_rpyt = torch.tensor(B_rpyt_data, dtype=torch.float32, device=args_cli.device)
A_pinv = torch.linalg.pinv(A_rpyt)

rot_pos_data = allocation.get("rotor_positions_body")
if rot_pos_data is None:
    rot_pos_data = allocation.get("rotor_positions")
rot_pos = torch.tensor(rot_pos_data, dtype=torch.float32, device=args_cli.device)

rot_dirs_data = allocation.get("rotor_directions_body")
if rot_dirs_data is None:
    rot_dirs_data = allocation.get("rotor_directions")
rot_dirs = torch.tensor(rot_dirs_data, dtype=torch.float32, device=args_cli.device)

spin_signs_data = allocation["spin_signs"]
if isinstance(spin_signs_data, dict):
    spin_signs = torch.tensor([float(spin_signs_data[name]) for name in rotor_names], dtype=torch.float32, device=args_cli.device)
else:
    spin_signs = torch.tensor(spin_signs_data, dtype=torch.float32, device=args_cli.device)

moment_coeff = allocation.get("moment_coeff")
if moment_coeff is None:
    torque_constants = allocation.get("torque_constants")
    if torque_constants is not None:
        torque_constants = torch.tensor(torque_constants, dtype=torch.float32, device=args_cli.device)
    else:
        torque_constants = torch.full((num_rotors,), 0.05, dtype=torch.float32, device=args_cli.device)
else:
    torque_constants = torch.full((num_rotors,), float(moment_coeff), dtype=torch.float32, device=args_cli.device)

mass_kg = args_cli.mass_kg
if mass_kg is None:
    for key in ["mass_kg", "total_mass_kg", "estimated_mass_kg"]:
        if key in allocation:
            mass_kg = float(allocation[key])
            break
if mass_kg is None:
    mass_kg = 1.89
    print("[WARN] Mass not found in allocation JSON. Falling back to mass_kg=1.89. Override with --mass_kg if needed.")

drone_cfg = ArticulationCfg(
    prim_path="/World/drone",
    actuators={},
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.2,
            angular_damping=0.2,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(args_cli.x, args_cli.y, args_cli.z),
    ),
)

drone = Articulation(drone_cfg)

sim.reset()

device = torch.device(args_cli.device)

print(f"Loaded USD: {usd_path}")
print(f"Loaded allocation: {allocation_json_path}")
print(f"Morphology ID: {allocation.get('morphology_id', 'unknown')}")
print(f"Rotors: {rotor_names}")
print(f"Mass [kg]: {mass_kg:.4f}")
print(f"Articulation body names: {drone.body_names}")

if args_cli.base_body_name not in drone.body_names:
    raise ValueError(f"Body '{args_cli.base_body_name}' not found in articulation. Available bodies: {drone.body_names}")
base_body_id = drone.body_names.index(args_cli.base_body_name)
print(f"Applying wrench to body '{args_cli.base_body_name}' with id {base_body_id}")

forces = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
torques = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)

target_yaw = math.radians(args_cli.yaw_deg)
max_tilt = math.radians(args_cli.max_tilt_deg)
g = 9.81

def quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([
        ww + xx - yy - zz,
        2 * (xy - wz),
        2 * (xz + wy),
        2 * (xy + wz),
        ww - xx + yy - zz,
        2 * (yz - wx),
        2 * (xz - wy),
        2 * (yz + wx),
        ww - xx - yy + zz,
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))

def quat_to_euler_xyz(q: torch.Tensor):
    w, x, y, z = q.unbind(-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

step_count = 0
dt = sim.get_physics_dt()

while simulation_app.is_running():
    root_state = drone.data.root_state_w
    pos_w = root_state[:, 0:3]
    quat_wxyz = root_state[:, 3:7]
    lin_vel_w = root_state[:, 7:10]
    ang_vel_w = root_state[:, 10:13]

    roll, pitch, yaw = quat_to_euler_xyz(quat_wxyz)
    R_wb = quat_wxyz_to_rotmat(quat_wxyz)
    ang_vel_b = torch.bmm(R_wb.transpose(1, 2), ang_vel_w.unsqueeze(-1)).squeeze(-1)

    x = pos_w[:, 0]
    y = pos_w[:, 1]
    z = pos_w[:, 2]
    vx = lin_vel_w[:, 0]
    vy = lin_vel_w[:, 1]
    vz = lin_vel_w[:, 2]

    ex = args_cli.hover_x - x
    ey = args_cli.hover_y - y
    ez = args_cli.hover_z - z

    ax_cmd = args_cli.kp_x * ex + args_cli.kd_x * (-vx)
    ay_cmd = args_cli.kp_y * ey + args_cli.kd_y * (-vy)

    roll_cmd = torch.clamp(-ay_cmd / g, min=-max_tilt, max=max_tilt)
    pitch_cmd = torch.clamp(ax_cmd / g, min=-max_tilt, max=max_tilt)

    thrust_cmd = mass_kg * (g + args_cli.kp_z * ez + args_cli.kd_z * (-vz))
    thrust_cmd = torch.clamp(thrust_cmd, min=0.0, max=num_rotors * max_rotor_thrust)

    roll_err = roll_cmd - roll
    pitch_err = pitch_cmd - pitch
    yaw_err = wrap_to_pi(torch.full_like(yaw, target_yaw) - yaw)

    tau_x = args_cli.kp_roll * roll_err + args_cli.kd_roll * (-ang_vel_b[:, 0])
    tau_y = args_cli.kp_pitch * pitch_err + args_cli.kd_pitch * (-ang_vel_b[:, 1])
    tau_z = args_cli.kp_yaw * yaw_err + args_cli.kd_yaw * (-ang_vel_b[:, 2])

    desired_rpyt = torch.stack([thrust_cmd, tau_x, tau_y, tau_z], dim=-1)
    rotor_thrusts = torch.matmul(desired_rpyt, A_pinv.transpose(0, 1))
    rotor_thrusts = torch.clamp(rotor_thrusts, min=0.0, max=max_rotor_thrust)

    force_b = torch.sum(rotor_thrusts.unsqueeze(-1) * rot_dirs.unsqueeze(0), dim=1)
    torque_from_offset = torch.cross(rot_pos.unsqueeze(0), rotor_thrusts.unsqueeze(-1) * rot_dirs.unsqueeze(0), dim=-1).sum(dim=1)
    reaction_torque = torch.zeros((1, 3), dtype=torch.float32, device=device)
    reaction_torque[:, 2] = torch.sum(spin_signs.unsqueeze(0) * torque_constants.unsqueeze(0) * rotor_thrusts, dim=-1)
    torque_b = torque_from_offset + reaction_torque

    forces[:, 0, :] = force_b
    torques[:, 0, :] = torque_b

    drone.permanent_wrench_composer.set_forces_and_torques(
        body_ids=[base_body_id],
        forces=forces,
        torques=torques,
    )

    drone.write_data_to_sim()
    sim.step()
    drone.update(dt)

    if step_count % max(1, args_cli.debug_every) == 0:
        print(
            f"step={step_count} pos=({x.item():.3f},{y.item():.3f},{z.item():.3f}) vel=({vx.item():.3f},{vy.item():.3f},{vz.item():.3f}) "
            f"rpy=({roll.item():.3f},{pitch.item():.3f},{yaw.item():.3f}) cmd_rp=({roll_cmd.item():.3f},{pitch_cmd.item():.3f}) "
            f"thrust={thrust_cmd.item():.3f}"
        )

    step_count += 1

simulation_app.close()
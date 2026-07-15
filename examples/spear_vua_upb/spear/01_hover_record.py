#!/usr/bin/env python3
"""Recording variant of 01_load_single_usd_apply_wrench_hover.py.

Same base-link wrench hover controller, plus a drone-following Camera sensor
whose RGB frames are streamed to an mp4. Requires --enable_cameras on the CLI.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hover controller with mp4 recording via a following camera.")
parser.add_argument("--usd", type=str, required=True)
parser.add_argument("--allocation_json", type=str, required=True)
parser.add_argument("--x", type=float, default=0.0)
parser.add_argument("--y", type=float, default=0.0)
parser.add_argument("--z", type=float, default=1.5)
parser.add_argument("--hover_x", type=float, default=0.0)
parser.add_argument("--hover_y", type=float, default=0.0)
parser.add_argument("--hover_z", type=float, default=1.5)
parser.add_argument("--yaw_deg", type=float, default=0.0)
parser.add_argument("--max_rotor_thrust", type=float, default=15.0)
parser.add_argument("--mass_kg", type=float, default=None)
parser.add_argument("--base_body_name", type=str, default="base_link")
parser.add_argument("--kp_x", type=float, default=0.8)
parser.add_argument("--kd_x", type=float, default=1.2)
parser.add_argument("--kp_y", type=float, default=0.8)
parser.add_argument("--kd_y", type=float, default=1.2)
parser.add_argument("--kp_z", type=float, default=8.0)
parser.add_argument("--kd_z", type=float, default=5.0)
parser.add_argument("--kp_roll", type=float, default=8.0)
parser.add_argument("--kd_roll", type=float, default=2.0)
parser.add_argument("--kp_pitch", type=float, default=8.0)
parser.add_argument("--kd_pitch", type=float, default=2.0)
parser.add_argument("--kp_yaw", type=float, default=2.0)
parser.add_argument("--kd_yaw", type=float, default=0.6)
parser.add_argument("--max_tilt_deg", type=float, default=12.0)
parser.add_argument("--debug_every", type=int, default=60)
# recording-specific
parser.add_argument("--video_path", type=str, required=True, help="Output mp4 path.")
parser.add_argument("--max_steps", type=int, default=500, help="Simulate this many steps then stop.")
parser.add_argument("--record_every", type=int, default=2, help="Capture one frame every N sim steps.")
parser.add_argument("--cam_width", type=int, default=1280)
parser.add_argument("--cam_height", type=int, default=720)
parser.add_argument("--cam_offset", type=float, nargs=3, default=[3.0, 3.0, 1.2], help="Camera eye offset from drone.")
parser.add_argument("--follow", action="store_true", help="Camera follows the drone (default: fixed camera).")
parser.add_argument("--cam_eye", type=float, nargs=3, default=[2.8, 2.8, 2.4], help="Fixed camera eye (world).")
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.0, 0.0, 1.5], help="Fixed camera look-at (world).")
parser.add_argument("--auto_mass", action="store_true", default=True, help="Use the articulation's actual total mass for feed-forward.")
parser.add_argument("--lock_stiffness", type=float, default=800.0, help="Stiffness locking fold/extend joints rigid.")
parser.add_argument("--lock_damping", type=float, default=60.0, help="Damping for the locking actuators.")
parser.add_argument("--lock_joints_regex", type=str, nargs="+",
                    default=[".*_fold_joint", ".*_extend_joint", ".*_rotor_spin_joint"],
                    help="Joint-name regex(es) to hold rigid. Must match >=1 joint on the loaded USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import json
import numpy as np
import torch
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg

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

B_rpyt_data = allocation.get("allocation_matrix_rpyt") or allocation.get("B_rpyt")
if B_rpyt_data is None:
    raise KeyError("Could not find hover allocation matrix ('allocation_matrix_rpyt' or 'B_rpyt').")
A_rpyt = torch.tensor(B_rpyt_data, dtype=torch.float32, device=args_cli.device)
A_pinv = torch.linalg.pinv(A_rpyt)

rot_pos_data = allocation.get("rotor_positions_body") or allocation.get("rotor_positions")
rot_pos = torch.tensor(rot_pos_data, dtype=torch.float32, device=args_cli.device)

rot_dirs_data = allocation.get("rotor_directions_body") or allocation.get("rotor_directions")
rot_dirs = torch.tensor(rot_dirs_data, dtype=torch.float32, device=args_cli.device)

spin_signs_data = allocation["spin_signs"]
if isinstance(spin_signs_data, dict):
    spin_signs = torch.tensor([float(spin_signs_data[n]) for n in rotor_names], dtype=torch.float32, device=args_cli.device)
else:
    spin_signs = torch.tensor(spin_signs_data, dtype=torch.float32, device=args_cli.device)

moment_coeff = allocation.get("moment_coeff")
if moment_coeff is None:
    tc = allocation.get("torque_constants")
    torque_constants = torch.tensor(tc, dtype=torch.float32, device=args_cli.device) if tc is not None \
        else torch.full((num_rotors,), 0.05, dtype=torch.float32, device=args_cli.device)
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
    print("[WARN] Mass not found in allocation JSON. Falling back to mass_kg=1.89.")

drone_cfg = ArticulationCfg(
    prim_path="/World/drone",
    actuators={
        # Lock the floppy fold (revolute), extend (prismatic) and rotor_spin (continuous)
        # joints rigidly at their neutral pose so the frame keeps the geometry the
        # allocation matrix was computed for.
        "lock": ImplicitActuatorCfg(
            joint_names_expr=args_cli.lock_joints_regex,
            stiffness=args_cli.lock_stiffness,
            damping=args_cli.lock_damping,
            effort_limit=1000.0,
            velocity_limit=100.0,
        ),
    },
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, linear_damping=0.2, angular_damping=0.2, max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(args_cli.x, args_cli.y, args_cli.z)),
)
drone = Articulation(drone_cfg)

# following camera
camera_cfg = CameraCfg(
    prim_path="/World/record_cam",
    update_period=0.0,
    height=args_cli.cam_height,
    width=args_cli.cam_width,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0, focus_distance=400.0, horizontal_aperture=24.0, clipping_range=(0.05, 1.0e6),
    ),
)
camera = Camera(cfg=camera_cfg)

sim.reset()

device = torch.device(args_cli.device)

# Hold every joint at its default (neutral) pose so the locking actuators keep the frame rigid.
default_joint_pos = drone.data.default_joint_pos.clone()
drone.set_joint_position_target(default_joint_pos)

# Use the drone's actual total mass for the gravity feed-forward (overrides the JSON/fallback).
actual_mass = float(drone.root_physx_view.get_masses()[0].sum().item())
if args_cli.mass_kg is not None:
    mass_kg = args_cli.mass_kg
    mass_src = "cli"
elif args_cli.auto_mass:
    mass_kg = actual_mass
    mass_src = "articulation"
else:
    mass_src = "json/fallback"

print(f"Loaded USD: {usd_path}")
print(f"Morphology ID: {allocation.get('morphology_id', 'unknown')}")
print(f"Rotors: {rotor_names}")
print(f"Actual total mass from physics: {actual_mass:.4f} kg | using mass_kg={mass_kg:.4f} ({mass_src})")

if args_cli.base_body_name not in drone.body_names:
    raise ValueError(f"Body '{args_cli.base_body_name}' not found. Available: {drone.body_names}")
base_body_id = drone.body_names.index(args_cli.base_body_name)

forces = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
torques = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
cam_offset = torch.tensor([args_cli.cam_offset], dtype=torch.float32, device=device)  # (1,3)

target_yaw = math.radians(args_cli.yaw_deg)
max_tilt = math.radians(args_cli.max_tilt_deg)
g = 9.81


def quat_wxyz_to_rotmat(q):
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([
        ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz,
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def quat_to_euler_xyz(q):
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def wrap_to_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


fps = max(1, round((1.0 / sim.get_physics_dt()) / max(1, args_cli.record_every)))
Path(args_cli.video_path).parent.mkdir(parents=True, exist_ok=True)
writer = imageio.get_writer(args_cli.video_path, fps=fps, codec="libx264", quality=8, macro_block_size=16)
print(f"Recording {args_cli.max_steps} steps -> {args_cli.video_path} (fps={fps})")

cam_eye_fixed = torch.tensor([args_cli.cam_eye], dtype=torch.float32, device=device)
cam_target_fixed = torch.tensor([args_cli.cam_target], dtype=torch.float32, device=device)
if not args_cli.follow:
    camera.set_world_poses_from_view(cam_eye_fixed, cam_target_fixed)

step_count = 0
dt = sim.get_physics_dt()
frames_written = 0

while simulation_app.is_running() and step_count < args_cli.max_steps:
    root_state = drone.data.root_state_w
    pos_w = root_state[:, 0:3]
    quat_wxyz = root_state[:, 3:7]
    lin_vel_w = root_state[:, 7:10]
    ang_vel_w = root_state[:, 10:13]

    roll, pitch, yaw = quat_to_euler_xyz(quat_wxyz)
    R_wb = quat_wxyz_to_rotmat(quat_wxyz)
    ang_vel_b = torch.bmm(R_wb.transpose(1, 2), ang_vel_w.unsqueeze(-1)).squeeze(-1)

    x, y, z = pos_w[:, 0], pos_w[:, 1], pos_w[:, 2]
    vx, vy, vz = lin_vel_w[:, 0], lin_vel_w[:, 1], lin_vel_w[:, 2]

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
    drone.permanent_wrench_composer.set_forces_and_torques(body_ids=[base_body_id], forces=forces, torques=torques)

    # camera: follow the drone, or leave at the fixed pose set before the loop
    if args_cli.follow:
        camera.set_world_poses_from_view(pos_w + cam_offset, pos_w)

    drone.write_data_to_sim()
    sim.step()
    drone.update(dt)
    camera.update(dt)

    if step_count % max(1, args_cli.record_every) == 0:
        rgb = camera.data.output["rgb"]
        if rgb is not None and rgb.shape[0] > 0:
            frame = rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            writer.append_data(frame)
            frames_written += 1

    if step_count % max(1, args_cli.debug_every) == 0:
        print(f"step={step_count} pos=({x.item():.2f},{y.item():.2f},{z.item():.2f}) "
              f"thrust={thrust_cmd.item():.2f} frames={frames_written}")

    step_count += 1

writer.close()
print(f"DONE: wrote {frames_written} frames to {args_cli.video_path}")
simulation_app.close()

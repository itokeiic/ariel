#!/usr/bin/env python3
"""Diagnostic: rich per-step logging of the hover controller (no camera)."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, required=True)
parser.add_argument("--allocation_json", type=str, required=True)
parser.add_argument("--z", type=float, default=1.5)
parser.add_argument("--hover_z", type=float, default=1.5)
parser.add_argument("--max_rotor_thrust", type=float, default=15.0)
parser.add_argument("--base_body_name", type=str, default="base_link")
parser.add_argument("--max_steps", type=int, default=70)
parser.add_argument("--kp_x", type=float, default=0.8); parser.add_argument("--kd_x", type=float, default=1.2)
parser.add_argument("--kp_y", type=float, default=0.8); parser.add_argument("--kd_y", type=float, default=1.2)
parser.add_argument("--kp_z", type=float, default=8.0); parser.add_argument("--kd_z", type=float, default=5.0)
parser.add_argument("--kp_roll", type=float, default=8.0); parser.add_argument("--kd_roll", type=float, default=2.0)
parser.add_argument("--kp_pitch", type=float, default=8.0); parser.add_argument("--kd_pitch", type=float, default=2.0)
parser.add_argument("--kp_yaw", type=float, default=2.0); parser.add_argument("--kd_yaw", type=float, default=0.6)
parser.add_argument("--max_tilt_deg", type=float, default=12.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math, json
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg

sim = SimulationContext(SimulationCfg(dt=0.01, device=args_cli.device))
sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

allocation = json.load(open(Path(args_cli.allocation_json).resolve()))
rotor_names = allocation["rotor_names"]; num_rotors = len(rotor_names)
max_rotor_thrust = float(args_cli.max_rotor_thrust)
A_rpyt = torch.tensor(allocation.get("allocation_matrix_rpyt") or allocation.get("B_rpyt"), dtype=torch.float32, device=args_cli.device)
A_pinv = torch.linalg.pinv(A_rpyt)
rot_pos = torch.tensor(allocation.get("rotor_positions_body") or allocation.get("rotor_positions"), dtype=torch.float32, device=args_cli.device)
rot_dirs = torch.tensor(allocation.get("rotor_directions_body") or allocation.get("rotor_directions"), dtype=torch.float32, device=args_cli.device)
ss = allocation["spin_signs"]
spin_signs = torch.tensor([float(ss[n]) for n in rotor_names] if isinstance(ss, dict) else ss, dtype=torch.float32, device=args_cli.device)
torque_constants = torch.full((num_rotors,), float(allocation.get("moment_coeff", 0.05)), dtype=torch.float32, device=args_cli.device)

drone_cfg = ArticulationCfg(
    prim_path="/World/drone", actuators={},
    spawn=sim_utils.UsdFileCfg(usd_path=str(Path(args_cli.usd).resolve()),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.2, angular_damping=0.2, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.z)))
drone = Articulation(drone_cfg)
sim.reset()
device = torch.device(args_cli.device)
mass_kg = float(drone.root_physx_view.get_masses()[0].sum().item())
com = drone.root_physx_view.get_coms()[0]  # per-body coms
print(f"mass={mass_kg:.4f} num_bodies={drone.num_bodies} num_joints={drone.num_joints}")
print(f"base_link COM (body 0): {com[0].tolist()}")
base_body_id = drone.body_names.index(args_cli.base_body_name)
forces = torch.zeros((1, 1, 3), device=device); torques = torch.zeros((1, 1, 3), device=device)
max_tilt = math.radians(args_cli.max_tilt_deg); g = 9.81


def quat_to_euler(q):
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin(torch.clamp(2*(w*y-z*x), -1, 1))
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw

def rotmat(q):
    w, x, y, z = q.unbind(-1)
    ww,xx,yy,zz=w*w,x*x,y*y,z*z; wx,wy,wz=w*x,w*y,w*z; xy,xz,yz=x*y,x*z,y*z
    return torch.stack([ww+xx-yy-zz,2*(xy-wz),2*(xz+wy),2*(xy+wz),ww-xx+yy-zz,2*(yz-wx),2*(xz-wy),2*(yz+wx),ww-xx-yy+zz],-1).reshape(q.shape[:-1]+(3,3))

dt = sim.get_physics_dt()
for step in range(args_cli.max_steps):
    rs = drone.data.root_state_w
    pos_w, quat, vlin, vang = rs[:, 0:3], rs[:, 3:7], rs[:, 7:10], rs[:, 10:13]
    roll, pitch, yaw = quat_to_euler(quat)
    R = rotmat(quat)
    wb = torch.bmm(R.transpose(1, 2), vang.unsqueeze(-1)).squeeze(-1)
    x, y, z = pos_w[:, 0], pos_w[:, 1], pos_w[:, 2]
    vx, vy, vz = vlin[:, 0], vlin[:, 1], vlin[:, 2]
    ax = args_cli.kp_x*(0-x)+args_cli.kd_x*(-vx); ay = args_cli.kp_y*(0-y)+args_cli.kd_y*(-vy)
    roll_cmd = torch.clamp(-ay/g, -max_tilt, max_tilt); pitch_cmd = torch.clamp(ax/g, -max_tilt, max_tilt)
    thrust_cmd = torch.clamp(mass_kg*(g+args_cli.kp_z*(args_cli.hover_z-z)+args_cli.kd_z*(-vz)), 0.0, num_rotors*max_rotor_thrust)
    tau_x = args_cli.kp_roll*(roll_cmd-roll)+args_cli.kd_roll*(-wb[:, 0])
    tau_y = args_cli.kp_pitch*(pitch_cmd-pitch)+args_cli.kd_pitch*(-wb[:, 1])
    tau_z = args_cli.kp_yaw*(0-yaw)+args_cli.kd_yaw*(-wb[:, 2])
    desired = torch.stack([thrust_cmd, tau_x, tau_y, tau_z], -1)
    rt = torch.clamp(torch.matmul(desired, A_pinv.transpose(0, 1)), 0.0, max_rotor_thrust)
    force_b = torch.sum(rt.unsqueeze(-1)*rot_dirs.unsqueeze(0), 1)
    tq = torch.cross(rot_pos.unsqueeze(0), rt.unsqueeze(-1)*rot_dirs.unsqueeze(0), dim=-1).sum(1)
    tq[:, 2] += torch.sum(spin_signs*torque_constants*rt, -1)
    forces[:, 0, :] = force_b; torques[:, 0, :] = tq
    drone.permanent_wrench_composer.set_forces_and_torques(body_ids=[base_body_id], forces=forces, torques=torques)
    drone.write_data_to_sim(); sim.step(); drone.update(dt)
    if step < 40 or step % 10 == 0:
        print(f"s={step:2d} z={z.item():+.3f} vz={vz.item():+.3f} r={roll.item():+.3f} p={pitch.item():+.3f} "
              f"wx={wb[0,0].item():+.3f} wy={wb[0,1].item():+.3f} | Tc={thrust_cmd.item():5.2f} "
              f"tx={tau_x.item():+.3f} ty={tau_y.item():+.3f} | rt=[{','.join(f'{v:.2f}' for v in rt[0].tolist())}] "
              f"Fz={force_b[0,2].item():+.2f} Tqx={tq[0,0].item():+.3f} Tqy={tq[0,1].item():+.3f}")
print("DIAG DONE")
simulation_app.close()

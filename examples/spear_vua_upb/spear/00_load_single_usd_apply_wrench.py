#!/usr/bin/env python3
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Load one articulated drone USD and apply body wrench using saved allocation metadata.")
parser.add_argument("--usd", type=str, required=True, help="Path to a single drone USD file.")
parser.add_argument("--allocation_json", type=str, required=True, help="Path to allocation JSON for this morphology.")
parser.add_argument("--z", type=float, default=1.5, help="Spawn height.")
parser.add_argument("--hover_action", type=float, default=0.2, help="Normalized motor action in [-1, 1] used for all rotors.")
parser.add_argument("--max_rotor_thrust", type=float, default=15.0, help="Maximum thrust per rotor.")
parser.add_argument("--use_full_wrench", action="store_true", help="Use saved full allocation matrix instead of explicit force/torque summation.")
parser.add_argument("--base_body_name", type=str, default="base_link", help="Body name to which the net wrench is applied.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg

from morphing_drone_wrench_model import MorphingDroneWrenchModel


sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([4.0, 4.0, 3.0], [0.0, 0.0, 1.0])


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
        pos=(0.0, 0.0, args_cli.z),
    ),
)

drone = Articulation(drone_cfg)

sim.reset()

device = torch.device(args_cli.device)

wrench_model = MorphingDroneWrenchModel(
    allocation_json_path=allocation_json_path,
    num_envs=1,
    device=device,
    use_full_wrench=args_cli.use_full_wrench,
    max_rotor_thrust=args_cli.max_rotor_thrust,
)

print(f"Loaded USD: {usd_path}")
print(f"Loaded allocation: {allocation_json_path}")
print(f"Morphology ID: {wrench_model.morphology_id}")
print(f"Rotors: {wrench_model.rotor_names}")
print(f"Rank full: {wrench_model.rank_full}, rank rpyt: {wrench_model.rank_rpyt}")
print(f"Condition full: {wrench_model.cond_full:.4f}, condition rpyt: {wrench_model.cond_rpyt:.4f}")
print(f"Articulation body names: {drone.body_names}")

if args_cli.base_body_name not in drone.body_names:
    raise ValueError(
        f"Body '{args_cli.base_body_name}' not found in articulation. Available bodies: {drone.body_names}"
    )
base_body_id = drone.body_names.index(args_cli.base_body_name)
print(f"Applying wrench to body '{args_cli.base_body_name}' with id {base_body_id}")

forces = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
torques = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)

motor_actions = torch.full(
    (1, wrench_model.num_rotors),
    fill_value=args_cli.hover_action,
    dtype=torch.float32,
    device=device,
)

t = 0.0
dt = sim.get_physics_dt()

while simulation_app.is_running():
    rotor_thrusts = wrench_model.motor_commands_to_thrusts(motor_actions)
    force_b, torque_b = wrench_model.compute_body_wrench(rotor_thrusts)

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
    t += dt

simulation_app.close()
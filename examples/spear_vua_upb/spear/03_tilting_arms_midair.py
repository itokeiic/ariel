
"""
Standalone Isaac Lab script: load adaptive drone USD, lift it, and tilt
a subset of arms mid-air using revolute joint position targets.

Run with:
  ./isaaclab.sh -p tilt_arms_midair.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, required=True,
                     help="Path to the converted drone USD, e.g. conversion/usd/adaptive_7rotor_drone.usd")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


DRONE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=args_cli.usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "arm_tilt": ImplicitActuatorCfg(
            joint_names_expr=["base_to_arm_.*_arm_joint"],
            effort_limit=8.0,
            velocity_limit=3.0,
            stiffness=40.0,
            damping=4.0,
        ),
        "motor_tilt": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*_arm_to_motor_joint"],
            effort_limit=6.0,
            velocity_limit=3.0,
            stiffness=30.0,
            damping=3.0,
        ),
        "rotor_spin": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*_motor_to_rotor_joint"],
            effort_limit=2.5,
            velocity_limit=400.0,
            stiffness=0.0,
            damping=0.05,
        ),
    },
)


@configclass
class DroneSceneCfg(InteractiveSceneCfg):
    ground = sim_utils.GroundPlaneCfg()
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    arm_tilt_ids, arm_tilt_names = robot.find_joints("base_to_arm_.*_arm_joint")
    rotor_ids, rotor_names = robot.find_joints("arm_.*_motor_to_rotor_joint")

    print(f"Found arm tilt joints: {arm_tilt_names}")
    print(f"Found rotor spin joints: {rotor_names}")

    tilt_target = torch.zeros(scene.num_envs, len(arm_tilt_ids), device=sim.device)
    spin_target = torch.full((scene.num_envs, len(rotor_ids)), 60.0, device=sim.device)

    step = 0
    lift_steps = 200          # let it hover/climb before tilting
    tilt_start = 200
    tilt_end = 400
    tilt_subset = [0, 2]      # indices into arm_tilt_ids to actuate mid-air
    tilt_angle = 0.3          # radians

    while simulation_app.is_running():
        if step < lift_steps:
            robot.set_joint_velocity_target(spin_target, joint_ids=rotor_ids)
        elif tilt_start <= step <= tilt_end:
            progress = (step - tilt_start) / max(1, (tilt_end - tilt_start))
            tilt_target[:, tilt_subset] = tilt_angle * progress
            robot.set_joint_position_target(tilt_target, joint_ids=arm_tilt_ids)
            robot.set_joint_velocity_target(spin_target, joint_ids=rotor_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args_cli.device if hasattr(args_cli, "device") else "cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 1.0])

    scene_cfg = DroneSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete, starting simulation...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
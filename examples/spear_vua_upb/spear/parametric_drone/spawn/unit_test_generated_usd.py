import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize one drone USD.")

source_group = parser.add_mutually_exclusive_group(required=False)
source_group.add_argument(
    "--usd",
    type=str,
    default=None,
    help="Full path to one USD file.",
)
source_group.add_argument(
    "--usd_name",
    type=str,
    default=None,
    help="USD filename inside --usd_dir, for example novel_hex_drone.usd",
)

parser.add_argument(
    "--usd_dir",
    type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/conversion/usd",
    help="Directory used with --usd_name or default auto-discovery.",
)
parser.add_argument("--x", type=float, default=0.0, help="Spawn x position.")
parser.add_argument("--y", type=float, default=0.0, help="Spawn y position.")
parser.add_argument("--z", type=float, default=1.5, help="Spawn z position.")
parser.add_argument("--prim_path", type=str, default="/World/drone", help="Prim path for the spawned USD.")
parser.add_argument("--no_physics_override", action="store_true", help="Spawn the USD without rigid/articulation property overrides.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def resolve_usd_path(args) -> Path:
    usd_dir = Path(args.usd_dir).resolve()

    if args.usd is not None:
        usd_path = Path(args.usd).resolve()
    elif args.usd_name is not None:
        usd_path = (usd_dir / args.usd_name).resolve()
    else:
        if not usd_dir.exists():
            raise FileNotFoundError(f"USD directory does not exist: {usd_dir}")
        usd_files = sorted(usd_dir.glob("*.usd"))
        if not usd_files:
            raise FileNotFoundError(f"No USD files found in: {usd_dir}")
        usd_path = usd_files[0]

    if not usd_path.exists():
        raise FileNotFoundError(f"USD file does not exist: {usd_path}")
    return usd_path


usd_path = resolve_usd_path(args_cli)

sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([3.0, 3.0, 2.0], [args_cli.x, args_cli.y, max(0.5, args_cli.z - 0.5)])

ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/GroundPlane", ground_cfg)

if args_cli.no_physics_override:
    drone_cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
    )
else:
    drone_cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping=1000.0,
            angular_damping=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    )

drone_cfg.func(
    prim_path=args_cli.prim_path,
    cfg=drone_cfg,
    translation=(args_cli.x, args_cli.y, args_cli.z),
)

print(f"Loaded USD: {usd_path}")
print(f"Spawned at: ({args_cli.x:.3f}, {args_cli.y:.3f}, {args_cli.z:.3f})")
print(f"Prim path: {args_cli.prim_path}")
print(f"Physics override: {not args_cli.no_physics_override}")

sim.reset()

while simulation_app.is_running():
    sim.step()

simulation_app.close()
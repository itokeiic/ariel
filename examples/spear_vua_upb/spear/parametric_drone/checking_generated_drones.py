import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# Launch Isaac Sim through Isaac Lab
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Visualize multiple drone airframes loaded at shared roots and switched over time."
)
parser.add_argument(
    "--usd_dir",
    type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated_gp/usd",
    help="Directory containing generated drone USD files.",
)
parser.add_argument("--spacing", type=float, default=3.0, help="Grid spacing between drone roots.")
parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid.")
parser.add_argument("--z", type=float, default=1.5, help="Spawn height.")
parser.add_argument("--num_roots", type=int, default=4, help="How many shared drone roots to spawn.")
parser.add_argument("--hold_sec", type=float, default=1.5, help="Seconds to hold each airframe before switching.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Import simulator-dependent modules only after app launch
# -----------------------------------------------------------------------------
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import UsdGeom, Gf


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_stage():
    return omni.usd.get_context().get_stage()


def define_xform(path: str):
    stage = get_stage()
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def get_or_add_translate_op(prim):
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xform.AddTranslateOp()


def get_or_add_scale_op(prim):
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            return op
    return xform.AddScaleOp()


def set_visibility(prim, visible: bool):
    imageable = UsdGeom.Imageable(prim)
    imageable.GetVisibilityAttr().Set("inherited" if visible else "invisible")


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------
sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([10.0, 10.0, 7.0], [4.0, 4.0, 1.5])

ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/GroundPlane", ground_cfg)

usd_dir = Path(args_cli.usd_dir).resolve()
if not usd_dir.exists():
    raise FileNotFoundError(f"USD directory does not exist: {usd_dir}")

usd_files = sorted(usd_dir.glob("*.usd"))
if not usd_files:
    raise FileNotFoundError(f"No USD files found in: {usd_dir}")

print(f"Found {len(usd_files)} USD files in: {usd_dir}")

stage = get_stage()
roots = []

for i in range(args_cli.num_roots):
    row = i // args_cli.cols
    col = i % args_cli.cols

    x = col * args_cli.spacing
    y = row * args_cli.spacing
    z = args_cli.z

    root_path = f"/World/drone_{i:02d}"
    root_prim = define_xform(root_path)
    root_translate_op = get_or_add_translate_op(root_prim)
    root_translate_op.Set(Gf.Vec3d(x, y, z))

    forms = []

    for j, usd_path in enumerate(usd_files):
        form_path = f"{root_path}/form_{j:02d}"

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
            prim_path=form_path,
            cfg=drone_cfg,
            translation=(0.0, 0.0, 0.0),
        )

        form_prim = stage.GetPrimAtPath(form_path)
        scale_op = get_or_add_scale_op(form_prim)
        translate_op = get_or_add_translate_op(form_prim)

        scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        set_visibility(form_prim, j == 0)

        forms.append(
            {
                "name": usd_path.name,
                "prim": form_prim,
                "scale_op": scale_op,
                "translate_op": translate_op,
            }
        )

    roots.append(
        {
            "idx": i,
            "root_path": root_path,
            "forms": forms,
            "time_offset": i * 0.35,
        }
    )

    print(f"Spawned {root_path} at ({x:.2f}, {y:.2f}, {z:.2f}) with {len(forms)} airframes")

sim.reset()


# -----------------------------------------------------------------------------
# Main loop: hard switch, exactly one visible airframe per root
# -----------------------------------------------------------------------------
t = 0.0
dt = sim.get_physics_dt()

while simulation_app.is_running():
    for root in roots:
        forms = root["forms"]
        n = len(forms)

        idx = int((t + root["time_offset"]) / args_cli.hold_sec) % n

        for k, form in enumerate(forms):
            visible = (k == idx)
            set_visibility(form["prim"], visible)

            if visible:
                form["scale_op"].Set(Gf.Vec3f(1.0, 1.0, 1.0))
                form["translate_op"].Set(Gf.Vec3d(0.0, 0.0, 0.0))
            else:
                form["scale_op"].Set(Gf.Vec3f(1.0, 1.0, 1.0))
                form["translate_op"].Set(Gf.Vec3d(0.0, 0.0, 0.0))

    sim.step()
    t += dt

simulation_app.close()
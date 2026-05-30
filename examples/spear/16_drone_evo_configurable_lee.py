"""Drone morphology evolution with the Lee geometric controller.

Companion to `9_drone_evo_configurable.py`, but PPO is removed entirely.
Every candidate body is evaluated by a single deterministic rollout using
the Lee geometric controller flying a B-spline trajectory through the
quintic gate circuit. Fitness = number of gates passed during the rollout.

Each evaluation takes ~1 second per individual (no RL training), so a
50-generation × 10-pop run completes in a few minutes on CPU.

Quick start
-----------
    uv run examples/spear/16_drone_evo_configurable_lee.py

CLI overrides:
    uv run examples/spear/16_drone_evo_configurable_lee.py \\
        --seed 7 --out-dir __data__/my_run

NOTE: do NOT add ``from __future__ import annotations`` to this file —
ariel's @EAOperation decorator needs real (not stringified) type hints
at import time.
"""
# ─────────────────────────────────────────────────────────────────────────────
# Standard-library / third-party imports  (do not edit)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.progress import track

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator"))
from planner_generator import generate_paths_from_coefficients  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "examples" / "d_drones"))
from _ctrl_helpers import GateChecker  # noqa: E402

from ariel.body_phenotypes.drone import (
    crossover_drones,
    initialize_drones,
    mutate_drones,
    parent_tag,
    truncation_select,
)
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.genome import deserialize_genome
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from ariel.simulation.drone.controllers.lee_control.lee_controller import (
    LeeGeometricControl,
)
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import (
    Trajectory,
)
from ariel.simulation.drone.controllers.utils.wind_model import Wind
from ariel.simulation.drone.drone_interface import DroneInterface

# ─────────────────────────────────────────────────────────────────────────────
# Minimal CLI
# ─────────────────────────────────────────────────────────────────────────────
curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(
    description="Drone morphology evolution + Lee controller (gates passed fitness)"
)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/drone_evo_lee/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 1 — EA  (evolutionary algorithm)
# ═════════════════════════════════════════════════════════════════════════════

EA_POP_SIZE  = 10
EA_GENS      = 20 #50

N_ARMS_MIN = 6
N_ARMS_MAX = 6

# Lee controller gains are tuned for planar X-quads. Random non-planar
# morphologies (large arm elevation / motor disc pitch) don't fly under
# these fixed gains and would score 0 gates, so the elevation/pitch
# columns are clamped to ±15°. Arm length / azimuth / spin direction
# stay wide so the EA still has meaningful geometry to evolve.
_DEG15 = np.deg2rad(15.0)
PARAMETER_LIMITS = np.array([
    [0.055, 0.17],     # arm length
    [-np.pi, np.pi],   # arm azimuth
    [-_DEG15, _DEG15], # arm elevation       (was: ±π/2)
    [-np.pi, np.pi],   # motor disc azimuth
    [-_DEG15, _DEG15], # motor disc pitch    (was: ±π)
    [0, 1],            # spin direction (binary)
])

APPEND_ARM_CHANCE = 0.0
BILATERAL_SYMMETRY = None
REPAIR = False
MUTATION_SCALES = None

CUSTOM_GENERATION_OPS = None


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 2 — GATE TRACK
# ═════════════════════════════════════════════════════════════════════════════

GATE_PATH_STEPS = 15    # gates sampled per circuit
GATE_PATH_SCALE = 5.0   # metres — spatial scale of the circuit
GATE_Z_HEIGHT   = -1.5  # NED z (metres)


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 3 — LEE CONTROLLER
#
#  Per-individual rollout: a single deterministic flight of the Lee
#  geometric controller along a B-spline through the quintic gates.
#  Attitude/rate gains are auto-scaled from each body's inertia.
# ═════════════════════════════════════════════════════════════════════════════

SIM_TIME     = 20.0    # seconds per rollout
SIM_DT       = 0.005   # integration timestep
LEE_POS_GAIN = 14.3    # position P gain (default from 3_simulate_lee.py)
LEE_VEL_GAIN = 9.0     # velocity P gain (default from 3_simulate_lee.py)
GATE_SIZE    = 1.0     # gate opening size (m), used by GateChecker
PROP_SIZE    = 2

# ── Shaped fitness weights ────────────────────────────────────────────────────
# fitness = GATE_BONUS * gates_passed
#         + survival_fraction              ∈ [0, 1], 1.0 if no divergence
#         - TRACK_WEIGHT * mean_tracking_error  (mean ||pos - desired_pos|| over rollout)
#
# The shaping makes fitness continuous even for bodies that pass 0 gates:
# a hovering drone scores ~1.0; a body that tracks the B-spline closely
# scores higher; a body that diverges instantly scores ~0. Gate passes
# dominate the score (10 per gate) once any morphology can fly the course.
GATE_BONUS    = 10.0
TRACK_WEIGHT  = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Below this line: experiment logic
# ─────────────────────────────────────────────────────────────────────────────

DATA   = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DB_PATH = DATA / f"database_{RUN_ID}.db"

_COEFFS_PATH = (
    _REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator" / "quintic_coeffs.npy"
)
COEFFS = np.load(_COEFFS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Gate track helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quintic_to_gates(
    coeffs: np.ndarray,
    n_gates: int,
    scale: float,
    z: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one quintic path → gate_pos (N×3) and gate_yaw (N,)."""
    OVERSAMPLE = max(n_gates * 8, 64)
    MIN_SPACING = 0.3
    paths, yaws_dense_arr = generate_paths_from_coefficients(
        coeffs, num_generate=1, steps=OVERSAMPLE, seed=seed, clip_range=(-1.0, 1.0),
    )
    xy_dense  = paths[0] * scale
    yaw_dense = yaws_dense_arr[0]

    kept = [0]
    for idx in range(1, len(xy_dense)):
        if np.linalg.norm(xy_dense[idx] - xy_dense[kept[-1]]) >= MIN_SPACING:
            kept.append(idx)
        if len(kept) == n_gates:
            break
    if len(kept) < n_gates:
        kept = list(np.linspace(0, len(xy_dense) - 1, n_gates, dtype=int))

    xy       = xy_dense[kept]
    gate_pos = np.column_stack([xy, np.full(n_gates, z)]).astype(np.float64)
    gate_yaw = yaw_dense[kept].astype(np.float64)
    return gate_pos, gate_yaw


class _QuinticGateConfig:
    """Minimal GateConfig-shaped object for Trajectory(...)."""

    def __init__(self, gate_pos, gate_yaw, gate_size, starting_pos):
        self.gate_pos     = np.asarray(gate_pos)
        self.gate_yaw     = np.asarray(gate_yaw)
        self.gate_size    = float(gate_size)
        self.starting_pos = np.asarray(starting_pos)


# ─────────────────────────────────────────────────────────────────────────────
# Genome handler
# ─────────────────────────────────────────────────────────────────────────────

genome_handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(N_ARMS_MIN, N_ARMS_MAX),
    parameter_limits=PARAMETER_LIMITS,
    append_arm_chance=APPEND_ARM_CHANCE,
    bilateral_plane_for_symmetry=BILATERAL_SYMMETRY,
    repair=REPAIR,
    mutation_scales_percentage=MUTATION_SCALES,
    rnd=np.random.default_rng(args.seed),
)


# ─────────────────────────────────────────────────────────────────────────────
# Generate the single quintic gate track shared across the run
# ─────────────────────────────────────────────────────────────────────────────

_gate_pos, _gate_yaw = _quintic_to_gates(
    COEFFS, GATE_PATH_STEPS, GATE_PATH_SCALE, GATE_Z_HEIGHT, seed=args.seed,
)
_gate_cfg = _QuinticGateConfig(
    gate_pos=_gate_pos,
    gate_yaw=_gate_yaw,
    gate_size=GATE_SIZE,
    starting_pos=_gate_pos[0] + np.array([0.0, -1.0, 0.0]),
)


# ─────────────────────────────────────────────────────────────────────────────
# Lee rollout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_lee_stack(propellers):
    """Construct drone_property + controller + trajectory + gate checker.

    Returns (drone_property, ctrl, traj, gate_checker, wind). Raises on bad morphology
    (e.g. singular allocation), caller wraps in try/except.
    """
    drone_property = DroneInterface(0, propellers=propellers)
    wind = Wind("None")
    ctrl = LeeGeometricControl(
        drone_property, yawType=1, orient="NED", auto_scale_gains=True,
        pos_P_gain=np.array([LEE_POS_GAIN] * 3),
        vel_P_gain=np.array([LEE_VEL_GAIN] * 3),
    )
    traj = Trajectory(drone_property, "xyz_pos", np.array([15, 3, 1]), gate_config=_gate_cfg)

    start_pos, _, _ = traj.bspline_trajectory.evaluate(0.0)
    _, vel_050, _   = traj.bspline_trajectory.evaluate(0.05)
    initial_yaw = (
        float(np.arctan2(vel_050[1], vel_050[0]))
        if np.linalg.norm(vel_050[:2]) > 1e-3
        else float(_gate_cfg.gate_yaw[0])
    )
    drone_property.drone_sim.set_state(
        position=start_pos,
        velocity=np.zeros(3),
        attitude=np.array([0.0, 0.0, initial_yaw]),
        angular_velocity=np.zeros(3),
    )
    drone_property._update_state_variables()

    gate_checker = GateChecker(
        _gate_cfg.gate_pos, _gate_cfg.gate_yaw, _gate_cfg.gate_size,
    )

    # Seed first command (same as 3_simulate_lee.py:154-155)
    sDes = traj.desiredState(0.0, SIM_DT, drone_property)
    ctrl.controller(sDes, drone_property, traj.ctrlType, SIM_DT)

    return drone_property, ctrl, traj, gate_checker, wind


def _eval_with_lee(genotype: dict) -> tuple[float, int, dict]:
    """Decode genome → Lee rollout → return (fitness, n_motors, components).

    fitness = GATE_BONUS * gates_passed + survival_fraction - TRACK_WEIGHT * mean_tracking_error
    components is a dict with the individual terms for logging.
    """
    bad = (float("-inf"), 0, {"gates": 0, "survival": 0.0, "tracking_err": float("inf")})

    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return bad

    if not np.isfinite(genome.arms[:, 0]).any():
        return bad

    try:
        bp         = spherical_angular_to_blueprint(genome.arms, propsize=PROP_SIZE)
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return bad

    n_motors = len(propellers)

    try:
        drone_property, ctrl, traj, gate_checker, wind = _build_lee_stack(propellers)
    except Exception:
        return float("-inf"), n_motors, {"gates": 0, "survival": 0.0, "tracking_err": float("inf")}

    n_steps        = int(SIM_TIME / SIM_DT)
    tracking_sum   = 0.0
    steps_done     = 0

    try:
        t, i = 0.0, 1
        while i <= n_steps:
            drone_property.update(t, SIM_DT, ctrl.w_cmd, wind)
            t_new = SIM_DT * i
            sDes = traj.desiredState(t_new, SIM_DT, drone_property)
            ctrl.controller(sDes, drone_property, traj.ctrlType, SIM_DT)
            gate_checker.check_gate_passing(drone_property.pos)
            # sDes[:3] is desired position from the trajectory
            tracking_sum += float(np.linalg.norm(drone_property.pos - sDes[:3]))
            steps_done   += 1
            t = t_new
            i += 1
    except Exception:
        # Numerical divergence (Euler-singularity guard in DroneSimulator.step).
        pass

    survival      = steps_done / n_steps if n_steps > 0 else 0.0
    tracking_err  = tracking_sum / steps_done if steps_done > 0 else float("inf")
    gates         = int(gate_checker.gates_passed)
    fitness       = GATE_BONUS * gates + survival - TRACK_WEIGHT * tracking_err

    return float(fitness), n_motors, {
        "gates": gates, "survival": survival, "tracking_err": tracking_err,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EA evaluation operation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_lee(population: Population) -> Population:
    """Run a Lee-controlled rollout for every unevaluated body; assign fitness."""
    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for _i, ind in enumerate(track(to_eval, description="Lee rollout…", console=console)):
        t0 = time.time()
        fit, n_motors, comp = _eval_with_lee(ind.genotype)
        ind.fitness = fit
        display_id = ind.id if ind.id is not None else f"#{_i + 1}"
        console.log(
            f"  id={display_id}  motors={n_motors}  "
            f"fit={fit:.3f}  gates={comp['gates']}  "
            f"surv={comp['survival']:.2f}  trk_err={comp['tracking_err']:.2f}m  "
            f"({time.time()-t0:.2f}s)"
        )

    finite = [ind.fitness_ for ind in population.alive.evaluated if np.isfinite(ind.fitness_)]  # type: ignore[arg-type]
    if finite:
        console.log(f"    batch done — max={max(finite):.3f}  mean={np.mean(finite):.3f}")
    return population


eval_op = EAOperation(evaluate_with_lee)


# ─────────────────────────────────────────────────────────────────────────────
# Build generation pipeline
# ─────────────────────────────────────────────────────────────────────────────

if CUSTOM_GENERATION_OPS is not None:
    generation_ops = [eval_op if op is None else op for op in CUSTOM_GENERATION_OPS]
else:
    generation_ops = [
        parent_tag(n=EA_POP_SIZE),
        crossover_drones(template_handler=genome_handler),
        mutate_drones(template_handler=genome_handler),
        eval_op,
        truncation_select(n=EA_POP_SIZE),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold blue]Drone Body Evolution + Lee controller (gates passed)")
console.log(
    f"pop={EA_POP_SIZE}  gens={EA_GENS}  arms={N_ARMS_MIN}–{N_ARMS_MAX}  "
    f"sim_time={SIM_TIME}s  sim_dt={SIM_DT}s  "
    f"gate_steps={GATE_PATH_STEPS}  scale={GATE_PATH_SCALE}  "
    f"pos_gain={LEE_POS_GAIN}  vel_gain={LEE_VEL_GAIN}"
)
console.log(f"DB → {DB_PATH}")

initial_pop = Population([Individual() for _ in range(EA_POP_SIZE)])
init_op     = initialize_drones(template_handler=genome_handler)

console.log("Initializing + evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

finite0 = [ind.fitness_ for ind in initial_pop if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
if finite0:
    console.log(f"Initial — max={max(finite0):.1f}  mean={np.mean(finite0):.2f}")

console.rule("[bold green]Evolving")
ea = EA(
    population=initial_pop,
    operations=generation_ops,
    num_steps=EA_GENS,
    db_file_path=DB_PATH,
    db_handling="delete",
)
ea.run()


# ─────────────────────────────────────────────────────────────────────────────
# Save best individual
# ─────────────────────────────────────────────────────────────────────────────

ea.fetch_population(only_alive=False, requires_eval=False)
finite = [ind for ind in ea.population if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
if not finite:
    console.log("[red]No valid individuals — aborting.[/red]")
    raise SystemExit(1)

best        = sorted(finite, key=lambda ind: ind.fitness_, reverse=True)[0]
best_genome = deserialize_genome(best.genotype)
best_bp     = spherical_angular_to_blueprint(best_genome.arms, propsize=PROP_SIZE)

bp_path = DATA / f"best_blueprint_{RUN_ID}.json"
best_bp.save_json(bp_path)
console.log(f"Best blueprint → {bp_path}")

np.save(DATA / f"gate_pos_{RUN_ID}.npy", _gate_cfg.gate_pos)
np.save(DATA / f"gate_yaw_{RUN_ID}.npy", _gate_cfg.gate_yaw)

console.log(
    f"[bold green]Done.[/bold green]  "
    f"Best fitness: {best.fitness_:.3f}  DB → {DB_PATH}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Fitness history PNG
# ─────────────────────────────────────────────────────────────────────────────

try:
    import sqlite3
    from collections import defaultdict

    import matplotlib.pyplot as plt

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT time_of_birth, fitness_ FROM individual "
        "WHERE fitness_ IS NOT NULL ORDER BY time_of_birth"
    ).fetchall()
    conn.close()

    gen_fits: dict[int, list[float]] = defaultdict(list)
    for gen, fit in rows:
        gen_fits[int(gen)].append(float(fit))

    gens     = sorted(gen_fits)
    best_fit = [max(gen_fits[g]) for g in gens]
    mean_fit = [float(np.mean(gen_fits[g])) for g in gens]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gens, best_fit, "o-", color="steelblue", label="best")
    ax.plot(gens, mean_fit, "s--", color="tomato",   label="mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (gate_bonus*gates + survival - track_weight*track_err)")
    ax.set_title("Drone morphology evolution — Lee controller shaped fitness")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fitness_png = DATA / f"fitness_history_{RUN_ID}.png"
    fig.savefig(fitness_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.log(f"Fitness history → {fitness_png}")
except Exception as exc:
    console.log(f"[yellow]Fitness plot skipped: {exc}[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# MP4 video — best individual flying the gate track via Lee controller
# ─────────────────────────────────────────────────────────────────────────────

try:
    import ariel.simulation.drone.controllers.utils as ctrl_utils

    best_propellers = blueprint_to_propellers(best_bp, convention="ned")
    drone_property, ctrl, traj, gate_checker, wind = _build_lee_stack(best_propellers)

    n_steps = int(SIM_TIME / SIM_DT) + 1
    t_all          = np.zeros(n_steps)
    pos_all        = np.zeros((n_steps, 3))
    quat_all       = np.zeros((n_steps, 4))
    sDes_traj_all  = np.zeros((n_steps, len(traj.sDes)))

    t, i = 0.0, 1
    while i < n_steps:
        t_all[i]         = t
        pos_all[i]       = drone_property.pos
        quat_all[i]      = drone_property.quat
        sDes_traj_all[i] = traj.sDes

        drone_property.update(t, SIM_DT, ctrl.w_cmd, wind)
        t_new = SIM_DT * i
        sDes = traj.desiredState(t_new, SIM_DT, drone_property)
        ctrl.controller(sDes, drone_property, traj.ctrlType, SIM_DT)
        gate_checker.check_gate_passing(drone_property.pos)

        t = t_new
        i += 1

    waypoints = np.array(_gate_cfg.gate_pos, dtype=float)

    # Try .mp4 first; fall back to .gif if ffmpeg isn't available.
    import matplotlib.animation as _mpl_anim
    has_ffmpeg = _mpl_anim.writers.is_available("ffmpeg")
    ext = "mp4" if has_ffmpeg else "gif"
    save_path = DATA / f"best_flight_{RUN_ID}.{ext}"

    anim = ctrl_utils.sameAxisAnimation(
        t_all, waypoints, pos_all, quat_all, sDes_traj_all, SIM_DT,
        drone_property.params, 15, 3, 1, "NED",
        gate_pos=np.array(_gate_cfg.gate_pos),
        gate_yaw=np.array(_gate_cfg.gate_yaw),
        gate_size=_gate_cfg.gate_size,
        save_path=str(save_path),
    )
    del anim
    console.log(
        f"Best flight rollout — gates_passed={gate_checker.gates_passed}, video → {save_path}"
    )
except Exception as exc:
    console.log(f"[yellow]Video skipped: {exc}[/yellow]")

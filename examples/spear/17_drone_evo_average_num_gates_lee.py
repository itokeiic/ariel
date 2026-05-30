"""Drone morphology evolution with the Lee controller and multi-trajectory fitness.

Combines `15_drone_evo_average_num_gates.py` (10 quintic circuits, fitness
averaged across all) with `16_drone_evo_configurable_lee.py` (Lee geometric
controller — no PPO training). Each candidate body is evaluated by running
N_TRAJECTORIES=10 deterministic Lee-controlled rollouts (one per circuit);
fitness = mean shaped fitness across the 10 circuits.

Outputs:
  - best_blueprint_*.json      — body of the best individual
  - gate_pos_*_NN.npy / gate_yaw_*_NN.npy  (NN = 00–09)
  - fitness_history_*.png      — best/mean fitness vs. cumulative fitness evals
  - avg_gates_history_*.png    — best/mean avg-gates-passed vs. cumulative fitness evals
  - best_flight_*.mp4          — MuJoCo render of best body on all 10 circuits sequentially

Quick start
-----------
    uv run examples/spear/17_drone_evo_average_num_gates_lee.py

CLI overrides:
    uv run examples/spear/17_drone_evo_average_num_gates_lee.py \\
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
    description="Drone morphology evolution + Lee controller + 10-trajectory avg fitness"
)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/drone_evo_lee_avg/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 1 — EA  (evolutionary algorithm)
# ═════════════════════════════════════════════════════════════════════════════

EA_POP_SIZE  = 10
EA_GENS      = 20 # 50

N_ARMS_MIN = 6
N_ARMS_MAX = 6

# Lee controller gains are tuned for planar X-quads. Random non-planar
# morphologies (large arm elevation / motor disc pitch) don't fly under
# these fixed gains, so the elevation/pitch columns are clamped to ±15°.
# Arm length / azimuth / spin direction stay wide so the EA still has
# meaningful geometry to evolve.
_DEG15 = np.deg2rad(15.0)
PARAMETER_LIMITS = np.array([
    [0.055, 0.17],     # arm length
    [-np.pi, np.pi],   # arm azimuth
    [-_DEG15, _DEG15], # arm elevation       (clamped vs ex. 9's ±π/2)
    [-np.pi, np.pi],   # motor disc azimuth
    [-_DEG15, _DEG15], # motor disc pitch    (clamped vs ex. 9's ±π)
    [0, 1],            # spin direction (binary)
])

APPEND_ARM_CHANCE = 0.0
BILATERAL_SYMMETRY = None
REPAIR = False
MUTATION_SCALES = None

CUSTOM_GENERATION_OPS = None


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 2 — GATE TRACKS
#
#  N_TRAJECTORIES distinct quintic circuits are generated at startup;
#  fitness for each candidate body is the MEAN shaped fitness across all
#  N_TRAJECTORIES circuits (deterministic, same circuits for everyone).
# ═════════════════════════════════════════════════════════════════════════════

N_TRAJECTORIES  = 10    # number of gate circuits to average fitness over
GATE_PATH_STEPS = 15    # gates sampled per circuit
GATE_PATH_SCALE = 5.0   # metres — spatial scale of each circuit
GATE_Z_HEIGHT   = -1.5  # NED z (metres)


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 3 — LEE CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

SIM_TIME     = 20.0    # seconds per rollout (per circuit)
SIM_DT       = 0.005   # integration timestep
LEE_POS_GAIN = 14.3    # position P gain (default from 3_simulate_lee.py)
LEE_VEL_GAIN = 9.0     # velocity P gain (default from 3_simulate_lee.py)
GATE_SIZE    = 1.0     # gate opening size (m), used by GateChecker
PROP_SIZE    = 2

# ── Shaped fitness weights ────────────────────────────────────────────────────
# per-circuit fitness =
#     GATE_BONUS * gates_passed
#     + survival_fraction
#     - TRACK_WEIGHT * mean_tracking_error
#     + (COMPLETION_BONUS * remaining_time_fraction  if course completed)
#
# The rollout breaks early once all gates have been passed once (no second
# lap). COMPLETION_BONUS rewards bodies that finish the course faster:
# remaining_time_fraction = (n_steps - steps_done) / n_steps  ∈ [0, 1].
# A body that completes instantly gets +COMPLETION_BONUS; one that just
# barely completes at the last step gets +0.
# Individual fitness = mean of per-circuit shaped fitness across N_TRAJECTORIES.
GATE_BONUS       = 10.0
TRACK_WEIGHT     = 0.1
COMPLETION_BONUS = 10.0   # +10 max for instant completion, scaled by remaining time


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

def _multi_quintic_to_gates(
    coeffs: np.ndarray,
    n_trajectories: int,
    n_gates: int,
    scale: float,
    z: float,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate n_trajectories gate circuits from the quintic planner.

    Returns a list of (gate_pos [N×3], gate_yaw [N]) tuples.
    """
    OVERSAMPLE = max(n_gates * 8, 64)
    MIN_SPACING = 0.3
    paths, yaws_dense_arr = generate_paths_from_coefficients(
        coeffs, num_generate=n_trajectories, steps=OVERSAMPLE,
        seed=seed, clip_range=(-1.0, 1.0),
    )
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for t in range(n_trajectories):
        xy_dense  = paths[t] * scale
        yaw_dense = yaws_dense_arr[t]

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
        result.append((gate_pos, gate_yaw))
    return result


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
# Generate the N_TRAJECTORIES gate circuits shared across the run
# ─────────────────────────────────────────────────────────────────────────────

_all_gate_sets = _multi_quintic_to_gates(
    COEFFS, N_TRAJECTORIES, GATE_PATH_STEPS, GATE_PATH_SCALE, GATE_Z_HEIGHT, seed=args.seed,
)
_all_gate_cfgs = [
    _QuinticGateConfig(
        gate_pos=gp,
        gate_yaw=gy,
        gate_size=GATE_SIZE,
        starting_pos=gp[0] + np.array([0.0, -1.0, 0.0]),
    )
    for (gp, gy) in _all_gate_sets
]


# ─────────────────────────────────────────────────────────────────────────────
# Lee rollout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_lee_stack(propellers, gate_cfg):
    """Construct drone_property + controller + trajectory + gate checker for one circuit.

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
    traj = Trajectory(drone_property, "xyz_pos", np.array([15, 3, 1]), gate_config=gate_cfg)

    start_pos, _, _ = traj.bspline_trajectory.evaluate(0.0)
    _, vel_050, _   = traj.bspline_trajectory.evaluate(0.05)
    initial_yaw = (
        float(np.arctan2(vel_050[1], vel_050[0]))
        if np.linalg.norm(vel_050[:2]) > 1e-3
        else float(gate_cfg.gate_yaw[0])
    )
    drone_property.drone_sim.set_state(
        position=start_pos,
        velocity=np.zeros(3),
        attitude=np.array([0.0, 0.0, initial_yaw]),
        angular_velocity=np.zeros(3),
    )
    drone_property._update_state_variables()

    gate_checker = GateChecker(
        gate_cfg.gate_pos, gate_cfg.gate_yaw, gate_cfg.gate_size,
    )

    # Seed first command (same as 3_simulate_lee.py:154-155)
    sDes = traj.desiredState(0.0, SIM_DT, drone_property)
    ctrl.controller(sDes, drone_property, traj.ctrlType, SIM_DT)

    return drone_property, ctrl, traj, gate_checker, wind


def _rollout_one_circuit(propellers, gate_cfg) -> dict:
    """Run a single Lee-controlled rollout on one gate circuit.

    Returns a dict with: fit, gates, survival, tracking_err.
    All fields are populated even on mid-rollout divergence (-inf is
    only returned if the controller stack can't be built at all).
    """
    try:
        drone_property, ctrl, traj, gate_checker, wind = _build_lee_stack(propellers, gate_cfg)
    except Exception:
        return {"fit": float("-inf"), "gates": 0,
                "survival": 0.0, "tracking_err": float("inf")}

    n_steps      = int(SIM_TIME / SIM_DT)
    n_gates      = gate_checker.num_gates
    tracking_sum = 0.0
    steps_done   = 0
    completed    = False

    try:
        t, i = 0.0, 1
        while i <= n_steps:
            drone_property.update(t, SIM_DT, ctrl.w_cmd, wind)
            t_new = SIM_DT * i
            sDes = traj.desiredState(t_new, SIM_DT, drone_property)
            ctrl.controller(sDes, drone_property, traj.ctrlType, SIM_DT)
            gate_checker.check_gate_passing(drone_property.pos)
            tracking_sum += float(np.linalg.norm(drone_property.pos - sDes[:3]))
            steps_done   += 1
            t = t_new
            i += 1
            # Stop once the drone has flown through every gate once — the
            # B-spline trajectory and GateChecker both cycle back otherwise,
            # which would inflate gate counts with extra laps.
            if gate_checker.gates_passed >= n_gates:
                completed = True
                break
    except Exception:
        # Numerical divergence (Euler-singularity guard in DroneSimulator.step).
        pass

    survival      = steps_done / n_steps if n_steps > 0 else 0.0
    tracking_err  = tracking_sum / steps_done if steps_done > 0 else float("inf")
    gates         = int(gate_checker.gates_passed)
    # Reward fast completion: full bonus for instant finish, 0 for using all of SIM_TIME.
    time_bonus    = COMPLETION_BONUS * (1.0 - survival) if completed else 0.0
    fit           = GATE_BONUS * gates + survival - TRACK_WEIGHT * tracking_err + time_bonus

    return {"fit": fit, "gates": gates,
            "survival": survival, "tracking_err": tracking_err,
            "completed": completed, "time_bonus": time_bonus}


def _eval_with_lee(genotype: dict) -> tuple[float, int, dict]:
    """Decode genome → run Lee rollout on each of N_TRAJECTORIES circuits → mean fitness."""
    bad_comp = {"avg_gates": 0.0, "avg_surv": 0.0, "avg_trk_err": float("inf"),
                "gates_per_circuit": [0] * N_TRAJECTORIES}

    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return float("-inf"), 0, bad_comp

    if not np.isfinite(genome.arms[:, 0]).any():
        return float("-inf"), 0, bad_comp

    try:
        bp         = spherical_angular_to_blueprint(genome.arms, propsize=PROP_SIZE)
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return float("-inf"), 0, bad_comp

    n_motors = len(propellers)

    fits: list[float]      = []
    gates_list: list[int]  = []
    surv_list: list[float] = []
    trk_list: list[float]  = []
    completed_list: list[bool] = []

    for gate_cfg in _all_gate_cfgs:
        out = _rollout_one_circuit(propellers, gate_cfg)
        fits.append(out["fit"])
        gates_list.append(out["gates"])
        surv_list.append(out["survival"])
        trk_list.append(out["tracking_err"])
        completed_list.append(out["completed"])

    finite_fits = [f for f in fits if np.isfinite(f)]
    finite_trk  = [t for t in trk_list if np.isfinite(t)]
    mean_fit = float(np.mean(finite_fits)) if finite_fits else float("-inf")

    comp = {
        "avg_gates":   float(np.mean(gates_list)),
        "avg_surv":    float(np.mean(surv_list)),
        "avg_trk_err": float(np.mean(finite_trk)) if finite_trk else float("inf"),
        "gates_per_circuit": gates_list,
        "n_completed":  int(sum(completed_list)),
    }
    return mean_fit, n_motors, comp


# ─────────────────────────────────────────────────────────────────────────────
# EA evaluation operation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_lee(population: Population) -> Population:
    """Run a 10-circuit Lee-rollout for every unevaluated body; assign mean fitness."""
    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for _i, ind in enumerate(track(to_eval, description="Lee rollout ×10…", console=console)):
        t0 = time.time()
        fit, n_motors, comp = _eval_with_lee(ind.genotype)
        ind.fitness = fit
        # Persist multi-trajectory metrics so the plots can be built post-run.
        ind.tags["avg_gates"]         = float(comp["avg_gates"])
        ind.tags["avg_trk_err"]       = float(comp["avg_trk_err"])
        ind.tags["gates_per_circuit"] = list(comp["gates_per_circuit"])
        ind.tags["n_completed"]       = int(comp["n_completed"])
        display_id = ind.id if ind.id is not None else f"#{_i + 1}"
        console.log(
            f"  id={display_id}  motors={n_motors}  "
            f"fit={fit:.3f}  avg_gates={comp['avg_gates']:.2f}  "
            f"completed={comp['n_completed']}/{N_TRAJECTORIES}  "
            f"avg_surv={comp['avg_surv']:.2f}  avg_trk={comp['avg_trk_err']:.2f}m  "
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

console.rule("[bold blue]Drone Body Evolution + Lee controller (avg fitness over 10 circuits)")
console.log(
    f"pop={EA_POP_SIZE}  gens={EA_GENS}  arms={N_ARMS_MIN}–{N_ARMS_MAX}  "
    f"sim_time={SIM_TIME}s  sim_dt={SIM_DT}s  "
    f"n_trajectories={N_TRAJECTORIES}  gate_steps={GATE_PATH_STEPS}  "
    f"scale={GATE_PATH_SCALE}  pos_gain={LEE_POS_GAIN}  vel_gain={LEE_VEL_GAIN}"
)
console.log(f"DB → {DB_PATH}")

initial_pop = Population([Individual() for _ in range(EA_POP_SIZE)])
init_op     = initialize_drones(template_handler=genome_handler)

console.log("Initializing + evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

finite0 = [ind.fitness_ for ind in initial_pop if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
if finite0:
    console.log(f"Initial — max={max(finite0):.3f}  mean={np.mean(finite0):.3f}")

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
# Save best individual + all 10 gate tracks
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

for idx, (gp, gy) in enumerate(_all_gate_sets):
    np.save(DATA / f"gate_pos_{RUN_ID}_{idx:02d}.npy", gp)
    np.save(DATA / f"gate_yaw_{RUN_ID}_{idx:02d}.npy", gy)

console.log(
    f"[bold green]Done.[/bold green]  Best fitness: {best.fitness_:.3f}  "
    f"best avg_gates: {best.tags.get('avg_gates', 0):.2f}  DB → {DB_PATH}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — fitness vs. cumulative fitness function evaluations
# Plot 2 — avg gates passed vs. cumulative fitness function evaluations
# ─────────────────────────────────────────────────────────────────────────────

try:
    import sqlite3
    from collections import defaultdict

    import matplotlib.pyplot as plt

    # ---- Plot 1: fitness ----------------------------------------------------
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT time_of_birth, fitness_ FROM individual "
        "WHERE fitness_ IS NOT NULL ORDER BY time_of_birth"
    ).fetchall()
    conn.close()

    gen_fits: dict[int, list[float]] = defaultdict(list)
    for gen, fit in rows:
        gen_fits[int(gen)].append(float(fit))

    gens      = sorted(gen_fits)
    n_evals   = [g * EA_POP_SIZE for g in gens]
    best_fit  = [max(gen_fits[g]) for g in gens]
    mean_fit  = [float(np.mean(gen_fits[g])) for g in gens]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_evals, best_fit, "o-", color="steelblue", label="best")
    ax.plot(n_evals, mean_fit, "s--", color="tomato",   label="mean")
    ax.set_xlabel("Number of fitness function evaluations")
    ax.set_ylabel("Fitness (mean shaped fitness over 10 circuits)")
    ax.set_title("Drone morphology evolution — Lee controller, 10-circuit avg fitness")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fitness_png = DATA / f"fitness_history_{RUN_ID}.png"
    fig.savefig(fitness_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.log(f"Fitness history → {fitness_png}")

    # ---- Plot 2: avg gates passed -------------------------------------------
    # Pull avg_gates from each individual's tags (persisted as JSON in DB).
    gen_gates: dict[int, list[float]] = defaultdict(list)
    for ind in ea.population:
        if ind.fitness_ is None or not np.isfinite(ind.fitness_):
            continue
        g = int(ind.time_of_birth)
        ag = ind.tags.get("avg_gates")
        if ag is not None:
            gen_gates[g].append(float(ag))

    if gen_gates:
        gens2     = sorted(gen_gates)
        n_evals2  = [g * EA_POP_SIZE for g in gens2]
        best_ag   = [max(gen_gates[g]) for g in gens2]
        mean_ag   = [float(np.mean(gen_gates[g])) for g in gens2]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(n_evals2, best_ag, "o-", color="steelblue", label="best")
        ax.plot(n_evals2, mean_ag, "s--", color="tomato",   label="mean")
        ax.set_xlabel("Number of fitness function evaluations")
        ax.set_ylabel("Average gates passed (over 10 circuits)")
        ax.set_title("Drone morphology evolution — avg gates passed per circuit")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        gates_png = DATA / f"avg_gates_history_{RUN_ID}.png"
        fig.savefig(gates_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.log(f"Avg-gates history → {gates_png}")
    else:
        console.log("[yellow]Avg-gates plot skipped: no tag data found[/yellow]")

except Exception as exc:
    console.log(f"[yellow]Fitness plots skipped: {exc}[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison vs CANONICAL HEXACOPTER on the same 10 circuits
#   - Build a standard 6-arm planar hex (60° apart, alternating CCW/CW)
#   - Run both bodies (canonical + evolved best) on each of the 10 circuits
#   - Print a per-circuit summary table
#   - Save comparison_gates_*.png — bar chart, canonical vs evolved per circuit
#   - Save comparison_flight_*.mp4 — side-by-side MuJoCo render (left=canonical,
#     right=evolved). Replaces the single-body best_flight video; the evolved
#     drone is visible in the right pane.
# ─────────────────────────────────────────────────────────────────────────────

best_propellers = blueprint_to_propellers(best_bp, convention="ned")

# ── 1. Build canonical hexacopter via the same genome → blueprint pipeline ────
# Genome row layout (from decoders.py): [magnitude, arm_az, arm_pitch,
# motor_az, motor_pitch, direction]. direction >= 0.5 → CW, else CCW.
_CANONICAL_ARM_LEN = float(np.mean(PARAMETER_LIMITS[0]))   # midpoint of evolved range
canonical_arms = np.array([
    [_CANONICAL_ARM_LEN, i * np.pi / 3.0, 0.0, 0.0, 0.0, float(i % 2)]
    for i in range(6)
])
canonical_bp         = spherical_angular_to_blueprint(canonical_arms, propsize=PROP_SIZE)
canonical_propellers = blueprint_to_propellers(canonical_bp, convention="ned")
canonical_bp.save_json(DATA / f"canonical_blueprint_{RUN_ID}.json")
console.log(
    f"Canonical hex: 6 arms × {_CANONICAL_ARM_LEN:.3f}m at 60° apart, "
    f"alternating CCW/CW, prop_size={PROP_SIZE}"
)

# ── 2. Per-circuit rollouts for both bodies ──────────────────────────────────
console.rule("[bold cyan]Comparison rollouts on all 10 circuits")
canonical_results = [_rollout_one_circuit(canonical_propellers, cfg) for cfg in _all_gate_cfgs]
evolved_results   = [_rollout_one_circuit(best_propellers,      cfg) for cfg in _all_gate_cfgs]

canonical_gates = [r["gates"] for r in canonical_results]
evolved_gates   = [r["gates"] for r in evolved_results]
canonical_fits  = [r["fit"]   for r in canonical_results]
evolved_fits    = [r["fit"]   for r in evolved_results]

# ── 3. Console summary table ─────────────────────────────────────────────────
console.log(
    f"{'circuit':>7} | {'can_gates':>9} {'evo_gates':>9} | "
    f"{'can_fit':>10} {'evo_fit':>10}"
)
for i in range(N_TRAJECTORIES):
    console.log(
        f"{i:>7} | {canonical_gates[i]:>9} {evolved_gates[i]:>9} | "
        f"{canonical_fits[i]:>10.3f} {evolved_fits[i]:>10.3f}"
    )
console.log(
    f"{'mean':>7} | {np.mean(canonical_gates):>9.2f} {np.mean(evolved_gates):>9.2f} | "
    f"{np.mean(canonical_fits):>10.3f} {np.mean(evolved_fits):>10.3f}"
)

# ── 4. Bar chart: gates passed per circuit ───────────────────────────────────
try:
    import matplotlib.pyplot as plt

    x = np.arange(N_TRAJECTORIES + 1)
    can_bars = list(canonical_gates) + [float(np.mean(canonical_gates))]
    evo_bars = list(evolved_gates)   + [float(np.mean(evolved_gates))]
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, can_bars, width, label="canonical hex", color="gray")
    ax.bar(x + width / 2, evo_bars, width, label="evolved best", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i:02d}" for i in range(N_TRAJECTORIES)] + ["mean"])
    ax.set_xlabel("Circuit")
    ax.set_ylabel("Gates passed")
    ax.set_title("Canonical hexacopter vs evolved best — gates passed per circuit")
    ax.legend()
    ax.grid(True, alpha=0.4, axis="y")
    plt.tight_layout()
    cmp_png = DATA / f"comparison_gates_{RUN_ID}.png"
    fig.savefig(cmp_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.log(f"Comparison bar chart → {cmp_png}")
except Exception as exc:
    console.log(f"[yellow]Comparison bar chart skipped: {exc}[/yellow]")


# ── 5. Side-by-side MP4 — left=canonical, right=evolved ──────────────────────
try:
    import math

    import cv2
    import mujoco

    from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
    from ariel.simulation.environments import SimpleFlatWorld
    from ariel.utils.video_recorder import VideoRecorder

    HALF_W, HALF_H = 720, 540
    cmp_mp4 = DATA / f"comparison_flight_{RUN_ID}.mp4"
    recorder = VideoRecorder(
        file_name=cmp_mp4.stem, output_folder=cmp_mp4.parent,
        width=HALF_W * 2, height=HALF_H, fps=30,
    )
    steps_per_frame = max(1, int(round(1.0 / (recorder.fps * SIM_DT))))

    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.array([0.0, 0.0, 0.5])
    cam.azimuth   = 90.0
    cam.elevation = -75.0
    cam.distance  = 22.0

    def _logged_rollout(propellers, gate_cfg, max_steps):
        """Lee rollout with early-stop on full lap; returns (pos_ned[:k], euler[:k])."""
        try:
            drone_property, ctrl, traj, gate_checker, wind = _build_lee_stack(propellers, gate_cfg)
        except Exception:
            return None
        n_gates   = gate_checker.num_gates
        pos_ned   = np.zeros((max_steps, 3), dtype=np.float64)
        euler_log = np.zeros((max_steps, 3), dtype=np.float64)
        t, i = 0.0, 1
        try:
            while i < max_steps:
                pos_ned[i]   = drone_property.pos
                euler_log[i] = drone_property.euler
                drone_property.update(t, SIM_DT, ctrl.w_cmd, wind)
                t_new = SIM_DT * i
                sDes  = traj.desiredState(t_new, SIM_DT, drone_property)
                ctrl.controller(sDes, drone_property, traj.ctrlType, SIM_DT)
                gate_checker.check_gate_passing(drone_property.pos)
                t = t_new; i += 1
                if gate_checker.gates_passed >= n_gates:
                    break
        except Exception:
            pass
        return pos_ned[:i], euler_log[:i]

    def _ned_to_enu(pos_ned, euler_log):
        pos_enu = pos_ned.copy()
        pos_enu[:, 2] = -pos_enu[:, 2]
        phi, theta, psi = euler_log[:, 0], euler_log[:, 1], euler_log[:, 2]
        cy, sy = np.cos(psi / 2), np.sin(psi / 2)
        cp, sp = np.cos(theta / 2), np.sin(theta / 2)
        cr, sr = np.cos(phi / 2), np.sin(phi / 2)
        quat_enu = np.stack([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            -(cr * sp * cy + sr * cp * sy),
            cr * cp * sy - sr * sp * cy,
        ], axis=-1)
        return pos_enu, quat_enu

    def _mean_arm(bp):
        ls = [bp.payload(a).length for a in bp.children(bp.root_id)]  # type: ignore[arg-type]
        return float(np.mean(ls)) if ls else 0.06

    def _build_world(bp, gate_cfg, initial_pos_enu, mean_arm, body_name):
        spec = blueprint_to_mjspec(
            bp, motor_mass=0.01, arm_mass=0.034 * mean_arm, body_name=body_name,
        )
        world = SimpleFlatWorld()
        world.spawn(
            spec,
            position=(float(initial_pos_enu[0]), float(initial_pos_enu[1]), float(initial_pos_enu[2])),
            correct_collision_with_floor=False,
        )
        for gi in range(len(gate_cfg.gate_pos)):
            gx = float(gate_cfg.gate_pos[gi, 0])
            gy = float(gate_cfg.gate_pos[gi, 1])
            gz = float(-gate_cfg.gate_pos[gi, 2])
            yaw = float(gate_cfg.gate_yaw[gi])
            qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
            gb = world.spec.worldbody.add_body(name=f"gate_{gi}", pos=[gx, gy, gz], quat=[qw, 0, 0, qz])
            gb.add_geom(name=f"gate_plane_{gi}",  type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.01, 0.75, 0.75], rgba=(1.0, 0.55, 0.0, 0.25),
                        contype=0, conaffinity=0)
            gb.add_geom(name=f"gate_marker_{gi}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.06, 0.0, 0.0],   rgba=(1.0, 0.1, 0.1, 1.0),
                        contype=0, conaffinity=0)
        model = world.spec.compile()
        data  = mujoco.MjData(model)
        model.opt.timestep = SIM_DT
        return model, data

    def _set_pose(model, data, p_enu, q_enu):
        data.qpos[0:3] = [float(p_enu[0]), float(p_enu[1]), float(p_enu[2])]
        data.qpos[3:7] = [float(q_enu[0]), float(q_enu[1]), float(q_enu[2]), float(q_enu[3])]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def _label(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    canonical_mean_arm = _mean_arm(canonical_bp)
    best_mean_arm      = _mean_arm(best_bp)
    max_steps          = int(SIM_TIME / SIM_DT) + 1

    t_render = time.time()
    for circuit_idx, gate_cfg in enumerate(_all_gate_cfgs):
        console.log(f"Rendering circuit {circuit_idx + 1}/{N_TRAJECTORIES} …")

        can_log = _logged_rollout(canonical_propellers, gate_cfg, max_steps)
        evo_log = _logged_rollout(best_propellers,      gate_cfg, max_steps)
        if can_log is None or evo_log is None:
            console.log(f"[yellow]  circuit {circuit_idx} rollout failed[/yellow]")
            continue

        can_pos_ned, can_eul = can_log
        evo_pos_ned, evo_eul = evo_log
        if len(can_pos_ned) < 2 or len(evo_pos_ned) < 2:
            console.log(f"[yellow]  circuit {circuit_idx} too few steps[/yellow]")
            continue

        can_pos_enu, can_quat_enu = _ned_to_enu(can_pos_ned, can_eul)
        evo_pos_enu, evo_quat_enu = _ned_to_enu(evo_pos_ned, evo_eul)

        can_model, can_data = _build_world(
            canonical_bp, gate_cfg, can_pos_enu[0], canonical_mean_arm, "canonical_hex",
        )
        evo_model, evo_data = _build_world(
            best_bp, gate_cfg, evo_pos_enu[0], best_mean_arm, "evolved_drone",
        )

        n_frames = max(len(can_pos_enu), len(evo_pos_enu))
        with mujoco.Renderer(can_model, width=HALF_W, height=HALF_H) as r_can, \
             mujoco.Renderer(evo_model, width=HALF_W, height=HALF_H) as r_evo:
            for idx in range(0, n_frames, steps_per_frame):
                ci = min(idx, len(can_pos_enu) - 1)
                ei = min(idx, len(evo_pos_enu) - 1)
                _set_pose(can_model, can_data, can_pos_enu[ci], can_quat_enu[ci])
                _set_pose(evo_model, evo_data, evo_pos_enu[ei], evo_quat_enu[ei])
                r_can.update_scene(can_data, camera=cam)
                r_evo.update_scene(evo_data, camera=cam)
                combined = np.hstack([r_can.render(), r_evo.render()]).copy()
                _label(combined, "CANONICAL HEX",                  20, 35)
                _label(combined, "EVOLVED",                        HALF_W + 20, 35)
                _label(combined,
                       f"circuit {circuit_idx + 1}/{N_TRAJECTORIES}",
                       HALF_W - 110, HALF_H - 20)
                recorder.write(frame=combined)

    recorder.release()
    console.log(f"Comparison video rendered in {time.time() - t_render:.1f}s → {cmp_mp4}")

except Exception as exc:
    console.log(f"[yellow]Comparison video skipped: {exc}[/yellow]")

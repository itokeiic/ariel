"""Drone morphology evolution with average-gates-passed fitness.

Generates 10 different gate circuits via the quintic planner at startup.
Each candidate body is trained with PPO on the first circuit, then evaluated
deterministically on all 10 circuits.  Fitness = average number of gates
passed per episode, averaged across all 10 circuits.

After the EA run the script saves:
  - best_blueprint_*.json      — body of the best individual
  - best_policy_*.zip          — trained PPO weights
  - gate_pos_*_NN.npy / gate_yaw_*_NN.npy  (NN = 00–09)
  - fitness_history_*.png      — best/mean avg-gates vs. evaluations
  - best_flight_*.mp4          — MuJoCo render of best drone on circuit 00

Quick start
-----------
    uv run examples/spear/15_drone_evo_average_num_gates.py

CLI overrides:
    uv run examples/spear/15_drone_evo_average_num_gates.py \\
        --device cuda:0 --seed 7 --out-dir __data__/my_run

NOTE: do NOT add ``from __future__ import annotations`` to this file —
ariel's @EAOperation decorator needs real (not stringified) type hints
at import time.
"""
# ─────────────────────────────────────────────────────────────────────────────
# Standard-library / third-party imports  (do not edit)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import base64
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from stable_baselines3 import PPO

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator"))
from planner_generator import generate_paths_from_coefficients  # noqa: E402

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
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv

# ─────────────────────────────────────────────────────────────────────────────
# Minimal CLI  (run-level overrides — not the main config surface)
# ─────────────────────────────────────────────────────────────────────────────
curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(
    description="Drone morphology evolution — avg gates passed fitness"
)
parser.add_argument("--device",  default="cpu",
                    help="Torch device: 'cpu' or 'cuda:0'  (default: cpu)")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/drone_evo_avg_gates/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 1 — EA  (evolutionary algorithm)
# ═════════════════════════════════════════════════════════════════════════════

EA_POP_SIZE  = 10
EA_GENS      = 5

N_ARMS_MIN = 6
N_ARMS_MAX = 6

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],            # arm length
    [-np.pi, np.pi],          # arm azimuth
    [-np.pi / 2, np.pi / 2],  # arm elevation
    [-np.pi, np.pi],          # motor disc azimuth
    [-np.pi, np.pi],          # motor disc pitch
    [0, 1],                   # spin direction (binary)
])

APPEND_ARM_CHANCE = 0.0
BILATERAL_SYMMETRY = None
REPAIR = False
MUTATION_SCALES = None

CUSTOM_GENERATION_OPS = None


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 2 — GATE TRACKS
#
#  The planner generates N_TRAJECTORIES distinct quintic circuits.
#  PPO is trained on circuit 0; fitness is the average gates passed
#  across all N_TRAJECTORIES circuits.
# ═════════════════════════════════════════════════════════════════════════════

N_TRAJECTORIES = 10     # how many gate circuits to generate

GATE_PATH_STEPS = 15    # gates sampled per circuit
GATE_PATH_SCALE = 5.0   # metres — spatial scale of each circuit
GATE_Z_HEIGHT   = -1.5  # NED z (metres)


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 3 — PPO / RL
# ═════════════════════════════════════════════════════════════════════════════

PPO_STEPS      = 1_000_000
PPO_NUM_ENVS   = 2000
PPO_EVAL_STEPS = 40_000
PROP_SIZE      = 2

PPO_NET_ARCH    = dict(pi=[256, 256], vf=[256, 256])
PPO_ACTIVATION  = torch.nn.SiLU
PPO_N_EPOCHS    = 20
PPO_GAMMA       = 0.999
PPO_LR          = 1e-3
PPO_CLIP_RANGE  = 0.2
PPO_ENT_COEF    = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Below this line: experiment logic  — edit only if you know what you're doing
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
        gate_pos = np.column_stack([xy, np.full(n_gates, z)]).astype(np.float32)
        gate_yaw = yaw_dense[kept].astype(np.float32)
        result.append((gate_pos, gate_yaw))
    return result


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
# Gate sets and environment factory
# ─────────────────────────────────────────────────────────────────────────────

_all_gate_sets = _multi_quintic_to_gates(
    COEFFS, N_TRAJECTORIES, GATE_PATH_STEPS, GATE_PATH_SCALE, GATE_Z_HEIGHT, seed=args.seed,
)


def _make_env(propellers: Any, gate_pos: np.ndarray, gate_yaw: np.ndarray) -> TorchDroneGateEnv:
    start_pos = (gate_pos[0] + np.array([0.0, -1.0, 0.0])).astype(np.float32)
    return TorchDroneGateEnv(
        propellers=propellers,
        num_envs=PPO_NUM_ENVS,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        device=args.device,
        dt=0.01,
        seed=args.seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PPO training + multi-track evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _model_to_b64(model: PPO) -> str:
    import os
    tmp = Path(tempfile.mktemp(suffix=".zip"))
    try:
        model.save(str(tmp))
        return base64.b64encode(tmp.read_bytes()).decode("ascii")
    finally:
        tmp.unlink(missing_ok=True)


def _b64_to_policy(b64: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(b64))


cfg: dict[str, Any] = {
    "ppo_steps":   PPO_STEPS,
    "num_envs":    PPO_NUM_ENVS,
    "eval_steps":  PPO_EVAL_STEPS,
    "prop_size":   PROP_SIZE,
    "device":      args.device,
    "gate_sets":   _all_gate_sets,
    "policy_bank": {},
    "policy_bank_fitness": {},
}


def _train_and_eval(genotype: dict, cfg: dict) -> tuple[float, Any, int]:
    """Decode genome → train PPO on circuit 0 → evaluate on all circuits."""
    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return float("-inf"), None, 0

    if not np.isfinite(genome.arms[:, 0]).any():
        return float("-inf"), None, 0

    try:
        bp         = spherical_angular_to_blueprint(genome.arms, propsize=cfg["prop_size"])
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return float("-inf"), None, 0

    n_motors     = len(propellers)
    gate_sets    = cfg["gate_sets"]
    train_gp, train_gy = gate_sets[0]
    env          = _make_env(propellers, train_gp, train_gy)
    n_steps      = max(64, cfg["ppo_steps"] // (cfg["num_envs"] * 10))
    rollout_size = n_steps * cfg["num_envs"]
    _target      = max(4096, rollout_size // 4)
    batch_size   = next(b for b in range(_target, 0, -1) if rollout_size % b == 0)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=PPO_ACTIVATION,
            net_arch=PPO_NET_ARCH,
            log_std_init=0.0,
        ),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        learning_rate=PPO_LR,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        device=cfg["device"],
        verbose=0,
    )

    # Warm-start from best known policy for this motor count.
    warm_b64: str | None = cfg["policy_bank"].get(n_motors)
    if warm_b64 is not None:
        _tmp = Path(tempfile.mktemp(suffix=".zip"))
        try:
            _tmp.write_bytes(base64.b64decode(warm_b64))
            _warm = PPO.load(str(_tmp), env=env, device=cfg["device"])
            model.policy.load_state_dict(_warm.policy.state_dict())
            del _warm
        except Exception:
            pass
        finally:
            _tmp.unlink(missing_ok=True)

    model.learn(total_timesteps=cfg["ppo_steps"])

    # Evaluate deterministically on each of the N_TRAJECTORIES circuits.
    # infos[i]["num_gates_passed"] is np.ndarray shape (num_envs,); [i] gives
    # the gates passed by env i in its just-completed episode.
    avg_per_track: list[float] = []
    eval_iters = max(cfg["eval_steps"] // cfg["num_envs"], 500)
    for gp, gy in gate_sets:
        track_env = _make_env(propellers, gp, gy)
        obs       = track_env.reset()
        gates_per_ep: list[float] = []
        for _ in range(eval_iters):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = track_env.step(action)
            for i, done in enumerate(dones):
                if done:
                    gates_per_ep.append(float(infos[i]["num_gates_passed"][i]))
        avg_per_track.append(float(np.mean(gates_per_ep)) if gates_per_ep else 0.0)

    fitness = float(np.mean(avg_per_track))
    return fitness, model, n_motors


# ─────────────────────────────────────────────────────────────────────────────
# EA evaluation operation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_ppo(population: Population) -> Population:
    """Train a PPO policy for every unevaluated body; assign avg-gates fitness."""
    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for _i, ind in enumerate(track(to_eval, description="PPO evaluation…", console=console)):
        t0  = time.time()
        fit, model, n_motors = _train_and_eval(ind.genotype, cfg)
        ind.fitness = fit
        warm_tag = ""
        if model is not None:
            ind.tags["policy_b64"] = _model_to_b64(model)
            if np.isfinite(fit):
                prev = cfg["policy_bank_fitness"].get(n_motors, float("-inf"))
                if fit > prev:
                    cfg["policy_bank"][n_motors]         = ind.tags["policy_b64"]
                    cfg["policy_bank_fitness"][n_motors] = fit
                    warm_tag = " [new bank best]"
        ws = "warm" if cfg["policy_bank"].get(n_motors) and not warm_tag else "cold"
        display_id = ind.id if ind.id is not None else f"#{_i + 1}"
        console.log(
            f"  id={display_id}  motors={n_motors}  "
            f"avg_gates={fit:.3f}  start={ws}  ({time.time()-t0:.1f}s){warm_tag}"
        )

    finite = [ind.fitness_ for ind in population.alive.evaluated if np.isfinite(ind.fitness_)]  # type: ignore[arg-type]
    if finite:
        console.log(f"    batch done — max={max(finite):.4f}  mean={np.mean(finite):.4f}")
    return population


eval_op = EAOperation(evaluate_with_ppo)

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

console.rule("[bold blue]Drone Body Evolution + PPO (avg gates passed)")
console.log(
    f"pop={EA_POP_SIZE}  gens={EA_GENS}  arms={N_ARMS_MIN}–{N_ARMS_MAX}  "
    f"ppo_steps={PPO_STEPS:,}  num_envs={PPO_NUM_ENVS}  "
    f"gate_steps={GATE_PATH_STEPS}  scale={GATE_PATH_SCALE}  "
    f"n_trajectories={N_TRAJECTORIES}  device={args.device}"
)
console.log(f"DB → {DB_PATH}")

initial_pop = Population([Individual() for _ in range(EA_POP_SIZE)])
init_op     = initialize_drones(template_handler=genome_handler)

console.log("Initializing + evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

finite0 = [ind.fitness_ for ind in initial_pop if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
if finite0:
    console.log(f"Initial — max={max(finite0):.4f}  mean={np.mean(finite0):.4f}")

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

for idx, (gp, gy) in enumerate(_all_gate_sets):
    np.save(DATA / f"gate_pos_{RUN_ID}_{idx:02d}.npy", gp)
    np.save(DATA / f"gate_yaw_{RUN_ID}_{idx:02d}.npy", gy)

policy_path: Path | None = None
if "policy_b64" in best.tags:
    policy_path = DATA / f"best_policy_{RUN_ID}.zip"
    _b64_to_policy(best.tags["policy_b64"], policy_path)
    console.log(f"Best policy    → {policy_path}")

console.log(
    f"[bold green]Done.[/bold green]  "
    f"Best avg_gates_passed: {best.fitness_:.4f}  DB → {DB_PATH}"
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
    n_evals  = [g * EA_POP_SIZE for g in gens]
    best_fit = [max(gen_fits[g]) for g in gens]
    mean_fit = [float(np.mean(gen_fits[g])) for g in gens]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_evals, best_fit, "o-", color="steelblue", label="best")
    ax.plot(n_evals, mean_fit, "s--", color="tomato",   label="mean")
    ax.set_xlabel("Number of fitness evaluations")
    ax.set_ylabel("Avg gates passed")
    ax.set_title("Drone morphology evolution — avg gates passed")
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
# MP4 video — best individual flying gate circuit 00
# ─────────────────────────────────────────────────────────────────────────────

if policy_path is None or not policy_path.exists():
    console.log("Video skipped: no policy ZIP found.")
else:
    try:
        import math

        import mujoco

        from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
        from ariel.simulation.environments import SimpleFlatWorld
        from ariel.simulation.tasks.drone_gate_env import DroneGateEnv as _DroneGateEnvCPU
        from ariel.utils.video_recorder import VideoRecorder

        vid_gp, vid_gy = _all_gate_sets[0]
        vid_start = (vid_gp[0] + np.array([0.0, -1.0, 0.0])).astype(np.float32)
        best_propellers = blueprint_to_propellers(best_bp, convention="ned")

        cpu_env = _DroneGateEnvCPU(
            propellers=best_propellers,
            num_envs=1,
            gates_pos=vid_gp,
            gate_yaw=vid_gy,
            start_pos=vid_start,
            dt=0.01,
            seed=args.seed,
        )
        model_vid = PPO.load(str(policy_path), env=cpu_env, device="cpu")
        console.log("Rendering video — rolling out best policy on circuit 00 …")

        DT      = 0.01
        N_STEPS = PPO_EVAL_STEPS // PPO_NUM_ENVS
        pos_ned  = np.zeros((N_STEPS, 3), dtype=np.float32)
        euler_log = np.zeros((N_STEPS, 3), dtype=np.float32)

        obs = cpu_env.reset()
        for i in range(N_STEPS):
            pos_ned[i]   = cpu_env.world_states[0, 0:3]
            euler_log[i] = cpu_env.world_states[0, 6:9]
            action, _ = model_vid.predict(obs, deterministic=True)
            obs, _, _, _ = cpu_env.step(action)

        # NED → ENU
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

        arm_lengths = [best_bp.payload(a).length for a in best_bp.children(best_bp.root_id)]  # type: ignore[arg-type]
        mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06

        drone_spec = blueprint_to_mjspec(
            best_bp,
            motor_mass=0.01,
            arm_mass=0.034 * mean_arm_len,
            body_name="evolved_quad",
        )
        world_mj = SimpleFlatWorld()
        world_mj.spawn(
            drone_spec,
            position=(float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])),
            correct_collision_with_floor=False,
        )

        for _gi in range(len(vid_gp)):
            _gx  = float(vid_gp[_gi, 0])
            _gy  = float(vid_gp[_gi, 1])
            _gz  = float(-vid_gp[_gi, 2])
            _yaw = float(vid_gy[_gi])
            _qw  = math.cos(_yaw / 2.0)
            _qz  = math.sin(_yaw / 2.0)
            _gate_body = world_mj.spec.worldbody.add_body(
                name=f"gate_{_gi}",
                pos=[_gx, _gy, _gz],
                quat=[_qw, 0.0, 0.0, _qz],
            )
            _gate_body.add_geom(
                name=f"gate_plane_{_gi}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.75, 0.75],
                rgba=(1.0, 0.55, 0.0, 0.25),
                contype=0,
                conaffinity=0,
            )
            _gate_body.add_geom(
                name=f"gate_marker_{_gi}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.06, 0.0, 0.0],
                rgba=(1.0, 0.1, 0.1, 1.0),
                contype=0,
                conaffinity=0,
            )

        model_mj = world_mj.spec.compile()
        data_mj  = mujoco.MjData(model_mj)
        model_mj.opt.timestep = DT

        def _set_pose(idx: int) -> None:
            p = pos_enu[idx]
            q = quat_enu[idx]
            data_mj.qpos[0:3] = [float(p[0]), float(p[1]), float(p[2])]
            data_mj.qpos[3:7] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            data_mj.qvel[:] = 0.0
            mujoco.mj_forward(model_mj, data_mj)

        mp4 = DATA / f"best_flight_{RUN_ID}.mp4"
        recorder = VideoRecorder(
            file_name=mp4.stem, output_folder=mp4.parent,
            width=720, height=540, fps=30,
        )
        steps_per_frame = max(1, int(round(1.0 / (recorder.fps * DT))))
        t_render = time.time()
        with mujoco.Renderer(model_mj, width=recorder.width, height=recorder.height) as renderer:
            for idx in range(0, len(pos_enu), steps_per_frame):
                _set_pose(idx)
                renderer.update_scene(data_mj)
                recorder.write(frame=renderer.render())
        recorder.release()
        console.log(f"Video rendered in {time.time() - t_render:.1f}s → {mp4}")

    except Exception as exc:
        console.log(f"[yellow]Video skipped: {exc}[/yellow]")

console.log(
    f"\nTo visualise:\n"
    f"  uv run examples/spear/8_visualize_gate_track.py  --run-dir {DATA}"
)

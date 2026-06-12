#!/bin/bash
# Slurm array job: run examples/spear/17_drone_evo_average_num_gates_lee.py
# 20 times (one per array task) with seeds 0..19, each writing a separate
# out-dir. Submit from the repo root on the login node:
#
#     sbatch --partition=batch cluster/run_ea_array.sbatch
#
# Hex cluster (ci-group): use the `batch` partition — CPU, 4-day limit, Ripper
# 2-7. Avoid `short` (2 h cap; a run can take ~1.5-2.5 h). Use a GPU partition
# (`batch-gpu`/`gpu`) only if you want the comparison videos rendered (then add
# --gres=gpu:1 and export MUJOCO_GL=egl). Resource flags below are defaults;
# override on the command line as needed.
#
#SBATCH --job-name=ariel-ea
#SBATCH --array=0-19
#SBATCH --output=cluster/logs/ea_seed%a_%A.out
#SBATCH --error=cluster/logs/ea_seed%a_%A.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# No --partition / --gres here: pass --partition=batch on the sbatch command line.
# The EA is CPU-only; the end-of-run MuJoCo video is optional (see MUJOCO_GL).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Keep BLAS/OMP from oversubscribing the allocated cores (each task is one EA).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg   # headless matplotlib for the fitness/avg-gates/bar plots

# End-of-run side-by-side video uses MuJoCo's GL renderer. On a GPU node set
# MUJOCO_GL=egl; on a CPU-only node leave it unset (or =osmesa) — either way the
# render is wrapped in try/except in the script, so if GL is unavailable the
# video is skipped and the EA results (best_blueprint, plots, gate tracks) are
# still written. Default to egl; override by exporting MUJOCO_GL before sbatch.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

SEED="${SLURM_ARRAY_TASK_ID}"
OUT="__data__/drone_evo_lee_avg/slurm_seed$(printf '%02d' "$SEED")"

echo "[$(date '+%F %T')] host=$(hostname) seed=${SEED} out=${OUT} cpus=${SLURM_CPUS_PER_TASK:-?}"

# --no-sync: reuse the .venv built once on the login node (uv sync) instead of
# letting 20 concurrent tasks each touch/rebuild the shared environment.
uv run --no-sync examples/spear/17_drone_evo_average_num_gates_lee.py \
    --seed "${SEED}" \
    --out-dir "${OUT}"

echo "[$(date '+%F %T')] done seed=${SEED}"

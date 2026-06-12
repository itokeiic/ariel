# Running example 17 (drone EA) on a Slurm cluster

Runs `examples/spear/17_drone_evo_average_num_gates_lee.py` as a 20-task Slurm
array (seeds 0..19), each an independent evolution writing its own out-dir under
`__data__/drone_evo_lee_avg/slurm_seedNN/`.

The script is CPU-only. The end-of-run MuJoCo comparison video is optional and
wrapped in try/except, so it is skipped on CPU-only nodes without failing the
run; the EA outputs (`best_blueprint_*.json`, `fitness_history_*.png`,
`avg_gates_history_*.png`, `comparison_gates_*.png`, `gate_pos/yaw_*.npy`,
`database_*.db`) are always written.

## Deploy (run once on the ripper1 login node)

```bash
# 1. Clone your fork (branch spear_examples). Use SSH if your GitHub key is on
#    ripper1, otherwise the https URL.
git clone -b spear_examples git@github.com:itokeiic/ariel.git ariel
#   or: git clone -b spear_examples https://github.com/itokeiic/ariel.git ariel
cd ariel

# 2. Build the pinned environment once (Python 3.12.9 + deps from uv.lock).
#    Needs internet on the login node (you have it).
uv sync

# 3. Sanity-check the env (fast, no Slurm): import + 1-second help.
uv run --no-sync examples/spear/17_drone_evo_average_num_gates_lee.py --help

# 4. Submit the 20-task array on the `batch` partition (CPU, 4-day limit).
mkdir -p cluster/logs
sbatch --partition=batch cluster/run_ea_array.sbatch
```

### Partition choice (Hex / ci-group cluster)

Use **`batch`** — CPU, 4-day limit, Ripper 2-7. The EA is CPU-only and each run
takes ~1.5-2.5 h, so avoid `short` (2 h cap). Use a GPU partition
(`batch-gpu` or `gpu`) only to render the comparison videos on the cluster:

```bash
MUJOCO_GL=egl sbatch --partition=batch-gpu --gres=gpu:1 cluster/run_ea_array.sbatch
```

## Monitor / collect

```bash
squeue -u "$USER"                       # running/queued tasks
tail -f cluster/logs/ea_seed00_*.out    # one task's log
# best fitness + avg_gates per seed once finished:
for d in __data__/drone_evo_lee_avg/slurm_seed*/; do
  uv run --no-sync python - "$d" <<'PY'
import sys, glob, json, sqlite3, math
d = sys.argv[1]
db = glob.glob(d + "database_*.db")[0]
rows = sqlite3.connect(db).execute(
    "SELECT fitness_, tags_ FROM individual WHERE fitness_ IS NOT NULL").fetchall()
rows = [r for r in rows if r[0] is not None and not math.isinf(r[0])]
best = max(rows, key=lambda r: r[0])
ag = json.loads(best[1]).get("avg_gates")
print(f"{d}: best_fitness={best[0]:.3f} avg_gates={ag}")
PY
done
```

## Notes / knobs

- Resources in `run_ea_array.sbatch` (`--time=04:00:00 --cpus-per-task=2
  --mem=8G`) are defaults; override on the `sbatch` command line. A single
  aligned run took ~1h20m on a workstation, so 4h has margin.
- Seeds are the array indices (`--array=0-19`). Change the range to run more/
  fewer, or edit the seed mapping in the script.
- For videos on GPU nodes, submit with `MUJOCO_GL=egl sbatch ...` and a
  `--gres=gpu:1`; otherwise videos are skipped (rerender the best body locally).
- `uv run --no-sync` reuses the login-node `.venv`; do **not** drop `--no-sync`
  or 20 tasks may race rebuilding the shared environment.

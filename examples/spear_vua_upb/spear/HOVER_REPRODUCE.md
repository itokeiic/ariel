# Reproducing a stable hover for an evolved drone in Isaac Lab

This guide walks through, end to end, how to load one evolved drone morphology into
Isaac Lab, run the base-wrench PD hover controller, understand why it diverges with the
default gains, fix it, and record a stable-hover video.

**End result:** the drone holds `z = 1.50 m` at hover thrust (`≈ m·g`), and an 8 s
clip is written to `hover_morph00_stable.mp4`.

> Paths below are for the machine this was developed on. The three machine-specific
> ones are called out in **Step 0** — change them to match your setup; everything else
> is relative to the repo.

---

## Step 0 — Machine-specific paths

| What | Path on this machine |
|------|----------------------|
| Repo root (`$REPO`) | `/home/keiichi-ito/Documents/sandbox/ariel` |
| Conda env (Python 3.11) | `ariel-isaaclab-train` |
| Isaac Sim (binary, 5.1.0) | `/home/keiichi-ito/Documents/sandbox/isaac-sim-standalone-5.1.0-linux-x86_64` |
| Isaac Lab repo | `/home/keiichi-ito/Documents/sandbox/IsaacLab` (has `_isaac_sim` symlink → the Isaac Sim dir) |

Everything in this guide runs from the example directory:

```bash
cd /home/keiichi-ito/Documents/sandbox/ariel/examples/spear_vua_upb/spear
```

Requirements: an NVIDIA GPU (this was validated on an RTX 2000 Ada, 8 GB) and the
`ariel-isaaclab-train` conda env, which has `isaaclab` **and** `ariel` installed
(editable). Video encoding uses `imageio` + `imageio-ffmpeg`, already in that env.

---

## Step 1 — Wire up the Isaac Sim runtime (the non-obvious part)

The conda env has `isaaclab` on the path but **not** `isaacsim` — the runtime lives in
the binary Isaac Sim, and this env has no `conda activate` hook to add it. You must
source Isaac Sim's env script **after** activating the env, every shell:

```bash
conda activate ariel-isaaclab-train
source /home/keiichi-ito/Documents/sandbox/isaac-sim-standalone-5.1.0-linux-x86_64/setup_conda_env.sh
```

Verify:

```bash
python -c "import isaacsim, isaaclab; print('both import OK')"
```

**Gotchas**
- In **non-interactive** shells, `conda activate <env>` / `conda run -n <env> python`
  can silently resolve to the wrong env's Python. If in doubt, call the interpreter by
  absolute path: `/home/keiichi-ito/miniconda3/envs/ariel-isaaclab-train/bin/python`.
- To make the wiring permanent (skip the `source` step), from the Isaac Lab repo run
  `./isaaclab.sh --conda ariel-isaaclab-train` once — it installs the missing
  `activate.d` hook.

---

## Step 2 — The asset: USD + allocation JSON

One generic controller flies any morphology by reading a per-drone **allocation JSON**
(rotor geometry, spin directions, and the `thrust/roll/pitch/yaw` allocation matrix).
Runnable USD + allocation pairs live under:

```
parametric_drone/generated_gp/usd/gptm_family_42_morph_00.usd
parametric_drone/generated_gp/allocation/gptm_family_42_morph_00.json
```

We use `gptm_family_42_morph_00` throughout — a "leveled vectoring quad": rotor
directions are essentially vertical `(0,0,1)`, positions form a stretched X (pitch arms
`x = ±0.14 m`, roll arms `y = ±0.045 m`), spin signs the textbook `+ - - +`. That
allocation JSON has **no mass field**, so the base script falls back to `1.89 kg`
(the true mass is `1.8914 kg`, so the fallback happens to be fine here).

---

## Step 3 — Run the base example (this diverges — expected)

The stock hover controller:

```bash
python 01_load_single_usd_apply_wrench_hover.py \
  --usd parametric_drone/generated_gp/usd/gptm_family_42_morph_00.usd \
  --allocation_json parametric_drone/generated_gp/allocation/gptm_family_42_morph_00.json \
  --headless
```

It **does not hover** — it climbs away:

```
step=0    pos=(0.00, 0.00, 1.50)    thrust=20.43
step=120  pos=(-0.99, -0.71, 4.98)  thrust=0.00
step=480  pos=(-23.98, 3.15, 47.81) thrust=0.00   # ~48 m up
```

Note the paradox: thrust command reads 0 while it keeps accelerating upward.

---

## Step 4 — Diagnose (optional, but this is the "why")

`diag_hover.py` runs the same controller with rich per-step logging and no camera
(so it boots faster):

```bash
python diag_hover.py \
  --usd parametric_drone/generated_gp/usd/gptm_family_42_morph_00.usd \
  --allocation_json parametric_drone/generated_gp/allocation/gptm_family_42_morph_00.json \
  --max_steps 70 --headless
```

Look at the roll rate `wx` and per-rotor thrusts from ~step 3 on:

```
s=4  wx=+3.317  tx=-6.812  rt=[0.00, 15.00, 0.00, 15.00]  Fz=+29.99
s=5  wx=-1.657  tx=+3.214  rt=[15.00, 0.00, 15.00, 0.00]  Fz=+29.99
s=6  wx=+3.275  tx=-6.860  rt=[0.00, 15.00, 0.00, 15.00]  Fz=+29.99
```

The roll loop oscillates **bang-bang every timestep**; two rotors are always pinned at
15 N, so the vertical force `Fz` is locked at **~30 N** regardless of the thrust
command.

---

## Step 5 — Root cause

The morphology is **stretched**: pitch arms at `x = ±0.14 m` but roll arms at only
`y = ±0.045 m` (3× shorter). So the maximum *deliverable* roll torque is tiny
(`≈ 1.4 N·m`, the ceiling you see in the diagnostic), while the default attitude gains
(`kp_roll=8, kd_roll=2`) *demand* `±7 N·m`. The `kd_roll·wx` term dominates and drives a
Nyquist-frequency limit cycle → the rotors saturate → `Fz ≈ 30 N ≫ 18.5 N` hover →
runaway climb. Pitch, with its 3× longer arm, stays smooth on the same gains.

It was **not** the mass (1.8914 ≈ 1.89 fallback) and **not** the drone's unactuated
"floppy" joints (locking them rigid changed nothing).

---

## Step 6 — The fix + record the stable hover

Lower the roll gains so demand stays inside the deliverable envelope. `01_hover_record.py`
is a recording variant of the base script; it adds a camera and, for robustness,
lock-actuators on the articulation's joints and an auto-read of the drone's real mass
(neither is what fixes the hover — the roll gains are — but they're good hygiene).

```bash
python 01_hover_record.py \
  --usd parametric_drone/generated_gp/usd/gptm_family_42_morph_00.usd \
  --allocation_json parametric_drone/generated_gp/allocation/gptm_family_42_morph_00.json \
  --video_path hover_morph00_stable.mp4 \
  --kp_roll 2.5 --kd_roll 0.3 \
  --cam_eye 1.2 1.2 1.9 --cam_target 0.05 0.2 1.5 \
  --max_steps 800 --record_every 2 --headless --enable_cameras
```

`--headless --enable_cameras` is required: headless so no window opens, `--enable_cameras`
so the camera sensor actually renders frames.

---

## Step 7 — Expected result

```
Actual total mass from physics: 1.8914 kg | using mass_kg=1.8914 (articulation)
step=0    pos=(0.00, 0.00, 1.50)  thrust=20.45
step=400  pos=(0.05, 0.20, 1.50)  thrust=18.55
step=760  pos=(0.05, 0.21, 1.50)  thrust=18.55
DONE: wrote 400 frames to hover_morph00_stable.mp4
```

- **z holds at 1.50 m** (the target) for the full 8 s — no climb.
- **thrust settles to 18.55 N = m·g** — true hover equilibrium.
- Position holds near `(0.05, 0.21, 1.50)`. The ~0.2 m offset in y is steady-state
  error — this controller is pure PD with no integral term. Add an integral (position
  I-gain) if you need it dead-centered.

Output: `hover_morph00_stable.mp4` (1280×720, H.264, 8.0 s, 50 fps).

---

## Camera framing knobs (`01_hover_record.py`)

| Flag | Meaning | Default |
|------|---------|---------|
| `--cam_eye X Y Z` | Fixed camera position (world) | `2.8 2.8 2.4` |
| `--cam_target X Y Z` | Fixed look-at point (world) | `0.0 0.0 1.5` |
| `--follow` | Camera follows the drone instead of fixed | off |
| `--cam_offset X Y Z` | Eye offset from drone when `--follow` | `3.0 3.0 1.2` |
| `--cam_width` / `--cam_height` | Render resolution | `1280` / `720` |

Moving the eye closer to the target zooms in (this guide uses `1.2 1.2 1.9`, ≈ 1.7 m
out). Use `--follow` for a chase cam — useful for the *un*-fixed divergent runs, where a
fixed camera would lose the drone off-screen.

---

# Part 2 — Trying other morphologies

Everything above flies `morph_00`. Two ways to fly a *different* drone.

## 2a. Longer arms, zero code — reuse an existing morph

The GP family `gptm_family_42_morph_00…09` is a smooth sweep from compact (short arms) to
extended (long arms). `morph_09` has ~2× morph_00's arm reach with rotors splayed to ~45°
(a more symmetric X), so it has ample roll authority and hovers on the **default** gains —
no `--kp_roll/--kd_roll` override. Both its USD and allocation already exist:

```bash
python 01_hover_record.py \
  --usd parametric_drone/generated_gp/usd/gptm_family_42_morph_09.usd \
  --allocation_json parametric_drone/generated_gp/allocation/gptm_family_42_morph_09.json \
  --video_path hover_morph09_stable.mp4 \
  --cam_eye 1.6 1.6 2.1 --cam_target 0.05 0.1 1.5 \
  --max_steps 800 --record_every 2 --headless --enable_cameras
```

Result: `hover_morph09_stable.mp4`. This confirms morph_00's instability was **geometry,
not gains** — same controller, longer roll arm → stable.

Arm length is set in `parametric_drone/generate_gp_morphologies_vectoring.py`
(gene endpoints `arm_length_closed/_open`, lines 82–83; per-morph `lerp`, line 126). To
exceed the family's range, edit those, re-run that generator, then convert URDF→USD (the
converter command appears in Part 2b).

## 2b. A custom tilted-thrust hexrotor — generate USD + allocation

`02_generate_novel_morphology_with_rotor_cant.py` is a self-contained generator that builds
a **rigid 6-rotor drone with a fixed alternating tangential rotor cant** and — the key part —
emits a matching **allocation JSON** from the same poses, so it's directly flyable by the
hover controller.

**One-time setup** (in addition to Step 1's env wiring):
```bash
pip install xacro                                              # not preinstalled here
export PATH="$CONDA_PREFIX/bin:$PATH"                          # so the xacro console script is found
export ISAACLAB_PATH=/home/keiichi-ito/Documents/sandbox/IsaacLab   # generator locates the converter via this
```

**Generate** (writes xacro → URDF → USD + allocation into `conversion/`):
```bash
python 02_generate_novel_morphology_with_rotor_cant.py
```
Expect `B_rpyt rank 4/4`, mass ≈ 1.04 kg; it prints the exact fly command at the end.

**Fly + record** (default gains — the symmetric hex has ample roll authority):
```bash
python 01_hover_record.py \
  --usd conversion/usd/hex_cant15_6rotor_drone.usd \
  --allocation_json conversion/allocation/hex_cant15_6rotor_drone.json \
  --video_path hover_hex_cant15.mp4 \
  --lock_joints_regex ".*_motor_to_rotor_joint" \
  --cam_eye 1.5 1.5 2.1 --cam_target 0.0 0.0 1.5 \
  --max_steps 800 --record_every 2 --headless --enable_cameras
```
`--lock_joints_regex` must match ≥1 joint on the USD; this hex's only free joints are the
6 `*_motor_to_rotor_joint` rotor spins. Result: `hover_hex_cant15.mp4` — hovers on default
gains.

**Key knobs in the generator's `main()`:** `num_motors` (3–8), `rotor_cant_deg` (cant
magnitude; 15° gives mild yaw authority — raise for more), `cant_alternate` (alternating
sign keeps net horizontal force *and* net yaw at zero → clean hover). The allocation
(`rotor_positions`, `rotor_directions`, `B_rpyt`) is computed by `compute_allocation()`
from the same MotorSpec poses the xacro bakes, so geometry and control never desync. The
frame is rigid (`enable_arm_tilt=False`, `enable_motor_tilt=False`) — the only tilt is the
fixed rotor cant.

---

## File map

| File | What it is |
|------|-----------|
| `01_load_single_usd_apply_wrench_hover.py` | Base hover controller (stock; diverges on morph_00 with default gains) |
| `01_hover_record.py` | Recording variant: same controller + camera → mp4, joint-lock actuators (`--lock_joints_regex`), auto-mass |
| `diag_hover.py` | Per-step diagnostic logger (no camera) used to find the roll instability |
| `02_generate_novel_morphology_with_rotor_cant.py` | Self-contained generator: rigid canted **hexrotor** + allocation-JSON export (Part 2b) |
| `parametric_drone/generated_gp/{usd,allocation}/gptm_family_42_morph_00.*` | morph_00 asset + allocation (short arms) |
| `parametric_drone/generated_gp/{usd,allocation}/gptm_family_42_morph_09.*` | morph_09 asset + allocation (long arms, Part 2a) |
| `conversion/{usd,allocation}/hex_cant15_6rotor_drone.*` | Generated tilted-thrust hex USD + allocation (Part 2b) |
| `hover_morph00.mp4` | morph_00 **divergent** run (following camera) — the "before" |
| `hover_morph00_stable.mp4` | morph_00 **stable** hover (fixed close camera) — the "after" |
| `hover_morph09_stable.mp4` | morph_09 (long arms) hover on default gains (Part 2a) |
| `hover_hex_cant15.mp4` | Tilted-thrust hexrotor hover on default gains (Part 2b) |

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'isaacsim'`** — you skipped the `source ...
  setup_conda_env.sh` in Step 1, or a non-interactive shell picked the wrong Python.
- **`ModuleNotFoundError: No module named 'isaaclab'`** — wrong conda env active.
- **Blank / empty video** — you omitted `--enable_cameras`.
- **`Body 'base_link' not found`** — pass `--base_body_name <name>`; the script prints
  the available body names on load.
- **USD `Unresolved reference prim path .../visuals/...` warnings** — benign
  composition warnings from these generated assets; ignore them.
- **It still climbs away (morph_00)** — you didn't pass the roll-gain overrides
  (`--kp_roll 2.5 --kd_roll 0.3`). Not needed for morph_09 or the hex (Part 2).
- **`xacro: command not found` / `No module named 'xacro'`** (Part 2b) — `pip install
  xacro`, then use the `$CONDA_PREFIX/bin/xacro` console script and ensure the env bin is
  on `PATH` (`export PATH="$CONDA_PREFIX/bin:$PATH"`). There is no `python -m xacro`.
- **`Could not find IsaacLab root ...`** (Part 2b) — set `export
  ISAACLAB_PATH=<your IsaacLab repo>`.
- **`ValueError` / no joints found for an actuator group** — your `--lock_joints_regex`
  matched zero joints on the loaded USD. Pass a pattern that matches (the script prints the
  articulation's body names on load; check the URDF for joint names).

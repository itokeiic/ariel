# Drone morphology optimisation for gate racing — state of the work

Written 2026-08-16, covering commits `a6b16c0..2470819` on branch `spear`.
Update the range when you add to it; a stale range is how a reader ends up
trusting a section that a later commit superseded.
Read this before touching the drone EA, the Lee controller, or the B-spline gate
trajectory: several defects were found in code that had already produced
results, and some of those results should not be trusted.

---

## 1. What the work is for

The SPEAR line of work aims at drones that **change shape in flight**. The step
before that is knowing whether shape matters at all: given a gate-racing task,
which arm geometry flies it best? A static arm angle is the same thing as
holding a revolute joint at a fixed angle, so an answer transfers directly to a
joint target once the articulated plant exists.

Two constraints shape everything below:

* The consortium runs on **Isaac Lab / PhysX**, so the blueprint has to emit
  articulated USD, and any control work has to be portable to that plant.
* ARIEL's own drone plant is a reduced ODE model. It is fast, and it is fine
  for the axes swept here, but it has a hard limitation (§3.2).

---

## 2. The headline experimental result

**Static arm geometry is invisible to this task across almost all of its
operating range, and where it becomes visible, it does so through a cliff.**

Measured on a SPEAR-matched quadrotor (0.83 kg, 5" props, 0.20 m arms,
thrust-to-weight 5.2), three probe bodies spanning the feasible half-angle range
(narrow 20.4°, X 45°, wide 69.6°), on parameterised slaloms:

| turn | speed | gates (sequential) | gates (flown) | tracking | rotor clipping |
|------|-------|--------------------|---------------|----------|----------------|
| 60–120° | 2.0–3.8 m/s | 14.00 ± **0.00** | 15.00 ± **0.00** | 0.07–0.33 m | 0.1% |
| 90° | 3.90 m/s | 9.44 ± 2.67 | 15.00 ± 0.00 | 0.387 m | 0.7% |
| 90° | 4.00 m/s | 3.89 ± 0.33 | 13.22 ± 2.00 | 0.668 m | 8.5% |

Nine operating points spanning a 2× range of speed and turn angle give
*identical* scores for all three bodies. A 5% change in speed then takes the
task from "everyone is perfect" to "everyone is struggling".

### Why geometry is invisible

The Lee controller inverts the **same allocation matrix** the plant integrates,
and `auto_scale_gains=True` re-derives attitude gains per body from its inertia
to hit a fixed 12 rad/s bandwidth. With an exact model, the allocation cancels
geometry: a longer arm simply receives proportionally different thrust commands.
Torque margin across the family is 35–77×, so no body is ever authority-starved
at the commanded bandwidth.

Geometry can therefore only matter through **actuator saturation**, and the
binding limit is not the one we expected:

* Upper bound (not enough thrust): **0.0%** of steps at 4 and 6 m/s.
* Lower bound (allocation wants a rotor to *pull*): **29.1%** at 6 m/s.

The constraint is **differential range, not total thrust**. Raising
thrust-to-weight would not help. A layout producing more moment per newton of
differential needs less differential and clips less, which is exactly the
geometry-sensitive channel — and clipping does vary monotonically with layout
(1.2% → 3.2% across arm length, tracking `α_roll` 272.8 → 125.2 rad/s²).

### Scoring by each body's own limiting speed (the cleanest test)

Fixed-speed scoring sits on the cliff above, so each body was instead scored by
**the fastest speed at which it still completes the course**, found by bisection
(monotone in speed; checked, 0 violations in 21 bodies / 168 rollouts). This is
continuous by construction: no operating point to calibrate, no ceiling, and at a
body's own limit the two gate-counting rules agree. Uniform courses, since a
uniform slalom is one corner repeated and has no layout to overfit; generalisation
is checked by sweeping turn angle instead.

Max completing speed, 7 arm half-angles x 3 corner sharpnesses:

| half-angle | roll agility (rad/s^2/N) | 60° | 90° | 120° |
|---|---|---|---|---|
| 20.44° (H-frame, fore-aft) | 313.3 | **4.688** | **4.062** | **3.812** |
| 45.00° (X frame) | 159.9 | 4.625 | 3.938 | 3.688 |
| 69.56° (wide, lateral) | 121.3 | 4.688 | 3.938 | 3.688 |

Max speed falls with corner sharpness as the difficulty parameter intends, but
the **morphology spread is 0.06-0.125 m/s -- one to two quantisation steps**. A
2.6x range in roll angular acceleration buys about **3%** more speed, and the
argmax does not move with corner sharpness: the same body wins at 60°, 90° and
120°, so there is no evidence here that different corner sharpness wants a
different airframe.

The diagnostic at each body's own limit is the more useful result, because the
bodies **fail for different reasons**:

| body | max speed | tracking at limit | lower-bound clipping |
|---|---|---|---|
| narrow (20.44°) | 4.062 | 0.706 m | **19.2%** |
| X (45.00°) | 3.938 | 0.414 m | **0.94%** |
| wide (69.56°) | 3.938 | 0.437 m | 5.06% |

The narrow frame runs until it saturates -- clipping a fifth of all steps,
tracking degraded, still completing. The X and wide frames die at under 1-5%
clipping and *better* tracking: they stop before running out of authority,
killed by lateral error at corner reversals rather than by actuator limits.

**Why the effect is small.** `auto_scale_gains=True` derives each body's attitude
gains from its own inertia to hit a fixed 12 rad/s bandwidth, so the controller
deliberately equalises attitude response across morphologies. With 35-77x torque
margin on top, a 2.6x authority difference is spent on control *effort* (the
19% vs 1% clipping spread) rather than on performance. The finding is therefore
not "geometry does not matter" but the sharper: **a per-body-tuned model-based
controller designs static geometry out of the closed loop.** Morphology shows up
only where the controller cannot compensate -- a changed achievable wrench set
(tilted rotors, needs the normal-aware plant) or the transients of the shape
change itself.

### Confirmed: the controller was hiding the morphology (`--fixed-gains`)

Re-running the identical sweep with `auto_scale_gains=False` -- one controller
for every body, so closed-loop bandwidth becomes `sqrt(K_rot/I)` and varies
across the family -- supports the explanation above:

| slalom | spread, auto-scaled | spread, fixed gains | ratio | argmax auto -> fixed |
|---|---|---|---|---|
| 60° | 0.062 m/s | 0.312 m/s | **5.0x** | 20.44° -> 20.44° |
| 90° | 0.125 m/s | 0.312 m/s | **2.5x** | 20.44° -> 20.44° |
| 120° | 0.125 m/s | 0.250 m/s | **2.0x** | 20.44° -> **28.63°** |

Two things change beyond the spread:

* **The ordering becomes physical.** At 90° with auto-scaled gains, five of
  seven bodies scored an identical 3.938 m/s. With fixed gains the column is
  monotone in roll agility -- 4.000, 4.000, 3.938, 3.875, 3.812, 3.750, 3.688 --
  every step down in `alpha_roll` costs speed.
* **The best body changes with corner sharpness.** The narrowest frame wins at
  60° and 90°, but at 120° `t=28.63°` beats it (3.625 vs 3.562). Under fixed
  gains the narrowest body is heavily overdamped (zeta 2.16 against 0.82 for the
  wide one), so at the sharpest corners, where reversals come fastest, its
  sluggishness costs more than its authority buys. This is the
  "different airframe for different task" effect, and it is invisible under
  per-body-tuned control.

**Morphology optimisation on this stack is therefore a co-design problem.**
Optimising a body against a controller that re-tunes itself per body measures
almost nothing, because the controller absorbs the design. Two caveats: the
effect is larger but still modest (6-8% against 1.5-3%), and fixed gains is not
the "right" experiment either -- it measures which body suits *those* gains.
Neither extreme is the answer; the joint problem is.

Monotonicity held on all 21 bodies in both runs (0 violations, 168 rollouts
each).

### The agility ordering is the opposite of the torque ordering

Roll *torque* authority rises as the quad widens (1.735 → 3.896 N·m from t=24°
to 66°), but inertia rises faster, so roll *angular acceleration* **falls**
(2921 → 1325 rad/s²). A tight slalom needs the latter: at 6 m/s on a 2.4 m leg
the drone has ~0.4 s to swing from +69° to −69° of bank. Comparing operating
points confirms it — 120°@3.5 m/s and 90°@4.0 m/s demand nearly the same
lateral acceleration (8.9 vs 9.5 m/s²) but clip 0.1% and 8.5% respectively, so
what binds is the *rate* of attitude change, which scales with speed rather than
with turn angle.

---

## 3. Defects found in existing code

Six, in code that had been producing published numbers. Each is committed with
its evidence.

### 3.1 Arm-pitch convention mismatch (`a6b16c0`)

ARIEL's spherical genome stores arm pitch as an **elevation** (0 = horizontal);
the ported airevolve geometry code read it as a **polar angle from +z**. A
6-arm drone at r = 0.17 m, 60° apart, elevation 0 mapped to six cylinders
stacked at (0, 0, 0.17). Collision repair therefore saw fabricated collisions,
burned all 100 iterations, and wrote back geometry in a frame the decoder does
not share.

Post-repair collisions on random populations went **10/10 → 0/10**. Worse, the
pre-fix EA was *selecting for* the resulting degenerate bodies: the best
individual of a pre-fix run had a minimum motor-motor distance of 0.0198 m
against a 0.0533 m requirement (overlapping rotors) and arm elevations of 86°
and 112° against a declared ±15° clamp. **Treat every example-17 result from
before this fix as invalid.**

### 3.2 The plant ignores thrust direction (`a6b16c0`, guarded not fixed)

`dynamics_params.derive_reference_params` builds the plant from each rotor's
`loc` but **ignores `dir[0:3]`**, applying all thrust along the body axis, while
`DroneConfiguration._compute_allocation_matrices` honours it when building the
`Bf`/`Bm` that `get_params()` hands the controller as `mixerFM`. A canted rotor
is therefore *allocated* as tilted and *simulated* as axial.

`_guard_axial_thrust` now warns once per process (or raises under
`ARIEL_STRICT_THRUST_NORMALS=1`) with the deviation and the fraction of thrust
not simulated. **This is still open**: a normal-aware plant is the real fix, and
it is why the sweeps here are confined to arm azimuth and length, the axes the
plant models correctly. Arm elevation and motor cant are *not* trustworthy on
this plant.

### 3.3 `blueprint_to_usd` had never run (`73fcb21`)

It raised `NameError` on any blueprint containing a `RotorNode` — i.e. every
blueprint any decoder produces. The consortium-facing backend existed on paper
only; `examples/d_drones/17_visualize_usd_blueprint.py` could not have worked.

### 3.4 B-spline gates were never interpolated (`75bd8bb`)

The class claims "exact interpolation" at `tension=1.0`, but its correction is a
local approximation valid only when consecutive gates are near-collinear. On a
slalom the curve missed a gate centre by up to **0.64 m** against a 0.5 m
half-width — the reference path flew *outside* the gate it was meant to pass.
`fit_offsets_to_gates()` solves the collocation system properly; worst-case miss
drops to ~5e-3 m.

### 3.5 Reported velocity did not match reported position (`75bd8bb`)

For non-periodic splines the loop phase advanced `u` by
`u_range − startup_consumed` while returning `du_dt` for the un-reduced
`u_range / loop_time`. Reported velocity was **22% too large**. Harmless while
callers used position only; feeding it forward commands a speed the path does
not have.

### 3.6 The startup ramp overshot cruise speed (`215316d`)

The quintic ramp had to cover `loop_speed × startup_time` while starting from
rest and ending *at* loop_speed — a mean equal to the endpoint maximum, which
forces the middle above it. Measured peaks **2.1–2.4× cruise**, so a "4 m/s"
course demanded 5.67 m/s during startup and broke the vehicle before it reached
cruise. Fixing it took a 90° slalom at 4 m/s from 2/15 gates to 12/15.

### 3.7 Lee control: zero velocity setpoint, world-z thrust (`b1fff51`)

* `LeePositionController` hard-coded `setpoint_velocity` to zeros, so a moving
  reference was tracked with a stop-here target. Measured lag error/speed of
  0.32/0.27/0.24/0.24 s at 2/4/6/8 m/s — a constant time lag no gain removes
  (7× the position gain moved tracking 10%).
* `_wrench_to_motor_commands` took the **world** z-component of the desired
  force. PX4 projects onto the **body** z axis, which the attitude loop tilts to
  meet the force; world z is correct only in level flight. At 71° of bank a
  34.9 N force command produced an 11.1 N thrust command. **This one changes
  behaviour for every Lee user** (~1% at the 7–10° tilts of the old courses, so
  prior results shift slightly rather than qualitatively).

Velocity/acceleration feedforward and the `max_accel` clamp are **opt-in**
(`velocity_feedforward=True`, `max_accel=...`); library defaults are unchanged.
The clamp defaulted to 5.0 m/s² — below what an aggressive course demands
(21 m/s²) and far below what the airframe delivers (50 m/s²), so while it binds
every morphology is limited by the same constant.

---

## 4. What was built

| piece | where | why |
|---|---|---|
| `JointSpec` on `ArmNode`/`MotorNode` | `body_phenotypes/drone/blueprint.py` | mid-flight morphing needs joints in the IR; defaults to `fixed`, so existing blueprints are unchanged |
| Articulated USD export | `body_phenotypes/drone/backends.py` | flat rigid-body links + `PhysicsRevoluteJoint` with drives. Names match `02_generate_novel_morphology.py`, so the consortium's Isaac scripts (`00`/`01` `--base_body_name base_link`, `03`'s `find_joints`) drive blueprint output unmodified |
| Backend-agnostic plant interface | `simulation/drone/plant.py` | `DroneState`/`DroneCommand`/`RotorGeometry`/`DronePlant`. `rotor_frames()` is load-bearing: adapters report live geometry each step so `B(q)` is rebuilt rather than cached, which structurally prevents §3.2's class of bug |
| Maneuverability metrics | `simulation/drone/plant.py` | airevolve's `min(eig(Bm Bmᵀ))` and `rank(Bm)`, plus an inertia-normalised form |
| `payload_mass` | `simulation/drone/drone_configuration.py` (+simulator, interface) | the mass model never reads `CorePlateNode.mass`, so a blueprint quad weighs 0.093 kg (TWR 8.9) against SPEAR's 0.83–1.25 kg. 0.667 kg of payload gives 0.829 kg at TWR 5.24 |
| Parameterised slalom | `simulation/tasks/slalom_course.py` | difficulty as one number; see §5 |
| Design sweep | `examples/spear/19_morphology_design_sweep.py` | the experiment itself, with `--tracking-check`, `--calibrate` and `--map-only` modes |

### The maneuverability metric has a caveat worth knowing

For **coplanar** rotors the smallest eigenvalue of `Bm Bmᵀ` is the *yaw* one, and
yaw authority comes from the propeller drag ratio `km/kf`, which no in-plane
geometry changes. Eigenvalues on a SPEAR-matched quad are
`[5.1e-04, 0.08, 0.08]` with the small eigenvector exactly `(0,0,1)`. The metric
is therefore **identical across a whole azimuth sweep and across a 2× change in
arm length**. It reports "can this airframe yaw", not "is it agile". Pass
`inertia=` for the `I⁻¹Bm` form, which does vary (342 → 16 as arms go
0.12 → 0.25 m), or read per-axis rows of `moment_allocation()` when the task
loads one axis in particular. It comes into its own for tilted rotors.

---

## 5. Task design decisions, and the measurements behind them

* **Difficulty is the turn angle, at fixed leg length.** At fixed *spacing* a
  sharper turn also lengthens the legs, so the corner gets *wider* — radius
  4.00/2.31/2.00/2.31 m for 30/60/90/120°, i.e. non-monotonic. At fixed leg it
  falls 4.64 → 1.39 m.
* **Gate yaw is the bisector of adjacent legs.** `GateChecker` counts a pass
  only when the drone crosses the plane with normal `(cos yaw, sin yaw, 0)`; the
  `SlalomGates` preset in `controllers/utils/gate_configs.py` uses a
  `[1,0,-1,0]·π/2` pattern that does not follow the path and would mis-detect.
* **Spacing must exceed gate size.** Shrinking a course to raise difficulty
  makes gate planes overlap and "passing a gate" ambiguous — this is why scaling
  the quintic courses down was abandoned.
* **Lead-out of 2 legs.** The trajectory clamps at its final waypoint, so a
  drone arriving at speed sailed past a course ending on the last scored gate —
  closest approach 2.14–2.65 m for *every* body. Gate 14 was unreachable by
  construction, which capped `flown` counting at 14/15 for everyone and made it
  unable to discriminate at all.
* **Multi-course averaging** (`--n-courses`, default 5). The simulator is
  deterministic, so layout is the only sampling error; seeded courses jitter the
  per-corner turn angle. Note the jittered set is genuinely harder than the
  uniform one (a run is governed by its *hardest* corner, and jitter raises the
  max while leaving the mean).

### Objective function

`--objective normalized` (default): gates scored out of 100, plus a quality
budget worth **half a gate**, split four ways — `track` (`exp(−err/(gate/2))`),
`margin` (`1 − clipping fraction`), `survive`, `speed`. Capping the budget below
one gate keeps it lexicographic in practice (14 gates with perfect quality =
96.67 ranks below 15 gates with worst quality = 100.00) while staying continuous
for an EA. Within a gate tier — most comparisons, on this landscape — the
quality terms are the entire signal.

`margin` exists because of §2: it is the only term that responds to the
constraint that actually binds.

`--objective legacy` reproduces example 17's formula
(`10·gates + survival − 0.1·track + completion bonus`), whose term magnitudes
are 30–150 / 0.6–1.0 / −0.008 to −0.39, i.e. the gate count with rounding errors
attached.

### Gate counting is a real choice

`--gate-counting sequential|flown`. Sequential stops at the first miss (racing
semantics, high variance: one corner costs `10 × remaining gates`); flown counts
every gate passed within half a gate, in any order. On identical flights they
differ sharply — 12 vs 14 gates at 4 m/s uniform, 4.00 vs 13.00 on a jittered
set. Both are recorded in every CSV whichever is active, so runs can be
re-scored without re-flying.

---

## 6. How to run it

```bash
# does the controller track at all at speed X?
uv run examples/spear/19_morphology_design_sweep.py --tracking-check \
    --feedforward --max-accel 40

# where does the task separate morphologies?
uv run examples/spear/19_morphology_design_sweep.py --calibrate \
    --cal-speeds 2,3,4 --cal-turns 60,90,120 --feedforward --max-accel 40

# geometry only, no rollouts: feasibility + metrics over (half-angle x length)
uv run examples/spear/19_morphology_design_sweep.py --map-only

# the sweep itself
uv run examples/spear/19_morphology_design_sweep.py --turn-deg 90 --speed 4.0 \
    --gate-counting flown --feedforward --max-accel 40
```

Use the repo's uv venv (`uv run`). The conda `ariel` env lacks `fcl` and cannot
run the repair path. `ariel-isaaclab-train` plus Isaac's bundled USD is needed
only for USD validation:

```bash
USDLIBS=$(ls -d <isaac>/extscache/omni.usd.libs-*/)
PYTHONPATH="$USDLIBS" LD_LIBRARY_PATH="${USDLIBS}bin:<conda-env>/lib" \
  <conda-env>/bin/python your_script.py
```

---

## 7. Open items

1. **Normal-aware plant (§3.2).** Guarded, not fixed. Blocks any experiment on
   arm elevation, motor cant, or tilted rotors — i.e. most of the interesting
   morphing directions. The intended resolution is to move the plant to
   Isaac/PhysX rather than extend the reduced ODE, since PhysX models thrust
   direction, time-varying inertia and joint dynamics natively.
2. **Repair clips before collision repair**, so repaired genomes escape their
   declared clamps (~22° against a 15° clamp). Deferred deliberately.
3. **Rotor-body collisions are unchecked.** The collision test only compares arm
   cylinders with each other; nothing stops a rotor intersecting the core. The
   EA's `inner_boundary_radius = 0.055` permits it. The sweep checks
   `L ≥ core_radius + 1.1·prop_radius` itself.
4. **Repair's contract changes for morphing bodies**: collision-free at `q = 0`
   guarantees nothing across a joint envelope.
5. **The task is a cliff** at fixed speed (§2) -- nine operating points from 2.0
   to 3.8 m/s scored every body identically, and a 5% speed increase collapsed
   them all. Resolved by the max-speed objective (`--max-speed-sweep`), which is
   continuous by construction. Keep this in mind before designing any new
   fixed-speed experiment on this task.
6. **Co-design.** `--fixed-gains` (§2) showed the controller was hiding most of
   the morphology effect, so body and controller should be optimised jointly.
   Nothing in the EA does this yet. Superseded question, kept for the record:
   `auto_scale_gains` normalises
   attitude bandwidth per body, which is the leading explanation for the 3%
   effect size. `--fixed-gains` runs the same sweep with one controller for all
   bodies, where closed-loop bandwidth becomes `sqrt(K_rot/I)` and varies 2.6x
   across the family (25.9 rad/s at zeta 2.16 for the narrow frame, 9.9 rad/s at
   zeta 0.82 for the wide one). If the spread jumps, morphology optimisation on
   this stack is really a **co-design** problem and body and controller cannot be
   optimised separately.
7. **Lee tracking above ~4 m/s.** Cruise tracking is excellent once settled
   (0.07 m at 4 m/s), but feedforward is not tuned — enabling it removes the lag
   and introduces speed overshoot and altitude sag, so the loop wants redesigning
   as `a_des = a_ff + Kp·e_p + Kd·e_v` with gains derived for tracking rather
   than point-holding.

# Drone morphology optimisation for gate racing — state of the work

Written 2026-08-16, covering commits `a6b16c0..2470819` on branch `spear`.
Update the range when you add to it; a stale range is how a reader ends up
trusting a section that a later commit superseded.
Read this before touching the drone EA, the Lee controller, or the B-spline gate
trajectory: several defects were found in code that had already produced
results, and some of those results should not be trusted.

> **The one open bug that bounds what this codebase can currently claim.**
> The plant ignores each rotor's thrust direction (§3.2). It simulates all
> thrust along the body axis while the *controller* allocates using the true
> normals, so a canted rotor is allocated as tilted and flown as axial. It is
> **guarded, not fixed** -- `_guard_axial_thrust` warns, or raises under
> `ARIEL_STRICT_THRUST_NORMALS=1`.
>
> It is a **regression introduced in ARIEL**, not inherited: airevolve's plant,
> from which this one descends, computes `F_body = Bf @ U` and honours the
> normals correctly. A twenty-line reference implementation therefore exists.
>
> Consequences: every result here is confined to arm azimuth and arm length,
> the two axes the plant models correctly. **Arm elevation, motor cant, tilted
> rotors and any fully-actuated design are not simulable today** -- which is
> most of the interesting morphing directions, and the reason §8 item 1 is the
> highest-value open item rather than a footnote.

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
| 28.63° | 233.1 | 4.625 | 4.000 | 3.688 |
| 36.81° | 187.9 | 4.625 | 3.938 | 3.688 |
| 45.00° (X frame) | 159.9 | 4.625 | 3.938 | 3.688 |
| 53.19° | 141.6 | 4.625 | 3.938 | 3.688 |
| 61.37° | 129.3 | **4.688** | 3.938 | 3.688 |
| 69.56° (wide, lateral) | 121.3 | **4.688** | 3.938 | 3.688 |

Note the 60° column: the best score is shared by the narrowest *and* the two
widest bodies, with the middle of the family one step below. That is
non-monotone in every geometric quantity, at exactly one quantisation step, and
is best read as rounding rather than as structure.

Max speed falls with corner sharpness as the difficulty parameter intends, but
the **morphology spread is 0.06-0.125 m/s -- one to two quantisation steps**. A
2.6x range in roll angular acceleration buys about **3%** more speed, and the
argmax does not move with corner sharpness. That last point needs one
qualification: at 90° and 120° the narrowest body wins outright, but at 60° the
best score is a **three-way tie** (20.44°, 61.37° and 69.56° all reach
4.688 m/s), so "wins" there means "ties for first". Either way there is no
evidence that different corner sharpness wants a different airframe -- under
auto-scaled gains.

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

### SUPERSEDED (2026-08-16): the numbers below were measured against a controller bottleneck

Everything in this section up to here, and the `--fixed-gains` comparison that
follows, was measured with the attitude loop at 12 rad/s and the position loop
at the inherited `omega_n = 3.78`. At that tuning motor clipping is 0.1-2%, i.e.
the **controller** is the binding constraint and the airframe is not being asked
for anything its geometry could differentiate. The findings are kept because the
comparison between the two regimes is itself the result, but the effect sizes
must not be quoted as properties of the airframes.

Retuning the cascade (see §5.2) doubles every body's speed and moves the
constraint to the airframe -- clipping 15-21% for all seven bodies. At that
operating point, on the 90 deg slalom:

| half-angle | roll agility | at 12 rad/s | clip | **at 24 rad/s** | clip |
|---|---|---|---|---|---|
| 20.44° | 313.3 | 4.062 | 19.2% | 8.414 | 20.5% |
| 28.63° | 233.1 | 4.000 | 11.5% | 8.523 | 21.2% |
| **36.81°** | 187.9 | 3.938 | 0.9% | **8.578** | 20.3% |
| 45.00° | 159.9 | 3.938 | 0.9% | 8.469 | 19.1% |
| 53.19° | 141.6 | 3.938 | 1.2% | 8.359 | 18.2% |
| 61.37° | 129.3 | 3.938 | 1.8% | 8.305 | 17.3% |
| 69.56° | 121.3 | 3.938 | 5.1% | 8.086 | 15.4% |

Two corrections to the superseded story:

* **Spread**: 0.125 -> 0.492 m/s, 3.2% -> 5.9% of the mean. At 0.0625 m/s
  resolution that is ~8 quantisation steps, so the shape is resolved rather
  than noise.
* **The optimum moves and becomes interior**: 20.44° -> **36.81°**, with a
  smooth single peak. "Narrower is monotonically better" was an artefact of
  measuring where geometry could not bind. The peak sits biased toward roll but
  not maximally (alpha_roll 187.9 against alpha_pitch 140.9), which is what a
  task that weaves laterally while accelerating longitudinally should want, and
  it is a far better landscape for an EA: smooth, single-peaked, interior.

### The result the chain was aimed at: corner sharpness selects different airframes

With the cascade tuned (§5.2) and every body clipping 15-21%, i.e. airframe-
limited rather than controller-limited, max completing speed across three corner
sharpnesses:

| half-angle | alpha_roll | 60° | 90° | 120° |
|---|---|---|---|---|
| 20.44° | 313.3 | 10.859 | 8.414 | 6.973 |
| 28.63° | 233.1 | 11.516 | 8.523 | **7.078** |
| 36.81° | 187.9 | **12.172** | **8.578** | **7.078** |
| 45.00° | 159.9 | 12.125 | 8.469 | 6.973 |
| 53.19° | 141.6 | **12.172** | 8.359 | 6.867 |
| 61.37° | 129.3 | **12.172** | 8.305 | 6.691 |
| 69.56° | 121.3 | 12.078 | 8.086 | 6.516 |

**The extremes swap ends.** The narrowest frame (highest roll agility) is *last*
at 60° -- 1.3 m/s and 11% behind -- and mid-pack at 120°. The widest frame is
competitive at 60° and *last* at 120°. The peak migrates with sharpness: a
plateau across 36.8-61.4° at 60°, 36.81° at 90°, and 28.63° at 120°.

The mechanism is the one §7.5 predicts. Sharper corners demand faster bank
reversals, which rewards roll angular acceleration; gentle corners flown at
12 m/s do not, and there the narrow frame's weak pitch authority (`alpha_pitch`
120.8 against 313.3 in roll) costs it.

**So no single static geometry is best across the difficulty range** -- which is
the first experimental support in this work for making the geometry adjustable
in flight. Measured before the cascade was tuned, the same sweep said the same
body won everywhere; that answer was an artefact of the controller-limited
regime.

Two limits on this: the 60° plateau is at resolution (its top four differ by
less than the 0.0625 m/s bisection tolerance), and the 120° peak is a tie
between 28.63° and 36.81°. The *direction* of the shift is consistent across all
three columns; the exact argmax at 60° and 120° is not resolved.

### Confirmed: the controller was hiding the morphology (`--fixed-gains`)

Re-running the identical sweep with `auto_scale_gains=False` -- one controller
for every body, so closed-loop bandwidth becomes `sqrt(K_rot/I)` and varies
across the family -- supports the explanation above:

| slalom | spread, auto-scaled | spread, fixed gains | ratio | argmax auto -> fixed |
|---|---|---|---|---|
| 60° | 0.062 m/s | 0.312 m/s | **5.0x** | 20.44° -> 20.44° |
| 90° | 0.125 m/s | 0.312 m/s | **2.5x** | 20.44° -> 20.44° |
| 120° | 0.125 m/s | 0.250 m/s | **2.0x** | 20.44° -> **28.63°** |

Max completing speed with one controller for every body (same layout as the
table above, so the two can be read side by side):

| half-angle | roll agility (rad/s^2/N) | 60° | 90° | 120° |
|---|---|---|---|---|
| 20.44° (H-frame, fore-aft) | 313.3 | **4.812** | **4.000** | 3.562 |
| 28.63° | 233.1 | 4.750 | **4.000** | **3.625** |
| 36.81° | 187.9 | 4.750 | 3.938 | 3.562 |
| 45.00° (X frame) | 159.9 | 4.688 | 3.875 | 3.562 |
| 53.19° | 141.6 | 4.625 | 3.812 | 3.500 |
| 61.37° | 129.3 | 4.562 | 3.750 | 3.438 |
| 69.56° (wide, lateral) | 121.3 | 4.500 | 3.688 | 3.375 |

Compare column by column with the auto-scaled table. Under auto-scaling the 90°
and 120° columns are almost constant -- five and six of seven bodies share a
single value. Under fixed gains the 60° and 90° columns descend monotonically
with roll agility, and the 60° column spans 4.500-4.812 m/s where auto-scaling
spanned 4.625-4.688.

The 120° column is the exception, and deliberately so: it descends from 28.63°
downward, but the narrowest body sits *below* its neighbour (3.562 against
3.625). That single inversion is the argmax flip discussed below -- the point
where the narrowest frame's overdamping under fixed gains costs more than its
authority returns.

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

Seven, in code that had been producing published numbers. Each is committed
with its evidence.

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

**It is a regression, not an inherited limitation, and the fix has a working
reference.** ARIEL's `DroneSimulator` descends from airevolve's (TU Delft
lineage), and *airevolve's plant does not have this bug*. It builds the dynamics
straight from the allocation matrices:

```python
F_body = Bf_sym @ U      # 3-D force, thrust along each rotor's own axis
M_body = Bm_sym @ U
I_inv  = Matrix(self.config.get_inertia_inverse(method="svd"))   # full tensor
```

ARIEL replaced that with the scalar `k_p_signed`/`k_q_signed` coefficients of
`derive_reference_params` (commit `4ed7e5d`, "Migration started."), mirroring
`experimentation/reference_drone_sim.py` (absent from this repository) -- a
quadrotor-specific reference whose
own comment concedes it "hardcodes signs for a specific motor layout". The trade
made there:

| | airevolve | ARIEL reference-form |
|---|---|---|
| thrust direction | **honoured** | ignored |
| inertia | **full tensor** (SVD inverse) | diagonal, baked into scalars |
| `omega x I omega` | dropped | dropped |
| aerodynamic drag | absent | **present** (`k_x`, `k_y`) |
| motor lag + sqrt-poly command | absent | **present** (`tau`) |
| parity test against the reference | — | **yes** |

It gained motor and drag fidelity and lost 3-D geometry fidelity; the loss does
not appear to have been noticed, and the parity test now pins the reduced
behaviour in place.

**Fix path.** Two obstacles previously recorded here -- the yaw linearisation and
the diagonal-inertia assumption -- were artefacts of the scalar form and
disappear with it: `Bm @ U` gives all three moment axes quadratically and
consistently, and airevolve already inverts the full tensor. Keep ARIEL's motor
lag, drag and sqrt-poly mapping, and replace the force/moment computation with
`F_body = Bf.W^2` and `omega_dot = I^-1 (Bm.W^2)`. The reference implementation
is about twenty lines in
`airevolve/simulator/simulation/drone_simulator.py:118-150`.

**There is no safety net for this change.** The docstrings cite
`unit_tests/test_dynamics_parity.py`, `experimentation/reference_drone_sim.py`
and an `optimal_quad_control_RL` reference as verifying the current dynamics "at
machine precision (<1e-9 rel err)". *None of them exists in this repository or
in its history*; those citations are now marked unverifiable in the source. So
the fix should begin by writing the characterisation test that does not exist:
record planar-quad trajectories now, require them afterwards. Build in one
expected difference rather than discovering it -- force, roll and pitch should
match to machine precision, but **yaw will change by design**, since the reduced
form linearises yaw at hover (`k_r_signed . W`, plus a `dW` reaction term) while
`Bm @ U` is quadratic in `W`.

**Scope.** airevolve's own morphology results are unaffected -- its plant flies
canted rotors correctly. The exposure is ARIEL-only.

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
| `JointSpec` on `ArmNode`/`MotorNode` | `src/ariel/body_phenotypes/drone/blueprint.py` | mid-flight morphing needs joints in the IR; defaults to `fixed`, so existing blueprints are unchanged |
| Articulated USD export | `src/ariel/body_phenotypes/drone/backends.py` | flat rigid-body links + `PhysicsRevoluteJoint` with drives. Names match `examples/spear_vua_upb/spear/02_generate_novel_morphology.py`, so the consortium's Isaac scripts (`examples/spear_vua_upb/spear/00_load_single_usd_apply_wrench.py` and its siblings default `--base_body_name base_link`, `03`'s `find_joints`) drive blueprint output unmodified |
| Backend-agnostic plant interface | `src/ariel/simulation/drone/plant.py` | `DroneState`/`DroneCommand`/`RotorGeometry`/`DronePlant`. `rotor_frames()` is load-bearing: adapters report live geometry each step so `B(q)` is rebuilt rather than cached, which structurally prevents §3.2's class of bug |
| Maneuverability metrics | `src/ariel/simulation/drone/plant.py` | airevolve's `min(eig(Bm Bmᵀ))` and `rank(Bm)`, plus an inertia-normalised form |
| `payload_mass` | `src/ariel/simulation/drone/drone_configuration.py` (+simulator, interface) | the mass model never reads `CorePlateNode.mass`, so a blueprint quad weighs 0.093 kg (TWR 8.9) against SPEAR's 0.83–1.25 kg. 0.667 kg of payload gives 0.829 kg at TWR 5.24 |
| Parameterised slalom | `src/ariel/simulation/tasks/slalom_course.py` | difficulty as one number; see §5 |
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
  `SlalomGates` preset in `src/ariel/simulation/drone/controllers/utils/gate_configs.py` uses a
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

### 5.1 The two controller parameters, and why they are not settable

The attitude loop is a second-order system in torque
(`src/ariel/simulation/drone/controllers/lee_control/lee_controller.py`, the comment above the auto-scaling branch):

```
I·theta_ddot + K_angvel·theta_dot + K_rot·theta = 0
    =>  omega_n = sqrt(K_rot / I),   zeta = K_angvel / (2·sqrt(K_rot·I))
```

* **`omega_n`** is the attitude bandwidth: how fast the vehicle nulls an
  attitude error, settling in roughly `1/omega_n`. It matters here because at
  4 m/s on a 2.4 m leg the drone has ~0.6 s per corner and must reverse its bank
  each time, so attitude response has to be several times faster than the corner
  rate or it arrives late and misses laterally.
* **`zeta`** is the damping ratio: below 1 it overshoots and rings, at 1 it is
  the fastest approach without overshoot, above 1 it is sluggish. This is the
  term behind the 120° argmax flip -- the narrowest frame lands at zeta 2.16
  under fixed gains, so despite the highest nominal bandwidth it responds slowly.

**Neither is a parameter.** `omega_n_att = 12.0` is a hard-coded local inside
the `auto_scale_gains` branch (`src/ariel/simulation/drone/controllers/lee_control/lee_controller.py:95`), and `zeta` appears only
as the literal `2.0` in `rate_P_gain = 2.0 * I_diag * omega_n_att` -- that factor
*is* the choice `zeta = 1`. Substituting `K_rot = I·omega_n^2` and
`K_angvel = 2·I·omega_n` makes `I` cancel out of `omega_n` and pins `zeta = 1`,
which is exactly how auto-scaling makes every morphology respond identically.

With `auto_scale_gains=False` the gains fall back to fixed defaults
(`att = [0.3, 0.3, 0.1]`, `rate = [0.05, 0.05, 0.03]`) and the two quantities
become *emergent* from each body's inertia -- 25.9 rad/s at zeta 2.16 for the
narrow frame against 9.9 rad/s at zeta 0.82 for the wide one.

**For co-design, expose them.** Accept `omega_n_att` and `zeta` as constructor
arguments (defaulting to 12.0 and 1.0 so nothing existing moves) and compute
`att = I·omega_n^2`, `rate = 2·zeta·I·omega_n`. That gives an EA two physically
meaningful, inertia-normalised genes that mean the same thing across
morphologies. Searching the six raw gain numbers instead would not: a gain
value does not mean the same thing on a body with 7x the roll inertia, which is
the whole reason auto-scaling exists.

### 5.2 Cascade tuning: the position loop must sit below the attitude loop

The inherited position gains (14.3, 9.0) are `omega_n = 3.78 rad/s` at
`zeta = 1.19`, carried from a 93 g drone on near-straight courses. Scanning max
completing speed over both bandwidths on the X quad, 90 deg slalom, 0/54
non-monotone bisections:

| attitude omega_n | best speed | at position omega_n | ratio | clipping |
|---|---|---|---|---|
| 6 | 2.375 m/s | 1.30 | 4.6x | 0.0% |
| 12 | 5.125 m/s | 1.60 | 7.5x | 2.0% |
| **24** | **8.438 m/s** | 2.00 | 12x | **19.2%** |
| 36 | 7.500 m/s | 4.00 | 9x | 24.6% |

Three things this fixes and one it reveals:

* **The position loop was roughly twice as stiff as it should be.** At a fixed
  12 rad/s attitude loop, retuning alone gives +32% (3.875 -> 5.125 m/s). The
  optimum is a 6-10x separation from the inner loop; at or above it (`ratio <=
  1.3x`) the two loops fight and the vehicle is **unflyable at any speed**, and
  far below it (15x) the loop is too soft to correct.
* **The ceiling scales with the cascade** until the airframe binds: doubling
  attitude bandwidth roughly doubles achievable speed, 2.375 -> 5.125 -> 8.438.
* **The crossover is at ~24 rad/s**, where clipping reaches 19%. Beyond it
  (36 rad/s) the loop demands more than the motors deliver and speed *falls*.
* **Morphology enters here.** Every body is commanded the same `omega_n_att`;
  whether it can deliver depends on its inertia and torque authority. That is a
  mechanism connecting `alpha_roll` / `I^-1 Bm` to a performance number rather
  than a correlation -- and it is why measurements below the crossover cannot
  see geometry.

Use `--att-omega-n 24 --pos-omega-n 2.0 --pos-zeta 1.0` for any morphology
experiment on this task.

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

# the sweep itself, at a fixed speed
uv run examples/spear/19_morphology_design_sweep.py --turn-deg 90 --speed 4.0 \
    --gate-counting flown --feedforward --max-accel 40

# what produced the headline results: score each body by its own limiting speed
uv run examples/spear/19_morphology_design_sweep.py --max-speed-sweep \
    --sweep-turns 60,90,120 --points 7 --speed-tol 0.0625 \
    --feedforward --max-accel 40

# the same, with one controller for every body (the co-design evidence)
uv run examples/spear/19_morphology_design_sweep.py --max-speed-sweep --fixed-gains \
    --sweep-turns 60,90,120 --points 7 --speed-tol 0.0625 \
    --feedforward --max-accel 40
```

`--feedforward` and `--max-accel 40` are needed on every run that flies faster
than about 2 m/s: the library defaults are a zero velocity setpoint and a
5 m/s^2 acceleration clamp (§3.7), and without them the drone lags the reference
by a fixed time and every morphology is limited by the same constant. Both
sweeps above take 168 rollouts, roughly 25 minutes each.

Use the repo's uv venv (`uv run`). The conda `ariel` env lacks `fcl` and cannot
run the repair path. `ariel-isaaclab-train` plus Isaac's bundled USD is needed
only for USD validation:

```bash
USDLIBS=$(ls -d <isaac>/extscache/omni.usd.libs-*/)
PYTHONPATH="$USDLIBS" LD_LIBRARY_PATH="${USDLIBS}bin:<conda-env>/lib" \
  <conda-env>/bin/python your_script.py
```


### 6.1 Every flag, and why it exists

Defaults in brackets. Most of these exist because of a specific measurement in
§2 or §3, named in the last column -- a flag whose reason is not recorded is a
flag nobody can safely change.

**Mode** (pick one; without any, the script runs the design map and both sweeps)

| flag | default | why it exists |
|---|---|---|
| `--tracking-check` | off | Establishes the usable speed range on one canonical body *before* spending a sweep. Added after a full sweep was run at a speed where the controller could not track, making every body look identical. |
| `--calibrate` | off | Grid over speed x turn angle, ranked by **spread across bodies** rather than mean score. Exists because a task everyone passes and a task everyone fails are equally useless for selecting a morphology, and we hit both. |
| `--max-speed-sweep` | off | Scores each body by its own limiting speed (bisection). Replaces fixed-speed scoring, which sits on a cliff (§2). |
| `--map-only` | off | Feasibility + geometry metrics with no rollouts, so the design space can be inspected in seconds. |

**Course**

| flag | default | why it exists |
|---|---|---|
| `--turn-deg` | 90 | The difficulty knob. Corner radius is `leg/(2 sin(turn/2))`, so this sets demanded lateral acceleration `v^2/R`. |
| `--leg` | 2.4 | Held fixed while turn angle varies, because at fixed *spacing* a sharper turn also lengthens the legs and the corner gets wider -- the knob becomes non-monotonic (§5). |
| `--n-gates` | 15 | Scored gates. Longer courses cost proportionally more per rollout. |
| `--gate-size` | 1.0 | Gate opening. Must stay below the resulting gate spacing or gate planes overlap and a pass becomes ambiguous; `slalom_gates` asserts this. |
| `--n-courses` | 5 | Layouts averaged per evaluation. The simulator is deterministic, so layout is the only sampling error. **Set 1 for max-speed sweeps**: a uniform slalom is one corner repeated, so there is nothing to overfit, and jitter would make each limit depend on whichever corner the RNG made sharpest. |
| `--seed` | 0 | Base seed for the course set. |
| `--jitter` | 0.25 | Per-corner turn-angle spread when `--n-courses > 1`. Note it raises effective difficulty even at constant mean, because a run is governed by its hardest corner. |

**Speed and the bisection**

| flag | default | why it exists |
|---|---|---|
| `--speed` | 6.0 | Nominal speed for fixed-speed modes. Traversal time is `startup + path_length/speed`, so this is a true speed, not a time budget. |
| `--speed-lo` / `--speed-hi` | 2.0 / 6.0 | Initial bisection bracket. A body failing at `lo` is reported as unflyable rather than scored. |
| `--speed-cap` | 12.0 | The bracket **widens upward** rather than clipping, since a ceiling would make good bodies tie -- the exact failure the max-speed objective exists to avoid. This bounds the widening. |
| `--speed-tol` | 0.125 | Bisection resolution. Each halving costs one rollout; 0.0625 was used for the reported sweeps because the morphology spread is only 1-2 steps wide at 0.125. |
| `--sim-margin` | 1.6 | Rollout time as a multiple of traversal time, so a lagging drone can still finish. Tracking error is accumulated only over the course itself -- past `total_time` the reference clamps and the drone overruns, which turned a 0.37 m cruise error into a reported 4.3 m. |
| `--cal-speeds` / `--cal-turns` | 2,3,4 / 60,90,120 | The `--calibrate` grid. |
| `--sweep-turns` | 60,90,120 | Corner sharpnesses for `--max-speed-sweep`. This is the *structured* generalisation check that replaces random jitter. |

**Controller** (see §3.7 -- the first two are required above ~2 m/s)

| flag | default | why it exists |
|---|---|---|
| `--feedforward` | **off** | Passes the trajectory's velocity and acceleration to the position controller. The library hard-codes a zero velocity setpoint, so a moving reference is tracked with a stop-here target and the drone lags by a fixed ~0.25 s that no gain increase removes. Off by default because the feedback gains are not tuned for it (§8 item 8). |
| `--max-accel` | 5.0 | Commanded-acceleration clamp. The library default is below what an aggressive course demands (21 m/s^2) and far below what the airframe delivers (50), so while it binds **every morphology is limited by the same constant** and no sweep can see geometry. Use 40. |
| `--fixed-gains` | off | Disables `auto_scale_gains`, so one controller flies every body. This is the co-design evidence: it widens the morphology spread 2-5x and makes the ordering monotone in roll agility (§2). |
| `--pos-gain` / `--vel-gain` | 14.3 / 9.0 | Position and velocity gains, inherited from example 17. Exposed during the tracking investigation; 7x the position gain moved tracking by 10%, which is how gain tuning was ruled out as the cause of the lag. |

**Scoring**

| flag | default | why it exists |
|---|---|---|
| `--objective` | `normalized` | `normalized` = gates out of 100 plus a quality budget worth half a gate. `legacy` reproduces example 17's formula, whose non-gate terms are three orders of magnitude smaller than its gate term. |
| `--gate-counting` | `sequential` | `sequential` stops at the first miss (racing semantics); `flown` counts every gate passed in any order. On identical flights these differ by 12 vs 14 gates, so it is a choice about the task. Both are recorded in every CSV whichever is active. |
| `--points` | 11 | Morphologies per sweep, spread evenly across the feasible half-angle range. |
| `--out-dir` | timestamped | Where CSVs land. |

---

## 7. Variables and metrics: what they mean and where they come from

Written for a reader comfortable with differential equations but not
necessarily with control or vibration theory. Everything below is either
measured by `examples/spear/19_morphology_design_sweep.py` or read off the geometry.

### 7.1 The second-order system, in one page

Every quantity called a *bandwidth* or a *damping ratio* here comes from one
equation. Take a mass on a spring with a dashpot:

$$m\ddot{x} + c\dot{x} + kx = 0$$

Divide by $m$ and rename the two coefficients:

$$\ddot{x} + 2\zeta\omega_n\dot{x} + \omega_n^2 x = 0,
\qquad \omega_n=\sqrt{k/m}, \qquad \zeta=\frac{c}{2\sqrt{km}}$$

$\omega_n$ (rad/s) is the **natural frequency**, and $\zeta$ (dimensionless) the
**damping ratio**. Substituting $x=e^{st}$ gives the characteristic roots

$$s = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2-1}$$

which is the whole story:

| $\zeta$ | roots | behaviour |
|---|---|---|
| $<1$ | complex pair | **underdamped** -- overshoots and rings, overshoot $\approx e^{-\pi\zeta/\sqrt{1-\zeta^2}}$ |
| $=1$ | real, repeated | **critically damped** -- fastest approach with no overshoot |
| $>1$ | real, distinct | **overdamped** -- no overshoot, but sluggish |

Both roots have real part $-\zeta\omega_n$, so disturbances decay like
$e^{-\zeta\omega_n t}$: **settling time $\approx 4/(\zeta\omega_n)$**. And
driving the system with a sinusoid, it follows inputs slower than $\omega_n$ and
attenuates and lags faster ones -- which is why $\omega_n$ is loosely called the
loop's *bandwidth*.

Why this governs a drone: a feedback controller that drives an error to zero
*is* such a system, by construction. Choosing gains is choosing $\omega_n$ and
$\zeta$; the only question is which physical quantity plays the role of $m$.

### 7.2 The attitude loop: $m$ is the inertia

For one axis, small angles, and a diagonal inertia tensor, rigid-body rotation
is $I\ddot{\theta} = \tau$. Lee's controller commands a torque from the attitude
and rate errors,

$$\tau = -K_\text{rot}\,\theta_\text{err} - K_\text{angvel}\,\dot{\theta}_\text{err}$$

so the error obeys

$$I\ddot{\theta} + K_\text{angvel}\dot{\theta} + K_\text{rot}\theta = 0
\qquad\Longrightarrow\qquad
\omega_n=\sqrt{K_\text{rot}/I},\quad
\zeta=\frac{K_\text{angvel}}{2\sqrt{K_\text{rot}I}}$$

Here the inertia plays the part of the mass, and that is the key to a result in
§2. `auto_scale_gains` does not pick gains directly; it picks a *response* and
solves for the gains that produce it, choosing

$$K_\text{rot} = I\,\omega_n^2, \qquad K_\text{angvel} = 2\zeta I\,\omega_n$$

(the code fixes $\zeta=1$, writing the second as $2I\omega_n$). Substituting
both into the closed-loop equation:

$$I\ddot{\theta} + \underbrace{2\zeta I\omega_n}_{K_\text{angvel}}\dot{\theta} + \underbrace{I\omega_n^2}_{K_\text{rot}}\theta = 0$$

Every term now carries a factor of $I$, so dividing through by $I$ (which is
strictly positive) removes it entirely:

$$\frac{I}{I}\ddot{\theta} + \frac{2\zeta I\omega_n}{I}\dot{\theta} + \frac{I\omega_n^2}{I}\theta = 0
\qquad\Longrightarrow\qquad
\ddot{\theta} + 2\zeta\omega_n\dot{\theta} + \omega_n^2\theta = 0$$

The inertia has vanished from the equation of motion. Whatever $I$ a body has,
its attitude error decays with the same $\omega_n$ and the same $\zeta$ --
identical settling time, identical overshoot. Physically: the gains are scaled
so that a heavier-to-rotate body is commanded proportionally more torque for the
same error, and the two effects cancel exactly.

That is why per-body-tuned control hides morphology. The controller is built to
erase the very difference the experiment is trying to measure.

**Contrast: fixed gains.** With $K_\text{rot}$ and $K_\text{angvel}$ held
constant across bodies, dividing by $I$ leaves it behind in both coefficients:

$$\ddot{\theta} + \frac{K_\text{angvel}}{I}\dot{\theta} + \frac{K_\text{rot}}{I}\theta = 0
\qquad\Longrightarrow\qquad
\omega_n = \sqrt{\frac{K_\text{rot}}{I}}, \quad
\zeta = \frac{K_\text{angvel}}{2\sqrt{K_\text{rot}I}}$$

Now both depend on the airframe, and a 7x spread in roll inertia across the
swept family becomes a real spread in response. With the library defaults
$K_\text{rot}=0.3$, $K_\text{angvel}=0.05$:

| half-angle | $I_{xx}$ | auto-scaled $\omega_n$ / $\zeta$ | fixed-gain $\omega_n$ / $\zeta$ |
|---|---|---|---|
| 20.44° (narrow) | 4.459e-04 | 12.0 / 1.00 | 25.9 / 2.16 (overdamped) |
| 45.00° (X) | 1.768e-03 | 12.0 / 1.00 | 13.0 / 1.09 |
| 69.56° (wide) | 3.091e-03 | 12.0 / 1.00 | 9.9 / 0.82 (underdamped) |

Note the fixed-gain column is not simply "the narrow frame is better": it gets a
higher bandwidth *and* becomes overdamped, so its response is stiffer but
sluggish. That competition is what produces the argmax flip at 120° in §2.

### 7.2b From three scalars to the 6-DOF law

§7.2 treats one axis at a time, with scalar $K_\text{rot}$ and $K_\text{angvel}$.
The implementation is the full attitude law on $SO(3)$, where the errors are
3-vectors and the gains are vectors applied elementwise. The two views connect
as follows.

**What the controller actually computes.** From the current rotation $R$ and the
desired $R_d$ (`compute_body_torque` in
`src/ariel/simulation/drone/controllers/lee_control/base_lee_controller.py`):

$$e_R = \tfrac{1}{2}\left(R^\top R_d - (R^\top R_d)^\top\right)^{\vee},
\qquad
e_\Omega = \Omega - R^\top R_d\,\Omega_d,
\qquad
M = -K_R \odot e_R - K_\Omega \odot e_\Omega + \Omega\times I\Omega$$

where $(\cdot)^\vee$ is the inverse of the skew map and $\odot$ is elementwise
multiplication. $e_R$ is an attitude error expressed as a 3-vector in the body
frame, so it needs no small-angle assumption to be defined -- that is the point
of the geometric formulation.

**Why elementwise gains are diagonal matrices.** Writing
$K_R = \operatorname{diag}(k_{R,1},k_{R,2},k_{R,3})$, the elementwise product is
the matrix product $K_R e_R$. Nothing in the code ever forms an off-diagonal
gain, so the control is three independent single-axis laws sharing one error
vector.

**Why the scalar analysis is recovered.** Rigid-body rotation in the body frame
is

$$I\dot{\Omega} + \Omega\times I\Omega = M$$

The $+\Omega\times I\Omega$ term in the control law is there to cancel the same
term in the plant, leaving $I\dot{\Omega} = -K_R e_R - K_\Omega e_\Omega$. For
small attitude errors, $R^\top R_d \approx \mathbb{1} + \widehat{\delta\theta}$
gives $e_R \approx -\delta\theta$ and $e_\Omega \approx -\dot{\delta\theta}$, so

$$I\,\ddot{\delta\theta} + K_\Omega\,\dot{\delta\theta} + K_R\,\delta\theta = 0$$

If $I$ is diagonal, this is **three uncoupled copies of §7.2**, one per axis:

$$I_{ii}\,\ddot{\delta\theta}_i + K_{\Omega,i}\,\dot{\delta\theta}_i + K_{R,i}\,\delta\theta_i = 0
\qquad\Longrightarrow\qquad
\omega_{n,i}=\sqrt{K_{R,i}/I_{ii}},\quad
\zeta_i=\frac{K_{\Omega,i}}{2\sqrt{K_{R,i}I_{ii}}}$$

`auto_scale_gains` sets $K_R = \operatorname{diag}(I)\,\omega_n^2$ and
$K_\Omega = 2\zeta\operatorname{diag}(I)\,\omega_n$ elementwise, so the
cancellation of §7.2 happens **independently on each axis** and all three land on
the same $\omega_n$ and $\zeta$. That is why a single pair of numbers describes
the attitude response of an airframe whose three axes have quite different
inertias.

Note what is **not** equal across the three axes: the gains themselves. They
scale *with* each axis's inertia, and differ exactly as the inertias differ --
for the X quad $K_R = (1.019, 1.025, 2.023)$, the yaw gain nearly twice roll and
pitch because a planar rotor layout has $I_{zz}\approx I_{xx}+I_{yy}$. Equal
gains on unequal axes would give three *different* bandwidths, which is the
situation `auto_scale_gains` exists to avoid. What is held equal is the
closed-loop response $(\omega_n,\zeta)$, not the numbers producing it.

**Three caveats this hides**, in increasing order of importance here:

1. *Large errors.* $\lVert e_R\rVert$ saturates (it is bounded by 1), so the
   effective proportional gain falls away from hover. The second-order reading is
   local; the $SO(3)$ law remains valid.
2. *Off-diagonal inertia.* `auto_scale_gains` takes `np.diag(quad.params["IB"])`,
   so the decoupling above assumes the body axes are principal axes. Exact for
   the family swept here, not for an asymmetric one -- see §7.2c, which is also
   where the fix lives.
3. *The gyroscopic term is cancelled but never simulated.* The control law adds
   $+\Omega\times I\Omega$ to cancel a term the reduced plant does not have:
   `dynamics_params` integrates $\dot{\Omega} = I^{-1}M$ with no
   $\Omega\times I\Omega$ (`d_p = Mx`, `d_q = My`, `d_r = Mz`). So on this plant
   the compensation is not a cancellation but an **injected torque**. It is small
   for the bodies swept here -- at a hard reversal rate of
   $\Omega=(6,3,1)$ rad/s it is 0.012-0.051 N·m against 1.7-3.9 N·m of available
   roll torque, i.e. under 3% -- and it vanishes exactly at the X quad where
   $I_{xx}=I_{yy}$. It would not be negligible for a strongly asymmetric or
   fast-spinning body, and it is a second instance of the controller and plant
   disagreeing about the model (§3.2 is the first).

### 7.2c Principal axes, and the matrix-gain fix (`--matrix-gains`)

The requirement behind §7.2b's per-axis decoupling is not symmetry as such but
that the **body axes are principal axes**, i.e. that the products of inertia
vanish. Coplanar rotors with a z-symmetric core give $I_{xz}=I_{yz}=0$ for free;
$I_{xy}=0$ needs a mirror plane containing z. The family swept in this document
has both mirrors, so the decoupling is exact -- measured
`max|off-diag| = 0.00e+00`, not merely small. Break either mirror and the axes
couple through $I$, while the default gains -- `np.diag(quad.params["IB"])` --
assume they do not.

Achieved bandwidth against a 24 rad/s target, for the symmetric X quad and for
three single-arm perturbations of it. Read as the eigenvalues of $I^{-1}K_R$,
the modal frequencies of $I\dot{\Omega} = -K_R e_R$. Reading
$\sqrt{K_{R,ii}/I_{ii}}$ per axis instead is **circular**: with
$K_R=\operatorname{diag}(I)\omega_n^2$ it returns $\omega_n$ by construction,
whatever the coupling.

| body | max off-diagonal | diagonal gains | matrix gains |
|---|---|---|---|
| symmetric X quad (swept family) | 0.00e+00 | 24.00 | 24.00 |
| arm 0 azimuth +30 deg | 2.10e-04 | 22.66 - 25.60 | **24.00** |
| arm 0 lengthened to 0.26 m | 3.10e-04 | 22.40 - 26.00 | **24.00** |
| arm 0 elevated 15 deg (out of plane) | 1.51e-04 | 22.95 - 25.00 | **24.00** |

Each perturbation moves **one** arm along one degree of freedom, so every row is
reproducible. Symmetric perturbations are useless as illustrations: alternating
$\pm15^\circ$ of arm elevation leaves the products of inertia at exactly zero,
which is why the swept family is principal to begin with.

Regenerate with `uv run --no-sync python docs/tools/principal_axes_table.py`,
which rewrites `docs/data/principal_axes.csv` and
`docs/data/principal_axes_sensitivity.csv`.
`tests/unit/test_docs/test_principal_axes_table.py` pins the committed CSV
against the generator and asserts the markdown above is the table it emits, so a
change to the inertia or decoder path fails the suite instead of silently
ageing the document.

> *Superseded 2026-08-17.* An earlier version of this table reported
> `9.70e-05 / 23.37-24.69`, `3.68e-05 / 23.76-24.25` and
> `2.89e-04 / 22.11-26.03` for loosely-described bodies ("unequal arm lengths",
> "arms tilted out of plane"). Those came from an ad-hoc script that was not
> kept and the bodies were never specified, so the rows could not be reproduced;
> the tilted-arms row was additionally wrong, since a symmetric tilt pattern has
> zero products of inertia and therefore shows no spread at all. Regenerated by
> `docs/tools/principal_axes_table.py`. The conclusion is unchanged -- diagonal
> gains spread the bandwidth on a non-principal body, matrix gains do not -- and
> the spread is in fact **wider** than first reported.

The parameterisation itself is not the limitation. Using the **full tensor**,
$K_R=\omega_n^2 I$ and $K_\Omega=2\zeta\omega_n I$, gives
$I^{-1}K_R=\omega_n^2\mathbb{1}$ **exactly for any symmetric positive-definite
$I$** -- no diagonality assumed. `--matrix-gains` enables it; the gains become
3x3 and `compute_body_torque` applies them as matrix-vector products.

Two measured consequences. On the symmetric family it is a **no-op**: the X quad
scores 8.438 m/s either way, bit-identical, so every result in this document
stands. On an asymmetric body (arm 0 moved 30 deg) it is *not* simply better --
max speed 6.562 -> 6.469 m/s -- but tracking improves (0.530 -> 0.511 m) and
clipping falls (13.9% -> 11.0%). The diagonal form was handing one axis about 3%
more bandwidth than requested, which bought a little speed at the cost of control
effort. **For evolving morphologies this is the point**: an EA over asymmetric
bodies would otherwise reward whichever airframe accidentally receives the most
bandwidth, confounding geometry with tuning exactly as `auto_scale_gains` at
fixed bandwidth confounds it the other way.


#### Finding the principal axes -- and why the controller does not

The principal axes are the **eigenvectors of $I$**. Because $I$ is real and
symmetric, the spectral theorem guarantees they always exist, are real, and are
mutually orthogonal, so *every* rigid body has them; the only question is whether
they coincide with the frame you chose:

$$I = V\Lambda V^\top,\qquad V^\top I V = \Lambda =
\operatorname{diag}(I_1,I_2,I_3)$$

with the columns of $V$ the principal axes written in body coordinates.
`np.linalg.eigvalsh` / `eigh` (the symmetric solver) is the practical route. For
the in-plane pair there is also the closed form -- Mohr's circle --

$$\tan 2\theta = \frac{2I_{xy}}{I_{xx} - I_{yy}},$$

and one shortcut worth knowing, which is what §7.2c actually uses: **a mirror
plane forces the products of inertia involving its normal to vanish**. That is
how the swept family is shown to be principal without computing anything.

| body | principal moments $I_1,I_2,I_3$ (kg m^2) | misalignment |
|---|---|---|
| symmetric X quad (swept family) | 0.00177, 0.00178, 0.00351 | 0.00 deg |
| arm 0 azimuth +30 deg | 0.00134, 0.00220, 0.00351 | 14.63 deg |
| arm 0 lengthened to 0.26 m | 0.00179, 0.00241, 0.00416 | 44.49 deg |
| arm 0 elevated 15 deg (out of plane) | 0.00177, 0.00180, 0.00348 | 39.54 deg |

Misalignment is $\theta$ folded into $[0,45^\circ]$, since relabelling which
principal axis is called $x$ is not a physical rotation.

**The controller computes none of this.** $K_R=\omega_n^2 I$ is
*basis-independent*:

$$I^{-1}K_R = I^{-1}(\omega_n^2 I) = \omega_n^2\mathbb{1}$$

for any symmetric positive-definite $I$. Rotating into the principal frame,
scaling each axis by its own moment and rotating back gives
$V(\omega_n^2\Lambda)V^\top = \omega_n^2 I$ -- the same matrix. The
eigendecomposition is real, and it cancels out of the answer (measured
$\max|I^{-1}K_R - \omega_n^2\mathbb{1}| = 1.1e-13$ across all four bodies).

Three reasons to prefer that over adopting the principal frame:

1. The rotors, the allocation matrix and the state estimate all live in the body
   frame. Using the principal frame means rotating $e_R$, $e_\Omega$, the output
   torque **and** $B$ into it and back out.
2. **For a morphing airframe $V$ is time-varying.** Moving one arm
   $30^\circ$ rotates the principal frame $14.63^\circ$. A control frame that
   turns with the morphology introduces a $\dot V$ term that the present
   formulation simply does not have.
3. **The principal *direction* is ill-conditioned near degeneracy, and this
   airframe sits there** -- see below.

$I_{xx}\approx I_{yy}$ to within $0.6\%$ ($I_{xx}-I_{yy} = -1.10\mathrm{e}{-}05$),
so Mohr's denominator is tiny and almost any coupling dominates it. Perturbing
$I_{xy}$ alone, leaving everything else fixed:

| $I_{xy}$ | misalignment of principal axes | diagonal-gain bandwidth |
|---|---|---|
| 0.0e+00 | 0.0 deg | 24.00 |
| 1.0e-06 | 5.1 deg | 23.99 - 24.01 |
| 1.0e-05 | 30.6 deg | 23.93 - 24.07 |
| 1.0e-04 | 43.4 deg | 23.35 - 24.71 |
| 3.1e-04 | 44.5 deg | 22.14 - 26.42 |

A coupling of $10^{-5}$ -- far too small to matter dynamically, costing $0.3\%$
of bandwidth -- already swings the principal axes $30^\circ$. The orientation of
the principal frame is a poor thing to build a controller on: it moves violently
under perturbations that barely move the principal moments or the closed-loop
response. Note also that lengthening an arm at $45^\circ$ leaves
$I_{xx}-I_{yy}$ *unchanged* -- it loads both axes equally -- while growing
$I_{xy}$, which is why that body reaches a near-maximal $44.49^\circ$.

$K_R=\omega_n^2 I$ needs none of these decisions, which is the argument for it
under evolving morphology.


### 7.3 The position loop: $m$ cancels entirely

The outer loop commands an *acceleration*,

$$a_\text{des} = a_\text{ff} + K_\text{pos}e_p + K_\text{vel}\dot{e}_p$$

and mass enters only afterwards, when force is computed as $F = m(a-g)$
(`src/ariel/simulation/drone/controllers/lee_control/acceleration_control.py`). The position error therefore obeys

$$\ddot{e}_p + K_\text{vel}\dot{e}_p + K_\text{pos}e_p = 0
\qquad\Longrightarrow\qquad
\omega_n = \sqrt{K_\text{pos}}, \quad \zeta = \frac{K_\text{vel}}{2\sqrt{K_\text{pos}}}$$

No inertia or mass appears, which is why these two numbers transfer across
morphologies unchanged and the raw gains do not. The inherited pair
$(K_\text{pos},K_\text{vel}) = (14.3, 9.0)$ is $\omega_n$ = 3.78 rad/s at
$\zeta$ = 1.19.

$a_\text{ff}$ is the trajectory's own acceleration. Without it the feedback
terms must *generate* the cornering acceleration from an error that has already
happened, which is a lag; with it they only trim the residual.

### 7.4 Why the two loops must be separated

The position loop's actuator *is* the attitude loop: to accelerate sideways the
vehicle must first tilt. Writing $a_\text{des}$ as though the tilt appears
instantly is only valid when the attitude loop settles much faster than the
position loop moves -- i.e. $\omega_n^\text{att} \gg \omega_n^\text{pos}$. When
they are comparable, the outer loop reacts to an error the inner loop has not
yet had time to correct, and commands more of the same: the loops fight.

Measured on the X quad (§5.2): best at a **6-12x separation**, unflyable at any
speed when the ratio drops to 1.3x, and too soft to correct at 15x. This is the
single largest performance factor found in this work -- retuning alone doubled
achievable speed twice over.

### 7.5 From rotor thrusts to accelerations: the allocation matrix

Each rotor $i$ has a position $r_i$ and a unit thrust axis $n_i$ in the body
frame, and produces thrust $f_i \ge 0$ along $n_i$ plus a reaction torque about
its own axis. Stacking:

$$\begin{bmatrix}F\\M\end{bmatrix} = B f, \qquad
B_{F,i} = n_i, \qquad
B_{M,i} = r_i \times n_i - \text{spin}_i\,\frac{k_m}{k_f}\,n_i$$

with $\text{spin}_i = \pm1$ for ccw/cw. $B$ is `allocation_matrix()` in
`src/ariel/simulation/drone/plant.py`; the controller inverts it to turn a
desired wrench into thrusts, and it is the object the plant's own dynamics
should agree with (§3.2, where they do not).

**Angular-acceleration allocation.** Dividing the moment block by inertia gives
what the airframe can actually *do*:

$$A = I^{-1}B_M \qquad [\text{rad·s}^{-2}\text{ per newton}]$$

* **`alpha_roll`** $= \lVert A_{[0]}\rVert$, the norm of the roll row. Since
  $\max_{\lVert f\rVert=1} \lvert A_{[0]}\cdot f\rvert = \lVert A_{[0]}\rVert$,
  it is the largest roll angular acceleration obtainable per unit of thrust
  effort. `alpha_pitch` and `alpha_yaw` are rows 1 and 2.
* Torque alone gives the **opposite** ordering: widening a quad raises roll
  torque (1.735 -> 3.896 N·m across the family) because lever arms grow, but
  inertia grows faster, so roll *acceleration* falls (313.3 -> 121.3). A slalom
  at 8.5 m/s allows roughly a quarter-second per corner to reverse bank, so the
  acceleration is what matters.
* **`maneuverability`** $=\lambda_\text{min}(B_M B_M^\top)$, airevolve's
  descriptor: the weakest axis of the moment allocation. For **coplanar** rotors
  that axis is always yaw, whose authority comes from $k_m/k_f$ and is
  unchanged by in-plane geometry -- measured constant at 5.1e-04 across the
  entire azimuth sweep *and* a 2x change in arm length. Pass `inertia=` for the
  $I^{-1}B_M$ form, which does vary. It becomes informative for tilted rotors.

### 7.6 Course and airframe quantities

For a slalom of leg $s$ and turn angle $\theta$, consecutive legs run at
$\pm\theta/2$ to the course axis, giving

$$d = s\cos(\theta/2), \qquad A = \tfrac{s}{2}\sin(\theta/2), \qquad
R = \frac{s}{2\sin(\theta/2)}, \qquad a_\text{lat} = \frac{v^2}{R}$$

for gate spacing, lateral amplitude, corner radius and the lateral acceleration
the course demands at speed $v$. Holding $s$ fixed (rather than $d$) is what
makes $R$ fall monotonically with $\theta$.

On the airframe side, with $n$ rotors of maximum thrust $T_\text{max}$,

$$\text{TWR} = \frac{n\,T_\text{max}}{mg}, \qquad
a_\text{lat}^\text{max} = g\sqrt{\text{TWR}^2-1}$$

The second follows from tilting by $\phi$ while holding altitude: the vertical
component supports weight, the horizontal turns, so $a_\text{lat}=g\tan\phi$ and
the ceiling is set by $\cos\phi = 1/\text{TWR}$. For the SPEAR-matched quad
(0.829 kg, TWR 5.24) that is 51 m/s^2 -- but see §2: the binding limit in
practice is the *lower* motor bound, not total thrust.

### 7.7 What a rollout measures

| quantity | definition | why it is reported |
|---|---|---|
| `gates` | sequential count; stops at the first miss | racing semantics |
| `gates_flown` | gates passed within half a gate, any order | flight quality without the lockout |
| `tracking_err` | mean $\lVert p-p_\text{ref}\rVert$, **course phase only** | past `total_time` the reference clamps and the drone overruns; including it turned 0.37 m into 4.3 m |
| `saturation_lo` | fraction of steps with a motor pinned at the lower bound | the allocation asked a rotor to *pull*: it ran out of differential range. This is the geometry-sensitive channel |
| `saturation_hi` | fraction pinned at maximum | not enough total thrust. Measured 0.0% where the lower bound was 29% |
| `survival` | steps flown / steps allotted | crash penalty |
| `completed` | every gate passed | gates the speed and completion terms |

### 7.8 The two objectives

**Normalized** (`--objective normalized`): gates scored out of 100 plus a
quality budget worth *half a gate*, so no combination of quality can outrank one
more gate while the function stays continuous for an EA:

$$\text{fitness} = 100\frac{g}{N} + \frac{1}{2}\cdot\frac{100}{N}\cdot
\frac{q_\text{track}+q_\text{margin}+q_\text{survive}+q_\text{speed}}{4}$$

with $q_\text{track}=e^{-\text{err}/(\text{gate}/2)}$ and
$q_\text{margin}=1-\text{clipping}$.

**Max completing speed** (`--max-speed-sweep`): the fastest $v$ at which a body
still finishes, found by bisection because completion is monotone in speed
(checked, 0 violations in 100+ bodies). Continuous by construction, with no
operating point to calibrate -- it replaced fixed-speed scoring after nine
operating points scored every body identically (§2).

---

### 7.9 Provenance: which command produced which number

Every headline table above is reproducible from the CSV beside it. The raw runs
land in `__data__/`, which is gitignored, so the files behind the results in this
document are copied into `docs/data/` (52 KB total) and are tracked. All runs use
the SPEAR-matched quad (`PAYLOAD_MASS = 0.667`, 5" props, `N_ARMS = 4`) and the
uniform slalom unless a course set is named.

| result | data file | command |
|---|---|---|
| §2 corner sharpness, 90° | `docs/data/tuned_90deg.csv` | `--max-speed-sweep --sweep-turns 90 --points 7 --speed-tol 0.0625 --speed-lo 3 --speed-hi 10 --att-omega-n 24 --pos-omega-n 2.0 --pos-zeta 1.0 --feedforward --max-accel 40` |
| §2 corner sharpness, 60° | `docs/data/tuned_60deg.csv` | as above but `--sweep-turns 60 --speed-lo 8 --speed-hi 14 --speed-cap 25` |
| §2 corner sharpness, 120° | `docs/data/tuned_60_120deg.csv` | as above but `--sweep-turns 60,120 --speed-lo 3 --speed-hi 12` (the 60° column of this file is **censored at the 12 m/s cap**; use `tuned_60deg.csv`) |
| §2 pre-tuning sweep (superseded) | `docs/data/pretuning_60_90_120deg.csv` | `--max-speed-sweep --sweep-turns 60,90,120 --points 7 --speed-tol 0.0625 --feedforward --max-accel 40` |
| §2 `--fixed-gains` comparison | `docs/data/pretuning_fixed_gains.csv` | as above plus `--fixed-gains` |
| §5.2 position-gain scan | `docs/data/gain_scan_att12.csv`, `gain_scan_att12_low.csv` | `--gain-scan --scan-omega 2,3.78,5,7,9,12 --scan-zeta 0.7,1.19,1.6` and `--scan-omega 0.8,1.2,1.6,2.0,2.6,3.2 --scan-zeta 1.0,1.19,1.4` |
| §5.2 attitude-bandwidth scan | `docs/data/gain_scan_att{6,24,36}.csv` | `--gain-scan --att-omega-n {6,24,36} --scan-zeta 1.0` with `--scan-omega` bracketing a 6-12x ratio |
| §7.2c principal axes and matrix gains | `docs/data/principal_axes.csv` | `uv run --no-sync python docs/tools/principal_axes_table.py` (no rollouts; closed-form from the inertia tensor) |
| §2 fixed-speed cliff | `docs/data/calibration_grid.csv`, `calibration_fine.csv` | `--calibrate --cal-speeds 2,2.5,3,3.5,4 --cal-turns 60,90,120 --n-courses 3` and `--cal-speeds 3.6,3.7,3.8,3.9 --cal-turns 90` |

**Checking this document.** `docs/tools/check_doc_claims.py` verifies the parts
of the prose that are mechanically checkable against the repository -- cited
paths and line numbers, CLI flags, identifiers, and markdown that renders as
something other than it looks. It exists because prose drifts from code
silently: re-reading does not catch a renamed function or a citation to a file
that never existed, since the same reasoning that wrote the sentence finds it
plausible. Every check runs in the unit suite
(`tests/unit/test_docs/test_doc_claims.py`), so this is enforced rather than
merely available; run it directly with

```
uv run --no-sync python docs/tools/check_doc_claims.py docs/drone_morphology_gate_racing.md
```

The rendering lint applies to every document under `docs/`. The claim checks are
opted into per document, because the older Sphinx pages use bare filenames as
prose and would report findings that are not defects.

Two conventions worth knowing when reading the CSVs:

* **`bracket`** records how a max-speed number was obtained: `ok` means bisection
  converged inside the bracket, `above_cap` means the body hit `--speed-cap` and
  the value is a **lower bound**, `below_lo` means it could not fly the slowest
  bracket speed. A four-way tie at exactly the cap is censoring, not a finding --
  this happened once and is the reason the 60° column was re-run.
* **`monotone`** records whether the bisection's assumption held: every tested
  speed above the limit failed and every one below passed. 0 violations across
  every run reported here.

---

## 8. Open items

1. **Normal-aware plant (§3.2) -- the priority.** Guarded, not fixed. Blocks any experiment on
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
4. **The controller cancels a gyroscopic term the plant does not have** (§7.2b).
   The attitude law adds $+\Omega\times I\Omega$; `dynamics_params` integrates
   $\dot{\Omega}=I^{-1}M$ without it, so the term is injected rather than
   cancelled. Under 3% of available torque for the symmetric bodies swept here,
   and exactly zero at the X quad, but it grows with asymmetry and rate. Fixed
   for free by the normal-aware plant work (item 1), which restores the full
   Euler equation.
5. **Repair's contract changes for morphing bodies**: collision-free at `q = 0`
   guarantees nothing across a joint envelope.
6. **The task is a cliff** at fixed speed (§2) -- nine operating points from 2.0
   to 3.8 m/s scored every body identically, and a 5% speed increase collapsed
   them all. Resolved by the max-speed objective (`--max-speed-sweep`), which is
   continuous by construction. Keep this in mind before designing any new
   fixed-speed experiment on this task.
7. **Co-design.** `--fixed-gains` (§2) showed the controller was hiding most of
   the morphology effect, so body and controller should be optimised jointly.
   Nothing in the EA does this yet. See §5.1 for the two parameters to put in
   the genome and why they are not settable today.

   *Superseded question, kept for the record:* `auto_scale_gains` normalises
   attitude bandwidth per body, which is the leading explanation for the 3%
   effect size. `--fixed-gains` runs the same sweep with one controller for all
   bodies, where closed-loop bandwidth becomes `sqrt(K_rot/I)` and varies 2.6x
   across the family. If the spread jumps, morphology optimisation on this stack
   is really a co-design problem. *(Answered 2026-08-16: it does -- 2-5x.)*
8. **RESOLVED (2026-08-16) -- Lee tracking.** Not a loop redesign: once the
   upstream defects were fixed, what remained was that the position loop's
   bandwidth had never been set relative to the attitude loop it sits outside
   (§5.2, §7.4). Retuning took the X quad from 3.875 to 8.438 m/s. Use
   `--att-omega-n 24 --pos-omega-n 2.0 --pos-zeta 1.0`. *Original entry:*
   Cruise tracking is excellent once settled
   (0.07 m at 4 m/s), but feedforward is not tuned — enabling it removes the lag
   and introduces speed overshoot and altitude sag, so the loop wants redesigning
   as `a_des = a_ff + Kp·e_p + Kd·e_v` with gains derived for tracking rather
   than point-holding.

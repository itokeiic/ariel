# The plant ignores rotor thrust direction

**Status: FIXED 2026-09-10.** Both plants now integrate each rotor along its own
thrust axis. `_guard_axial_thrust` has been deleted, per §7 item 4 — there is
nothing left to warn about. The work order below is kept as the record of what
was wrong and why; §9 records what actually landed, including two further
defects found during the fix that this document did not anticipate.

Code shown in §2–§5 is verbatim from commit `84410c7`, i.e. **before** the fix.
Read it as history, not as current state.

---

## 1. What is wrong, in one sentence

The controller allocates a wrench assuming each rotor pushes along **its own
thrust axis**, and the plant integrates every rotor as if it pushed along **one
fixed body axis**. For any tilted rotor the two describe different vehicles.

## 2. Where the normal is dropped

`src/ariel/simulation/drone/dynamics_params.py`, in `derive_reference_params()` — the assumed axis is declared at line 50:

```python
_ASSUMED_THRUST_AXIS = np.array([0.0, 0.0, -1.0])
```

and each rotor is reduced to a position and a spin sign at lines 197–206:

```python
x_i = float(prop["loc"][0])
y_i = float(prop["loc"][1])
spin = _spin_sign(prop["dir"][3])

# M_x = sum_i (-y_i · F_z_i) / Ixx = sum_i (-y_i · k_f · W_i²) / Ixx
k_p_signed.append(-y_i * k_f / Ixx)
# M_y = sum_i (+x_i · F_z_i) / Iyy = sum_i (+x_i · k_f · W_i²) / Iyy
k_q_signed.append(+x_i * k_f / Iyy)
# M_z (steady, linearized at hover): spin_i · 2 · k_m · W_hover / Izz
k_r_signed.append(spin * 2.0 * k_m * W_hover / Izz)
```

The tell is on the `spin` line: it reads **`prop["dir"][3]`**, the `"ccw"`/`"cw"`
character, but never **`prop["dir"][0:3]`**, the thrust direction vector. Each
rotor becomes `(x_i, y_i)` plus a sign. Orientation is discarded here, and
nothing downstream can recover it.

`_guard_axial_thrust()` (line 59 of the same file) exists only to warn when a
rotor's real normal disagrees with `_ASSUMED_THRUST_AXIS`. It warns once per
process, or raises under `ARIEL_STRICT_THRUST_NORMALS=1`.

## 3. Where the single axis is applied

`src/ariel/simulation/drone/drone_simulator.py`, building the symbolic equations of motion at lines 222–254:

```python
T = -k_w * sum_W2
...
accel = Matrix([0, 0, self.g]) + R @ Matrix([Dx, Dy, T])
...
d_p = Mx
d_q = My
d_r = Mz
```

`Matrix([Dx, Dy, T])` is the whole story. The only rotor-derived body force is
the scalar `T` on the z component; `Dx` and `Dy` are aerodynamic drag, not
thrust. A rotor tilted 30° contributes exactly as much z-force as an untilted
one and produces no in-plane force at all. The moments `Mx`, `My`, `Mz` come
from the position-only scalars above, so they are wrong for a tilted rotor in
the same way.

## 4. Who this actually bites

| bodies sampled | n | median tilt | max tilt | tilted > 1° | thrust not simulated (median) |
|---|---|---|---|---|---|
| azimuth sweep (this document) | 7 | 0.0° | 0.0° | 0% | 0% |
| EA, example 17 limits (±15° elevation and cant) | 300 | 22.3° | 41.5° | 100% | 38% |
| EA, handler defaults (±90° elevation, ±180° cant) | 300 | 80.2° | 90.0° | 100% | 99% |

The azimuth sweep in [drone_morphology_gate_racing.md](drone_morphology_gate_racing.md)
is **exactly** unaffected -- sweeping arm azimuth moves rotors *within* the
plane, and that family has no motor cant, so its deviation is 0.00°, not merely
small. Nothing published from it changes.

The evolutionary search space is a different matter: under the EA's own genome
limits every sampled body tilts, and at the median more than a third of each
rotor's thrust is allocated by the controller and never applied by the plant.
That is the difference between a limitation to fix eventually and a bound on
every EA result the pipeline produces.

> *History note, 2026-09-11:* the EA rows above were sampled through a decoder
> that leaked arm pitch into thrust direction (docs/decoder_thrust_composition.md).
> Regenerated after that fix, the example-17 row reads a median tilt of 12.9°
> (max 15.0°) with 22% of each rotor's thrust in-plane, and the handler-defaults
> row 78.7° and 98%; the current table is in `docs/drone_morphology_gate_racing.md`
> §4.1. The conclusion stands: every sampled EA body still tilts, so every EA
> result was bounded by this plant defect.

## 5. Why the fix is contained

The correct object already exists one layer up, and is already stored on the
simulator.

`src/ariel/simulation/drone/drone_configuration.py` builds it properly (line 250):

```python
self.Bf[:, idx] = k_f * w_max**2 * prop_dir

# Moment allocation (cross product + propeller torque)
moment_from_thrust = np.cross(prop_r, k_f * w_max**2 * prop_dir)
moment_from_torque = k_m * w_max**2 * prop_rot * prop_dir
self.Bm[:, idx] = moment_from_thrust + moment_from_torque
```

and `src/ariel/simulation/drone/drone_simulator.py` already holds it (line 98):

```python
self.Bf, self.Bm = self.config.get_allocation_matrices()
```

`self.Bf` and `self.Bm` are on the object and unused by the dynamics. The change
is to build the body force and moment from them —

```
F_body = Bf @ U
M_body = Bm @ U           ->   Omega_dot = I^-1 @ M_body
```

— instead of from `k_w`, `k_p_signed`, `k_q_signed`, `k_r_signed`.

**There is a working reference.** airevolve's plant (TU Delft lineage, the same
ancestor as this one) does exactly that and does not have this bug. Read it for
the form; do not modify that repository.

## 6. What must survive the rewrite

These are the reasons this is not a copy-paste from airevolve:

| must keep | where it lives now | note |
|---|---|---|
| aerodynamic drag | `Dx`, `Dy` in the `accel` line | absent from airevolve's plant |
| first-order motor lag | `tau` in the params dict | absent from airevolve's plant |
| sqrt-polynomial motor command | params dict | absent from airevolve's plant |
| full inertia tensor | currently baked as `/Ixx`, `/Iyy`, `/Izz` | moving to `Bm` means applying `I^-1` explicitly, which also removes the plant's diagonal-inertia assumption -- the same limitation `--matrix-gains` removed on the controller side |

## 7. Verification plan

No test currently pins the reduced-plant form, so nothing needs re-basing. Write
these with the fix:

1. **Axial parity.** For a coplanar quad with zero cant, the new dynamics must
   reproduce the old to numerical tolerance. If this fails, the rewrite changed
   something it should not have.
2. **Hover equilibrium.** Thrust summing to `m·g` holds altitude, for a tilted
   body too -- where the current plant drifts.
3. **Tilted-rotor case that currently fails.** The `elevation` body in
   `reference_morphologies` tilts its thrust axis 11.37° and cannot complete a
   course at any speed tested. After the fix it should fly.
4. **Guard goes quiet.** `_guard_axial_thrust` should be deleted, not silenced:
   once the plant honours normals there is nothing to warn about.

## 8. Why this was not done sooner

It was deprioritised on the reasoning that the work would move to Isaac
Lab/PhysX, where the plant would be replaced anyway. That condition never held:
the experiments run on this reduced plant, not on PhysX (see
[drone_morphology_gate_racing.md](drone_morphology_gate_racing.md) §4.1). The
premise was wrong, not the reasoning.

Note also that moving to PhysX does **not** automatically fix this. It depends
where forces are applied: a lumped body wrench computed from cached rotor
geometry reproduces the same class of defect in a new place. Rotor pose, thrust
axis, mass and inertia are all functions of joint configuration and must be
re-read each step -- which is what `DronePlant.rotor_frames()` and
`mass_inertia()` are specified to do.

---

## 9. What landed (2026-09-10)

### The fix

`derive_reference_params` now also returns 3-D geometry — `rotor_dirs` (unit
thrust axes from `dir[0:3]`), `rotor_arms` (moment arms **about the CG**),
`rotor_spins` — plus the raw `k_f`, `k_m`, `W_hover`, `mass` and `inertia`. Both
plants build their wrench from those:

```
F_body = sum_i k_f W_i^2 n_i
M_body = sum_i [ r_i x (k_f W_i^2 n_i) + spin_i (2 k_m W_hover W_i) n_i
                 + spin_i (k_r_react Izz) dW_i n_i ]
Omega_dot = I^-1 M_body
```

Inverting the full tensor also drops the plant's diagonal-inertia assumption
(§6).

### Two defects this document did not anticipate

1. **A second implementation.** `src/ariel/simulation/tasks/torch_drone_gate_env.py`
   reimplements the same dynamics on tensors for GPU rollouts and read the same
   axial-only coefficients. §5 named only
   `src/ariel/simulation/drone/drone_simulator.py`; fixing one and
   not the other would have left two plants agreeing on coplanar bodies and
   diverging silently on tilted ones. Both are fixed.

2. **Moment arms were measured from the body origin, not the CG** — and
   `loc[2]` was dropped. This is independent of thrust direction: for a body
   whose CG is offset in-plane it is a live moment error *at zero cant*. On a
   test body with one arm moved 5 cm the in-plane arm error was 7.8 mm. Since
   the EA perturbs arm positions, this touched every asymmetric body it has
   ever evaluated.

While fixing the torch path, a third, unrelated defect surfaced: it
unnormalised the motor state with `params["w_min"]` (238.49 rad/s, the motor
model's physical minimum) where `DroneSimulator` uses `W_MIN_N` (0.0, the
normalisation floor). The two plants therefore **never** agreed, coplanar or
not, despite `TorchDroneGateEnv` advertising itself as a drop-in replacement.
Any RL result trained through the torch env predates this fix and was trained
against a different plant than the one its documentation claims.

### What was deliberately NOT changed

The yaw drag torque is still the reference's **hover linearisation**
(`dMz/dW ~ 2 k_m W_hover`), not the quadratic `k_m W^2` implied by `Bm`.
Changing it is a fidelity decision independent of thrust direction, it would
have broken the axial-parity gate, and every controller tuning calibrated
against the linearised yaw response would need revisiting. It remains open.

Also unchanged: there is still no gyroscopic term (`Omega x I Omega`) in the
rotational dynamics. Pre-existing, and likewise out of scope here.

### Verification (§7 of the plan)

| item | result |
|---|---|
| 1. axial parity | **exact** — 0.000e+00 deviation over 16 randomised states |
| 2. tilt actually does something | thrust magnitude conserved, direction rotates; the old model was byte-identical at 0° and 30° cant |
| 3. sympy vs torch | agree to ~1e-13 at 0/15/30/45° cant |
| 4. guard deleted | done, along with `_ASSUMED_THRUST_AXIS`, `_AXIAL_TOL_RAD`, `_NONAXIAL_WARNED` |

Tests: `tests/unit/test_simulation/test_plant_thrust_direction.py`. The parity
reference is recomputed from `params` inside the test rather than loaded from a
frozen file, so it pins the intent and stays readable.

### Item 3: SUPERSEDED 2026-09-10 — the prediction was right, the test body was not

> **Correction, same day.** The section below concluded that item 3's prediction
> failed and blamed the controller's 4-DOF mixer. That diagnosis was measured on
> a body carrying a **decoder** defect: `reference_morphologies["elevation"]`
> sets `motor_pitch = 0` on all four rotors, so its genome asks for axial
> thrust, but the decoder leaks arm pitch into thrust direction and gives arm 0
> an 11.37° tilt it never asked for. See
> [decoder_thrust_composition.md](decoder_thrust_composition.md).
>
> Re-flown with the composition corrected, against this same fixed plant, at
> `--speed-tol 0.0625` (probe: `docs/tools/elevation_corrected_probe.py`):
>
> | θ | as decoded | corrected (axial) | symmetric baseline |
> |---|---|---|---|
> | 60° | nan | **9.852 m/s** | 10.164 |
> | 90° | nan | **7.586 m/s** | 8.250 |
> | 120° | nan | **6.336 m/s** | 6.414 |
>
> **So item 3 passes and the plant fix does what it predicted.** It needed a body
> that actually tests it — correctly decoded, `elevation` is a purely positional
> perturbation (rotor 5.2 cm out of plane, all thrust axial), which exercises the
> out-of-plane moment arm and CG-relative arms this fix also corrected, not
> thrust direction.
>
> What survives from the analysis below: the mixer truncation is real. `mixerFM`
> keeps only `Bf`'s z row, so a **deliberately** canted rotor (non-zero
> `motor_pitch`) would still produce in-plane force the controller cannot
> command. That gap is open. It simply was not what this body was demonstrating,
> and the `cond(mixerFM) = 453.9` and "19.7% in-plane" figures below describe a
> geometry that should not exist.
>
> **Correction 2026-09-11 — bisection tolerance.** Every speed in this section
> as first written, and in the original text below, came from runs that did not
> pass `--speed-tol`, whose default was then 0.125; every reported sweep uses
> 0.0625. The coarser setting stops one halving early, so it reads either the
> same or exactly one step (0.039 m/s) low. That — **not drift** — is why the
> symmetric body read 10.125 / 6.375 against the report's 10.164 / 6.414.
> Re-run at 0.0625 it reproduces the report and the pre-fix table exactly,
> which also verifies end to end, through the whole bisection, that the fix
> leaves a coplanar symmetric body untouched. The default is now 0.0625.
>
> Like-for-like, both at 0.0625 (change against the pre-fix run in brackets):
>
> | body | 60° | 90° | 120° | pre-fix (2026-08-17) |
> |---|---|---|---|---|
> | symmetric X quad | 10.164 | 8.250 | 6.414 | 10.164 / 8.250 / 6.414 |
> | arm 0 azimuth +30° | 9.227 (+11.3%) | 7.039 (+11.8%) | 5.242 (+7.2%) | 8.289 / 6.297 / 4.891 |
> | arm 0 lengthened | 9.812 (-7.7%) | 7.938 (+9.1%) | 6.883 (+49.1%) | 10.633 / 7.273 / 4.617 |
> | arm 0 elevated 15° (as decoded) | nan | nan | nan | did not fly / did not fly / did not fly |
>
> One attribution in the original text also overreaches. It credits the azimuth
> and length changes to the CG-relative moment arms alone. Those two bodies are
> asymmetric, so they carry non-zero products of inertia, and the fix changed
> that too (the plant now inverts the full tensor instead of dividing by
> Ixx/Iyy/Izz). Both corrections touch only asymmetric bodies and this run does
> not separate them — **assumed**, not verified, that either dominates.
>
> **Resolved 2026-09-11 — 2×2 factorial** (`docs/tools/plant_fix_factorial.py`).
> Each body flown with the plant's moment arms and inertia reverted
> independently, controller untouched (its mixer uses CG-relative `Bm` and its
> gains read the configuration's inertia, so it is identical in every variant).
> The instrument is validated: `neither` reproduces the pre-fix table and `both`
> the post-fix run in all 12 cells. Max completing speed, m/s; one bisection
> step is 0.039:
>
> | body, θ | neither (pre-fix) | CG arms only | full inertia only | both (current) | CG effect | inertia effect | interaction |
> |---|---|---|---|---|---|---|---|
> | azimuth 60° | 8.289 | 9.227 | 8.289 | 9.227 | +0.938 | +0.000 | +0.000 |
> | azimuth 90° | 6.297 | 6.961 | 6.297 | 7.039 | +0.664 | +0.000 | +0.078 |
> | azimuth 120° | 4.891 | 5.281 | 4.891 | 5.242 | +0.391 | +0.000 | -0.039 |
> | length 60° | 10.633 | 10.320 | 10.164 | 9.812 | -0.312 | -0.469 | -0.039 |
> | length 90° | 7.273 | 8.289 | 7.781 | 7.938 | +1.016 | +0.508 | -0.859 |
> | length 120° | 4.617 | 6.844 | 4.656 | 6.883 | +2.227 | +0.039 | +0.000 |
>
> **CG-relative arms** carry the change for the azimuth body at every θ and for
> the length body at 90° and 120°. Mechanism, verified by linear algebra rather
> than inferred: shifting every arm by the same vector `cg` leaves differential
> roll, pitch and yaw commands exactly intact (gain 1.000, zero cross-talk), so
> the pre-fix error lived entirely in the collective-thrust channel. Each newton
> of collective thrust produced an unmodelled moment `cg × F` — roll/pitch
> -1.44/-2.49 mN·m per N for azimuth, -1.40/+1.40 for length. The controller's
> mixer has always used CG-relative arms, so it never compensated. It is not an
> authority change: roll/pitch authority differs by at most 0.34% between the
> two arm conventions.
>
> The disturbance is **small** — 0.47–0.66% of the available roll moment at
> hover, 2.5–3.5% at full thrust — yet removing it is worth up to
> 2.2 m/s. **That disproportion is not explained.** Saturation at the margin does
> not account for it everywhere: the pre-fix length body at 120° failed at low
> speed with 0.6% motor saturation and gained 2.2 m/s once the disturbance was
> removed. Also **not explained:** the length body's negative CG effect at 60°.
>
> **Full inertia** is zero for the azimuth body at every θ and large for the
> length body at 60° and 90°. The diagonal model is nearly right for azimuth
> (principal axes rotated +14.63°, in-plane moment split 42.4% in the diagonal
> against 48.5% true) and qualitatively wrong for length: it reads almost
> isotropic (0.5% split) while the body splits 29.5% along axes rotated
> -45.51° — the same 14.63° and 44.49° misalignments as the §7.2c table in
> `docs/drone_morphology_gate_racing.md`. Attitude gains are built from that
> diagonal unless `--matrix-gains` is passed, so after the fix the length body's
> plant and controller disagreed about its inertia. Pre-registered test: if that
> mismatch is the cause, flying the current plant with `--matrix-gains` recovers
> ≥ 10.1 m/s at 60° and ≥ 8.1 at 90°. Result: 10.320 / 8.367 / 6.727 — both
> thresholds met, and 60° lands exactly on the consistent-diagonal value
> (10.320). **Not explained:** at 120° matrix gains cost 0.156 m/s against
> diagonal gains.
>
> *Corrected 2026-09-11, same day.* The geometry figures first written in this
> block — parasitic moments of −7.3/−12.7 and −7.1/+7.1 mN·m per N, 0.5–0.7% at
> hover and 13–18% at full thrust ("largest exactly when the slalom is hardest"),
> a 1.3% authority difference, principal axes at 13.7° and −45.5°, splits of
> 44%/49% and 1%/29% — were computed without the 0.667 kg payload the sweep
> flies with, which pulls the CG toward the body origin. Recomputed with it they
> are as above, and the in-plane CG offsets are 2.87 mm (azimuth) and 1.98 mm
> (length), not the 14.7 mm and 10.0 mm in the original text below. The flight
> results are unaffected — every rollout went through the sweep, payload
> included. The correction removes the mechanism's *magnitude* argument and leaves
> its *identity* (a pure collective-thrust moment, no authority change) intact.
>
> Implication: with the plant now right, asymmetric bodies should be flown with
> `--matrix-gains`, or their inertia is modelled correctly in the plant and
> wrongly in the controller. Since 2026-09-11 `--matrix-gains` is the default. The gate-racing family is unaffected — its products
> of inertia are identically zero, where full and diagonal gains coincide.
>
> Raw output: `__data__/morphology_design_sweep/20260911_112644/`. Committed
> `docs/data/perturbation*` still hold the pre-fix run and were not overwritten. *(Later on 2026-09-11: regenerated after the
> decoder and gains decisions; see `docs/drone_morphology_gate_racing.md` §7.2d.)*
>
> Original text kept below.

### Item 3 was run, and its prediction appeared to fail (superseded, see above)

§7 item 3 predicted the `elevation` body (thrust axis tilted 11.37°) "should
fly" after the fix. Re-run 2026-09-10 via `examples/spear/19_morphology_design_sweep.py` with the
tuned settings
(`--perturbation-sweep --sweep-turns 60,90,120 --completion strict
--speed-lo 2 --speed-hi 12 --speed-cap 25 --att-omega-n 24 --pos-omega-n 2.0
--pos-zeta 1.0 --feedforward --max-accel 40`):

| body | 60° | 90° | 120° | before |
|---|---|---|---|---|
| symmetric X quad | 10.125 | 8.250 | 6.375 | 10.164 / 8.250 / 6.414 |
| arm 0 azimuth +30° | **9.188** | **7.000** | **5.203** | 8.289 / 6.297 / 4.891 |
| arm 0 lengthened | 9.812 | **7.938** | **6.844** | 10.633 / 7.273 / 4.617 |
| arm 0 elevated 15° | **nan** | **nan** | **nan** | did not fly ×3 |

**It still does not fly** — 100% lower-bound motor clipping, 132 m tracking
error at 90°. Fixing the plant is necessary but **not sufficient** for a tilted
body, for a reason the work order did not anticipate:

* `mixerFM = vstack([-Bf[2:3, :], Bm])` keeps only the **z row** of the force
  allocation. The controller therefore commands collective body-z thrust plus
  three moments — a 4-DOF underactuated mixer. For this body **19.7% of the
  tilted rotor's thrust is in-plane**, a force the controller can neither
  command nor account for.
* Before the fix the plant produced no in-plane force either, so the truncated
  mixer was *consistent* with the plant — both wrong, and agreeing. The fix
  makes the plant right and leaves the controller as the mismatched half.
* Conditioning degrades too: `cond(mixerFM)` is **453.9** for this body against
  ~89 for the three coplanar ones.

So tilted-rotor flight needs a 6-DOF allocation built from the full `Bf`, not
just its z row — or an explicit feed-forward of the in-plane rotor force as a
modelled disturbance. That is a controller change, tracked separately from this
one.

The azimuth and length bodies improved substantially (up to +48% at 120°), which
is the CG-relative moment-arm correction: their in-plane arms moved 14.7 mm and
10.0 mm respectively. The symmetric body's in-plane arms move 0.00 mm, and its
dynamics are parity-exact (4.5e-13) against the old model, so the gate-racing
report's Table 1 is unaffected. Its 10.164 -> 10.125 and 6.414 -> 6.375 shifts
are one bisection step (tolerance 0.0625) and are **not** attributable to this
fix — parity rules that out; they are unexplained drift between this run and the
2026-08-17 table and should be resolved before the perturbation table is
regenerated.

Raw output: `__data__/morphology_design_sweep/20260910_142133/`. The committed
`docs/data/perturbation*.md|csv` were deliberately NOT overwritten, so the
pre-fix record survives.

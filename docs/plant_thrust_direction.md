# The plant ignores rotor thrust direction

**Status: open.** Guarded, not fixed. This is the highest-priority defect in the
drone simulation stack, and this file is the work order for fixing it.

Code shown is verbatim from commit `84410c7`. Line numbers will drift once the
fix lands; the anchors quoted alongside them will not.

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

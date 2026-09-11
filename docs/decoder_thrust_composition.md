# The decoder leaks arm pitch into thrust direction

**Status: open.** Found 2026-09-10 while re-running the `elevation` reference
morphology against the fixed plant. Not guarded. This file is the work order.

Code and measurements are verbatim from the working tree at that date.

---

## 1. What is wrong, in one sentence

The spherical-angular decoder builds a motor's local pose by **subtracting** its
parent arm's Euler angles, which cancels the parent only when the two rotations
commute — so a genome with non-zero `arm_pitch` decodes to a thrust direction it
never asked for.

## 2. What the encoding promises

`src/ariel/body_phenotypes/drone/decoders.py`, lines 39–42, defines the genome
columns:

```
- arm_az     : azimuth of arm in XY plane (rad)
- arm_pitch  : elevation above XY plane (rad), 0 = horizontal
- motor_az   : motor thrust-vector azimuth in world XY (rad)
- motor_pitch: motor thrust-vector pitch (rad), 0 = thrust along +Z
```

The arm columns say **where the rotor sits**; the motor columns say **which way
it thrusts**, in world/body frame. The two are meant to be independent, which is
what lets the encoding express canted-rotor and fully-actuated designs that arm
geometry alone cannot.

## 3. Where it breaks

`src/ariel/body_phenotypes/drone/decoders.py` line 67:

```python
motor_local_rpy = (0.0, motor_pitch - arm_pitch, motor_az - arm_az)
```

with the comment "encoded as rpy that — when composed with the arm's frame —
yields the genome's world-frame (motor_az, motor_pitch)."

Subtracting angles inverts a rotation only when the rotations commute. Written
out, the composition is

```
R_arm @ R_motor = Rz(arm_az) Ry(-arm_pitch) Rz(motor_az - arm_az) Ry(motor_pitch - arm_pitch)
```

which equals the intended `Rz(motor_az) Ry(motor_pitch)` only in special cases:

* **`arm_pitch = 0`** — every rotation in sight is about z, they commute, and the
  cancellation is exact. Verified over 2000 random `(arm_az, motor_az,
  motor_pitch)` samples: max error **1.5e-06 deg**.
* **`arm_pitch != 0`** — `Ry` does not commute with the `Rz` factors around it,
  and a residual survives. It depends on `arm_az` as well as `arm_pitch`, so it
  is not a clean function of the elevation angle.

## 4. Who this bites

Thrust-direction error between what the genome specifies and what decodes,
sampled uniformly over each range:

| arm_pitch range | n | median error | max error | bodies off by > 1° |
|---|---|---|---|---|
| `arm_pitch = 0` (azimuth/length only) | 2000 | 0.0° | 1.5e-06° | 0% |
| example 17 limits (±15°) | 4000 | **7.4°** | 29.8° | **89%** |
| genome-handler defaults (±90°) | 4000 | **34.8°** | **178.6°** | **96%** |

A 178.6° error is a rotor pointing very nearly opposite to the direction its
genes asked for.

The gate-racing sweep and the azimuth/length perturbations are **exactly**
unaffected: they hold `arm_pitch = 0`, where the cancellation is exact. Nothing
published from those changes. The exposure is to any EA run that samples arm
elevation — which the handler defaults do, over the full ±90°.

## 5. The concrete case that exposed it

`reference_morphologies()["elevation"]` sets `arm_pitch = 15°` on arm 0 and
leaves `motor_pitch = 0` on **all four** rotors, i.e. the genome asks for four
axial rotors and one displaced out of plane. It decodes with arm 0's thrust axis
tilted **11.37°**.

That spurious tilt is why the body was flagged `axial_thrust=False`, described in
`src/ariel/simulation/drone/reference_morphologies.py` as a perturbation that "lifts the rotor out of plane
AND tilts its thrust axis", and recorded in `docs/data/perturbation_table.md` as
"did not fly" with plant "not simulated".

Re-flown with the composition corrected — thrust taken straight from
`(motor_az, motor_pitch)` — against the fixed plant, tuned operating point,
strict completion, `--speed-tol 0.0625` (the tolerance of every reported sweep),
through `max_completing_speed` in `examples/spear/19_morphology_design_sweep.py`
(probe: `docs/tools/elevation_corrected_probe.py`):

| θ | as decoded (11.37° spurious tilt) | corrected (axial) | symmetric baseline |
|---|---|---|---|
| 60° | nan | **9.852 m/s** | 10.164 |
| 90° | nan | **7.586 m/s** | 8.250 |
| 120° | nan | **6.336 m/s** | 6.414 |

Tracking error 0.44–0.55 m, motor clipping 12–16% — the same regime as the three
bodies that always flew, not marginal survival. Correctly decoded, this body is a
**purely positional** perturbation costing 3.1% of top speed at 60°,
8.0% at 90° and 1.2% at 120°.

> **SUPERSEDED 2026-09-11.** First recorded (2026-09-10) at the sweep's then
> default `--speed-tol 0.125`: 9.812 / 7.547 / 6.297 against a same-tolerance
> baseline of 10.125 / 8.250 / 6.375, described as "costing 3–9%". Two
> corrections: every reported sweep uses 0.0625, so those were not comparable
> with the report — each corrected value rises by exactly one bisection step
> (0.039 m/s) at 0.0625, as the extra halving predicts; and "3–9%" misstated the
> range even then, since the 120° cost was 1.2%.

## 6. Why the fix is contained, and why it is still a decision

The fix is to stop subtracting angles and compose properly: build the motor's
world orientation from `(motor_az, motor_pitch)`, then derive its local pose as

```
R_motor_local = R_arm^-1 @ R_motor_world
```

That is a few lines in `spherical_angular_to_blueprint`, and `arm_pitch = 0`
genomes are provably untouched by it (§3).

**But it changes the phenotype of every genome with non-zero `arm_pitch`**, by a
median of 7–35° of thrust direction. That is a change to the genotype→phenotype
map, so:

* every EA result that sampled arm elevation was searching a different space
  than the encoding documents, and is not comparable to results after the fix;
* any cached or checkpointed population carries the old mapping.

So this is a search-space decision, not a drop-in bug fix, and it should be made
deliberately rather than folded into unrelated work.

There is a second, prior question worth settling at the same time: **is the
absolute convention the one we want?** The alternative is arm-relative — the
motor mounts rigidly to the arm, so elevating the arm 15° tilts the thrust 15°,
which is what a physical build gives for free. The code documents and intends
absolute; nothing here argues it is wrong, only that it is not what the code
achieves.

## 7. Verification plan

1. **`arm_pitch = 0` is untouched.** Randomised `(arm_az, motor_az,
   motor_pitch)`; decoded thrust must match the pre-fix decoder to machine
   precision. This is the regression gate for everything already published.
2. **Round-trip.** For random `(arm_az, arm_pitch, motor_az, motor_pitch)`, the
   decoded thrust direction must equal `Rz(motor_az) Ry(motor_pitch) @ +Z` to
   machine precision. This is the property line 67 was trying to provide.
3. **The elevation body decodes axial.** All four thrust axes parallel to body z,
   arm 0's rotor still 5.2 cm out of plane.
4. **It flies.** The §5 table, reproduced from the real pipeline rather than a
   monkeypatched one.

## 8. Related

* `docs/plant_thrust_direction.md` — the plant defect fixed 2026-09-10. Its §9
  originally attributed this body's failure to the controller's 4-DOF mixer;
  that diagnosis was measured on a body carrying this decoder bug and is
  corrected there.
* The mixer limitation is nonetheless real and still open:
  `mixerFM = vstack([-Bf[2:3, :], Bm])` keeps only the z row of the force
  allocation, so a **deliberately** canted rotor (non-zero `motor_pitch`) still
  produces in-plane force the controller can neither command nor account for.
  This decoder fix does not address that; it removes the bodies that hit it by
  accident.

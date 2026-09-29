# Gate-passing tasks in ARIEL: the slalom course and the gate task

This guide explains the three Python files that define a gate-passing task in
ARIEL, how to use them, and where to change them. It assumes Python and NumPy,
and no prior knowledge of ARIEL.

*Written 2026-09-29, against commit `5cdae8e` and the merge that followed it.
Every code block below was run, and its printed output is copied from that run.*

## 1. What the task is

A drone must fly through a sequence of gates, in order. A gate is a circular
opening in a vertical plane. The task defines:

1. **the course**: where the gates are and which way each one faces;
2. **the pass test**: what counts as flying *through* a gate;
3. **the episode rules**: the order of gates, what ends an attempt, and the score.

The task deliberately says nothing about *how* the drone is flown. The same task
can be flown by a classical controller tracking a reference path, which is how
the results in `Reports/morphology_gate_racing.tex` were produced, or by a
reinforcement-learning policy, which is where this work is heading.

## 2. The three files

| question | file | main entry points |
|---|---|---|
| Where are the gates? | `src/ariel/simulation/tasks/slalom_course.py` | `slalom_gates()`, which returns a `SlalomCourse` |
| Did the drone pass a gate? | `src/ariel/simulation/drone/gate_metrics.py` | `in_plane_offset()`, `passes()`, and `crossing_report()` for a whole recorded flight |
| What are the rules, and the score? | `src/ariel/simulation/tasks/gate_task.py` | `GateTask`, with `reset()`, `update()` and `upcoming()` |

How they connect:

```text
slalom_gates(turn_deg, ...)  ->  SlalomCourse  (gate_pos, gate_yaw, gate_size, starting_pos)
                                      v
GateTask(course, num_envs, ...)  -- uses -->  gate_metrics.in_plane_offset / passes
      reset(start positions)
      update(positions, time)  -> events: passed / missed / done   (once per simulation step)
      upcoming(n)              -> next n gates, for a policy's observations
      completed, course_time   -> result
```

`GateTask` accepts any object with `gate_pos`, `gate_yaw` and `gate_size`, so a
course does not have to come from `slalom_gates` (section 7).

## 3. Conventions

* **Frame: North-East-Down (NED).** x points north (along the slalom), y east,
  z **down**. Altitude is therefore negative z: `z = -1.5` is 1.5 m above the
  ground.
* **Units:** metres, seconds, radians in code. Some printed values below are
  converted to degrees.
* **Gate heading `gate_yaw`:** the gate's normal is `(cos yaw, sin yaw, 0)`.
  This is the direction the drone must fly through it. Gates are always
  vertical, because the normal has no z component.
* **Pass direction:** a crossing counts only when the drone moves from behind
  the plane (negative signed distance) to in front of it. Crossing backwards
  does nothing.
* **Vertical offset sign:** positive means *below* the gate centre, following
  NED.

## 4. Quick start

Run these from the repository root after installing the project (`uv sync`).

### 4.1 Build a course

```python
import numpy as np
from ariel.simulation.tasks.slalom_course import slalom_gates

course = slalom_gates(90.0)                  # 90-degree turn at every gate
print(course.gate_pos.shape, course.path_pos.shape)
print(f"spacing {course.spacing:.2f} m, amplitude {course.amplitude:.2f} m, "
      f"path {course.path_length:.1f} m")
print("gate 0:", course.gate_pos[0], "yaw", round(float(np.degrees(course.gate_yaw[0])), 1), "deg")
print("spawn:", course.starting_pos)
```

```text
(15, 3) (17, 3)
spacing 1.70 m, amplitude 0.85 m, path 38.4 m
gate 0: [ 0.   0.  -1.5] yaw 45.0 deg
spawn: [-0.70710678 -0.70710678 -1.5       ]
```

There are 15 scored gates. `path_pos` has 17 points because it adds two
lead-out waypoints after the last gate (section 5.3).

### 4.2 Score a recorded flight

`docs/data/flight_logs/slalom_90deg.npz` is a real flight on this course,
recorded at 200 Hz: the X-configuration quadrotor (arm half-angle 45°) at
8.25 m/s, its limiting speed on this course, flown by the Lee geometric
controller. Feeding it to the task one sample at a time is
exactly how a simulator would use it.

```python
from ariel.simulation.tasks.gate_task import GateTask, Status

log = np.load("docs/data/flight_logs/slalom_90deg.npz")   # a real flight, 200 Hz
task = GateTask(course)                                   # one drone
task.reset(log["pos"][0], t=float(log["t"][0]))
for pos, t in zip(log["pos"][1:], log["t"][1:]):
    events = task.update(pos, float(t))                   # once per simulation step
    if events.done[0]:
        break

print(Status(task.status[0]).name, task.gates_passed[0], "gates,",
      f"course time {task.course_time[0]:.3f} s")
print("worst in-plane offset:", round(float(np.nanmax(task.crossing_offset[0])), 3), "m")
```

```text
FINISHED 15 gates, course time 7.225 s
worst in-plane offset: 0.493 m
```

All 15 gates were passed, the worst with 7 mm to spare: the opening's radius is
0.5 m.

### 4.3 What a miss looks like

The same flight shifted 0.3 m sideways, scored under both miss rules:

```python
shifted = log["pos"] + np.array([0.0, 0.3, 0.0])          # 0.3 m east of the real path
for rule in ("terminate", "continue"):
    task = GateTask(course, on_miss=rule)
    task.reset(shifted[0], t=float(log["t"][0]))
    for pos, t in zip(shifted[1:], log["t"][1:]):
        task.update(pos, float(t))
    print(f"{rule:9s}: {Status(task.status[0]).name}, passed "
          f"{np.flatnonzero(task.passed[0]).tolist()}")
```

```text
terminate: MISSED, passed [0, 1, 2, 3]
continue : FINISHED, passed [0, 1, 2, 3, 5, 7, 9, 11, 13]
```

With `on_miss="terminate"` (the default), the attempt ends at the first missed
gate, gate 4. With `"continue"`, every gate is judged separately: the shift
helps on one side of the weave and hurts on the other, so from gate 4 on,
alternate gates fail.

## 5. The course: `slalom_gates`

### 5.1 Parameters

| parameter | default | meaning |
|---|---|---|
| `turn_deg` | (required) | angle between consecutive legs, degrees. **The difficulty setting.** |
| `leg` | 2.4 | distance between consecutive gates, m. Held fixed as `turn_deg` varies. |
| `n_gates` | 15 | scored gates |
| `gate_size` | 1.0 | diameter of each circular opening, m |
| `z` | -1.5 | course altitude (NED, so 1.5 m up) |
| `start_offset` | 1.0 | spawn distance behind gate 0, m |
| `lead_out` | 2 | extra waypoints after the last gate (section 5.3) |
| `seed` | None | None gives the uniform course; an integer gives a randomised layout |
| `jitter` | 0.25 | with a seed, each corner's angle is scaled by a random factor in 1 ± `jitter` |

### 5.2 Geometry

Legs alternate at +θ/2 and −θ/2 from the course axis (x), so for leg length s:

* gate spacing along x: d = s·cos(θ/2)
* sideways amplitude of the weave: A = (s/2)·sin(θ/2)

At the default 2.4 m leg:

| θ | spacing | amplitude |
|---|---|---|
| 30° | 2.32 m | 0.31 m |
| 60° | 2.08 m | 0.60 m |
| 90° | 1.70 m | 0.85 m |
| 120° | 1.20 m | 1.04 m |

**Why the leg is fixed, not the spacing.** With a fixed leg the course has the
same length at every turn angle (38.4 m between the waypoints), so the turn
angle is the only thing that changes. With fixed spacing, sharper turns would
also make the course longer, mixing difficulty with flight length.
`docs/drone_morphology_gate_racing.md` §5 records the measurements behind this.

Each gate faces along the bisector of its two neighbouring legs, i.e. along the
direction of travel. That is why gate 0 in section 4.1 has a yaw of 45°: the
first leg runs at +θ/2.

### 5.3 Details that exist for a reason

* **Spacing must exceed the gate size.** Otherwise neighbouring gate planes
  overlap and "passing gate k" becomes ambiguous. `slalom_gates` raises
  `ValueError` if it does not. At the defaults this caps the turn angle at
  130.8°: 130° works, 131° raises.
* **Lead-out.** `path_pos` continues two legs past the last scored gate. A
  reference path that *ends* on the last gate makes a fast drone decelerate
  onto it and overshoot. `gate_pos` holds only the scored gates, and is what
  the pass test uses.
* **Uniform versus randomised.** Without a seed every corner has exactly
  `turn_deg`. With a seed each corner varies, for example 80.5° to 100.4° for
  `slalom_gates(90.0, seed=1)`, so a set of courses shares a difficulty but not
  a shape. Use randomised courses when training or optimising, so the result
  does not overfit one layout. `measured_turn_angles()` returns the actual
  angles.

### 5.4 What a `SlalomCourse` holds

| attribute | used by | meaning |
|---|---|---|
| `gate_pos`, `gate_yaw`, `gate_size` | the task | the scored gates |
| `starting_pos` | the task | spawn point |
| `spacing`, `amplitude`, `leg`, `turn_deg` | people | layout description |
| `path_pos`, `path_yaw`, `path_length` | the classical controller only | waypoints including the lead-out, for the reference path |
| `gate_circle_radius` | people | radius of the circle through three consecutive gates. Describes the layout only: the drone does not fly this circle |

The methods `trajectory_config()`, `traversal_time()` and `reference_demand()`
exist for the classical controller's reference path. A policy that chooses its
own path does not need them.

## 6. The pass test: `src/ariel/simulation/drone/gate_metrics.py`

### 6.1 Definition

At the simulation step where the drone first crosses a gate's plane, measure
its distance from the gate centre **within the plane**, sideways and vertical
together. The gate is passed if that distance is at most `gate_size / 2`.

Two functions implement this, and everything else calls them:

* `in_plane_offset(pos, gate_pos, gate_yaw)` returns `(signed, lateral,
  vertical, offset)`. `signed` is the distance along the gate normal, negative
  before the gate; `offset` is the in-plane distance from the centre. The
  distance along the normal is removed first, so a sample taken slightly past
  the plane is not penalised for how far past it landed.
* `passes(offset, gate_size, clearance=0.0)` returns whether
  `offset + clearance <= gate_size / 2`.

```python
from ariel.simulation.drone.gate_metrics import in_plane_offset, passes

gate, yaw = np.array([4.0, 0.0, -1.5]), 0.0
p = np.array([4.02, 0.30, -1.20])            # just past the plane, 0.3 m east, 0.3 m up
signed, lateral, vertical, offset = in_plane_offset(p, gate, yaw)
print(round(float(signed), 3), round(float(lateral), 3), round(float(vertical), 3),
      round(float(offset), 3))
print(bool(passes(offset, gate_size=1.0)), bool(passes(offset, 1.0, clearance=0.26)))
```

```text
0.02 0.3 0.3 0.424
True False
```

Both functions work on arrays of any leading shape, so one call scores many
drones at once.

### 6.2 Choices to know about

* **A circle, not a square.** A square of the same half-width admits 27% more
  area, all of it near the corners.
* **Altitude counts.** An earlier test in this repository checked only
  horizontal position and passed a drone flying below the gates
  (`docs/drone_morphology_gate_racing.md` §3.8).
* **The drone is a point.** With the default `clearance=0` only the drone's
  centre is tested. Its rotors can extend well beyond that: 0.26 m on the
  airframes in the report. Pass `clearance` to require the whole airframe to
  fit. A morphing drone can pass its current extent at every step.
* **Gates have no physical presence.** Nothing in the simulation collides with
  a gate. A miss is recorded; it does not stop the drone.
* **The first sample past the plane is scored.** The test does not interpolate
  between samples. At a 200 Hz step and 10 m/s the drone moves 5 cm per step,
  which is negligible; at a 50 Hz policy step it moves 20 cm, which can decide
  a borderline gate. Check at the physics rate where possible.

`crossing_report(positions, gate_pos, gate_yaw, gate_size)` applies the same
test to a whole recorded trajectory and returns one `GateCrossing` per gate. It
is the criterion the report's figures use.

## 7. The episode rules: `GateTask`

### 7.1 Life cycle

```python
from ariel.simulation.tasks.gate_task import course_bounds

N = 4                                                     # parallel environments
task = GateTask(course, num_envs=N, max_time=20.0,
                bounds=course_bounds(course, course.starting_pos, margin=2.0))
pos = np.tile(course.starting_pos, (N, 1))                # (N, 3)
task.reset(pos, t=0.0)

gate_pos, gate_yaw, valid = task.upcoming(2)              # policy input: next two gates
print(gate_pos.shape, gate_yaw.shape, valid.shape)        # (N, 2, 3) (N, 2) (N, 2)

pos = pos + np.array([0.5, 0.0, 0.0])                     # stand-in for one physics step
events = task.update(pos, t=0.005)
print(events.passed.shape, events.done, Status(task.status[0]).name)
```

```text
(4, 2, 3) (4, 2) (4, 2)
(4,) [False False False False] RUNNING
```

1. `GateTask(layout, num_envs, on_miss=..., bounds=..., max_time=...)` creates
   the task for `num_envs` drones at once.
2. `reset(pos, mask=None, start_gate=0, t=0.0)` starts episodes. `pos` sets
   which side of the first gate each drone is on. `mask` resets only some
   environments. `start_gate` lets training begin mid-course.
3. `update(pos, t, clearance=0.0)` is called **once after every simulation
   step**. It returns a `StepEvents` with, per environment: `passed`, `missed`,
   `gate` (which gate, or −1) and `done`.
4. Read the result from the attributes in section 7.3.

Everything is batched: arrays have a leading dimension of `num_envs`, and there
is no per-environment Python loop inside. One drone is simply `num_envs=1`.

### 7.2 Rules

* **Order.** Each drone has one target gate. Only the target's plane is
  watched; passing it moves the target on. Gates must therefore be flown in
  order.
* **Misses.** Crossing the target's plane outside the opening is a miss.
  `on_miss="terminate"` (default) ends the episode with status `MISSED`, as in
  racing and as a physical gate frame would. `"continue"` records the miss and
  moves on; this is how the report scores a flight.
* **Bounds.** `course_bounds(layout, *points, margin=2.0)` returns a box around
  the gates and any extra points. For the 90° course with its spawn this is
  z from −3.5 to +0.5. **The task knows nothing about the ground:** positive z
  is below ground in NED, so set the upper z bound to 0 yourself if the ground
  should end an episode.

An episode ends when its status leaves `RUNNING`:

| status | when |
|---|---|
| `FINISHED` | the last gate's plane was crossed (see `completed` for whether every gate was passed) |
| `MISSED` | a gate was missed, with `on_miss="terminate"` |
| `OUT_OF_BOUNDS` | the drone left the `bounds` box |
| `TIMEOUT` | `max_time` elapsed since `reset` |

### 7.3 Results

| attribute | shape | meaning |
|---|---|---|
| `status` | (N,) | a `Status` value |
| `target` | (N,) | index of the gate each drone is aiming at |
| `gates_passed` | (N,) | gates passed this episode |
| `completed` | (N,) | finished with every gate passed |
| `course_time` | (N,) | time from `reset` to the last gate, for completed episodes; infinity otherwise |
| `passed` | (N, n_gates) | per-gate verdict |
| `crossing_offset`, `crossing_time` | (N, n_gates) | per gate, at the crossing; NaN if never crossed |
| `upcoming(n)` | (N, n, 3), (N, n), (N, n) | next `n` gates' positions and headings, target first, plus a validity mask past the end |

### 7.4 What is deliberately left out

* **Reward.** It is a training choice, not part of the task. Build it from the
  `StepEvents` and `upcoming()`.
* **Observation frames.** `upcoming()` returns world-frame poses. Converting
  them to the body or gate frame belongs to the environment.
* **Dynamics, controller and reference path.**

## 8. Using your own gates

Any object with `gate_pos` (n, 3), `gate_yaw` (n,) and `gate_size` works. Here,
three gates at different heights, flown along a path through their centres:

```python
from types import SimpleNamespace

layout = SimpleNamespace(
    gate_pos=np.array([[4.0, 0.0, -1.5], [8.0, 2.0, -2.5], [12.0, 0.0, -1.5]]),
    gate_yaw=np.radians([0.0, 0.0, 0.0]),                 # all face +x (north)
    gate_size=1.0,
)
task = GateTask(layout)
start = np.array([0.0, 0.0, -1.5])
task.reset(start)
corners = np.vstack([start, layout.gate_pos, [14.0, 0.0, -1.5]])
path = np.vstack([np.linspace(a, b, 100) for a, b in zip(corners[:-1], corners[1:])])
for k, p in enumerate(path[1:], 1):
    task.update(p, t=0.01 * k)
print(Status(task.status[0]).name, task.gates_passed[0], "gates, offsets",
      np.round(task.crossing_offset[0], 3))
```

```text
FINISHED 3 gates, offsets [0. 0. 0.]
```

Rules for a layout of your own:

* **Point each gate along the direction of travel.** A gate facing the wrong
  way is never crossed "forwards", so it is never passed.
* **Keep gates far enough apart** that a drone passing gate k is still behind
  gate k+1's plane. Only the target's plane is watched, and it is watched from
  the moment the previous gate is passed.
* **List the gates in flying order.**

A setpoint-style task maps onto this directly: the next setpoint is
`upcoming(1)`, the target gate's centre. What differs is the pass test (a plane
crossing inside an opening, rather than reaching within a distance of a point)
and the rules for ending an episode.

## 9. Changing things

| change | where | notes |
|---|---|---|
| difficulty | `turn_deg`, and `leg` | sharper turns and shorter legs are harder; spacing must stay above `gate_size` |
| gate size | `gate_size` | same constraint |
| altitude changes along the course | a layout of your own (section 8) | `slalom_gates` keeps one altitude, `z` |
| gate shape, e.g. square | `in_plane_offset` / `passes` in `src/ariel/simulation/drone/gate_metrics.py` | one place changes the task and `crossing_report` together; see the caveat below |
| airframe size | `clearance` in `update()` | per environment, per step |
| score | `course_time` on `GateTask`, or your own from the per-gate arrays | |
| GPU batching (Isaac Lab) | `src/ariel/simulation/tasks/gate_task.py` | uses only array operations, no Python loop over environments; a torch port replaces NumPy calls one for one. Not yet done |

**Caveat when changing the pass test.** Two older implementations do *not* use
`src/ariel/simulation/drone/gate_metrics.py`:

* `examples/spear/19_morphology_design_sweep.py`, which produced the report's
  results, repeats the test inline in its `rollout` function.
  `tests/unit/test_simulation/test_gate_criterion.py` checks that the two agree
  on one real flight. Change both together.
* `DroneGateEnv` in `src/ariel/simulation/tasks/drone_gate_env.py`, the
  repository's reinforcement-learning environment, uses its own test: a
  1.5 m **box** around the gate centre, hard-coded, which ignores the
  course's `gate_size`. It does not use `GateTask`.

## 10. Status

* **Verified.** `GateTask` gives the same per-gate verdicts and offsets as
  `crossing_report` on all six recorded flights in `docs/data/flight_logs/`,
  both as flown and shifted sideways to produce misses. The tests also cover
  stopping at the first miss, batching, clearance, bounds, the time limit and
  starting mid-course.
* **Not yet used.** Nothing outside the tests calls `GateTask`. The report's
  results come from the sweep script's own copy of the same test.
* **Not done.** A torch version for Isaac Lab, and moving `DroneGateEnv` onto
  `GateTask`.

## 11. Tests

```bash
uv run pytest tests/unit/test_simulation/test_gate_task.py \
    tests/unit/test_simulation/test_gate_criterion.py \
    tests/unit/test_simulation/test_slalom_course.py
```

* `tests/unit/test_simulation/test_gate_task.py`: the task agrees with
  `crossing_report`, plus the episode rules.
* `tests/unit/test_simulation/test_gate_criterion.py`: the pass test itself
  (altitude counts, circle not square, offset projected into the plane), and
  agreement with the sweep's inline copy on a real flight.
* `tests/unit/test_simulation/test_slalom_course.py`: course length stays fixed
  as turn angle varies, and the classical controller's reference-path
  measurements.

## 12. Further reading

* `docs/drone_morphology_gate_racing.md`, the design record for this work:
  §3.8 for how the pass test was decided, §5 for the course design decisions,
  and §4 for what was built.
* `Reports/morphology_gate_racing.tex`, Courses and Scoring sections, for the
  task as used in the report.

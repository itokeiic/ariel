"""Does static arm geometry matter for flying a slalom?

Groundwork for mid-flight morphing: before making arm sweep a live DOF, find
out whether holding it at different angles changes anything, and where the
useful range is. Every configuration here is a *static* setting, which is
identical to holding a revolute arm joint at a fixed angle, so the answer
transfers directly to a joint target once the articulated plant exists.

What this fixes, relative to the first attempt at this question
--------------------------------------------------------------
That sweep found fitness invariant to morphology (0.000 spread across a full
rotation, 0.011 across a 3x arm-length change). Four reasons, all addressed:

* The quintic circuits turn a median of 0.4 deg -- they never ask the airframe
  for anything. Replaced by a slalom whose turn angle is an explicit knob
  (``ariel.simulation.tasks.slalom_course``).
* The drone was 0.093 kg with 2" props (thrust-to-weight 8.9), so a 2 m/s
  course used 3% of its lateral budget. Now SPEAR-matched: 0.83 kg, 5" props,
  0.20 m arms, TWR 5.2, matching ``symmetric_4rotor`` in
  ``examples/spear_vua_upb/spear``.
* Speed was implicit. Now a fixed nominal speed, so a tighter course is harder
  because of curvature rather than because it is also flown faster.
* ``cond(mixerFM)`` was the instrument, and it is constant for coplanar rotors.
  Replaced by per-axis angular acceleration plus airevolve's maneuverability.

Why azimuth and length are the honest axes to sweep here: ``DroneSimulator``'s
plant reads each rotor's ``loc`` but ignores its thrust normal (see
``dynamics_params._guard_axial_thrust``). Both axes move exactly what the plant
models and leave the normals axial, so the guard stays silent for every body
below. Arm elevation and motor cant would not be trustworthy until the plant is
normal-aware.

Run:
    uv run examples/spear/19_morphology_design_sweep.py --tracking-check
    uv run examples/spear/19_morphology_design_sweep.py --map-only
    uv run examples/spear/19_morphology_design_sweep.py --turn-deg 90 --speed 6
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "examples" / "d_drones"))
from _ctrl_helpers import GateChecker  # noqa: E402

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.ec.drone.genome_handlers.conversions.arm_conversions import (
    arms_to_cylinders_polar_angular,
)
from ariel.ec.drone.genome_handlers.operators.particle_repair_operator import (
    are_there_cylinder_collisions,
)
from ariel.simulation.drone.controllers.lee_control.lee_controller import (
    LeeGeometricControl,
)
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import (
    Trajectory,
)
from ariel.simulation.drone.controllers.utils.wind_model import Wind
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.drone_interface import DroneInterface
from ariel.simulation.drone.plant import (
    RotorGeometry,
    maneuverability,
    moment_allocation,
    rank_controllability,
)
from ariel.simulation.tasks.slalom_course import slalom_gates

console = Console()

# ── airframe: SPEAR symmetric_4rotor ────────────────────────────────────────
PROP_SIZE      = 5
PROP_RADIUS    = 0.0635          # 5" prop, matching every SPEAR xacro
CORE_RADIUS    = 0.05            # CorePlateNode default
PAYLOAD_MASS   = 0.667           # -> 0.829 kg total, TWR 5.24
ARM_LENGTH     = 0.20
N_ARMS         = 4               # a quad has no allocation redundancy
CLEARANCE_TOL  = 0.1             # same 10% margin the repair operator uses

# ── controller / rollout ────────────────────────────────────────────────────
SIM_DT         = 0.005
LEE_POS_GAIN   = 14.3
LEE_VEL_GAIN   = 9.0
STARTUP_TIME   = 3.0             # BSplineGateTrajectory default
# ── objective ───────────────────────────────────────────────────────────────
# "legacy" reproduces 17_drone_evo_average_num_gates_lee.py exactly:
#     10*gates + survival - 0.1*tracking_err + 10*(1-survival) if completed
# Term magnitudes there are 30-150 / 0.6-1.0 / -0.008 to -0.39 / 0-3.7, so it
# is the gate count with rounding errors attached, and it says nothing about
# the constraint that actually binds.
GATE_BONUS       = 10.0
TRACK_WEIGHT     = 0.1
COMPLETION_BONUS = 10.0

# "normalized" (default): gates score out of 100, plus a quality budget worth
# HALF a gate. The budget is capped there deliberately -- no combination of
# quality terms can outrank passing one more gate, so the objective stays
# lexicographic in practice while remaining continuous and differentiable-ish
# for an EA. Within a gate tier -- which is most comparisons, since this
# landscape is flat -- the quality terms are the whole signal.
QUALITY_TERMS = ("track", "margin", "survive", "speed")

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--turn-deg", type=float, default=90.0)
parser.add_argument("--leg", type=float, default=2.4)
parser.add_argument("--n-gates", type=int, default=15)
parser.add_argument("--gate-size", type=float, default=1.0)
parser.add_argument("--speed", type=float, default=6.0, help="nominal m/s")
parser.add_argument("--points", type=int, default=11)
parser.add_argument("--sim-margin", type=float, default=1.6,
                    help="rollout time = traversal time x this")
parser.add_argument("--tracking-check", action="store_true",
                    help="canonical X quad at several speeds; establishes the "
                         "usable speed range before any morphology work")
parser.add_argument("--max-speed-sweep", action="store_true",
                    help="score each morphology by the fastest speed at which "
                         "it still completes the course, found by bisection, "
                         "swept across --sweep-turns. Continuous by "
                         "construction: no operating point to calibrate, no "
                         "ceiling, and at a body's own limit the two gate "
                         "counting rules agree.")
parser.add_argument("--sweep-turns", default="60,90,120")
parser.add_argument("--speed-lo", type=float, default=2.0)
parser.add_argument("--speed-hi", type=float, default=6.0)
parser.add_argument("--speed-tol", type=float, default=0.125)
parser.add_argument("--speed-cap", type=float, default=12.0)
parser.add_argument("--calibrate", action="store_true",
                    help="grid over speed x turn angle, evaluating a few "
                         "morphologies at each point. Picks the operating point "
                         "where gate counts actually SPREAD across bodies -- a "
                         "point where everyone scores the ceiling, or everyone "
                         "collapses, cannot discriminate morphology however "
                         "good the objective is.")
parser.add_argument("--cal-speeds", default="2,3,4")
parser.add_argument("--cal-turns", default="60,90,120")
parser.add_argument("--map-only", action="store_true",
                    help="feasibility + maneuverability map only, no rollouts")
parser.add_argument("--pos-gain", type=float, default=LEE_POS_GAIN)
parser.add_argument("--vel-gain", type=float, default=LEE_VEL_GAIN)
parser.add_argument("--max-accel", type=float, default=5.0,
                    help="commanded-acceleration clamp, m/s^2. The library "
                         "default of 5.0 is below what a fast course demands "
                         "(21 m/s^2 at 6 m/s here) and below what the airframe "
                         "can deliver (50 m/s^2), so while it binds every "
                         "morphology is limited by the same constant. Raising "
                         "it alone does NOT fix tracking -- see --feedforward.")
parser.add_argument("--feedforward", action="store_true",
                    help="feed the trajectory's velocity and acceleration to "
                         "the position controller. Off by default because the "
                         "controller is not tuned for it: it removes the ~0.25 s "
                         "lag but overshoots in speed and sags in altitude, and "
                         "currently tracks worse overall. Needs control design "
                         "before it is usable.")
parser.add_argument("--n-courses", type=int, default=5,
                    help="layouts to average over. One fixed layout selects the "
                         "body that suits that layout; the simulator is "
                         "deterministic, so this is the only sampling there is.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--jitter", type=float, default=0.25,
                    help="per-corner turn-angle spread across the course set")
parser.add_argument("--gate-counting", choices=("sequential", "flown"),
                    default="sequential",
                    help="sequential: GateChecker order, so the first gate "
                         "missed ends the scoring run -- racing semantics, and "
                         "a high-variance objective (12 counted vs 14 flown at "
                         "4 m/s). flown: every gate the drone passed within "
                         "half a gate of, regardless of order -- measures "
                         "flight quality rather than run validity. The two can "
                         "select different morphologies, so it is a real choice "
                         "about what the task is.")
parser.add_argument("--objective", choices=("normalized", "legacy"),
                    default="normalized",
                    help="normalized: gates out of 100 plus a quality budget "
                         "worth half a gate (tracking, control margin, "
                         "survival, speed). legacy: example 17's formula, for "
                         "reproducing its numbers.")
parser.add_argument("--out-dir", default=None)
args = parser.parse_args()

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DATA = Path(args.out_dir or f"__data__/morphology_design_sweep/{RUN_ID}")
DATA.mkdir(parents=True, exist_ok=True)


def make_courses() -> list:
    """The course set every body is evaluated on."""
    if args.n_courses <= 1:
        return [slalom_gates(args.turn_deg, leg=args.leg, n_gates=args.n_gates,
                             gate_size=args.gate_size)]
    return [slalom_gates(args.turn_deg, leg=args.leg, n_gates=args.n_gates,
                         gate_size=args.gate_size, seed=args.seed + k,
                         jitter=args.jitter)
            for k in range(args.n_courses)]


# ── morphology ──────────────────────────────────────────────────────────────

def make_genome(half_angle: float, arm_length: float = ARM_LENGTH) -> np.ndarray:
    """Bilaterally symmetric quad: arms at +/-t and 180 +/- t.

    Columns: [length, arm_az, arm_elevation, motor_az, motor_pitch, spin].
    Elevation and motor pitch stay 0, so thrust normals remain axial and the
    reduced plant models this airframe correctly.
    """
    az = np.array([half_angle, np.pi - half_angle, np.pi + half_angle, -half_angle])
    g = np.zeros((N_ARMS, 6), dtype=float)
    g[:, 0] = arm_length
    g[:, 1] = az
    g[:, 3] = np.pi
    g[:, 5] = np.arange(N_ARMS) % 2
    return g


def feasible_half_angle_range(arm_length: float) -> tuple[float, float]:
    """Half-angles whose propeller discs clear each other, radians.

    Neighbouring rotors are 2*L*sin(t) and 2*L*cos(t) apart, and both must
    exceed 2*(1+tol)*r.
    """
    c = (1.0 + CLEARANCE_TOL) * PROP_RADIUS / arm_length
    if c >= 1.0:
        raise ValueError(f"no clearing half-angle exists at arm length {arm_length} m")
    return float(np.arcsin(c)), float(np.pi / 2 - np.arcsin(c))


def min_arm_length() -> float:
    """Shortest arm whose propeller disc clears the core plate.

    Not covered by the collision test, which only compares arm cylinders with
    each other -- nothing today stops a rotor from intersecting the body.
    """
    return CORE_RADIUS + (1.0 + CLEARANCE_TOL) * PROP_RADIUS


def rotors_overlap(genome: np.ndarray) -> bool:
    return bool(are_there_cylinder_collisions(
        arms_to_cylinders_polar_angular(genome, propeller_radius=PROP_RADIUS)
    ))


def rotor_hits_body(genome: np.ndarray) -> bool:
    return bool(np.any(genome[:, 0] < min_arm_length()))


def describe(genome: np.ndarray) -> dict:
    """Geometry-only metrics; no simulation."""
    bp = spherical_angular_to_blueprint(genome, propsize=PROP_SIZE)
    props_zup = blueprint_to_propellers(bp, convention="z_up")
    cfg = DroneConfiguration(props_zup, payload_mass=PAYLOAD_MASS)
    kf, km = cfg.propellers[0]["constants"]
    geom = RotorGeometry(
        positions=np.array([p["loc"] for p in props_zup], dtype=float),
        axes=np.array([p["dir"][:3] for p in props_zup], dtype=float),
        spins=np.array([1.0 if p["dir"][3] == "ccw" else -1.0 for p in props_zup]),
    )
    ratio = km / kf
    alpha = moment_allocation(geom, torque_to_thrust_ratio=ratio,
                              inertia=cfg.inertia_matrix)
    return {
        "mass": float(cfg.mass),
        "twr": float(kf * cfg.propellers[0]["wmax"] ** 2 * N_ARMS / (cfg.mass * 9.81)),
        "alpha_roll": float(np.linalg.norm(alpha[0])),
        "alpha_pitch": float(np.linalg.norm(alpha[1])),
        "alpha_yaw": float(np.linalg.norm(alpha[2])),
        "maneuverability": maneuverability(geom, torque_to_thrust_ratio=ratio),
        "maneuverability_I": maneuverability(geom, torque_to_thrust_ratio=ratio,
                                             inertia=cfg.inertia_matrix),
        "rank_Bm": rank_controllability(geom, torque_to_thrust_ratio=ratio),
    }


# ── rollout ─────────────────────────────────────────────────────────────────

def _build_stack(propellers, course, total_time):
    drone = DroneInterface(0, propellers=propellers, payload_mass=PAYLOAD_MASS)
    wind = Wind("None")
    ctrl = LeeGeometricControl(
        drone, yawType=1, orient="NED", auto_scale_gains=True,
        # Both off by default in the library, and both must be on to fly an
        # aggressive trajectory: without feedforward the controller lags a
        # moving reference by ~0.25 s, and the 5 m/s^2 default clamp sits far
        # below what this course demands (21 m/s^2) or the airframe can deliver
        # (50 m/s^2), so every morphology would be limited by the same constant.
        velocity_feedforward=args.feedforward,
        max_accel=args.max_accel,
        pos_P_gain=np.array([args.pos_gain] * 3),
        vel_P_gain=np.array([args.vel_gain] * 3),
    )
    # The trajectory flies the lead-out; only the detector sees scored gates.
    # Without it the reference clamps on the final gate, and a drone arriving at
    # speed overshoots by 2.1-2.7 m -- making that gate unreachable for every
    # morphology and capping all of them at 14/15.
    traj = Trajectory(drone, "xyz_pos", np.array([15, 3, 1]),
                      gate_config=course.trajectory_config())
    # The shipped tension-based offsets only approximate interpolation. On a
    # 90 deg slalom the reference misses a gate centre by 0.52 m against a
    # 0.5 m half-width, i.e. the path the drone is asked to fly goes *outside*
    # the gate. Solve the offsets exactly instead.
    traj.bspline_trajectory.fit_offsets_to_gates()
    traj.bspline_trajectory.total_time = total_time

    drone.drone_sim.set_state(
        position=np.asarray(course.starting_pos, dtype=np.float64),
        velocity=np.zeros(3), attitude=np.zeros(3), angular_velocity=np.zeros(3),
    )
    drone._update_state_variables()
    checker = GateChecker(course.gate_pos, course.gate_yaw, course.gate_size)
    checker.check_gate_passing(drone.pos)
    ctrl.controller(traj.desiredState(0.0, SIM_DT, drone), drone, traj.ctrlType, SIM_DT)
    return drone, ctrl, traj, checker, wind


def rollout(propellers, course, speed) -> dict:
    total_time = course.traversal_time(speed, startup_time=STARTUP_TIME)
    sim_time = total_time * args.sim_margin
    try:
        drone, ctrl, traj, checker, wind = _build_stack(propellers, course, total_time)
    except Exception:
        return {"fit": None, "gates": 0, "survival": 0.0,
                "tracking_err": float("inf"), "completed": False,
                "completed_flown": False, "saturation": 1.0,
                "saturation_hi": 1.0, "saturation_lo": 1.0, "gates_flown": 0}

    n_steps = int(sim_time / SIM_DT)
    w_lo, w_hi = float(drone.params["minWmotor"]), float(drone.params["maxWmotor"])
    tracking_sum, steps, sat, completed = 0.0, 0, 0, False
    completed_flown = False
    # Split by bound: the upper one means 'not enough thrust', the lower one
    # means the allocation wanted a rotor to pull -- i.e. it ran out of
    # differential range for the commanded moment. On this course the lower
    # bound is what binds (29% of steps at 6 m/s against 0% upper), and it is
    # the geometry-sensitive one: more moment per newton of differential means
    # less need to drive a rotor toward zero.
    sat_hi = sat_lo = 0
    # Tracking error is accumulated only while the course is running. After
    # total_time the reference clamps at the final gate while the drone is
    # still arriving, so post-course error swamps everything: at 4 m/s the
    # cruise mean is 0.37 m and the post-course mean is 10.5 m, and a rollout
    # average would report the latter as if it described tracking.
    course_steps = 0
    # Gates the drone actually flew through, independent of ordering. The
    # sequential checker stops advancing at the first gate missed, so one bad
    # corner reads as a near-zero score: at 4 m/s it counted 2/15 while the
    # drone flew within half a gate of 11.
    closest = np.full(len(course.gate_pos), np.inf)
    try:
        t, i = 0.0, 1
        while i <= n_steps:
            drone.update(t, SIM_DT, ctrl.w_cmd, wind)
            t_new = SIM_DT * i
            sDes = traj.desiredState(t_new, SIM_DT, drone)
            ctrl.controller(sDes, drone, traj.ctrlType, SIM_DT)
            checker.check_gate_passing(drone.pos)
            if t_new <= total_time:
                tracking_sum += float(np.linalg.norm(drone.pos - sDes[:3]))
            w = np.asarray(ctrl.w_cmd, dtype=float)
            pinned_hi = bool(np.any(w >= w_hi - 1e-6))
            pinned_lo = bool(np.any(w <= w_lo + 1e-6))
            if pinned_hi or pinned_lo:
                sat += 1
            sat_hi += pinned_hi
            sat_lo += pinned_lo
            steps += 1
            if t_new <= total_time:
                course_steps += 1
            closest = np.minimum(
                closest, np.linalg.norm(course.gate_pos - drone.pos, axis=1))
            t, i = t_new, i + 1
            if checker.gates_passed >= checker.num_gates:
                completed = True
                break
            # Terminate identically whichever rule is scored later, so both are
            # read off the same flight rather than two runs that ended
            # differently.
            if np.all(closest <= course.gate_size / 2.0):
                completed_flown = True
                break
    except Exception:
        pass

    survival = steps / n_steps if n_steps else 0.0
    tracking_err = tracking_sum / course_steps if course_steps else float("inf")
    gates = int(checker.gates_passed)
    gates_flown = int(np.sum(closest <= course.gate_size / 2.0))
    return {
        "fit": None,   # scored by score(), which needs the course
        "gates": gates, "survival": survival, "tracking_err": tracking_err,
        "completed": completed, "completed_flown": completed or completed_flown,
        "saturation": sat / steps if steps else 1.0,
        "saturation_hi": sat_hi / steps if steps else 1.0,
        "saturation_lo": sat_lo / steps if steps else 1.0,
        "gates_flown": gates_flown,
    }


def score(out: dict, course, n_gates: int, objective: str,
          gate_counting: str = "sequential") -> dict:
    """Objective value plus its decomposition, so a score is auditable.

    Args:
        out: a rollout result.
        course: the course flown (for gate_size, the natural error scale).
        n_gates: gates on the course.
        objective: "normalized" or "legacy".
        gate_counting: "sequential" (GateChecker order; the first miss ends the
            run) or "flown" (every gate passed within half a gate, in any
            order). These can rank morphologies differently -- a body that
            clips one corner but flies the rest cleanly is near-worthless under
            the first and near-perfect under the second -- so it is a choice
            about the task, not a detail.

    Quality terms, each in [0, 1], higher better:
        track   exp(-err / (gate_size/2)) -- a mean error of half a gate
                scores 1/e. Uses the gate as the length scale rather than an
                arbitrary constant, so it transfers across course sizes.
        margin  1 - clipping fraction. The share of steps where the allocation
                got the wrench it asked for. This is the geometry-sensitive
                term: a layout with more moment per newton of differential
                needs less differential and clips less.
        survive 1 if the course was completed, else the fraction flown before
                divergence.
        speed   remaining time fraction if completed, else 0.
    """
    gates = out["gates_flown"] if gate_counting == "flown" else out["gates"]
    completed = out["completed_flown"] if gate_counting == "flown" else out["completed"]
    if objective == "legacy":
        bonus = COMPLETION_BONUS * (1.0 - out["survival"]) if completed else 0.0
        fit = (GATE_BONUS * gates + out["survival"]
               - TRACK_WEIGHT * out["tracking_err"] + bonus)
        return {"fitness": float(fit)}

    if not np.isfinite(out["tracking_err"]):
        track_q = 0.0
    else:
        track_q = float(np.exp(-out["tracking_err"] / max(course.gate_size / 2.0, 1e-9)))
    margin_q = float(1.0 - min(max(out["saturation"], 0.0), 1.0))
    survive_q = 1.0 if completed else float(min(max(out["survival"], 0.0), 1.0))
    speed_q = float(1.0 - out["survival"]) if completed else 0.0

    gate_score = 100.0 * gates / n_gates
    per_gate = 100.0 / n_gates
    budget = 0.5 * per_gate / len(QUALITY_TERMS)   # all four together < one gate
    quality = budget * (track_q + margin_q + survive_q + speed_q)

    return {
        "fitness": float(gate_score + quality),
        "gate_score": gate_score,
        "quality": float(quality),
        "q_track": track_q, "q_margin": margin_q,
        "q_survive": survive_q, "q_speed": speed_q,
        "gates_counted": gates,
    }


def evaluate(genome, courses, speed) -> dict:
    """Mean score over a set of courses, under both counting rules.

    Averaging over layouts is what stops the sweep selecting the body that
    happens to suit one gate arrangement -- with a deterministic simulator that
    is the only sampling error there is. Both counting rules are scored from the
    same rollouts, so the choice can be revisited without re-flying anything.
    """
    props = blueprint_to_propellers(
        spherical_angular_to_blueprint(genome, propsize=PROP_SIZE), convention="ned")
    outs, both = [], []
    for course in courses:
        o = rollout(props, course, speed)
        outs.append(o)
        both.append({m: score(o, course, len(course.gate_pos), args.objective, m)
                     for m in ("sequential", "flown")})

    def mean_of(key, mode=None):
        if mode is None:
            return float(np.mean([o[key] for o in outs]))
        return float(np.mean([b[mode][key] for b in both]))

    scored = {
        "fitness": mean_of("fitness", args.gate_counting),
        "fitness_sequential": mean_of("fitness", "sequential"),
        "fitness_flown": mean_of("fitness", "flown"),
    }
    if args.objective == "normalized":
        for k in ("gate_score", "quality", "q_track", "q_margin", "q_survive", "q_speed"):
            scored[k] = mean_of(k, args.gate_counting)

    rolled = {k: mean_of(k) for k in
              ("gates", "gates_flown", "survival", "tracking_err",
               "saturation", "saturation_hi", "saturation_lo",
               "completed", "completed_flown")}
    return {**describe(genome), **scored, **rolled}


# ── modes ───────────────────────────────────────────────────────────────────

def tracking_check() -> None:
    """Canonical X quad across speeds — is the controller usable up there?"""
    courses = make_courses()
    course = courses[0]
    genome = make_genome(np.pi / 4)
    d = describe(genome)
    console.rule(f"tracking check — X quad, {args.turn_deg:.0f}° slalom, "
                 f"R={course.radius:.2f} m")
    console.log(f"airframe: {d['mass']:.3f} kg, TWR {d['twr']:.2f}, "
                f"lateral budget {9.81*np.sqrt(max(d['twr']**2-1,0)):.0f} m/s²")
    rows = []
    for speed in (2.0, 4.0, 6.0, 8.0):
        a_lat = course.lateral_acceleration(speed)
        r = evaluate(genome, courses, speed)
        rows.append({"speed": speed, "a_lat": a_lat, **r})
        console.log(
            f"  {speed:>4.1f} m/s  a_lat={a_lat:5.1f}  gates={r['gates']:>2}/{args.n_gates} "
            f"trk={r['tracking_err']:6.3f} m  sat={r['saturation']*100:5.1f}%  "
            f"fit={r['fitness']:8.3f}  {'completed' if r['completed'] else ''}")
    _write_csv(rows, DATA / f"tracking_check_{RUN_ID}.csv")


def max_completing_speed(genome: np.ndarray, course) -> dict:
    """Fastest speed at which this body still completes the course.

    Bisection, which is valid because completion is monotone in speed: a body
    that finishes at 4 m/s finishes at 3. The calibration data supports that --
    all three probe bodies completed at every speed from 2.0 to 3.8 m/s and
    failed from 3.9 up, a single clean boundary -- but the assumption is
    checked rather than trusted, and a violation is reported instead of
    silently returning a wrong answer.

    Returns the limit speed, the flight at that limit, and the rollout count.
    """
    props = blueprint_to_propellers(
        spherical_angular_to_blueprint(genome, propsize=PROP_SIZE), convention="ned")
    tested: dict[float, dict] = {}

    def completes(v: float) -> bool:
        if v not in tested:
            tested[v] = rollout(props, course, v)
        return bool(tested[v]["completed_flown"])

    lo, hi = float(args.speed_lo), float(args.speed_hi)
    if not completes(lo):
        # Cannot fly the course even at the slowest bracket speed.
        return {"max_speed": float("nan"), "n_rollouts": len(tested),
                "bracket": "below_lo", "monotone": True, **tested[lo]}

    # Widen upward rather than clipping at hi -- a ceiling would make good
    # bodies tie, which is the failure this objective exists to avoid.
    while completes(hi) and hi < args.speed_cap:
        lo, hi = hi, min(hi * 1.5, float(args.speed_cap))
    if completes(hi):
        return {"max_speed": hi, "n_rollouts": len(tested),
                "bracket": "above_cap", "monotone": True, **tested[hi]}

    while hi - lo > float(args.speed_tol):
        mid = 0.5 * (lo + hi)
        if completes(mid):
            lo = mid
        else:
            hi = mid

    # Monotonicity: no tested speed above the limit may have completed, and
    # none below it may have failed.
    monotone = all((v <= lo) == o["completed_flown"] or abs(v - lo) < 1e-9
                   for v, o in tested.items())
    return {"max_speed": lo, "n_rollouts": len(tested), "bracket": "ok",
            "monotone": monotone, **tested[lo]}


def max_speed_sweep() -> list[dict]:
    """Max completing speed vs arm half-angle, at several turn angles.

    Uniform (unjittered) courses on purpose: a uniform slalom is the same
    corner repeated, so there is no layout to overfit to, and jitter would make
    the limit speed depend on whichever corner the RNG made sharpest.
    Generalisation is checked structurally instead -- by sweeping turn angle --
    which gives an interpretable curve rather than a mean over noise.
    """
    turns = [float(v) for v in args.sweep_turns.split(",")]
    lo_t, hi_t = feasible_half_angle_range(ARM_LENGTH)
    angles = np.linspace(lo_t, hi_t, args.points)
    rows: list[dict] = []
    for turn in turns:
        course = slalom_gates(turn, leg=args.leg, n_gates=args.n_gates,
                              gate_size=args.gate_size)
        console.rule(f"max-speed sweep — {turn:.0f}° slalom, R={course.radius:.2f} m")
        for t in angles:
            g = make_genome(float(t))
            if rotors_overlap(g):
                continue
            t0 = time.time()
            r = max_completing_speed(g, course)
            geo = describe(g)
            rows.append({"turn_deg": turn, "half_angle_deg": float(np.degrees(t)),
                         "max_speed": r["max_speed"], "n_rollouts": r["n_rollouts"],
                         "bracket": r["bracket"], "monotone": int(r["monotone"]),
                         "a_lat_at_limit": (r["max_speed"] ** 2 / course.radius
                                            if np.isfinite(r["max_speed"]) else float("nan")),
                         "tracking_err": r["tracking_err"],
                         "saturation_lo": r["saturation_lo"],
                         "saturation_hi": r["saturation_hi"],
                         "alpha_roll": geo["alpha_roll"], "alpha_pitch": geo["alpha_pitch"],
                         "maneuverability_I": geo["maneuverability_I"]})
            flag = "" if r["monotone"] else "  [red]NON-MONOTONE[/red]"
            console.log(f"  t={np.degrees(t):>5.1f}°  max speed {r['max_speed']:5.3f} m/s  "
                        f"a_lat={rows[-1]['a_lat_at_limit']:5.1f}  "
                        f"α_roll={geo['alpha_roll']:6.1f}  clip_lo={100*r['saturation_lo']:4.1f}%  "
                        f"({r['n_rollouts']} rollouts, {time.time()-t0:.0f}s){flag}")
        best = max((r for r in rows if r["turn_deg"] == turn and np.isfinite(r["max_speed"])),
                   key=lambda r: r["max_speed"], default=None)
        if best:
            console.log(f"  [bold green]best at {turn:.0f}°[/bold green]: "
                        f"t={best['half_angle_deg']:.1f}°  {best['max_speed']:.3f} m/s")
    _write_csv(rows, DATA / f"max_speed_{RUN_ID}.csv")
    return rows


def calibrate() -> list[dict]:
    """Find where the task separates morphologies.

    Every operating point is scored on the same three bodies -- the narrow,
    X and wide ends of the feasible half-angle range -- and what matters is
    not their mean score but the SPREAD between them. A course everyone flies
    perfectly and a course everyone fails are equally useless for selecting a
    morphology.
    """
    speeds = [float(v) for v in args.cal_speeds.split(",")]
    turns = [float(v) for v in args.cal_turns.split(",")]
    lo, hi = feasible_half_angle_range(ARM_LENGTH)
    probes = {"narrow": lo, "x": np.pi / 4, "wide": hi}
    rows = []
    console.rule("difficulty calibration")
    console.log(f"probe bodies: " + ", ".join(
        f"{k} t={np.degrees(v):.1f}°" for k, v in probes.items()))
    for turn in turns:
        for speed in speeds:
            try:
                courses = [slalom_gates(turn, leg=args.leg, n_gates=args.n_gates,
                                        gate_size=args.gate_size, seed=args.seed + k,
                                        jitter=args.jitter)
                           for k in range(args.n_courses)]
            except ValueError as exc:
                console.log(f"  turn {turn:.0f}° infeasible: {exc}")
                continue
            res = {k: evaluate(make_genome(t), courses, speed) for k, t in probes.items()}
            gseq = [res[k]["gates"] for k in probes]
            gflw = [res[k]["gates_flown"] for k in probes]
            row = {
                "turn_deg": turn, "speed": speed,
                "a_lat": courses[0].lateral_acceleration(speed),
                "gates_seq_mean": float(np.mean(gseq)),
                "gates_seq_spread": float(np.ptp(gseq)),
                "gates_flown_mean": float(np.mean(gflw)),
                "gates_flown_spread": float(np.ptp(gflw)),
                "fit_seq_spread": float(np.ptp([res[k]["fitness_sequential"] for k in probes])),
                "fit_flown_spread": float(np.ptp([res[k]["fitness_flown"] for k in probes])),
                "tracking_err": float(np.mean([res[k]["tracking_err"] for k in probes])),
                "clip_lo": float(np.mean([res[k]["saturation_lo"] for k in probes])),
            }
            rows.append(row)
            console.log(
                f"  turn {turn:>3.0f}°  {speed:>3.1f} m/s  a_lat={row['a_lat']:5.1f}  "
                f"seq {row['gates_seq_mean']:5.2f}±{row['gates_seq_spread']:<4.2f}  "
                f"flown {row['gates_flown_mean']:5.2f}±{row['gates_flown_spread']:<4.2f}  "
                f"trk={row['tracking_err']:5.3f}  clip={100*row['clip_lo']:4.1f}%")
    _write_csv(rows, DATA / f"calibration_{RUN_ID}.csv")
    if rows:
        best = max(rows, key=lambda r: r["gates_flown_spread"] + r["gates_seq_spread"])
        console.log(f"[bold green]most discriminating[/bold green]: turn {best['turn_deg']:.0f}°, "
                    f"{best['speed']:.1f} m/s (spread seq {best['gates_seq_spread']:.2f}, "
                    f"flown {best['gates_flown_spread']:.2f})")
    return rows


def design_map() -> list[dict]:
    """Feasibility + geometry metrics over (half-angle x arm length). No rollouts."""
    lengths = np.linspace(min_arm_length(), 0.25, 8)
    rows = []
    for L in lengths:
        lo, hi = feasible_half_angle_range(L)
        for t in np.linspace(np.deg2rad(10), np.deg2rad(80), 15):
            g = make_genome(float(t), arm_length=float(L))
            ov, body = rotors_overlap(g), rotor_hits_body(g)
            row = {"half_angle_deg": float(np.degrees(t)), "arm_length": float(L),
                   "rotor_overlap": int(ov), "rotor_body": int(body),
                   "analytic_ok": int(lo <= t <= hi)}
            if not (ov or body):
                row.update(describe(g))
            rows.append(row)
    _write_csv(rows, DATA / f"design_map_{RUN_ID}.csv")
    return rows


def sweep_half_angle() -> list[dict]:
    courses = make_courses()
    course = courses[0]
    lo, hi = feasible_half_angle_range(ARM_LENGTH)
    console.rule(f"half-angle sweep — t ∈ [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°], "
                 f"{args.turn_deg:.0f}° slalom at {args.speed} m/s")
    console.log(f"course: R={course.radius:.2f} m, a_lat={course.lateral_acceleration(args.speed):.1f} m/s², "
                f"traversal {course.traversal_time(args.speed, STARTUP_TIME):.2f} s")
    rows = []
    for t in np.linspace(lo, hi, args.points):
        g = make_genome(float(t))
        if rotors_overlap(g):
            console.log(f"  t={np.degrees(t):>5.1f}°  [red]OVERLAP[/red]")
            continue
        t0 = time.time()
        r = evaluate(g, courses, args.speed)
        rows.append({"half_angle_deg": float(np.degrees(t)), **r})
        console.log(
            f"  t={np.degrees(t):>5.1f}°  fit={r['fitness']:8.3f}  gates={r['gates']:>2}/{args.n_gates}  "
            f"trk={r['tracking_err']:6.3f}  sat={r['saturation']*100:5.1f}%  "
            f"α_roll={r['alpha_roll']:6.1f}  ({time.time()-t0:.1f}s)")
    _write_csv(rows, DATA / f"half_angle_{RUN_ID}.csv")
    return rows


def sweep_arm_length() -> list[dict]:
    courses = make_courses()
    console.rule(f"arm-length sweep — L ∈ [{min_arm_length():.3f}, 0.25] m, t = 45°")
    rows = []
    for L in np.linspace(min_arm_length(), 0.25, args.points):
        g = make_genome(np.pi / 4, arm_length=float(L))
        if rotors_overlap(g) or rotor_hits_body(g):
            console.log(f"  L={L:.3f}  [red]INFEASIBLE[/red]")
            continue
        r = evaluate(g, courses, args.speed)
        rows.append({"arm_length": float(L), **r})
        console.log(
            f"  L={L:.3f} m  fit={r['fitness']:8.3f}  gates={r['gates']:>2}/{args.n_gates}  "
            f"trk={r['tracking_err']:6.3f}  sat={r['saturation']*100:5.1f}%  "
            f"α_roll={r['alpha_roll']:6.1f}  λ_I={r['maneuverability_I']:.3g}")
    _write_csv(rows, DATA / f"arm_length_{RUN_ID}.csv")
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    console.log(f"  → {path}")


if __name__ == "__main__":
    if args.max_speed_sweep:
        max_speed_sweep()
    elif args.calibrate:
        calibrate()
    elif args.tracking_check:
        tracking_check()
    elif args.map_only:
        design_map()
    else:
        design_map()
        sweep_half_angle()
        sweep_arm_length()
    console.log("[bold green]done[/bold green]")

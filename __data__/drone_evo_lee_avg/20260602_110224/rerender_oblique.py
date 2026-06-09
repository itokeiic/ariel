"""Re-render the side-by-side comparison video from a less top-down angle,
using artifacts already saved in this run directory.

No EA re-run — just rollouts + render. Produces comparison_flight_oblique.mp4.
"""
from __future__ import annotations
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "goal_generator_ltu" / "polynomial_goal_generator"))
sys.path.insert(0, str(_REPO / "examples" / "d_drones"))

from _ctrl_helpers import GateChecker

from ariel.body_phenotypes.drone.backends import (
    blueprint_to_mjspec,
    blueprint_to_propellers,
)
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.controllers.lee_control.lee_controller import LeeGeometricControl
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import Trajectory
from ariel.simulation.drone.controllers.utils.wind_model import Wind
from ariel.simulation.drone.drone_interface import DroneInterface
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.video_recorder import VideoRecorder

# ── settings (match example 17) ──────────────────────────────────────────────
RUN_DIR        = Path(__file__).parent
RUN_TAG        = "20260602_110224"
TRAJ_TOTAL_TIME = 5.0   # open-spline traversal time (gentle pace gives a clean gate-0 fly-in)
SIM_TIME       = 20.0    # rollout window > TRAJ_TOTAL_TIME so the final gate (reached ~t=TRAJ_TOTAL_TIME) is crossed before the rollout ends; terminates early at the last gate anyway
SIM_DT         = 0.005
LEE_POS_GAIN   = 14.3
LEE_VEL_GAIN   = 9.0
GATE_SIZE      = 1.0
GATE_PLANE_EPS = 0.02   # 2 cm tolerance for near-plane crossing in rerender
GATE_ORIENTATION_TOL_DEG = 5.0
GATE_PROXIMITY_RADIUS = 0.0   # m; proximity-fallback pass radius for hairpin/overlapping gates (well inside the 0.5 m half-opening)
WRITE_GATE_DEBUG_CSV = True
DEBUG_GATE_CSV_CIRCUIT = 0  # zero-based circuit index; 0 -> circuit 1/10
FORCE_STRAIGHT_GATE0_APPROACH = True
GATE0_APPROACH_TIME = 1.5       # seconds: spawn -> gate0 linear approach
PROP_SIZE      = 2
N_TRAJECTORIES = 10
CANONICAL_ARM_LEN = 0.5 * (0.055 + 0.17)   # midpoint of PARAMETER_LIMITS[0]

# ── NEW camera — TOP-DOWN view ───────────────────────────────────────────────
CAM_AZIMUTH   = 90.0    # azimuth largely irrelevant when looking straight down
CAM_ELEVATION = -85.0   # nearly straight down (avoid −90° gimbal lock)
CAM_DISTANCE  = 18.0    # tighter zoom; circuit fits at this height
CAM_LOOKAT    = np.array([0.0, 0.0, 0.0])
START_HOLD_FRAMES = 30  # hold first frame per circuit (~1s at 30 fps)
DRAW_GATE_IDS = False
GATE_ID_FONT_SCALE = 0.45

# Use physical 3D gate-number tags (seven-segment boxes) so labels stay
# attached to gate locations regardless of camera projection.
DRAW_3D_GATE_IDS = True

HALF_W, HALF_H = 720, 540


class _QuinticGateConfig:
    def __init__(self, gp, gy, gs, sp):
        self.gate_pos     = np.asarray(gp)
        self.gate_yaw     = np.asarray(gy)
        self.gate_size    = float(gs)
        self.starting_pos = np.asarray(sp)
        # Open (clamped) B-spline, not a closed loop: the course is run once from
        # gate 0 to the last gate and the rollout terminates as soon as the final
        # gate is crossed. A periodic spline would add an unnecessary return leg
        # that, on the short/dense circuits, drives a large overshoot (and a
        # ~180deg velocity-yaw flip that destabilises altitude). BSplineGate
        # Trajectory reads this via getattr(gate_config, "periodic", True).
        self.periodic     = False


# ── load saved blueprints + gate configs ─────────────────────────────────────
best_bp = DroneBlueprint.load_json(str(RUN_DIR / f"best_blueprint_{RUN_TAG}.json"))
canonical_arms = np.array([
    [CANONICAL_ARM_LEN, i * np.pi / 3.0, 0.0, 0.0, 0.0, float(i % 2)] for i in range(6)
])
canonical_bp = spherical_angular_to_blueprint(canonical_arms, propsize=PROP_SIZE)

best_propellers      = blueprint_to_propellers(best_bp,      convention="ned")
canonical_propellers = blueprint_to_propellers(canonical_bp, convention="ned")

def _start_behind_gate0(gp, gy, offset=1.0):
    """Start 1 m behind gate 0 along course direction (gate0->gate1).

    This aligns the initial straight-in segment with the first course leg.
    Falls back to gate yaw if only one gate is available.
    """
    gate0 = np.asarray(gp[0], dtype=np.float64)
    if len(gp) > 1:
        d01 = np.asarray(gp[1], dtype=np.float64) - gate0
        d01[2] = 0.0
        n0 = d01 / max(np.linalg.norm(d01), 1e-12)
    else:
        n0 = np.array([np.cos(gy[0]), np.sin(gy[0]), 0.0], dtype=np.float64)
        n0 /= max(np.linalg.norm(n0), 1e-12)
    return gate0 - float(offset) * n0


def gate0_approach_normal(gate_cfg):
    """Unit XY normal defining the intended gate-0 approach direction."""
    gp = np.asarray(gate_cfg.gate_pos, dtype=np.float64)
    if len(gp) > 1:
        v = gp[1] - gp[0]
        v[2] = 0.0
        return v / max(np.linalg.norm(v), 1e-12)
    gy0 = float(gate_cfg.gate_yaw[0])
    return np.array([np.cos(gy0), np.sin(gy0), 0.0], dtype=np.float64)

gate_cfgs = []
for k in range(N_TRAJECTORIES):
    gp = np.load(RUN_DIR / f"gate_pos_{RUN_TAG}_{k:02d}.npy")
    gy = np.load(RUN_DIR / f"gate_yaw_{RUN_TAG}_{k:02d}.npy")
    gate_cfgs.append(_QuinticGateConfig(
        gp, gy, GATE_SIZE, _start_behind_gate0(gp, gy),
    ))


# ── rollout + scene helpers (copied from example 17's video block) ──────────
def build_lee_stack(propellers, gate_cfg):
    drone = DroneInterface(0, propellers=propellers)
    wind = Wind("None")
    ctrl = LeeGeometricControl(
        drone, yawType=1, orient="NED", auto_scale_gains=True,
        pos_P_gain=np.array([LEE_POS_GAIN] * 3),
        vel_P_gain=np.array([LEE_VEL_GAIN] * 3),
    )
    traj = Trajectory(drone, "xyz_pos", np.array([15, 3, 1]), gate_config=gate_cfg)
    # Finish the (open) traversal before the sim ends. The non-periodic spline
    # reaches the final gate exactly at total_time; leaving it at SIM_TIME means
    # the last gate is crossed right at the rollout's end and is missed. Pull
    # total_time in so the last gate is reached with a margin to spare.
    traj.bspline_trajectory.total_time = TRAJ_TOTAL_TIME
    # Override the spawn: place the drone 1 m BEHIND gate 0 along the COURSE
    # direction (gate 0 -> gate 1), facing along that same course direction.
    # The fly-in is then a straight shot through gate 0's centre, which keeps the
    # lateral error small enough to register the pass. Facing the gate_yaw normal
    # instead (as before) made the drone enter at an angle on circuits where the
    # gate yaw is not aligned with the course, grazing the gate edge (~1-5 cm out)
    # and missing gate 0. The Lee controller then pulls it onto the B-spline path.
    start_pos = np.asarray(gate_cfg.starting_pos, dtype=np.float64)
    incoming_heading = gate0_approach_normal(gate_cfg)
    initial_yaw = float(np.arctan2(incoming_heading[1], incoming_heading[0]))
    drone.drone_sim.set_state(
        position=start_pos, velocity=np.zeros(3),
        attitude=np.array([0.0, 0.0, initial_yaw]),
        angular_velocity=np.zeros(3),
    )
    drone._update_state_variables()
    checker = GateChecker(gate_cfg.gate_pos, gate_cfg.gate_yaw, gate_cfg.gate_size)
    # Seed signed-distance state at t=0 so a first-step crossing is not missed.
    check_gate_passing_eps(checker, drone)
    sDes = traj.desiredState(0.0, SIM_DT, drone)
    ctrl.controller(sDes, drone, traj.ctrlType, SIM_DT)
    return drone, ctrl, traj, checker, wind


def logged_rollout(propellers, gate_cfg, max_steps, debug_trace=False):
    drone, ctrl, traj, checker, wind = build_lee_stack(propellers, gate_cfg)
    n_gates   = checker.num_gates
    pos_ned   = np.zeros((max_steps, 3), dtype=np.float64)
    euler_log = np.zeros((max_steps, 3), dtype=np.float64)
    gate_debug_rows = []

    t, i = 0.0, 0
    try:
        while i < max_steps:
            pos_ned[i]   = drone.pos
            euler_log[i] = drone.euler
            drone.update(t, SIM_DT, ctrl.w_cmd, wind)
            t_new = SIM_DT * (i + 1)
            sDes = traj.desiredState(t_new, SIM_DT, drone)
            ctrl.controller(sDes, drone, traj.ctrlType, SIM_DT)

            passed, info = check_gate_passing_eps(checker, drone, return_info=True)

            if debug_trace:
                gate_debug_rows.append({
                    "step": i,
                    "t": t_new,
                    "x": float(drone.pos[0]),
                    "y": float(drone.pos[1]),
                    "z": float(drone.pos[2]),
                    "gate_index_checked": int(info["gate_index_checked"]),
                    "prev_signed_dist": info["prev_signed_dist"],
                    "signed_dist": info["signed_dist"],
                    "crossed_plane": int(info["crossed_plane"]),
                    "lateral_err": info["lateral_err"],
                    "vertical_err": info["vertical_err"],
                    "orientation_err_deg": info["orientation_err_deg"],
                    "passed": int(info["passed"]),
                    "next_gate_after": int(info["next_gate_after"]),
                    "gates_passed_total": int(checker.gates_passed),
                })
            t = t_new
            i += 1
            if checker.gates_passed >= n_gates:
                break
    except Exception:
        pass
    if debug_trace:
        return pos_ned[:i], euler_log[:i], checker.gates_passed, gate_debug_rows
    return pos_ned[:i], euler_log[:i], checker.gates_passed


def gate_target_normal(gate_cfg, gate_index):
    gate_y = float(gate_cfg.gate_yaw[gate_index])
    normal = np.array([np.cos(gate_y), np.sin(gate_y), 0.0], dtype=np.float64)
    return normal / max(np.linalg.norm(normal), 1e-12)


def drone_forward_axis_world(drone):
    forward = np.asarray(drone.dcm[:, 0], dtype=np.float64)
    return forward / max(np.linalg.norm(forward), 1e-12)


def horizontal_alignment_error_deg_to_incoming(drone_forward, gate_normal):
    """Yaw-plane alignment error to incoming direction (opposite gate normal)."""
    fxy = np.asarray(drone_forward[:2], dtype=np.float64)
    nxy = np.asarray(gate_normal[:2], dtype=np.float64)
    nf = np.linalg.norm(fxy)
    nn = np.linalg.norm(nxy)
    if nf < 1e-9 or nn < 1e-9:
        return 180.0
    fxy /= nf
    nxy /= nn
    angle_to_normal = float(np.degrees(np.arccos(np.clip(np.dot(fxy, nxy), -1.0, 1.0))))
    return abs(180.0 - angle_to_normal)


def _course_aligned_normal(checker, gate_index):
    """Gate-plane normal oriented along the local course direction (gate i ->
    i+1). gate_yaw alone is sometimes anti-aligned with the course; orienting it
    keeps a single forward pass crossing the plane in a consistent sense and
    leaves the opening geometry (lateral/vertical errors) unchanged."""
    gate_p = np.asarray(checker.gate_pos[gate_index], dtype=np.float64)
    gate_y = float(checker.gate_yaw[gate_index])
    normal = np.array([np.cos(gate_y), np.sin(gate_y), 0.0], dtype=np.float64)
    _gps = checker.gate_pos
    if gate_index < len(_gps) - 1:
        _course = np.asarray(_gps[gate_index + 1], dtype=np.float64) - gate_p
    else:
        _course = gate_p - np.asarray(_gps[gate_index - 1], dtype=np.float64)
    if float(np.dot(normal[:2], _course[:2])) < 0.0:
        normal = -normal
    return gate_p, normal


def _frame_errs(pt, gate_p, normal):
    """Signed plane distance + in-opening lateral / vertical offsets of a point."""
    s = float(np.dot(pt - gate_p, normal))
    lat = float(np.linalg.norm(pt[:2] - gate_p[:2] - s * normal[:2]))
    vert = float(abs(pt[2] - gate_p[2]))
    return s, lat, vert


def check_gate_passing_eps(checker, drone, eps=GATE_PLANE_EPS, return_info=False):
    """Frame-based gate-pass detector (direction-agnostic, step-size robust).

    A gate counts as passed when the drone's trajectory segment between the
    previous and current sample intersects the gate plane *inside* the ±0.5 m
    opening — regardless of crossing direction.

    This replaces the strict one-directional (- -> +) crossing test, which
    locked the sequential checker on densely spaced courses: the drone is often
    already on the far side of gate i's plane by the time gate i becomes the
    active target, so a - -> + crossing never recurs and every later gate is
    frozen out (e.g. circuit 04 gate 13, circuit 07 gate 3) even though the drone
    flew dead-centre through the opening. Intersecting the segment with the
    opening counts the actual fly-through; a catch-up loop advances through every
    gate a single segment clears (handles gates crossed within one step of each
    other). No centre-radius proximity hack is required.
    """
    pos = np.asarray(drone.pos, dtype=np.float64)
    prev_pos = getattr(checker, "_prev_pos", None)
    entry_gate = int(checker._next_gate)

    passed_any = False
    info_signed = info_lat = info_vert = info_orient = None
    info_crossed = False

    # Walk the current segment [prev_pos, pos] against the active gate, advancing
    # through every gate whose opening the segment passes through (catch-up).
    while checker._next_gate < checker.num_gates:
        gi = int(checker._next_gate)
        gate_p, normal = _course_aligned_normal(checker, gi)
        s_cur = float(np.dot(pos - gate_p, normal))
        if info_signed is None:
            info_signed = s_cur
        if prev_pos is None:
            break  # need a previous sample to form a segment

        s_prev = float(np.dot(prev_pos - gate_p, normal))
        hit = False

        # Primary: does the segment straddle the plane (any direction)?
        if (s_prev <= 0.0 <= s_cur) or (s_cur <= 0.0 <= s_prev):
            info_crossed = True
            denom = s_prev - s_cur
            f = min(max((s_prev / denom) if denom != 0.0 else 0.0, 0.0), 1.0)
            cross_pt = prev_pos + f * (pos - prev_pos)
            _, lat, vert = _frame_errs(cross_pt, gate_p, normal)
            if lat <= checker.gate_size / 2.0 and vert <= checker.gate_size / 2.0:
                hit = True
                info_lat, info_vert = lat, vert
                info_orient = horizontal_alignment_error_deg_to_incoming(
                    drone_forward_axis_world(drone), normal)

        # Grazing fallback: a sample sits inside the opening essentially on the
        # plane without a clean straddle (sampling artefact on a tangential pass).
        if not hit:
            _, lat_now, vert_now = _frame_errs(pos, gate_p, normal)
            if (abs(s_cur) <= float(eps)
                    and lat_now <= checker.gate_size / 2.0
                    and vert_now <= checker.gate_size / 2.0):
                hit = True
                info_lat, info_vert = lat_now, vert_now

        # Legacy centre-radius proximity fallback (no-op while radius is 0).
        if not hit and GATE_PROXIMITY_RADIUS > 0.0:
            if float(np.linalg.norm(pos - gate_p)) <= GATE_PROXIMITY_RADIUS:
                hit = True
                _, info_lat, info_vert = _frame_errs(pos, gate_p, normal)

        if not hit:
            break

        checker.gates_passed += 1
        checker._next_gate = (checker._next_gate + 1) % checker.num_gates
        passed_any = True
        if checker._next_gate == 0:  # wrapped -> non-periodic course complete
            break

    checker._prev_pos = pos
    checker._prev_signed_dist = info_signed  # kept for any external readers

    if return_info:
        return passed_any, {
            "gate_index_checked": entry_gate,
            "prev_signed_dist": None,
            "signed_dist": info_signed,
            "crossed_plane": bool(info_crossed),
            "lateral_err": info_lat,
            "vertical_err": info_vert,
            "orientation_err_deg": info_orient,
            "passed": bool(passed_any),
            "next_gate_after": int(checker._next_gate),
        }
    return passed_any


def write_gate_debug_csv(run_dir, run_tag, circuit_idx, label_name, rows):
    if not rows:
        return
    out_dir = run_dir / "debug_gate_csv"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"gate_debug_{run_tag}_circuit_{circuit_idx:02d}_{label_name}.csv"
    fieldnames = [
        "step",
        "t",
        "x",
        "y",
        "z",
        "gate_index_checked",
        "prev_signed_dist",
        "signed_dist",
        "crossed_plane",
        "lateral_err",
        "vertical_err",
        "orientation_err_deg",
        "passed",
        "next_gate_after",
        "gates_passed_total",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  wrote gate-debug CSV: {out}")


def write_gate_debug_events_csv(run_dir, run_tag, circuit_idx, label_name, rows):
    """Write only noteworthy gate-check rows to simplify manual audits."""
    if not rows:
        return
    event_rows = [
        row for row in rows
        if int(row["crossed_plane"]) == 1 or int(row["passed"]) == 1
    ]
    out_dir = run_dir / "debug_gate_csv"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"gate_debug_events_{run_tag}_circuit_{circuit_idx:02d}_{label_name}.csv"
    fieldnames = [
        "step",
        "t",
        "x",
        "y",
        "z",
        "gate_index_checked",
        "prev_signed_dist",
        "signed_dist",
        "crossed_plane",
        "lateral_err",
        "vertical_err",
        "orientation_err_deg",
        "passed",
        "next_gate_after",
        "gates_passed_total",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in event_rows:
            w.writerow(row)
    print(f"  wrote gate-event CSV: {out} (rows={len(event_rows)})")


def straight_gate0_sdes(gate_cfg, t):
    """Desired-state vector for a straight-line spawn->gate0 approach."""
    start = np.asarray(gate_cfg.starting_pos, dtype=np.float64)
    gate0 = np.asarray(gate_cfg.gate_pos[0], dtype=np.float64)
    n0 = gate0_approach_normal(gate_cfg)
    yaw0 = float(np.arctan2(n0[1], n0[0]))

    T = max(float(GATE0_APPROACH_TIME), 1e-6)
    alpha = min(max(float(t) / T, 0.0), 1.0)
    pos = (1.0 - alpha) * start + alpha * gate0
    vel = (gate0 - start) / T if alpha < 1.0 else np.zeros(3, dtype=np.float64)

    sDes = np.zeros(19, dtype=np.float64)
    sDes[0:3] = pos
    sDes[3:6] = vel
    sDes[12:15] = np.array([0.0, 0.0, yaw0])
    return sDes


def ned_to_enu(pos_ned, euler_log):
    pos_enu = pos_ned.copy(); pos_enu[:, 2] = -pos_enu[:, 2]
    phi, theta, psi = euler_log[:, 0], euler_log[:, 1], euler_log[:, 2]
    cy, sy = np.cos(psi/2), np.sin(psi/2)
    cp, sp = np.cos(theta/2), np.sin(theta/2)
    cr, sr = np.cos(phi/2), np.sin(phi/2)
    quat_enu = np.stack([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        -(cr*sp*cy + sr*cp*sy),
        cr*cp*sy - sr*sp*cy,
    ], axis=-1)
    return pos_enu, quat_enu


def mean_arm(bp):
    ls = [bp.payload(a).length for a in bp.children(bp.root_id)]
    return float(np.mean(ls)) if ls else 0.06


def build_world(bp, gate_cfg, init_pos_enu, m_arm, body_name):
    spec = blueprint_to_mjspec(bp, motor_mass=0.01, arm_mass=0.034 * m_arm, body_name=body_name)
    world = SimpleFlatWorld()
    world.spawn(spec,
                position=(float(init_pos_enu[0]), float(init_pos_enu[1]), float(init_pos_enu[2])),
                correct_collision_with_floor=False)
    for gi in range(len(gate_cfg.gate_pos)):
        gx = float(gate_cfg.gate_pos[gi, 0])
        gy = float(gate_cfg.gate_pos[gi, 1])
        gz = float(-gate_cfg.gate_pos[gi, 2])
        yaw = float(gate_cfg.gate_yaw[gi])
        qw, qz = math.cos(yaw/2.0), math.sin(yaw/2.0)
        gb = world.spec.worldbody.add_body(name=f"gate_{gi}", pos=[gx, gy, gz], quat=[qw, 0, 0, qz])
        # Half-extents = GATE_SIZE/2 so the drawn opening exactly matches the
        # pass window (lat/vert <= gate_size/2 in check_gate_passing_eps); the
        # visible frame and the scoring criterion stay in sync.
        _gh = GATE_SIZE / 2.0
        gb.add_geom(name=f"gate_plane_{gi}",  type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.01, _gh, _gh], rgba=(1.0, 0.55, 0.0, 0.25),
                    contype=0, conaffinity=0)
        gb.add_geom(name=f"gate_marker_{gi}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.06, 0.0, 0.0],   rgba=(1.0, 0.1, 0.1, 1.0),
                    contype=0, conaffinity=0)
        if DRAW_3D_GATE_IDS:
            add_gate_index_tag(gb, gi)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    return model, data


def set_pose(model, data, p_enu, q_enu):
    data.qpos[0:3] = [float(p_enu[0]), float(p_enu[1]), float(p_enu[2])]
    data.qpos[3:7] = [float(q_enu[0]), float(q_enu[1]), float(q_enu[2]), float(q_enu[3])]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def label(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


_SEGMENTS_BY_DIGIT = {
    0: ("a", "b", "c", "d", "e", "f"),
    1: ("b", "c"),
    2: ("a", "b", "g", "e", "d"),
    3: ("a", "b", "g", "c", "d"),
    4: ("f", "g", "b", "c"),
    5: ("a", "f", "g", "c", "d"),
    6: ("a", "f", "g", "e", "c", "d"),
    7: ("a", "b", "c"),
    8: ("a", "b", "c", "d", "e", "f", "g"),
    9: ("a", "b", "c", "d", "f", "g"),
}


def add_digit_7seg(body, prefix, digit, y_offset, z_offset):
    """Add one seven-segment digit to a gate body in local gate coordinates.

    Important: digits are built in the local XY plane (not YZ), so they remain
    visible under near top-down cameras where Z variation is visually collapsed.
    """
    seg_len = 0.10
    seg_thick = 0.018
    z_thick = 0.004

    # Segment centers in local (x, y), with constant z.
    xc = 0.0
    yc = y_offset
    dx = 0.06
    dy = 0.09
    seg_centers = {
        "a": (xc, yc + dy),
        "b": (xc + dx, yc + dy / 2),
        "c": (xc + dx, yc - dy / 2),
        "d": (xc, yc - dy),
        "e": (xc - dx, yc - dy / 2),
        "f": (xc - dx, yc + dy / 2),
        "g": (xc, yc),
    }

    for seg in _SEGMENTS_BY_DIGIT[int(digit)]:
        sx, sy = seg_centers[seg]
        if seg in ("a", "d", "g"):
            size = [seg_len / 2, seg_thick / 2, z_thick / 2]
        else:
            size = [seg_thick / 2, seg_len / 2, z_thick / 2]
        body.add_geom(
            name=f"{prefix}_{seg}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[float(sx), float(sy), float(z_offset)],
            size=size,
            rgba=(1.0, 0.95, 0.2, 1.0),
            contype=0,
            conaffinity=0,
        )


def add_gate_index_tag(body, gate_index):
    """Add one- or two-digit 3D index tag above a gate."""
    z_tag = 0.55
    if gate_index < 10:
        add_digit_7seg(body, f"gid{gate_index}_ones", gate_index, y_offset=0.0, z_offset=z_tag)
        return

    tens = gate_index // 10
    ones = gate_index % 10
    add_digit_7seg(body, f"gid{gate_index}_tens", tens, y_offset=-0.11, z_offset=z_tag)
    add_digit_7seg(body, f"gid{gate_index}_ones", ones, y_offset=0.11, z_offset=z_tag)


def _panel_bbox(panel_img):
    """Find visible render area inside a panel by excluding flat background."""
    bg = panel_img[0, 0].astype(np.int16)
    diff = np.abs(panel_img.astype(np.int16) - bg)
    mask = np.any(diff > 8, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, panel_img.shape[1] - 1, panel_img.shape[0] - 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _gate_uv(gate_xy):
    """Top-down camera-plane coordinates from world XY, matching camera azimuth."""
    theta = np.deg2rad(float(CAM_AZIMUTH))
    c, s = np.cos(theta), np.sin(theta)
    xy = np.asarray(gate_xy, dtype=np.float64) - np.asarray(CAM_LOOKAT[:2], dtype=np.float64)
    u = c * xy[:, 0] + s * xy[:, 1]
    v = -s * xy[:, 0] + c * xy[:, 1]
    return u, v


def draw_gate_ids(frame, gate_cfg, x_offset, bbox):
    """Draw gate indices near gate centers on one panel of the combined frame."""
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return

    gate_xy = np.asarray(gate_cfg.gate_pos)[:, :2]
    u, v = _gate_uv(gate_xy)
    du = max(float(u.max() - u.min()), 1e-6)
    dv = max(float(v.max() - v.min()), 1e-6)
    pad_u = 0.08 * du
    pad_v = 0.08 * dv
    u_min, u_max = float(u.min() - pad_u), float(u.max() + pad_u)
    v_min, v_max = float(v.min() - pad_v), float(v.max() + pad_v)

    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)
    for gi, (uu, vv) in enumerate(zip(u, v)):
        px = int(x0 + (uu - u_min) / max(u_max - u_min, 1e-6) * w)
        py = int(y1 - (vv - v_min) / max(v_max - v_min, 1e-6) * h)
        txt = str(gi)
        cv2.putText(frame, txt, (x_offset + px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, GATE_ID_FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (x_offset + px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, GATE_ID_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


# ── render ───────────────────────────────────────────────────────────────────
# --no-video: run the rollouts and regenerate the comparison bar chart only,
# skipping the (slow) MuJoCo frame rendering and the .mp4 entirely.
NO_VIDEO = "--no-video" in sys.argv
FPS = 30
mp4 = RUN_DIR / f"comparison_flight_topdown_{RUN_TAG}.mp4"
recorder = None
if not NO_VIDEO:
    recorder = VideoRecorder(file_name=mp4.stem, output_folder=mp4.parent,
                             width=HALF_W * 2, height=HALF_H, fps=FPS)
steps_per_frame = max(1, int(round(1.0 / (FPS * SIM_DT))))

canonical_gates = []
evolved_gates = []

cam = mujoco.MjvCamera()
cam.lookat[:]  = CAM_LOOKAT
cam.azimuth    = CAM_AZIMUTH
cam.elevation  = CAM_ELEVATION
cam.distance   = CAM_DISTANCE

can_mean_arm  = mean_arm(canonical_bp)
best_mean_arm = mean_arm(best_bp)
max_steps     = int(SIM_TIME / SIM_DT) + 1

t0 = time.time()
for k, gate_cfg in enumerate(gate_cfgs):
    print(f"  {'rollout' if NO_VIDEO else 'rendering'} circuit {k+1}/{N_TRAJECTORIES} …")
    debug_this = WRITE_GATE_DEBUG_CSV and (k == DEBUG_GATE_CSV_CIRCUIT)
    if debug_this:
        can_pos_ned, can_eul, can_gates, can_debug_rows = logged_rollout(
            canonical_propellers, gate_cfg, max_steps, debug_trace=True,
        )
        evo_pos_ned, evo_eul, evo_gates, evo_debug_rows = logged_rollout(
            best_propellers, gate_cfg, max_steps, debug_trace=True,
        )
        write_gate_debug_csv(RUN_DIR, RUN_TAG, k, "canonical", can_debug_rows)
        write_gate_debug_csv(RUN_DIR, RUN_TAG, k, "evolved", evo_debug_rows)
        write_gate_debug_events_csv(RUN_DIR, RUN_TAG, k, "canonical", can_debug_rows)
        write_gate_debug_events_csv(RUN_DIR, RUN_TAG, k, "evolved", evo_debug_rows)
    else:
        can_pos_ned, can_eul, can_gates = logged_rollout(canonical_propellers, gate_cfg, max_steps)
        evo_pos_ned, evo_eul, evo_gates = logged_rollout(best_propellers,      gate_cfg, max_steps)

    canonical_gates.append(float(can_gates))
    evolved_gates.append(float(evo_gates))

    if NO_VIDEO:
        continue
    if len(can_pos_ned) < 2 or len(evo_pos_ned) < 2:
        continue
    can_pos_enu, can_quat = ned_to_enu(can_pos_ned, can_eul)
    evo_pos_enu, evo_quat = ned_to_enu(evo_pos_ned, evo_eul)

    can_model, can_data = build_world(canonical_bp, gate_cfg, can_pos_enu[0], can_mean_arm, "canonical")
    evo_model, evo_data = build_world(best_bp,      gate_cfg, evo_pos_enu[0], best_mean_arm, "evolved")

    gate0_normal = gate0_approach_normal(gate_cfg)

    def signed_to_gate0_ned(pos_ned):
        return float(np.dot(pos_ned - gate_cfg.gate_pos[0], gate0_normal))

    n_frames = max(len(can_pos_enu), len(evo_pos_enu))
    with mujoco.Renderer(can_model, width=HALF_W, height=HALF_H) as r_can, \
         mujoco.Renderer(evo_model, width=HALF_W, height=HALF_H) as r_evo:
        # Hold the exact spawn frame to make initial geometry obvious.
        set_pose(can_model, can_data, can_pos_enu[0], can_quat[0])
        set_pose(evo_model, evo_data, evo_pos_enu[0], evo_quat[0])
        r_can.update_scene(can_data, camera=cam)
        r_evo.update_scene(evo_data, camera=cam)
        start_frame = np.hstack([r_can.render(), r_evo.render()]).copy()
        if DRAW_GATE_IDS:
            left_bbox = _panel_bbox(start_frame[:, :HALF_W])
            right_bbox = _panel_bbox(start_frame[:, HALF_W:])
            draw_gate_ids(start_frame, gate_cfg, 0, left_bbox)
            draw_gate_ids(start_frame, gate_cfg, HALF_W, right_bbox)
        label(start_frame, "CANONICAL HEX", 20, 35)
        label(start_frame, "EVOLVED",       HALF_W + 20, 35)
        label(start_frame, f"circuit {k+1}/{N_TRAJECTORIES}  START", HALF_W - 140, HALF_H - 20)
        label(start_frame, f"d0_can={signed_to_gate0_ned(can_pos_ned[0]):+.2f} m", 20, 65)
        label(start_frame, f"d0_evo={signed_to_gate0_ned(evo_pos_ned[0]):+.2f} m", HALF_W + 20, 65)
        for _ in range(START_HOLD_FRAMES):
            recorder.write(frame=start_frame)

        for idx in range(0, n_frames, steps_per_frame):
            ci = min(idx, len(can_pos_enu) - 1)
            ei = min(idx, len(evo_pos_enu) - 1)
            set_pose(can_model, can_data, can_pos_enu[ci], can_quat[ci])
            set_pose(evo_model, evo_data, evo_pos_enu[ei], evo_quat[ei])
            r_can.update_scene(can_data, camera=cam)
            r_evo.update_scene(evo_data, camera=cam)
            combined = np.hstack([r_can.render(), r_evo.render()]).copy()
            if DRAW_GATE_IDS:
                draw_gate_ids(combined, gate_cfg, 0, left_bbox)
                draw_gate_ids(combined, gate_cfg, HALF_W, right_bbox)
            label(combined, "CANONICAL HEX", 20, 35)
            label(combined, "EVOLVED",       HALF_W + 20, 35)
            label(combined, f"circuit {k+1}/{N_TRAJECTORIES}  (can={can_gates}, evo={evo_gates})",
                  HALF_W - 200, HALF_H - 20)
            label(combined, f"d0_can={signed_to_gate0_ned(can_pos_ned[ci]):+.2f} m", 20, 65)
            label(combined, f"d0_evo={signed_to_gate0_ned(evo_pos_ned[ei]):+.2f} m", HALF_W + 20, 65)
            recorder.write(frame=combined)

if recorder is not None:
    recorder.release()

try:
    import matplotlib.pyplot as plt

    x = np.arange(len(canonical_gates) + 1)
    can_bars = list(canonical_gates) + [float(np.mean(canonical_gates))]
    evo_bars = list(evolved_gates) + [float(np.mean(evolved_gates))]
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, can_bars, width, label="canonical hex", color="gray")
    ax.bar(x + width / 2, evo_bars, width, label="evolved best", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i:02d}" for i in range(len(canonical_gates))] + ["mean"])
    ax.set_xlabel("Circuit")
    ax.set_ylabel("Gates passed")
    ax.set_title("Canonical hexacopter vs evolved best - gates passed per circuit")
    ax.legend()
    ax.grid(True, alpha=0.4, axis="y")
    plt.tight_layout()

    cmp_png = RUN_DIR / f"comparison_gates_{RUN_TAG}.png"
    fig.savefig(cmp_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote comparison bar chart: {cmp_png}")
except Exception as exc:
    print(f"  comparison bar chart skipped: {exc}")

print(f"\nDone in {time.time()-t0:.1f}s → {'(chart only, no video)' if NO_VIDEO else mp4}")

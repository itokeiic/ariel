#!/usr/bin/env python3
"""Does the gyroscopic yaw torque explain the regenerated Table 1? (fixed plant)

    uv run --no-sync python docs/tools/gyroscopic_mechanism_fixed_plant.py [OUT.md]

docs/data/gyroscopic_mechanism.md tested the mechanism on the plant WITHOUT the
gyroscopic term (commit 8d3174b), with hover-tangent yaw drag. Since 4229360 the
plant integrates the term, rotor drag is k_m W^2, and Table 1 was regenerated on
that plant, so the mechanism is examined here on the plant that produced the
results. The plant is never patched; section 3 patches the controller.

Per controller step, for every Table 1 body:

  g_z       yaw component of -Omega x I Omega, the gyroscopic torque acting
  tau_need  the yaw torque the attitude law demands, excluding its
            +Omega x I Omega cancellation
  tau_z     the yaw torque the mixer is actually asked for (the law's output,
            cancellation included unless section 3 removes it)
  cap       the largest yaw torque, in a given direction, that the controller's
            own allocation can add to the commanded thrust, roll and pitch
            without leaving the motor limits [minWmotor, maxWmotor]

The output has three sections, all produced by one run.

1. PRE-REGISTERED 2026-09-16, before any data. Kept exactly as registered.
   Speeds: 60 deg 9.9, 90 deg 5.5, 120 deg 5.25 m/s -- just below the slowest
   body's Table 1 limit, so every body completes (asserted).
   (A) correlation. SUPPORTED if corr(g_z, tau_need) <= -0.3 for t=20.44 and
       28.63 AND >= +0.3 for t=61.37 and 69.56, at every turn. REFUTED if narrow
       and wide share a sign, or |corr| < 0.1 on either side. Else INCONCLUSIVE.
   (B) saturation. At 90 and 120 deg the share of steps with |tau_need| > cap
       (cap in tau_need's direction) is larger at t=20.44 than at both t=45.00
       and t=69.56. SUPPORTED at both turns; REFUTED at neither; INCONCLUSIVE
       if every share at a turn is < 1%.
   Found after seeing the data: (B) tests the demand without the cancellation,
   but the motors must deliver tau_z, cancellation included.

2. EXPLORATORY (post hoc, 2026-09-16; NOT a test). Section 1's flights with the
   share of steps where |tau_z| > cap (cap in tau_z's direction). It confounds
   geometry with how hard each body is pushed: at 90 and 120 deg the common
   speed is ~99% of the narrowest body's limit but 67-79% of the others'.

3. PRE-REGISTERED 2026-09-17, before any data. Removes that confound.
   Every body flies at 95% of its own Table 1 limit, under the standard
   controller and under one with the +Omega x I Omega cancellation removed.
   Measure: share of steps with |tau_z| > cap (cap in tau_z's direction).
   (P1) equal push: with the standard controller, at 90 and 120 deg, the share
        at t=20.44 exceeds the shares at t=45.00 and t=69.56. SUPPORTED at both
        turns; REFUTED at neither; INCONCLUSIVE if every share at a turn < 1%.
   (P2) cause: removing the cancellation lowers the share at t=20.44 at both 90
        and 120 deg, while at t=45.00 (gyroscopic term ~0) the share changes by
        at most 2 percentage points at both. SUPPORTED if all four hold;
        REFUTED if the t=20.44 share decreases at neither turn; else
        INCONCLUSIVE.
   60 deg is reported without a prediction.

The sign flip of correlations at t=45 follows from the sign of Ixx - Iyy alone;
the substantive quantity is corr(pq, tau).
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parents[2]
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1
       else REPO / "docs" / "data" / "gyroscopic_mechanism_fixed_plant.md")
TABLE1 = REPO / "docs" / "data" / "tuned_strict_completion.csv"
SPEEDS = ((60.0, 9.9), (90.0, 5.5), (120.0, 5.25))   # section 1 and 2
FRACTION = 0.95                                        # section 3

# The sweep parses its CLI at import: bind the Table 1 operating point first.
sys.argv = ["sweep", "--att-omega-n", "24", "--pos-omega-n", "2.0", "--pos-zeta", "1.0",
            "--feedforward", "--max-accel", "40", "--completion", "strict"]
sys.path.insert(0, str(REPO / "examples" / "spear"))
spec = importlib.util.spec_from_file_location(
    "sweep", REPO / "examples" / "spear" / "19_morphology_design_sweep.py")
sweep = importlib.util.module_from_spec(spec)
sys.modules["sweep"] = sweep
spec.loader.exec_module(sweep)

import ariel.simulation.drone.controllers.lee_control.base_lee_controller as blc  # noqa: E402
import ariel.simulation.drone.controllers.lee_control.lee_controller as lc  # noqa: E402
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers  # noqa: E402
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint  # noqa: E402
from ariel.simulation.tasks.slalom_course import slalom_gates  # noqa: E402

# Per step: (g_z, tau_need, tau_z, cap in tau_need's direction, cap in tau_z's direction)
LOG: list[tuple[float, float, float, float, float]] = []
_last: dict[str, np.ndarray] = {}
errors: list[str] = []
STATE = {"cancel": True, "removed_calls": 0}


def thrust_command(ctrl, quad) -> float:
    """The mixer's thrust, reproduced from `_wrench_to_motor_commands`."""
    force = ctrl.wrench_command[0:3]
    dcm = getattr(quad, "dcm", None)
    body_z = (np.array([0.0, 0.0, 1.0]) if dcm is None
              else np.asarray(dcm, dtype=float)[:, 2])
    proj = float(np.dot(force, body_z))
    thrust = max(-proj if ctrl.orient == "NED" else proj, 0.0)
    return float(np.clip(thrust, quad.params["minThr"], quad.params["maxThr"]))


def yaw_capacity(quad, thrust: float, mx: float, my: float, sign: float) -> float:
    """Largest yaw torque in direction `sign` addable within the motor limits.

    Uses the controller's own allocation: base = [T, Mx, My, 0], yaw direction =
    [0, 0, 0, 1], both through mixerFMinv and the wmax^2 scaling. If the base
    alone already violates a limit there is no yaw room, and the capacity is 0.
    """
    fm_inv = np.asarray(quad.params["mixerFMinv"], dtype=float)
    wmax2 = np.array([p["wmax"] for p in quad.drone_sim.config.propellers], float) ** 2
    lo, hi = quad.params["minWmotor"] ** 2, quad.params["maxWmotor"] ** 2
    base = (fm_inv @ np.array([thrust, mx, my, 0.0])) * wmax2
    d = sign * (fm_inv @ np.array([0.0, 0.0, 0.0, 1.0])) * wmax2
    cap = np.inf
    for b, di in zip(base, d):
        if di > 0.0:
            cap = min(cap, (hi - b) / di)
        elif di < 0.0:
            cap = min(cap, (lo - b) / di)
    return float(max(cap, 0.0))


_orig_torque = blc.BaseLeeController.compute_body_torque


def _torque(self, *a, **kw):
    law = np.asarray(_orig_torque(self, *a, **kw), dtype=float)
    w = np.asarray(self.robot_body_angvel, dtype=float)
    gyro = np.cross(w, np.asarray(self.robot_inertia, float) @ w)
    # compute_body_torque returns -K_R e_R - K_W e_W + Omega x I Omega with no
    # later clipping, so subtracting the term removes the cancellation exactly.
    cmd = law if STATE["cancel"] else law - gyro
    if not STATE["cancel"]:
        STATE["removed_calls"] += 1
    _last.update(cmd=cmd.copy(), need=law - gyro, gyro=gyro)
    return cmd


_orig_mix = lc.LeeGeometricControl._wrench_to_motor_commands


def _mix(self, quad) -> None:
    _orig_mix(self, quad)
    try:
        if "cmd" not in _last:
            return
        cmd, need, gyro = _last["cmd"], _last["need"], _last["gyro"]
        # The mixer must be allocating exactly the torque recorded as tau_z.
        if not np.allclose(np.asarray(self.wrench_command[3:6], float), cmd, atol=1e-12):
            raise AssertionError("mixer moment != recorded yaw command")
        thrust = thrust_command(self, quad)
        cap_need = yaw_capacity(quad, thrust, float(cmd[0]), float(cmd[1]),
                                1.0 if need[2] >= 0.0 else -1.0)
        cap_cmd = yaw_capacity(quad, thrust, float(cmd[0]), float(cmd[1]),
                               1.0 if cmd[2] >= 0.0 else -1.0)
        LOG.append((float(-gyro[2]), float(need[2]), float(cmd[2]), cap_need, cap_cmd))
    except Exception as e:           # rollout() swallows exceptions; keep a record
        errors.append(repr(e))
        raise


blc.BaseLeeController.compute_body_torque = _torque
lc.LeeGeometricControl._wrench_to_motor_commands = _mix


def props_for(t_deg: float):
    return blueprint_to_propellers(
        spherical_angular_to_blueprint(sweep.make_genome(np.radians(t_deg)),
                                       propsize=sweep.PROP_SIZE),
        convention="ned")


def course_for(turn: float):
    return slalom_gates(turn, leg=sweep.args.leg, n_gates=sweep.args.n_gates,
                        gate_size=sweep.args.gate_size)


def fly(t_deg: float, turn: float, speed: float, *, cancel: bool = True):
    """One rollout; returns (completed, per-step arrays)."""
    STATE["cancel"] = cancel
    LOG.clear()
    out = sweep.rollout(props_for(t_deg), course_for(turn), speed)
    STATE["cancel"] = True
    assert not errors, errors[:3]
    assert len(LOG) > 100, "rollout did not run"
    return bool(out["completed_strict"]), np.array(LOG)


# Instrument check, outside rollout()'s blanket except: at hover (T = m g, no
# roll or pitch), the capacity must equal the independent derivation for a
# symmetric quad with alternating spins -- two rotors at the lower limit, two
# carrying the rest of the thrust -- (k_m/k_f) m g - 4 k_m minWmotor^2.
_drone, _ctrl, *_ = sweep._build_stack(props_for(45.0), course_for(90.0), 8.0)
_k_f, _k_m = _drone.drone_sim.config.propellers[0]["constants"]
_mg = float(_drone.drone_sim.mass) * float(_drone.drone_sim.g)
_expected = (_k_m / _k_f) * _mg - 4.0 * _k_m * float(_drone.params["minWmotor"]) ** 2
for _sign in (1.0, -1.0):
    _cap = yaw_capacity(_drone, _mg, 0.0, 0.0, _sign)
    assert abs(_cap - _expected) < 1e-9 * max(1.0, _expected), (_sign, _cap, _expected)
print(f"capacity check: hover yaw capacity {_expected:.5f} N m, both directions", flush=True)
LOG.clear()

limits = {(float(r["turn_deg"]), f"{float(r['half_angle_deg']):.2f}"): float(r["max_speed"])
          for r in csv.DictReader(open(TABLE1))}
halves = sorted({float(r["half_angle_deg"]) for r in csv.DictReader(open(TABLE1))})
lines = [
    "Generated by `uv run --no-sync python docs/tools/gyroscopic_mechanism_fixed_plant.py`",
    "on the plant with the gyroscopic term and quadratic rotor drag (plant unpatched).",
    "The criteria, and the definitions of g_z, tau_need, tau_z and cap, are in the",
    "tool's docstring.",
    "",
    "## 1. Pre-registered 2026-09-16: common speed per turn",
    "",
    "| turn | speed (m/s) | t (deg) | completed | corr(g_z, tau_need) | steps with abs(tau_need) > cap "
    "| rms g_z (N m) | rms tau_need (N m) | median cap (N m) |",
    "|---|---|---|---|---|---|---|---|---|",
]
explore = ["", "## 2. Exploratory (post hoc, not a test): the same flights, total yaw command", "",
           "Confounded: at 90 and 120 deg the common speed is ~99% of the narrowest body's own "
           "limit but 67-79% of the others'.", "",
           "| turn | speed (m/s) | t (deg) | percent of own limit | steps with abs(tau_z) > cap "
           "| rms tau_z (N m) | corr(g_z, tau_z) |",
           "|---|---|---|---|---|---|---|"]
corr: dict[tuple[float, str], float] = {}
sat: dict[tuple[float, str], float] = {}
for turn, v in SPEEDS:
    for t in halves:
        key = (turn, f"{t:.2f}")
        done, a = fly(t, turn, v)
        assert done, f"{turn} deg t={t:.2f} did not complete at {v} m/s"
        g, need, tz, cap_need, cap_cmd = a.T
        corr[key] = float(np.corrcoef(g, need)[0, 1])
        sat[key] = float(np.mean(np.abs(need) > cap_need))
        lines.append(
            f"| {turn:.0f}° | {v} | {t:.2f} | {done} | {corr[key]:+.3f} | {sat[key]:.1%} | "
            f"{np.sqrt((g ** 2).mean()):.4f} | {np.sqrt((need ** 2).mean()):.4f} | "
            f"{np.median(cap_need):.4f} |")
        explore.append(
            f"| {turn:.0f}° | {v} | {t:.2f} | {100 * v / limits[key]:.1f}% | "
            f"{np.mean(np.abs(tz) > cap_cmd):.1%} | {np.sqrt((tz ** 2).mean()):.4f} | "
            f"{np.corrcoef(g, tz)[0, 1]:+.3f} |")
        print(lines[-1], flush=True)

narrow = [corr[(turn, k)] for turn, _ in SPEEDS for k in ("20.44", "28.63")]
wide = [corr[(turn, k)] for turn, _ in SPEEDS for k in ("61.37", "69.56")]
if max(narrow) <= -0.3 and min(wide) >= 0.3:
    verdict_a = "SUPPORTED"
elif (np.sign(np.mean(narrow)) == np.sign(np.mean(wide))
      or min(np.abs(narrow)) < 0.1 or min(np.abs(wide)) < 0.1):
    verdict_a = "REFUTED"
else:
    verdict_a = "INCONCLUSIVE"


def narrowest_largest(share: dict, turn: float) -> str:
    s20, s45, s70 = (share[(turn, k)] for k in ("20.44", "45.00", "69.56"))
    if max(s20, s45, s70) < 0.01:
        return "inconclusive"
    return "yes" if (s20 > s45 and s20 > s70) else "no"


def tally(per_turn: list[str]) -> str:
    if all(p == "yes" for p in per_turn):
        return "SUPPORTED"
    if all(p == "no" for p in per_turn):
        return "REFUTED"
    return "INCONCLUSIVE"


per_turn_b = [narrowest_largest(sat, turn) for turn in (90.0, 120.0)]
verdict_b = tally(per_turn_b)
lines += [
    "",
    f"(A) correlation: **{verdict_a}** -- narrow {min(narrow):+.3f} to {max(narrow):+.3f}, "
    f"wide {min(wide):+.3f} to {max(wide):+.3f}.",
    f"(B) saturation: **{verdict_b}** -- t=20.44 largest share at 90° / 120°: "
    f"{per_turn_b[0]} / {per_turn_b[1]}.",
]
lines += explore

# ---- Section 3: pre-registered 2026-09-17 -------------------------------------
confirm = ["", "## 3. Pre-registered 2026-09-17: each body at 95% of its own limit", "",
           "| turn | t (deg) | speed (m/s) | standard: completed | standard: abs(tau_z) > cap "
           "| no cancellation: completed | no cancellation: abs(tau_z) > cap | change (pp) |",
           "|---|---|---|---|---|---|---|---|"]
share_std: dict[tuple[float, str], float] = {}
share_nc: dict[tuple[float, str], float] = {}
for turn in (60.0, 90.0, 120.0):
    for t in halves:
        key = (turn, f"{t:.2f}")
        v = FRACTION * limits[key]
        done_std, a = fly(t, turn, v, cancel=True)
        share_std[key] = float(np.mean(np.abs(a[:, 2]) > a[:, 4]))
        before = STATE["removed_calls"]
        done_nc, b = fly(t, turn, v, cancel=False)
        assert STATE["removed_calls"] > before, "cancellation patch never reached the law"
        share_nc[key] = float(np.mean(np.abs(b[:, 2]) > b[:, 4]))
        confirm.append(
            f"| {turn:.0f}° | {t:.2f} | {v:.4f} | {done_std} | {share_std[key]:.1%} | {done_nc} | "
            f"{share_nc[key]:.1%} | {100 * (share_nc[key] - share_std[key]):+.1f} |")
        print(confirm[-1], flush=True)

per_turn_p1 = [narrowest_largest(share_std, turn) for turn in (90.0, 120.0)]
verdict_p1 = tally(per_turn_p1)
drops = [share_nc[(turn, "20.44")] < share_std[(turn, "20.44")] for turn in (90.0, 120.0)]
x_small = [abs(share_nc[(turn, "45.00")] - share_std[(turn, "45.00")]) <= 0.02
           for turn in (90.0, 120.0)]
if all(drops) and all(x_small):
    verdict_p2 = "SUPPORTED"
elif not any(drops):
    verdict_p2 = "REFUTED"
else:
    verdict_p2 = "INCONCLUSIVE"
confirm += [
    "",
    f"(P1) equal push: **{verdict_p1}** -- t=20.44 largest share at 90° / 120°: "
    f"{per_turn_p1[0]} / {per_turn_p1[1]}.",
    f"(P2) cause: **{verdict_p2}** -- t=20.44 share drops at 90° / 120°: "
    f"{drops[0]} / {drops[1]}; t=45.00 change within 2 pp: {x_small[0]} / {x_small[1]}.",
]
lines += confirm

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(l for l in lines if l.startswith("(")))
print(f"wrote {OUT}")

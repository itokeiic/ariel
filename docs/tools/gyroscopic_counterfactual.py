#!/usr/bin/env python3
"""Does the gyroscopic mismatch (open item 4) change Table 1?

    uv run --no-sync python docs/tools/gyroscopic_counterfactual.py VARIANT OUT.json
    uv run --no-sync python docs/tools/gyroscopic_counterfactual.py summarise DIR
    uv run --no-sync python docs/tools/gyroscopic_counterfactual.py mechanism OUT.md

The Lee controller adds +Omega x I Omega to its torque, to cancel that term in
Euler's equation, I dOmega/dt = M - Omega x I Omega. `DroneSimulator` integrates
dOmega/dt = I^-1 M without it, so on this plant the controller's term is not a
cancellation but an injected torque. Measured 2026-09-11 at each body's Table 1
limit: up to 25% of the body's roll torque, smallest at the X quad and growing
toward both ends of the family. Its size does not say which way it moves a
result, so this flies the Table 1 family under a 2x2 factorial:

  current      plant without the term, controller cancels it  == current code
  physical     plant with the term,    controller cancels it  (what PhysX gives)
  no_gyro      plant without the term, controller does not    (consistent, unphysical)
  uncancelled  plant with the term,    controller does not

That run showed the physical deficit is yaw saturation: yaw torque capacity at
hover is 0.092 N m on every body, the yaw part of the term reaches 0.2-0.4 N m on
asymmetric frames, and the controller's mixer clips each motor independently,
so an unachievable yaw demand also destroys thrust, roll and pitch. Real flight
stacks give up yaw first. Two further variants swap in such a mixer --
thrust, roll and pitch allocated first, the yaw torque then scaled to the largest
fraction that fits the motor limits, then the same per-motor clip:

  current_yawlast   plant without the term, controller cancels, yaw-last mixer
  physical_yawlast  plant with the term,    controller cancels, yaw-last mixer

With yaw unscaled the yaw-last mixer is identical to the original, which is
asserted on every such call.

All patches are applied at runtime; the library is not modified. `current`
must reproduce docs/data/tuned_strict_completion.csv cell for cell, which
validates the instrument. Output is written after every cell and a rerun skips
cells already present, so an interrupted run resumes.

Operating point: the Table 1 provenance row (docs/drone_morphology_gate_racing.md
section 7.9).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parents[2]
TABLE1 = REPO / "docs" / "data" / "tuned_strict_completion.csv"
VARIANTS = {  # name: (plant integrates the term, controller cancels it, mixer, plant yaw drag)
    "current": (False, True, "clip", "lin"),
    "physical": (True, True, "clip", "lin"),
    "no_gyro": (False, False, "clip", "lin"),
    "uncancelled": (True, False, "clip", "lin"),
    "current_yawlast": (False, True, "yaw_last", "lin"),
    "physical_yawlast": (True, True, "yaw_last", "lin"),
    # Open item 17: the plant's rotor drag torque is the tangent at hover,
    # 2 k_m W_hover W, while the mixer assumes k_m W^2 with the same k_m. In
    # Table 1 flights the tangent gives ~0.7 of the quadratic yaw torque.
    "current_quadyaw": (False, True, "clip", "quad"),
    "physical_quadyaw": (True, True, "clip", "quad"),
    "physical_quadyaw_yawlast": (True, True, "yaw_last", "quad"),
}


def table1() -> list[tuple[float, float, float]]:
    """(turn, half-angle, max_speed) for every Table 1 cell."""
    return [(float(r["turn_deg"]), float(r["half_angle_deg"]), float(r["max_speed"]))
            for r in csv.DictReader(open(TABLE1))]


def summarise(d: Path) -> str:
    runs = {v: json.loads((d / f"{v}.json").read_text()) for v in VARIANTS
            if (d / f"{v}.json").exists()}
    cells = table1()
    lines = ["| turn | t (deg) | Table 1 | " + " | ".join(runs) + " |",
             "|---|---|---|" + "---|" * len(runs)]
    for turn, t, v in cells:
        k = f"{turn:.0f}@{t:.2f}"
        vals = " | ".join(f"{runs[n][k]['max_speed']:.4f}" if k in runs[n] else "--"
                          for n in runs)
        lines.append(f"| {turn:.0f}° | {t:.2f} | {v:.4f} | {vals} |")
    lines += ["", "| turn | variant | argmax t (deg) | spread | ranking = Table 1 |",
              "|---|---|---|---|---|"]
    for turn in sorted({c[0] for c in cells}):
        col = [(t, v) for tn, t, v in cells if tn == turn]
        ref_order = sorted(range(len(col)), key=lambda i: (col[i][1], i))
        for n, run in [("Table 1", None), *runs.items()]:
            if run is None:
                sp = [v for _, v in col]
            else:
                ks = [f"{turn:.0f}@{t:.2f}" for t, _ in col]
                if not all(k in run for k in ks):
                    continue
                sp = [run[k]["max_speed"] for k in ks]
            best = max(sp)
            ties = [col[i][0] for i, s in enumerate(sp) if best - s < 0.0625]
            order = sorted(range(len(sp)), key=lambda i: (sp[i], i))
            arg = (f"{ties[0]:.2f}" if len(ties) == 1 else f"{ties[0]:.2f}-{ties[-1]:.2f}")
            lines.append(f"| {turn:.0f}° | {n} | {arg} | {100 * (best - min(sp)) / best:.1f}% | "
                         f"{'yes' if order == ref_order else 'no'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and sys.argv[1] == "summarise":
    print(summarise(Path(sys.argv[2])))
    raise SystemExit(0)

# `mechanism` flies the no_gyro configuration and correlates the yaw torque the
# term would add with the yaw torque the controller demands (see the routine below).
MECHANISM = sys.argv[1] == "mechanism"
VARIANT, OUT = ("no_gyro" if MECHANISM else sys.argv[1]), Path(sys.argv[2])
assert VARIANT in VARIANTS, VARIANT
PLANT_GYRO, CTRL_CANCELS, MIXER, YAW = VARIANTS[VARIANT]

# The sweep parses its CLI at import, so bind the Table 1 operating point first.
sys.argv = ["sweep", "--completion", "strict", "--speed-lo", "2", "--speed-hi", "12",
            "--speed-cap", "25", "--speed-tol", "0.0625", "--att-omega-n", "24",
            "--pos-omega-n", "2.0", "--pos-zeta", "1.0", "--feedforward", "--max-accel", "40"]
sys.path.insert(0, str(REPO / "examples" / "spear"))
spec = importlib.util.spec_from_file_location(
    "sweep", REPO / "examples" / "spear" / "19_morphology_design_sweep.py")
sweep = importlib.util.module_from_spec(spec)
sys.modules["sweep"] = sweep
spec.loader.exec_module(sweep)

import ariel.simulation.drone.controllers.lee_control.base_lee_controller as blc  # noqa: E402
import ariel.simulation.drone.drone_simulator as ds  # noqa: E402
from ariel.simulation.tasks.slalom_course import slalom_gates  # noqa: E402

# Every variant here is a patch on the plant as it was up to commit 8d3174b:
# no gyroscopic term, hover-tangent yaw drag, controller motor floor 75 rad/s.
# All three were fixed in the library on 2026-09-16, so on a later plant the
# patches would apply twice and `current` would no longer be Table 1. Refuse to
# run there. Probe: a symmetric body at one steady motor speed has no net moment,
# so any angular acceleration at non-zero body rates is the gyroscopic term.
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers as _b2p  # noqa: E402
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint as _s2b  # noqa: E402

_probe = ds.DroneSimulator(
    propellers=_b2p(_s2b(sweep.make_genome(np.radians(20.44)), propsize=sweep.PROP_SIZE),
                    convention="ned"),
    payload_mass=sweep.PAYLOAD_MASS)
_Wp = float(_probe.params["W_hover"])
_xp = np.zeros(12 + _probe.num_motors)
_xp[9:12] = (4.0, -3.0, 2.0)
_xp[12:] = 2.0 * (_Wp - ds.W_MIN_N) / (ds.W_MAX_N - ds.W_MIN_N) - 1.0
_up = np.full(_probe.num_motors, 2.0 * ds._invert_sqrt_poly(
    _Wp, _probe.params["w_max"], _probe.params["w_min"], _probe.params["k"]) - 1.0)
if np.abs(np.asarray(_probe.dynamics_func(_xp, _up), dtype=float).ravel()[9:12]).max() > 1e-6:
    raise SystemExit(
        "This plant already integrates the gyroscopic term (fixed 2026-09-16). The "
        "counterfactual patches the pre-fix plant; reproduce its data at commit 8d3174b.")

calls = {"plant": 0, "ctrl": 0}
# rollout() catches every exception, including those raised while building the
# drone, and scores the flight as a failure. A patch that raised would therefore
# read as "this body did not fly", so both wrappers record what they raise and
# every cell asserts the record is empty. (First run, 2026-09-11: a check that
# read self.state inside _setup_dynamics -- which __init__ calls before creating
# self.state -- failed every build silently until the call counter caught it.)
errors: list[str] = []
# (tau_z commanded, g_z the term would add) per controller call; mechanism mode only.
YAW_LOG: list[tuple[float, float]] = []

if PLANT_GYRO or YAW == "quad":
    _orig_setup = ds.DroneSimulator._setup_dynamics

    def _setup_with_gyro(self) -> None:
        _orig_setup(self)
        f = self.dynamics_func
        inertia = np.asarray(self.params["inertia"], dtype=float)
        inv = np.linalg.inv(inertia)

        n = self.num_motors
        spins = np.asarray(self.params["rotor_spins"], dtype=float)
        dirs = np.asarray(self.params["rotor_dirs"], dtype=float)
        k_m, w_hover = float(self.params["k_m"]), float(self.params["W_hover"])

        def with_gyro(state, action):
            calls["plant"] += 1
            try:
                out = np.array(f(state, action), dtype=float)
                w = np.asarray(state[9:12], dtype=float)   # body rates p, q, r
                if PLANT_GYRO:
                    out[9:12] -= inv @ np.cross(w, inertia @ w)
                if YAW == "quad":
                    # Swap the plant's rotor drag torque, spin * 2 k_m W_hover W
                    # about each rotor axis (drone_simulator.py), for k_m W^2.
                    W = ((np.asarray(state[12:12 + n], dtype=float) + 1.0) / 2.0
                         * (ds.W_MAX_N - ds.W_MIN_N) + ds.W_MIN_N)
                    corr = ((spins * k_m * (W ** 2 - 2.0 * w_hover * W))[:, None] * dirs).sum(0)
                    out[9:12] += inv @ corr
            except Exception as e:
                errors.append(f"plant: {e!r}")
                raise
            return out

        # No checks here: __init__ calls this before self.state exists.
        self._dynamics_without_gyro = f
        self.dynamics_func = with_gyro

    ds.DroneSimulator._setup_dynamics = _setup_with_gyro

    # Instrument check on a real Table 1 body, outside rollout()'s blanket except.
    # The term must land on d_p, d_q, d_r and nowhere else, and satisfy Euler's
    # equation there: I * (change in dOmega/dt) = -Omega x I Omega.
    from ariel.body_phenotypes.drone.backends import blueprint_to_propellers  # noqa: E402
    from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint  # noqa: E402

    _sim = ds.DroneSimulator(
        propellers=blueprint_to_propellers(
            spherical_angular_to_blueprint(sweep.make_genome(np.radians(20.44)),
                                           propsize=sweep.PROP_SIZE),
            convention="ned"),
        payload_mass=sweep.PAYLOAD_MASS)
    # Motors are set off hover so the quadratic and tangent yaw models differ.
    _x = np.array(_sim.state, dtype=float)
    _x[9:12] = (4.0, -3.0, 2.0)
    _p = _sim.params
    _n = _sim.num_motors
    _Wh, _km = float(_p["W_hover"]), float(_p["k_m"])
    _s = np.asarray(_p["rotor_spins"], dtype=float)
    _d = np.asarray(_p["rotor_dirs"], dtype=float)
    _W = _Wh * np.array([1.2 + 0.3 * (i % 2) for i in range(_n)])
    _x[12:12 + _n] = 2.0 * (_W - ds.W_MIN_N) / (ds.W_MAX_N - ds.W_MIN_N) - 1.0
    _u = np.zeros(_n)
    _I = np.asarray(_p["inertia"], dtype=float)
    _diff = (np.array(_sim.dynamics_func(_x, _u), dtype=float)
             - np.array(_sim._dynamics_without_gyro(_x, _u), dtype=float))
    _expected = np.zeros(3)
    if PLANT_GYRO:
        _gyro = -np.cross(_x[9:12], _I @ _x[9:12])
        assert np.abs(np.linalg.solve(_I, _gyro)).max() > 1.0, "gyro check is vacuous"
        _expected += _gyro
    if YAW == "quad":
        _quad = ((_s * _km * _W ** 2)[:, None] * _d).sum(0)
        _lin = ((_s * 2.0 * _km * _Wh * _W)[:, None] * _d).sum(0)
        assert abs(_quad[2] - _lin[2]) > 1e-3, "yaw check is vacuous: the models agree here"
        # At hover the two models must coincide: the tangent touches there.
        _hov = np.full(_n, _Wh)
        assert np.allclose(((_s * _km * (_hov ** 2 - 2.0 * _Wh * _hov))[:, None] * _d).sum(0),
                           0.0, atol=1e-12)
        _expected += _quad - _lin
    assert np.allclose(_diff[:9], 0.0) and np.allclose(_diff[12:], 0.0), _diff
    assert np.allclose(_I @ _diff[9:12], _expected), (_I @ _diff[9:12], _expected)
    print("plant patch verified: gyro", PLANT_GYRO, "yaw", YAW,
          "delta rates", _diff[9:12], flush=True)
    calls["plant"] = 0

if not CTRL_CANCELS:
    _orig_torque = blc.BaseLeeController.compute_body_torque

    def _torque_without_cancellation(self, *a, **kw):
        # compute_body_torque returns -K_R e_R - K_W e_W + Omega x I Omega with no
        # later clipping, so subtracting the same term afterwards removes it exactly.
        calls["ctrl"] += 1
        try:
            tau = _orig_torque(self, *a, **kw)
            w = np.asarray(self.robot_body_angvel, dtype=float)
            gyro = np.cross(w, np.asarray(self.robot_inertia, dtype=float) @ w)
            out = tau - gyro
            if MECHANISM:
                YAW_LOG.append((float(out[2]), float(-gyro[2])))
            return out
        except Exception as e:
            errors.append(f"controller: {e!r}")
            raise

    blc.BaseLeeController.compute_body_torque = _torque_without_cancellation

mix_stats = {"calls": 0, "unscaled_checked": 0, "yaw_scaled": 0}

if MIXER == "yaw_last":
    import ariel.simulation.drone.controllers.lee_control.lee_controller as lc  # noqa: E402

    _orig_mix = lc.LeeGeometricControl._wrench_to_motor_commands

    def _thrust_command(self, quad) -> float:
        """The original method's thrust, reproduced line for line (lee_controller.py)."""
        force = self.wrench_command[0:3]
        dcm = getattr(quad, "dcm", None)
        body_z = (np.array([0.0, 0.0, 1.0]) if dcm is None
                  else np.asarray(dcm, dtype=float)[:, 2])
        proj = float(np.dot(force, body_z))
        thrust = max(-proj if self.orient == "NED" else proj, 0.0)
        return float(np.clip(thrust, quad.params["minThr"], quad.params["maxThr"]))

    def _allocate_yaw_last(quad, t_vec):
        """Motor speeds for [thrust, Mx, My, Mz], yaw scaled to fit. Returns (w, scale)."""
        fm_inv = np.asarray(quad.params["mixerFMinv"], dtype=float)
        wmax2 = np.array([p["wmax"] for p in quad.drone_sim.config.propellers], float) ** 2
        lo, hi = quad.params["minWmotor"] ** 2, quad.params["maxWmotor"] ** 2
        base = (fm_inv @ np.array([t_vec[0], t_vec[1], t_vec[2], 0.0])) * wmax2
        dyaw = (fm_inv @ np.array([0.0, 0.0, 0.0, t_vec[3]])) * wmax2
        scale = 1.0
        for b, d in zip(base, dyaw):
            if d > 0.0:
                scale = min(scale, (hi - b) / d)
            elif d < 0.0:
                scale = min(scale, (lo - b) / d)
        scale = max(scale, 0.0)
        return np.sqrt(np.clip(base + scale * dyaw, lo, hi)), scale

    def _mix_yaw_last(self, quad) -> None:
        mix_stats["calls"] += 1
        try:
            _orig_mix(self, quad)
            w_orig = np.array(self.w_cmd, dtype=float)
            t_vec = np.array([_thrust_command(self, quad), *self.wrench_command[3:6]])
            w, scale = _allocate_yaw_last(quad, t_vec)
            if scale >= 1.0:
                mix_stats["unscaled_checked"] += 1
                if not np.allclose(w, w_orig, rtol=1e-9, atol=1e-9):
                    raise AssertionError(f"yaw_last != original with yaw unscaled: {w} vs {w_orig}")
            else:
                mix_stats["yaw_scaled"] += 1
            self.w_cmd = w
        except Exception as e:
            errors.append(f"mixer: {e!r}")
            raise

    lc.LeeGeometricControl._wrench_to_motor_commands = _mix_yaw_last

    # Instrument check on a real Table 1 body, outside rollout()'s blanket except:
    # a 0.3 N m yaw demand at hover (3x capacity) must leave thrust, roll and
    # pitch exactly as commanded and deliver a reduced, same-signed yaw; the
    # original mixer must not (otherwise this variant tests nothing).
    from ariel.body_phenotypes.drone.backends import blueprint_to_propellers  # noqa: E402
    from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint  # noqa: E402

    _course = slalom_gates(60.0, leg=sweep.args.leg, n_gates=sweep.args.n_gates,
                           gate_size=sweep.args.gate_size)
    _drone, _ctrl, *_ = sweep._build_stack(
        blueprint_to_propellers(
            spherical_angular_to_blueprint(sweep.make_genome(np.radians(20.44)),
                                           propsize=sweep.PROP_SIZE),
            convention="ned"),
        _course, 8.0)
    _fm = np.asarray(_drone.params["mixerFM"], dtype=float)
    _wmax2 = np.array([p["wmax"] for p in _drone.drone_sim.config.propellers], float) ** 2
    _hover = float(_drone.params["mass"]) * 9.81 if "mass" in _drone.params else float(_drone.drone_sim.mass) * 9.81
    _ctrl.wrench_command = np.array([0.0, 0.0, -_hover, 0.0, 0.0, 0.3])
    _orig_mix(_ctrl, _drone)
    _old = _fm @ (np.asarray(_ctrl.w_cmd, float) ** 2 / _wmax2)
    _ctrl.wrench_command = np.array([0.0, 0.0, -_hover, 0.0, 0.0, 0.3])
    _mix_yaw_last(_ctrl, _drone)
    _new = _fm @ (np.asarray(_ctrl.w_cmd, float) ** 2 / _wmax2)
    print("mixer check, commanded [T, Mx, My, Mz] =", [round(_hover, 4), 0.0, 0.0, 0.3],
          "\n  original delivers", _old.round(4), "\n  yaw_last delivers", _new.round(4), flush=True)
    assert abs(_new[0] - _hover) < 1e-6 * _hover and np.allclose(_new[1:3], 0.0, atol=1e-9), _new
    assert 0.0 < _new[3] < 0.3, _new
    assert abs(_old[0] - _hover) > 1e-3 * _hover or not np.allclose(_old[1:3], 0.0, atol=1e-6), (
        "the original mixer does not distort at this demand: the check is vacuous")
    mix_stats.update(calls=0, unscaled_checked=0, yaw_scaled=0)
    errors.clear()

if MECHANISM:
    # Does the term's yaw component oppose the demanded yaw on narrow frames and
    # assist it on wide ones? Pre-registered 2026-09-16, before any data:
    # SUPPORTED if corr(g_z, tau_z) <= -0.3 for both t=20.44 and 28.63 AND >= +0.3
    # for both t=61.37 and 69.56, at every turn; REFUTED if narrow and wide share a
    # sign or |corr| < 0.1 on either side. Note the sign flip at t=45 follows from
    # the sign of Ixx - Iyy alone; the substantive quantity is corr(pq, tau_z).
    from ariel.body_phenotypes.drone.backends import blueprint_to_propellers  # noqa: E402
    from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint  # noqa: E402

    speeds = ((60.0, 9.5), (90.0, 7.5), (120.0, 6.0))   # below every body's no_gyro limit
    halves = sorted({t for _, t, _ in table1()})
    lines = [
        "Generated by `uv run --no-sync python docs/tools/gyroscopic_counterfactual.py "
        "mechanism docs/data/gyroscopic_mechanism.md`.", "",
        "Configuration `no_gyro`: plant without the term and controller cancellation "
        "removed, so `tau_z` is only what yaw tracking demands. `g_z` is the yaw "
        "component of `-Omega x I Omega`, the torque the term would add. Pre-registered "
        "2026-09-16: SUPPORTED if corr <= -0.3 at t=20.44 and 28.63 and >= +0.3 at "
        "t=61.37 and 69.56, at every turn; REFUTED if narrow and wide share a sign or "
        "|corr| < 0.1 on either side. Yaw torque capacity at hover is 0.092 N m.", "",
        "| turn | speed (m/s) | t (deg) | completed | corr(g_z, tau_z) | <g_z tau_z>/<tau_z^2> "
        "| rms g_z (N m) | rms tau_z (N m) | steps with abs(g_z) > 0.092 N m |",
        "|---|---|---|---|---|---|---|---|---|"]
    corr: dict[tuple[float, str], float] = {}
    for turn, v in speeds:
        course = slalom_gates(turn, leg=sweep.args.leg, n_gates=sweep.args.n_gates,
                              gate_size=sweep.args.gate_size)
        for t in halves:
            props = blueprint_to_propellers(
                spherical_angular_to_blueprint(sweep.make_genome(np.radians(t)),
                                               propsize=sweep.PROP_SIZE),
                convention="ned")
            YAW_LOG.clear()
            out = sweep.rollout(props, course, v)
            assert len(YAW_LOG) > 100 and not errors, (len(YAW_LOG), errors[:3])
            tau = np.array([x[0] for x in YAW_LOG])
            g = np.array([x[1] for x in YAW_LOG])
            c = float(np.corrcoef(g, tau)[0, 1])
            corr[(turn, f"{t:.2f}")] = c
            lines.append(
                f"| {turn:.0f}° | {v:.1f} | {t:.2f} | {out['completed_strict']} | {c:+.3f} | "
                f"{(g * tau).mean() / (tau ** 2).mean():+.3f} | {np.sqrt((g ** 2).mean()):.4f} | "
                f"{np.sqrt((tau ** 2).mean()):.4f} | {np.mean(np.abs(g) > 0.092):.1%} |")
            print(lines[-1], flush=True)
    narrow = [corr[(turn, k)] for turn, _ in speeds for k in ("20.44", "28.63")]
    wide = [corr[(turn, k)] for turn, _ in speeds for k in ("61.37", "69.56")]
    if max(narrow) <= -0.3 and min(wide) >= 0.3:
        verdict = "SUPPORTED"
    elif (np.sign(np.mean(narrow)) == np.sign(np.mean(wide))
          or min(np.abs(narrow)) < 0.1 or min(np.abs(wide)) < 0.1):
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    lines += ["", f"Verdict: **{verdict}** -- narrow {min(narrow):+.3f} to {max(narrow):+.3f}, "
                  f"wide {min(wide):+.3f} to {max(wide):+.3f}."]
    OUT.write_text("\n".join(lines) + "\n")
    print(verdict, flush=True)
    raise SystemExit(0)

res: dict = json.loads(OUT.read_text()) if OUT.exists() else {}
for turn, t, _ in table1():
    key = f"{turn:.0f}@{t:.2f}"
    if key in res:
        continue
    course = slalom_gates(turn, leg=sweep.args.leg, n_gates=sweep.args.n_gates,
                          gate_size=sweep.args.gate_size)
    r = sweep.max_completing_speed(sweep.make_genome(np.radians(t)), course)
    # A patch that silently misses would make every variant equal `current`.
    if PLANT_GYRO:
        assert calls["plant"] > 0, "plant patch never reached the integrator"
    if not CTRL_CANCELS:
        assert calls["ctrl"] > 0, "controller patch never reached compute_body_torque"
    if MIXER == "yaw_last":
        assert mix_stats["calls"] > 0 and mix_stats["unscaled_checked"] > 0, mix_stats
    assert not errors, f"a patch raised inside rollout(): {errors[:3]}"
    res["_mixer"] = dict(mix_stats)
    res[key] = {"max_speed": float(r["max_speed"]), "monotone": bool(r["monotone"]),
                "bracket": str(r["bracket"]), "n_rollouts": int(r["n_rollouts"]),
                "trk": float(r["tracking_err"]), "sat_lo": float(r["saturation_lo"])}
    res["_calls"] = dict(calls)
    OUT.write_text(json.dumps(res, indent=1))
    print(VARIANT, key, r["max_speed"], flush=True)
print("done", VARIANT, calls, flush=True)

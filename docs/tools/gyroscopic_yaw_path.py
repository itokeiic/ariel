#!/usr/bin/env python3
"""How does the yaw gyroscopic coupling cost narrow frames? (corrected plant)

    uv run --no-sync python docs/tools/gyroscopic_yaw_path.py limits COND OUT.json   # COND: Zp | Zc
    uv run --no-sync python docs/tools/gyroscopic_yaw_path.py paths OUT.json
    uv run --no-sync python docs/tools/gyroscopic_yaw_path.py summarise DIR OUT.md

Data: docs/data/gyroscopic_yaw_path/ (Zp.json, Zc.json, paths.json), summarised in
docs/data/gyroscopic_yaw_path.md.

Why. gyroscopic_axis_attribution.py found that the yaw component of the gyroscopic
coupling, (Ixx - Iyy)/Izz * pq, produces most of the corrected model's reversal.
This asks (a) whether the effect comes from that torque acting in the plant or from
the controller's cancellation of it, and (b) whether it acts through heading lag
or crab angle at the gates.

How. Roll and pitch gyroscopic coupling are removed from BOTH plant and controller
in every condition. The yaw component is then:

  N   none                 plant without it, controller without its cancellation
  Z   both                 plant with it, controller cancelling it
  Zp  plant only           plant with it, controller not cancelling it
  Zc  controller only      controller "cancelling" a term the plant lacks

N and Z limits come from docs/data/gyroscopic_axis_attribution/ (`none`, `z`).

PRE-REGISTERED 2026-09-17, before any data.
(a) Narrow loss = limit(N) - limit(Z) at t=20.44: 2.109 (90 deg), 1.211 (120 deg).
    A condition REPRODUCES it if limit(N) - limit(cond) >= 0.5 x loss at both turns.
    A-plant: Zp reproduces, Zc does not. A-controller: Zc reproduces, Zp does not.
    A-both: neither alone reproduces.
(b) Within each body, same speed v = 0.95 x min(limit N, limit Z), flights under N
    and Z. At the 15 gate crossings, mean |heading error| HE = |wrap(psi - psi_des)|
    (psi_des = desired-state yaw) and mean |horizontal crab| CR =
    |wrap(psi - atan2(vy, vx))|. Ratio = metric(Z) / metric(N).
    B-heading SUPPORTED if, at both 90 and 120 deg, ratio >= 1.2 for t=20.44,
      <= 0.833 for t=69.56, and in [0.9, 1.1] for t=45.00. REFUTED if the t=20.44
      ratio < 1.0 at both turns. Otherwise INCONCLUSIVE.
    B-crab: same thresholds on CR.
    Instrument: if t=45.00 falls outside [0.9, 1.1], the metric is flagged INVALID
      at that turn (flights differ for reasons other than the yaw coupling).
Reported without prediction: 3-D sideslip beta = arcsin(v_body_y / |v|), gate
offset at crossing.
Result (2026-09-17): A-plant SUPPORTED (and Zc has the mirror effect: narrow gains,
wide loses); B-heading REFUTED; B-crab REFUTED. The flights of (b) are below the
narrow frame's limit, so how the failure develops above it is not observed.
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parents[2]
ATTRIBUTION = REPO / "docs" / "data" / "gyroscopic_axis_attribution"
TABLE_CORRECTED = REPO / "docs" / "data" / "tuned_strict_completion.csv"
HALVES = (20.44, 45.00, 69.56)
TURNS = (90.0, 120.0)
MASKS = {  # condition: (plant keeps yaw term, controller keeps yaw cancellation)
    "N": (False, False), "Z": (True, True), "Zp": (True, False), "Zc": (False, True)}


def load_limits(d: Path) -> dict[str, dict]:
    lim = {"N": json.loads((ATTRIBUTION / "none.json").read_text()),
           "Z": json.loads((ATTRIBUTION / "z.json").read_text())}
    for c in ("Zp", "Zc"):
        lim[c] = json.loads((d / f"{c}.json").read_text())
    return lim


def score(d: Path) -> dict:
    """Pre-registered scoring of (a) and (b)."""
    lim = load_limits(d)
    paths = json.loads((d / "paths.json").read_text())
    s = lambda c, turn, h: lim[c][f"{turn:.0f}@{h:.2f}"]["max_speed"]  # noqa: E731
    out: dict = {"limits": lim, "paths": paths, "a": {}}
    for c in ("Zp", "Zc"):
        cov = []
        for turn in TURNS:
            loss = s("N", turn, 20.44) - s("Z", turn, 20.44)
            got = s("N", turn, 20.44) - s(c, turn, 20.44)
            cov.append({"turn": turn, "loss": loss, "covered": got, "fraction": got / loss,
                        "reproduces": got >= 0.5 * loss})
        out["a"][c] = {"coverage": cov, "reproduces": all(x["reproduces"] for x in cov)}
    zp, zc = out["a"]["Zp"]["reproduces"], out["a"]["Zc"]["reproduces"]
    out["A"] = ("A-plant" if zp and not zc else "A-controller" if zc and not zp
                else "A-both" if not zp and not zc else "both reproduce")
    for name, metric in (("B-heading", "HE_gates_deg"), ("B-crab", "CR_gates_deg")):
        ratios = {f"{turn:.0f}@{h:.2f}": paths[f"{turn:.0f}@{h:.2f}"]["Z"][metric]
                  / paths[f"{turn:.0f}@{h:.2f}"]["N"][metric] for turn in TURNS for h in HALVES}
        per = []
        for turn in TURNS:
            r20, r45, r70 = (ratios[f"{turn:.0f}@{h:.2f}"] for h in HALVES)
            per.append({"turn": turn, "narrow": r20, "x": r45, "wide": r70,
                        "valid": 0.9 <= r45 <= 1.1})
        if not all(p["valid"] for p in per):
            verdict = "INVALID"
        elif all(p["narrow"] >= 1.2 and p["wide"] <= 0.833 for p in per):
            verdict = "SUPPORTED"
        elif all(p["narrow"] < 1.0 for p in per):
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"
        out[name] = {"ratios": ratios, "per_turn": per, "verdict": verdict}
    return out


def summarise(d: Path) -> str:
    sc = score(d)
    lim, paths = sc["limits"], sc["paths"]
    bad = [(c, k) for c in ("Zp", "Zc") for k, v in lim[c].items()
           if not v["monotone"] or v["bracket"] != "ok"]
    lines = [
        "Generated by `uv run --no-sync python docs/tools/gyroscopic_yaw_path.py summarise "
        "docs/data/gyroscopic_yaw_path docs/data/gyroscopic_yaw_path.md`.",
        "Criteria are in the tool's docstring (pre-registered 2026-09-17). Roll and pitch "
        "gyroscopic coupling are off in every condition.", "",
        f"Validity: non-monotone or bad bracket in Zp/Zc: {len(bad)}.", "",
        "## (a) Plant or controller: max completing speed (m/s)", "",
        "| turn | t (deg) | N none | Z both | Zp plant only | Zc controller only |",
        "|---|---|---|---|---|---|"]
    for turn in TURNS:
        for h in HALVES:
            k = f"{turn:.0f}@{h:.2f}"
            lines.append(f"| {turn:.0f}° | {h:.2f} | " +
                         " | ".join(f"{lim[c][k]['max_speed']:.3f}" for c in ("N", "Z", "Zp", "Zc")) + " |")
    lines += ["", "| condition | share of the narrow loss, 90° / 120° | reproduces |", "|---|---|---|"]
    for c in ("Zp", "Zc"):
        a = sc["a"][c]
        lines.append(f"| {c} | {100 * a['coverage'][0]['fraction']:.0f}% / "
                     f"{100 * a['coverage'][1]['fraction']:.0f}% | {a['reproduces']} |")
    lines += ["", f"Verdict (a): **{sc['A']}**.", "",
              "## (b) Heading and crab at the gates: N vs Z at the same speed", "",
              "| turn | t (deg) | speed (m/s) | gates N / Z | heading error N / Z (deg) | ratio "
              "| crab N / Z (deg) | ratio | 3-D sideslip N / Z (deg) | gate offset N / Z (m) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for turn in TURNS:
        for h in HALVES:
            k = f"{turn:.0f}@{h:.2f}"
            n, z = paths[k]["N"], paths[k]["Z"]
            lines.append(
                f"| {turn:.0f}° | {h:.2f} | {paths[k]['speed']:.3f} | {n['gates_passed']} / {z['gates_passed']} | "
                f"{n['HE_gates_deg']:.1f} / {z['HE_gates_deg']:.1f} | {z['HE_gates_deg'] / n['HE_gates_deg']:.2f} | "
                f"{n['CR_gates_deg']:.1f} / {z['CR_gates_deg']:.1f} | {z['CR_gates_deg'] / n['CR_gates_deg']:.2f} | "
                f"{n['beta_gates_deg']:.1f} / {z['beta_gates_deg']:.1f} | "
                f"{n['offset_gates_m']:.3f} / {z['offset_gates_m']:.3f} |")
    lines += ["", f"B-heading: **{sc['B-heading']['verdict']}**. B-crab: **{sc['B-crab']['verdict']}**."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "summarise":
    text = summarise(Path(sys.argv[2]))
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text)
    print(text)
    raise SystemExit(0)

# Capture this script's own arguments now: sys.argv is replaced below with the
# sweep's flags (the first scratch run on 2026-09-17 read its condition from the
# replaced list and failed).
ARGS = list(sys.argv)
MODE = ARGS[1]
STATE = {"plant_keep_z": False, "ctrl_keep_z": False}

sys.argv = ["sweep", "--completion", "strict", "--speed-lo", "2", "--speed-hi", "12",
            "--speed-cap", "25", "--speed-tol", "0.0625", "--att-omega-n", "24",
            "--pos-omega-n", "2.0", "--pos-zeta", "1.0", "--feedforward", "--max-accel", "40"]
import importlib.util  # noqa: E402

sys.path.insert(0, str(REPO / "examples" / "spear"))
spec = importlib.util.spec_from_file_location(
    "sweep", REPO / "examples" / "spear" / "19_morphology_design_sweep.py")
sweep = importlib.util.module_from_spec(spec)
sys.modules["sweep"] = sweep
spec.loader.exec_module(sweep)

import ariel.simulation.drone.controllers.lee_control.base_lee_controller as blc  # noqa: E402
import ariel.simulation.drone.controllers.trajectory_generation.trajectory as trj  # noqa: E402
import ariel.simulation.drone.drone_simulator as ds  # noqa: E402
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers  # noqa: E402
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint  # noqa: E402
from ariel.simulation.drone.gate_metrics import crossing_report  # noqa: E402
from ariel.simulation.tasks.slalom_course import slalom_gates  # noqa: E402

calls = {"plant": 0, "ctrl": 0}
errors: list[str] = []
YAW_DES: list[float] = []


def _plant_off() -> np.ndarray:
    return np.array([1.0, 1.0, 0.0 if STATE["plant_keep_z"] else 1.0])


def _ctrl_off() -> np.ndarray:
    return np.array([1.0, 1.0, 0.0 if STATE["ctrl_keep_z"] else 1.0])


_orig_setup = ds.DroneSimulator._setup_dynamics


def _setup(self) -> None:
    _orig_setup(self)
    f = self.dynamics_func
    inertia = np.asarray(self.params["inertia"], dtype=float)
    inv = np.linalg.inv(inertia)

    def g(state, action):
        calls["plant"] += 1
        try:
            out = np.array(f(state, action), dtype=float)
            w = np.asarray(state[9:12], dtype=float)
            out[9:12] += inv @ (_plant_off() * np.cross(w, inertia @ w))
        except Exception as e:
            errors.append(repr(e))
            raise
        return out

    self.dynamics_func = g


_orig_torque = blc.BaseLeeController.compute_body_torque


def _torque(self, *a, **kw):
    calls["ctrl"] += 1
    try:
        tau = np.asarray(_orig_torque(self, *a, **kw), dtype=float)
        w = np.asarray(self.robot_body_angvel, dtype=float)
        return tau - _ctrl_off() * np.cross(w, np.asarray(self.robot_inertia, float) @ w)
    except Exception as e:
        errors.append(repr(e))
        raise


_orig_des = trj.Trajectory.desiredState


def _des(self, *a, **kw):
    s = _orig_des(self, *a, **kw)
    YAW_DES.append(float(np.asarray(s)[14]))   # desired yaw (trajSelect yawType 3: along velocity)
    return s


ds.DroneSimulator._setup_dynamics = _setup
blc.BaseLeeController.compute_body_torque = _torque
trj.Trajectory.desiredState = _des


def props_for(t_deg: float):
    return blueprint_to_propellers(
        spherical_angular_to_blueprint(sweep.make_genome(np.radians(t_deg)),
                                       propsize=sweep.PROP_SIZE), convention="ned")


def course_for(turn: float):
    return slalom_gates(turn, leg=sweep.args.leg, n_gates=sweep.args.n_gates,
                        gate_size=sweep.args.gate_size)


def set_condition(cond: str) -> None:
    STATE["plant_keep_z"], STATE["ctrl_keep_z"] = MASKS[cond]


def check_plant(cond: str) -> None:
    """Outside rollout()'s blanket except: the kept angular acceleration is exact."""
    set_condition(cond)
    sim = ds.DroneSimulator(propellers=props_for(20.44), payload_mass=sweep.PAYLOAD_MASS)
    p = sim.params
    wh = float(p["W_hover"])
    x = np.zeros(12 + sim.num_motors)
    x[9:12] = (4.0, -3.0, 2.0)
    x[12:] = 2.0 * (wh - ds.W_MIN_N) / (ds.W_MAX_N - ds.W_MIN_N) - 1.0
    u = np.full(sim.num_motors,
                2.0 * ds._invert_sqrt_poly(wh, p["w_max"], p["w_min"], p["k"]) - 1.0)
    inertia = np.asarray(p["inertia"], dtype=float)
    got = np.asarray(sim.dynamics_func(x, u), dtype=float).ravel()[9:12]
    want = -np.linalg.solve(inertia, (1.0 - _plant_off()) * np.cross(x[9:12], inertia @ x[9:12]))
    assert np.allclose(got, want, atol=1e-9), (cond, got, want)
    print(f"{cond}: plant check ok {np.round(got, 3)}", flush=True)


def exact_half(h: float) -> float:
    return next(float(r["half_angle_deg"]) for r in csv.DictReader(open(TABLE_CORRECTED))
                if round(float(r["half_angle_deg"]), 2) == h)


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


if MODE == "limits":
    cond, out_path = ARGS[2], Path(ARGS[3])
    check_plant(cond)
    set_condition(cond)
    calls.update(plant=0, ctrl=0)
    res: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    for turn in TURNS:
        for h in HALVES:
            key = f"{turn:.0f}@{h:.2f}"
            if key in res:
                continue
            r = sweep.max_completing_speed(sweep.make_genome(np.radians(exact_half(h))),
                                           course_for(turn))
            assert calls["plant"] > 0 and calls["ctrl"] > 0 and not errors, (calls, errors[:3])
            res[key] = {"max_speed": float(r["max_speed"]), "monotone": bool(r["monotone"]),
                        "bracket": str(r["bracket"])}
            out_path.write_text(json.dumps(res, indent=1))
            print(cond, key, r["max_speed"], flush=True)
    print("done", cond, calls, flush=True)

elif MODE == "paths":
    out_path = Path(ARGS[2])
    for cond in ("N", "Z"):
        check_plant(cond)
    lim = {"N": json.loads((ATTRIBUTION / "none.json").read_text()),
           "Z": json.loads((ATTRIBUTION / "z.json").read_text())}
    res = {}
    for turn in TURNS:
        course = course_for(turn)
        n_g = len(course.gate_pos)
        for h in HALVES:
            key = f"{turn:.0f}@{h:.2f}"
            v = 0.95 * min(lim["N"][key]["max_speed"], lim["Z"][key]["max_speed"])
            res[key] = {"speed": v}
            for cond in ("N", "Z"):
                set_condition(cond)
                YAW_DES.clear()
                out = sweep.rollout(props_for(exact_half(h)), course, v, record=True)
                assert not errors, errors[:3]
                log = out["log"]
                t, pos, eul = log["t"], log["pos"].astype(float), log["euler"].astype(float)
                # desiredState runs once in _build_stack (t=0) and once per step.
                assert len(YAW_DES) == len(t) + 1, (len(YAW_DES), len(t))
                psi_des = np.array(YAW_DES[1:])
                vel = np.gradient(pos, t, axis=0)
                phi, th, psi = eul[:, 0], eul[:, 1], eul[:, 2]
                heading_err = np.abs(wrap(psi - psi_des))
                crab = np.abs(wrap(psi - np.arctan2(vel[:, 1], vel[:, 0])))
                # Body-frame velocity with the plant's R = Rz Ry Rx; beta from its y part.
                cph, sph, cth, sth, cps, sps = (np.cos(phi), np.sin(phi), np.cos(th),
                                                np.sin(th), np.cos(psi), np.sin(psi))
                vby = ((cps * sth * sph - sps * cph) * vel[:, 0]
                       + (sps * sth * sph + cps * cph) * vel[:, 1] + (cth * sph) * vel[:, 2])
                beta = np.abs(np.arcsin(np.clip(
                    vby / np.maximum(np.linalg.norm(vel, axis=1), 1e-9), -1.0, 1.0)))
                rep = crossing_report(pos, course.gate_pos, course.path_yaw[:n_g], course.gate_size)
                idx = [c.step for c in rep if c.crossed]
                res[key][cond] = {
                    "completed": bool(out["completed_strict"]),
                    "gates_passed": int(sum(c.passed for c in rep)),
                    "HE_gates_deg": float(np.degrees(heading_err[idx]).mean()),
                    "CR_gates_deg": float(np.degrees(crab[idx]).mean()),
                    "beta_gates_deg": float(np.degrees(beta[idx]).mean()),
                    "offset_gates_m": float(np.mean([c.offset for c in rep if c.crossed])),
                }
                print(cond, key, round(v, 4), res[key][cond], flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    print("done paths", calls, flush=True)

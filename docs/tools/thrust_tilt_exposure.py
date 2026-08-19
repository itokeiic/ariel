#!/usr/bin/env python3
"""How much of the EA's search space the axial-thrust plant cannot simulate.

    uv run --no-sync python docs/tools/thrust_tilt_exposure.py

Writes docs/data/thrust_tilt_exposure.md.

Section 3.2: the reduced plant applies every rotor's thrust along one body axis
and ignores ``dir[0:3]``, while the controller's mixer honours it. This measures
who that actually bites. The azimuth sweep is unaffected -- it moves rotors
within the plane -- but the evolutionary search space is not, and that is the
difference between "a limitation to fix later" and "a bound on every EA result".

Sampling is seeded, so the table is reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "data" / "thrust_tilt_exposure.md"
N_SAMPLE, SEED, PROP_SIZE = 300, 0, 5
BODY_Z = np.array([0.0, 0.0, 1.0])

_D15 = np.radians(15.0)
# Example 17's own limits, and the handler's defaults.
EX17_LIMITS = np.array([[0.055, 0.17], [-np.pi, np.pi], [-_D15, _D15],
                        [-np.pi, np.pi], [-_D15, _D15], [0.0, 1.0]])
DEFAULT_LIMITS = np.array([[0.055, 0.17], [-np.pi, np.pi], [-np.pi / 2, np.pi / 2],
                           [-np.pi, np.pi], [-np.pi, np.pi], [0.0, 1.0]])

CASES = (
    ("azimuth sweep (this document)", None, None),
    ("EA, example 17 limits (±15° elevation and cant)", EX17_LIMITS, (4, 6)),
    ("EA, handler defaults (±90° elevation, ±180° cant)", DEFAULT_LIMITS, (4, 6)),
)


def _deviation(genome: np.ndarray) -> float:
    """Worst angle between any rotor's thrust axis and the body z axis, degrees."""
    props = blueprint_to_propellers(
        spherical_angular_to_blueprint(genome, propsize=PROP_SIZE), convention="z_up")
    axes = np.array([p["dir"][:3] for p in props], dtype=float)
    return float(np.degrees(np.arccos(np.clip(np.abs(axes @ BODY_Z), 0.0, 1.0))).max())


def _sweep_devs() -> np.ndarray:
    """The bilaterally symmetric azimuth family this document actually flew."""
    from ariel.simulation.drone.reference_morphologies import x_quad_genome
    tol, prop_r, arm = 0.1, 0.0635, 0.20
    c = (1.0 + tol) * prop_r / arm
    lo, hi = np.arcsin(c), np.pi / 2 - np.arcsin(c)
    return np.array([_deviation(x_quad_genome(float(t), arm))
                     for t in np.linspace(lo, hi, 7)])


def rows() -> list[dict]:
    out = []
    for label, limits, narms in CASES:
        if limits is None:
            devs = _sweep_devs()
        else:
            h = SphericalAngularDroneGenomeHandler(
                min_max_narms=narms, parameter_limits=limits,
                rnd=np.random.default_rng(SEED))
            devs = np.array([_deviation(np.asarray(g, dtype=float).reshape(-1, 6))
                             for g in h.random_population(N_SAMPLE)])
        out.append({
            "case": label, "n": len(devs),
            "median": float(np.median(devs)), "max": float(devs.max()),
            "frac_tilted": float((devs > 1.0).mean()),
            # sin(tilt) is the in-plane component the plant drops entirely.
            "unsimulated_median": float(np.sin(np.radians(np.median(devs)))),
        })
    return out


def table(rs: list[dict]) -> str:
    lines = ["| bodies sampled | n | median tilt | max tilt | tilted > 1° | thrust not simulated (median) |",
             "|---|---|---|---|---|---|"]
    for r in rs:
        lines.append(
            f"| {r['case']} | {r['n']} | {r['median']:.1f}° | {r['max']:.1f}° | "
            f"{100 * r['frac_tilted']:.0f}% | {100 * r['unsimulated_median']:.0f}% |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rs = rows()
    md = table(rs)
    OUT.write_text(md)
    print(md)
    print(f"wrote {OUT.relative_to(REPO)}")

#!/usr/bin/env python3
"""Generate the achieved-bandwidth table in docs/drone_morphology_gate_racing.md 7.2c.

Writes docs/data/principal_axes.csv and prints the markdown table.

    uv run --no-sync python docs/tools/principal_axes_table.py

Why this is a script and not a paragraph of numbers: the first version of that
table was computed ad hoc and thrown away, the bodies were described only in
words ("unequal arm lengths"), and three of four rows could not afterwards be
reproduced -- one of them was wrong. Every row here is a named, single degree-of-
freedom perturbation of the X quad, so anyone can regenerate and check it.
tests/unit/test_principal_axes_table.py asserts the committed CSV still matches.

Bandwidth is the eigenvalues of I^-1 K_R, the modal frequencies of

    I dOmega/dt = -K_R e_R.

Reading sqrt(K_R,ii / I_ii) per axis instead is circular: with
K_R = diag(I) w^2 it returns w by construction, whatever the coupling.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "data" / "principal_axes.csv"

# Matches the sweep: SPEAR-matched quad, 5" props, 0.20 m arms, 24 rad/s target.
WN, HALF, L, PAYLOAD, PROP_SIZE = 24.0, np.pi / 4, 0.20, 0.667, 5
X = np.array([HALF, np.pi - HALF, np.pi + HALF, -HALF])


def inertia(azimuth, lengths, elevation=None) -> np.ndarray:
    g = np.zeros((4, 6))
    g[:, 0] = lengths
    g[:, 1] = azimuth
    if elevation is not None:
        g[:, 2] = elevation
    g[:, 3] = np.pi
    g[:, 5] = np.arange(4) % 2
    props = blueprint_to_propellers(
        spherical_angular_to_blueprint(g, propsize=PROP_SIZE), convention="z_up")
    return DroneConfiguration(props, payload_mass=PAYLOAD).inertia_matrix


def bandwidth(I: np.ndarray, *, matrix_gains: bool) -> np.ndarray:
    """Achieved closed-loop attitude bandwidth per mode, rad/s."""
    K_R = I * WN**2 if matrix_gains else np.diag(np.diag(I)) * WN**2
    return np.sqrt(np.sort(np.real(np.linalg.eigvals(np.linalg.solve(I, K_R)))))


def bodies() -> dict[str, np.ndarray]:
    """One-arm perturbations of the X quad, one degree of freedom each.

    Symmetric perturbations are useless as illustrations here: alternating
    +/-15 deg of arm elevation leaves the products of inertia at exactly zero,
    which is precisely why the swept family is principal to begin with.
    """
    return {
        "symmetric X quad (swept family)": inertia(X, L),
        "arm 0 azimuth +30 deg": inertia(X + np.radians([30, 0, 0, 0]), L),
        "arm 0 lengthened to 0.26 m": inertia(X, np.array([0.26, L, L, L])),
        "arm 0 elevated 15 deg (out of plane)":
            inertia(X, L, elevation=np.radians([15, 0, 0, 0])),
    }


def rows() -> list[dict]:
    out = []
    for name, I in bodies().items():
        d = bandwidth(I, matrix_gains=False)
        m = bandwidth(I, matrix_gains=True)
        out.append({
            "body": name,
            "max_off_diagonal": float(np.max(np.abs(I - np.diag(np.diag(I))))),
            "diag_wn_min": float(d[0]), "diag_wn_max": float(d[-1]),
            "matrix_wn_min": float(m[0]), "matrix_wn_max": float(m[-1]),
        })
    return out


def _span(lo: float, hi: float) -> str:
    return f"{lo:.2f}" if hi - lo < 5e-3 else f"{lo:.2f} - {hi:.2f}"


def markdown(rs: list[dict]) -> str:
    lines = ["| body | max off-diagonal | diagonal gains | matrix gains |",
             "|---|---|---|---|"]
    for r in rs:
        m = _span(r["matrix_wn_min"], r["matrix_wn_max"])
        lines.append(
            f"| {r['body']} | {r['max_off_diagonal']:.2e} | "
            f"{_span(r['diag_wn_min'], r['diag_wn_max'])} | "
            f"{m if r['max_off_diagonal'] == 0.0 else f'**{m}**'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rs = rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rs)
    print(markdown(rs))
    print(f"wrote {OUT.relative_to(REPO)}")

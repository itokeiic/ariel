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
OUT_SENS = REPO / "docs" / "data" / "principal_axes_sensitivity.csv"

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


def misalignment_deg(I: np.ndarray) -> float:
    """Rotation of the in-plane principal axes away from the body x/y axes.

    Mohr's circle: tan(2 theta) = 2 Ixy / (Ixx - Iyy). The raw theta carries a
    90 deg labelling ambiguity -- swapping which principal axis is called "x"
    is not a physical rotation -- so it is folded into [0, 45] deg. 0 means the
    body axes are already principal; 45 deg is the most misaligned they can be.
    """
    theta = 0.5 * np.degrees(np.arctan2(2.0 * I[0, 1], I[0, 0] - I[1, 1]))
    folded = theta % 90.0
    return 45.0 - abs(45.0 - folded)


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
        principal = np.linalg.eigvalsh(I)
        out.append({
            "body": name,
            "max_off_diagonal": float(np.max(np.abs(I - np.diag(np.diag(I))))),
            "diag_wn_min": float(d[0]), "diag_wn_max": float(d[-1]),
            "matrix_wn_min": float(m[0]), "matrix_wn_max": float(m[-1]),
            "I1": float(principal[0]), "I2": float(principal[1]),
            "I3": float(principal[2]),
            "misalignment_deg": misalignment_deg(I),
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


def sensitivity_rows(couplings=(0.0, 1e-6, 1e-5, 1e-4, 3.1e-4)) -> list[dict]:
    """How far the principal axes swing when a product of inertia is introduced.

    Perturbs Ixy of the symmetric X quad and nothing else. The point is the
    conditioning: this airframe has Ixx approximately Iyy, so the denominator of
    Mohr's formula is tiny and a coupling far too small to matter dynamically
    swings the principal DIRECTION most of its available range.
    """
    I0 = bodies()["symmetric X quad (swept family)"]
    out = []
    for c in couplings:
        I = I0.copy()
        I[0, 1] = I[1, 0] = c
        wn = bandwidth(I, matrix_gains=False)
        out.append({
            "I_xy": float(c),
            "misalignment_deg": misalignment_deg(I),
            "diag_wn_min": float(wn[0]), "diag_wn_max": float(wn[-1]),
        })
    return out


def markdown_principal(rs: list[dict]) -> str:
    lines = ["| body | principal moments $I_1,I_2,I_3$ (kg m^2) | misalignment |",
             "|---|---|---|"]
    for r in rs:
        lines.append(f"| {r['body']} | {r['I1']:.5f}, {r['I2']:.5f}, {r['I3']:.5f} "
                     f"| {r['misalignment_deg']:.2f} deg |")
    return "\n".join(lines) + "\n"


def markdown_sensitivity(rs: list[dict]) -> str:
    lines = ["| $I_{xy}$ | misalignment of principal axes | diagonal-gain bandwidth |",
             "|---|---|---|"]
    for r in rs:
        lines.append(f"| {r['I_xy']:.1e} | {r['misalignment_deg']:.1f} deg | "
                     f"{_span(r['diag_wn_min'], r['diag_wn_max'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rs = rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rs)
    sens = sensitivity_rows()
    with OUT_SENS.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sens[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sens)
    print(markdown(rs))
    print(markdown_principal(rs))
    print(markdown_sensitivity(sens))
    print(f"wrote {OUT.relative_to(REPO)} and {OUT_SENS.relative_to(REPO)}")

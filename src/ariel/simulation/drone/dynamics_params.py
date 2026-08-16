"""Reference-form dynamics parameter derivation.

Maps an airevolve propeller configuration into the 22-parameter reference
dynamics form (matching optimal_quad_control_RL/randomization.py:5-10's
`params_5inch` schema, but with per-motor coefficients computed from the
actual airevolve morphology rather than sysid'd from flight data).

This is the airevolve runtime equivalent of
`experimentation/reference_drone_sim.py:derive_params_2inch_from_airevolve`.
The runtime version generalizes to any prop size via
`get_extended_prop_params` (Phase 1) and to any 4-motor symmetric or
asymmetric quad. Generalization to N motors arrives in Phase 4.

Sign convention: we bake position and spin signs into the per-motor
coefficients themselves, so the downstream symbolic build is
morphology-agnostic. The reference's `_build_dynamics_func` hardcodes signs
for a specific motor layout (see reference_drone_sim.py:280-291); our
runtime pulls those signs out of the formula and into the parameter dict.

* `k_p_signed[i] = -y_i · k_f / Ixx` so that `Mx = sum_i k_p_signed[i] · W_i²`.
* `k_q_signed[i] = +x_i · k_f / Iyy` so that `My = sum_i k_q_signed[i] · W_i²`.
* `k_r_signed[i] = spin_i · 2 · k_m · W_hover / Izz` for the linear-W yaw term.
* `k_r_react_signed[i] = spin_i · k_r_react_borrow` for the dW yaw term.

Where spin_i = +1 for "ccw", -1 for "cw" (the convention was said to be
verified via the per-step parity test in unit_tests/test_dynamics_parity.py).

See `experimentation/RUNTIME_DYNAMICS_MIGRATION.md` Phase 2.1.

NOTE (2026-08-16): none of the artefacts cited below -- `unit_tests/test_dynamics_parity.py`, `experimentation/*`, and the
`optimal_quad_control_RL` reference -- exist in this repository, and none
has ever been in its history. The parity claims are therefore *unverified
provenance*, not verification: nothing here can reproduce them, and no test
guards this module's behaviour. Write a characterisation test before
changing the dynamics.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from .propeller_data import get_extended_prop_params

# Body-frame thrust axis this reduced model assumes for EVERY motor. In the
# NED convention used by DroneSimulator, "up" is -Z, so an untilted rotor has
# dir = (0, 0, -1). The moment coefficients below are the closed form of
# r_i x (k_f W² · THRUST_AXIS); see _guard_axial_thrust.
_ASSUMED_THRUST_AXIS = np.array([0.0, 0.0, -1.0])

# Deviation below which a motor counts as axial (radians). Purely numerical
# slack — 0.057°, well under any deliberate cant.
_AXIAL_TOL_RAD = 1e-3

_NONAXIAL_WARNED = False


def _guard_axial_thrust(propellers: list) -> None:
    """Warn (or raise) when thrust normals are not along the assumed axis.

    The reduced dynamics built from this parameter dict applies all thrust
    along the body thrust axis: the force is ``-k_w · sum(W²)`` on body Z and
    the moments are ``Mx = -y_i·k_f·W²``, ``My = +x_i·k_f·W²``. The per-motor
    thrust normal ``dir[0:3]`` is therefore IGNORED by the plant — while
    DroneConfiguration._compute_allocation_matrices DOES honour it when
    building Bf/Bm, which get_params() hands to the controller as mixerFM.
    A canted rotor is consequently allocated as tilted but simulated as axial.

    This guard makes that boundary visible instead of silent. Set
    ``ARIEL_STRICT_THRUST_NORMALS=1`` to turn it into a hard error.

    Args:
        propellers: list of propeller dicts (`loc`, `dir`, `propsize`).
    """
    global _NONAXIAL_WARNED

    max_tilt = 0.0
    for prop in propellers:
        n_i = np.asarray(prop["dir"][:3], dtype=float)
        norm_i = float(np.linalg.norm(n_i))
        if norm_i == 0.0:
            raise ValueError(
                f"derive_reference_params: zero-length thrust normal in {prop!r}"
            )
        cos_a = float(np.dot(n_i / norm_i, _ASSUMED_THRUST_AXIS))
        max_tilt = max(max_tilt, float(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    if max_tilt <= _AXIAL_TOL_RAD:
        return

    deg = np.degrees(max_tilt)
    # sin(tilt) of the thrust is the in-plane component the plant drops
    # entirely; 1-cos(tilt) is the shortfall along the thrust axis.
    detail = (
        f"thrust normals deviate from the assumed body axis "
        f"({', '.join(f'{v:g}' for v in _ASSUMED_THRUST_AXIS)}) "
        f"by up to {deg:.1f} deg. The reduced "
        f"dynamics ignores dir[0:3], so {np.sin(max_tilt) * 100:.0f}% of that "
        f"rotor's thrust (the in-plane component) is not simulated, while the "
        f"controller's mixerFM does account for it. Results for non-axial "
        f"rotors are not trustworthy."
    )

    if os.environ.get("ARIEL_STRICT_THRUST_NORMALS", "") not in ("", "0"):
        raise ValueError(f"derive_reference_params: {detail}")

    if not _NONAXIAL_WARNED:
        _NONAXIAL_WARNED = True
        warnings.warn(
            f"{detail} (further occurrences suppressed; set "
            f"ARIEL_STRICT_THRUST_NORMALS=1 to raise instead)",
            RuntimeWarning,
            stacklevel=3,
        )


def _spin_sign(rotation: str) -> float:
    """+1 for ccw, -1 for cw. Said to match the reference's convention
    (optimal_quad_control_RL, V0 parity test -- absent, see module docstring)."""
    if rotation == "ccw":
        return 1.0
    if rotation == "cw":
        return -1.0
    raise ValueError(f"unknown rotation direction: {rotation!r}")


def derive_reference_params(
    propellers: list,
    mass: float,
    inertia: np.ndarray,
    prop_size,
    gravity: float = 9.81,
) -> dict:
    """Derive a reference-form parameter dict for an airevolve drone config.

    Args:
        propellers: list of propeller dicts (`loc`, `dir`, `propsize`).
        mass: total drone mass in kg (from DroneConfiguration.mass).
        inertia: 3x3 inertia matrix in body frame.
        prop_size: prop size for fetching aerodynamic constants
            (k_drag, k_r_react, k, w_min). All propellers assumed same size.
        gravity: m/s².

    Returns:
        dict with keys:
            n_motors (int): number of motors
            k_w (float): thrust acceleration coefficient (a_z = -k_w · sum(W²))
            k_x, k_y (float): body-frame drag accel coefficients (single values, all motors)
            k_p_signed (list[float]): per-motor roll-moment coefficient (signed)
            k_q_signed (list[float]): per-motor pitch-moment coefficient (signed)
            k_r_signed (list[float]): per-motor yaw-moment coefficient (linear W, signed)
            k_r_react_signed (list[float]): per-motor yaw-moment from dW (signed)
            tau, k, w_min, w_max (float): motor model parameters

    Notes:
        * Asymmetric morphologies are supported via per-motor `k_p_i, k_q_i`
          computed from the actual `loc`. Spin asymmetry is supported via
          per-motor `k_r_signed`.
        * `k_x, k_y, k, w_min, k_r_react, tau` are single per-prop-size values
          (borrowed from the closest sysid set in propeller_data.py). Per-motor
          variation in these is a Phase 6 polish item.
        * `k_r_signed` is a *linearization* at hover throttle. For hover,
          `dMz/dW ≈ 2·k_m·W_hover`. Away from hover this loses fidelity; the
          reference accepts this as a sysid-level approximation.
    """
    n = len(propellers)
    if n == 0:
        raise ValueError("derive_reference_params: no propellers in config")

    # The plant below is axial-thrust only; say so out loud if it isn't true.
    _guard_axial_thrust(propellers)

    extended = get_extended_prop_params(prop_size)
    k_f, k_m = extended["constants"]
    w_max = float(extended["wmax"])

    Ixx = float(inertia[0, 0])
    Iyy = float(inertia[1, 1])
    Izz = float(inertia[2, 2])
    m = float(mass)

    # Hover motor speed (per motor) for linearizing the yaw torque term.
    F_hover_per_motor = m * gravity / n
    if F_hover_per_motor <= 0 or k_f <= 0:
        raise ValueError(
            f"derive_reference_params: invalid hover-thrust calc "
            f"(F_hover={F_hover_per_motor}, k_f={k_f})"
        )
    W_hover = float(np.sqrt(F_hover_per_motor / k_f))

    k_p_signed = []
    k_q_signed = []
    k_r_signed = []
    k_r_react_signed = []
    for prop in propellers:
        x_i = float(prop["loc"][0])
        y_i = float(prop["loc"][1])
        spin = _spin_sign(prop["dir"][3])

        # M_x = sum_i (-y_i · F_z_i) / Ixx = sum_i (-y_i · k_f · W_i²) / Ixx
        k_p_signed.append(-y_i * k_f / Ixx)
        # M_y = sum_i (+x_i · F_z_i) / Iyy = sum_i (+x_i · k_f · W_i²) / Iyy
        k_q_signed.append(+x_i * k_f / Iyy)
        # M_z (steady, linearized at hover): spin_i · 2 · k_m · W_hover / Izz
        k_r_signed.append(spin * 2.0 * k_m * W_hover / Izz)
        # M_z (motor-acceleration reaction): spin_i · k_r_react
        k_r_react_signed.append(spin * float(extended["k_r_react"]))

    return {
        "n_motors": n,
        "k_w": k_f / m,
        "k_x": float(extended["k_x_drag"]),
        "k_y": float(extended["k_y_drag"]),
        "k_p_signed": k_p_signed,
        "k_q_signed": k_q_signed,
        "k_r_signed": k_r_signed,
        "k_r_react_signed": k_r_react_signed,
        "tau": float(extended["tau"]),
        "k": float(extended["k"]),
        "w_min": float(extended["w_min"]),
        "w_max": w_max,
    }


# Reference's normalization constants (said to match
# optimal_quad_control_RL/quad_race_env.py:41-42; that reference and its parity
# test are absent -- see module docstring).
# These are intentionally independent of physical w_max; the motor state
# `w_i ∈ [-1, 1]` represents `W_i ∈ [W_MIN_N, W_MAX_N] = [0, 3000]` rad/s.
# At full throttle, Wc may exceed W_MAX_N; the state can go outside [-1, 1]
# during transients. This is intentional in the reference.
W_MIN_N = 0.0
W_MAX_N = 3000.0

"""Reference-form dynamics parameter derivation.

STATUS (2026-09-10): neither plant integrates the ``k_*_signed`` scalars
described below any more. ``DroneSimulator`` and ``torch_drone_gate_env`` build
the rotor wrench from ``rotor_dirs``, ``rotor_arms`` (measured about the CG) and
``rotor_spins``, which :func:`derive_reference_params` also returns, and apply
the full inertia tensor. The scalars are axial-only, drop ``loc[2]``, measure
arms from the body origin and assume principal axes; they are kept only so
tests/unit/test_simulation/test_plant_thrust_direction.py can rebuild the
pre-fix model for its parity check. Do not build new dynamics on them. See
docs/plant_thrust_direction.md.

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

SUPERSEDED IN PART (2026-09-10): a characterisation test now exists --
tests/unit/test_simulation/test_plant_thrust_direction.py pins exact axial
parity between the vector plant and this scalar form, the plant's response to
cant, CG-relative arms, and agreement between the sympy and torch plants. The
provenance of the scalar form itself remains unverified.
"""
from __future__ import annotations

import numpy as np

from .propeller_data import get_extended_prop_params

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
    center_of_gravity: np.ndarray | None = None,
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
            rotor_dirs (ndarray, (N,3)): unit thrust axis per rotor, from dir[0:3]
            rotor_arms (ndarray, (N,3)): rotor position about the centre of gravity
            rotor_spins (ndarray, (N,)): +1 cw / -1 ccw (DroneConfiguration's prop_rot)
            k_f, k_m, W_hover, mass, k_r_react (float), inertia (ndarray, (3,3)):
                raw constants the vector plants integrate

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

    # --- 3-D rotor geometry -------------------------------------------------
    # The scalar coefficients above reduce each rotor to (x_i, y_i) plus a spin
    # sign, which discards the thrust direction entirely and the z arm with it.
    # These arrays keep the geometry the plant needs to integrate a tilted
    # rotor correctly:
    #   * `rotor_dirs[i]`  unit thrust axis, from prop["dir"][0:3]
    #   * `rotor_arms[i]`  moment arm about the CENTRE OF GRAVITY, not the body
    #     origin. The scalar path uses raw loc[0], loc[1]; for a body whose cg
    #     is offset in-plane that is a live moment error even with zero tilt.
    #   * `rotor_spins[i]` +1 cw / -1 ccw, i.e. the sign convention of
    #     DroneConfiguration's `prop_rot`, so a drag torque is
    #     `spin * k * dir` and points along the rotor axis.
    cg = (np.zeros(3) if center_of_gravity is None
          else np.asarray(center_of_gravity, dtype=float).reshape(3))
    rotor_dirs, rotor_arms, rotor_spins = [], [], []
    for prop in propellers:
        d = np.asarray(prop["dir"][:3], dtype=float)
        nrm = float(np.linalg.norm(d))
        if nrm <= 0.0:
            raise ValueError(
                f"derive_reference_params: rotor thrust direction {prop['dir'][:3]!r} "
                f"has zero length; cannot normalise")
        rotor_dirs.append(d / nrm)
        rotor_arms.append(np.asarray(prop["loc"][:3], dtype=float) - cg)
        # prop_rot in DroneConfiguration: -1 for ccw, +1 for cw. _spin_sign is
        # the opposite convention, so negate to keep one convention here.
        rotor_spins.append(-_spin_sign(prop["dir"][3]))

    return {
        "n_motors": n,
        "k_w": k_f / m,
        # 3-D geometry and the raw constants the vector plant integrates.
        "rotor_dirs": np.asarray(rotor_dirs),
        "rotor_arms": np.asarray(rotor_arms),
        "rotor_spins": np.asarray(rotor_spins),
        "k_f": float(k_f),
        "k_m": float(k_m),
        "W_hover": W_hover,
        "mass": m,
        "inertia": np.asarray(inertia, dtype=float),
        "k_r_react": float(extended["k_r_react"]),
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

"""Test: the plant integrates each rotor along its own axis, with rigid-body rotation.

The plant used to reduce every rotor to `(x_i, y_i)` plus a spin sign and apply
all thrust along body -z, while the controller's `mixerFM` allocated using the
true normals -- two contradictory models of one drone inside one object. See
docs/plant_thrust_direction.md.

Pinned here:

* **Axial parity.** For a coplanar body the vector plant must reproduce the old
  scalar form exactly, plus the two terms deliberately added on 2026-09-16 and
  nothing else. The reference is recomputed from `self.params` rather than
  loaded from a frozen file, so it pins the *intent* and stays readable.
* **Tilt actually does something.** The old plant was invariant to rotor cant;
  a test that only checked parity would pass on the unfixed code.
* **The gyroscopic term** (2026-09-16, open item 4). A torque-free body with
  unequal inertias obeys Euler's equations.
* **Rotor drag torque is k_m W^2** (2026-09-16, open item 17), not the tangent at
  hover it replaced.
* **The controller's motor floor is the plant's**, not a constant from another
  drone.
"""

# Standard library
import warnings

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.simulation.drone.drone_simulator import (
    W_MAX_N,
    W_MIN_N,
    DroneSimulator,
    _invert_sqrt_poly,
)

ARM = 0.11


def _quad(cant_deg: float = 0.0) -> list[dict]:
    """A 5-inch quad whose rotors are canted `cant_deg` about the body x-axis."""
    a = np.radians(cant_deg)
    direction = [0.0, float(np.sin(a)), -float(np.cos(a))]
    locs = [[ARM, ARM, 0], [-ARM, ARM, 0], [-ARM, -ARM, 0], [ARM, -ARM, 0]]
    spins = ["ccw", "cw", "ccw", "cw"]
    return [{"loc": loc, "dir": [*direction, s], "propsize": 5}
            for loc, s in zip(locs, spins, strict=True)]


def _scalar_reference(sim: DroneSimulator, state: np.ndarray,
                      action: np.ndarray) -> np.ndarray:
    """The pre-fix reduced dynamics, from the same params dict.

    Kept deliberately literal -- this is the model the 2026-09-10 fix had to
    preserve for zero cant, so it is written the way it was written, not
    refactored. It has the hover-tangent yaw drag and no gyroscopic term.
    """
    p = sim.params
    n = sim.num_motors
    _, _, _, vx, vy, vz, phi, theta, psi, _, _, _ = state[:12]
    w = state[12:12 + n]

    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    vbx, vby, _ = R.T @ np.array([vx, vy, vz])

    W = (w + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
    U = (action + 1) / 2
    Wc = ((p["w_max"] - p["w_min"])
          * np.sqrt(p["k"] * U**2 + (1 - p["k"]) * U) + p["w_min"])
    d_W = (Wc - W) / p["tau"]

    T = -p["k_w"] * float(np.sum(W**2))
    Dx = -p["k_x"] * vbx * float(np.sum(W))
    Dy = -p["k_y"] * vby * float(np.sum(W))
    accel = np.array([0, 0, sim.g]) + R @ np.array([Dx, Dy, T])

    Mx = float(np.sum(np.asarray(p["k_p_signed"]) * W**2))
    My = float(np.sum(np.asarray(p["k_q_signed"]) * W**2))
    Mz = (float(np.sum(np.asarray(p["k_r_signed"]) * W))
          + float(np.sum(np.asarray(p["k_r_react_signed"]) * d_W)))
    return np.concatenate([accel, [Mx, My, Mz]])


def _terms_added_2026_09_16(sim: DroneSimulator, state: np.ndarray) -> np.ndarray:
    """The change in angular acceleration from the two deliberate additions.

    Written independently of the plant, from their physical statements:
    * rotor drag torque k_m W^2 about each rotor axis replaces 2 k_m W_hover W;
    * Euler's equation gains -Omega x I Omega.
    """
    p = sim.params
    n = sim.num_motors
    W = (state[12:12 + n] + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
    spins = np.asarray(p["rotor_spins"], dtype=float)
    dirs = np.asarray(p["rotor_dirs"], dtype=float)
    k_m, w_hover = float(p["k_m"]), float(p["W_hover"])
    drag_now = ((spins * k_m * W**2)[:, None] * dirs).sum(0)
    drag_before = ((spins * 2.0 * k_m * w_hover * W)[:, None] * dirs).sum(0)
    inertia = np.asarray(p["inertia"], dtype=float)
    omega = state[9:12]
    return np.linalg.solve(inertia, drag_now - drag_before - np.cross(omega, inertia @ omega))


def _steady(sim: DroneSimulator, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Motor state at speeds `W` and the action that holds them (dW = 0)."""
    p = sim.params
    w = 2.0 * (W - W_MIN_N) / (W_MAX_N - W_MIN_N) - 1.0
    action = np.array([2.0 * _invert_sqrt_poly(Wi, p["w_max"], p["w_min"], p["k"]) - 1.0
                       for Wi in W])
    return w, action


def test_axial_parity_with_the_scalar_form_plus_two_terms() -> None:
    """Zero cant: the vector plant is the scalar plant plus exactly two terms."""
    sim = DroneSimulator(propellers=_quad(0.0))
    n = sim.num_motors
    rng = np.random.default_rng(20260910)
    for _ in range(16):
        state = np.zeros(12 + n)
        state[0:3] = rng.uniform(-2, 2, 3)
        state[3:6] = rng.uniform(-5, 5, 3)
        # Stay clear of the Euler singularity at theta = +/- pi/2.
        state[6:9] = rng.uniform(-0.5, 0.5, 3)
        state[9:12] = rng.uniform(-2, 2, 3)
        state[12:] = rng.uniform(-1, 1, n)
        action = rng.uniform(-1, 1, n)

        got = np.asarray(sim.dynamics_func(state, action), dtype=float).ravel()
        expected = _scalar_reference(sim, state, action)
        # accelerations (3:6) are untouched; angular accelerations (9:12) gain
        # exactly the two 2026-09-16 terms.
        np.testing.assert_allclose(got[3:6], expected[:3], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(got[9:12], expected[3:] + _terms_added_2026_09_16(sim, state),
                                   rtol=1e-9, atol=1e-9)


def test_canted_rotors_produce_in_plane_force() -> None:
    """Tilt changes the force direction -- the old plant was blind to this.

    Canting every rotor by `a` about body x must rotate the thrust vector by
    `a` while preserving its magnitude, so the in-plane component grows as
    sin(a) and the axial component falls as cos(a).
    """
    n = 4
    state = np.zeros(12 + n)
    state[12:] = 0.2
    action = np.full(n, 0.2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the axial guard, still on for torch
        upright = np.asarray(
            DroneSimulator(propellers=_quad(0.0)).dynamics_func(state, action),
            dtype=float).ravel()
        g = DroneSimulator(propellers=_quad(0.0)).g
        thrust_mag = abs(upright[5] - g)      # remove gravity from d_vz

        for cant in (10.0, 20.0, 30.0):
            sim = DroneSimulator(propellers=_quad(cant))
            sd = np.asarray(sim.dynamics_func(state, action), dtype=float).ravel()
            a = np.radians(cant)
            assert sd[3] == pytest.approx(0.0, abs=1e-9), "no force expected along x"
            assert sd[4] == pytest.approx(thrust_mag * np.sin(a), rel=1e-6)
            assert sd[5] - g == pytest.approx(-thrust_mag * np.cos(a), rel=1e-6)


def test_torque_free_body_obeys_eulers_equations() -> None:
    """A spinning body with unequal inertias and no net torque precesses.

    A rectangular quad with all four rotors at one steady speed produces no net
    moment (arms cancel, drag torques cancel, dW = 0), so its angular
    acceleration is the gyroscopic term alone. The plant without the term
    returned zero here.
    """
    locs = [[0.15, 0.08, 0], [-0.15, 0.08, 0], [-0.15, -0.08, 0], [0.15, -0.08, 0]]
    props = [{"loc": loc, "dir": [0.0, 0.0, -1.0, s], "propsize": 5}
             for loc, s in zip(locs, ["ccw", "cw", "ccw", "cw"], strict=True)]
    sim = DroneSimulator(propellers=props)
    inertia = np.asarray(sim.params["inertia"], dtype=float)
    np.testing.assert_allclose(inertia - np.diag(np.diag(inertia)), 0.0, atol=1e-12)
    Ixx, Iyy, Izz = np.diag(inertia)
    assert abs(Ixx - Iyy) > 0.2 * max(Ixx, Iyy), "test body needs unequal inertias"

    n = sim.num_motors
    w, action = _steady(sim, np.full(n, float(sim.params["W_hover"])))
    state = np.zeros(12 + n)
    state[12:] = w

    at_rest = np.asarray(sim.dynamics_func(state, action), dtype=float).ravel()
    np.testing.assert_allclose(at_rest[9:12], 0.0, atol=1e-9)   # no net moment
    np.testing.assert_allclose(at_rest[12:], 0.0, atol=1e-9)    # motors steady

    p, q, r = 4.0, -3.0, 2.0
    state[9:12] = (p, q, r)
    got = np.asarray(sim.dynamics_func(state, action), dtype=float).ravel()[9:12]
    euler = np.array([(Iyy - Izz) * q * r / Ixx,
                      (Izz - Ixx) * r * p / Iyy,
                      (Ixx - Iyy) * p * q / Izz])
    np.testing.assert_allclose(got, euler, rtol=1e-9, atol=1e-9)
    assert np.abs(got).max() > 1.0, "check is vacuous: the term is ~0 here"


def test_rotor_drag_torque_is_quadratic_in_motor_speed() -> None:
    """Off hover, yaw torque is k_m W^2 per rotor, not the tangent 2 k_m W_hover W.

    Diagonal rotor pairs at 1.5x and 1.0x hover speed give a pure yaw moment.
    """
    sim = DroneSimulator(propellers=_quad(0.0))
    p = sim.params
    n = sim.num_motors
    w_hover, k_m = float(p["W_hover"]), float(p["k_m"])
    W = w_hover * np.array([1.5, 1.0, 1.5, 1.0])
    w, action = _steady(sim, W)
    state = np.zeros(12 + n)
    state[12:] = w

    got = np.asarray(sim.dynamics_func(state, action), dtype=float).ravel()
    moment = np.asarray(p["inertia"], dtype=float) @ got[9:12]
    spins = np.asarray(p["rotor_spins"], dtype=float)
    nz = np.asarray(p["rotor_dirs"], dtype=float)[:, 2]
    quadratic = float(np.sum(spins * k_m * W**2 * nz))
    tangent = float(np.sum(spins * 2.0 * k_m * w_hover * W * nz))

    np.testing.assert_allclose(moment[:2], 0.0, atol=1e-9)
    assert moment[2] == pytest.approx(quadratic, rel=1e-9)
    assert abs(quadratic - tangent) > 0.1 * abs(quadratic), "check is vacuous"


def test_controller_motor_floor_is_the_plants_idle_speed() -> None:
    """`minWmotor` was 75 rad/s, a Quadcopter_SimCon default for another drone.

    The plant's motor model never runs below `w_min`, so a lower controller floor
    let allocation assume headroom that commands silently lost.
    """
    sim = DroneSimulator(propellers=_quad(0.0))
    assert sim.get_params()["minWmotor"] == sim.params["w_min"]


def test_torch_plant_matches_the_sympy_plant() -> None:
    """The two plants agree, tilted rotors, gyroscopic term and yaw drag included.

    `TorchDroneGateEnv` advertises itself as a drop-in replacement for
    `DroneGateEnv`, so their dynamics must agree. Two defects broke that and
    are both fixed here: the torch path reimplemented the same axial-only
    rotor model, and it unnormalised the motor state against the motor's
    physical minimum (238.49 rad/s) instead of the normalisation floor
    (`W_MIN_N` = 0), which made the plants disagree even for a coplanar body.
    The random states have non-zero body rates and off-hover motor speeds, so
    both 2026-09-16 terms are exercised.
    """
    torch = pytest.importorskip("torch")
    from ariel.simulation.tasks.torch_drone_gate_env import (  # noqa: PLC0415
        _build_torch_dynamics,
    )

    rng = np.random.default_rng(7)
    for cant in (0.0, 15.0, 30.0, 45.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = DroneSimulator(propellers=_quad(cant))
        n = sim.num_motors
        dyn = _build_torch_dynamics(sim.params, n, sim.g,
                                    torch.device("cpu"), torch.float64)
        for _ in range(8):
            state = np.zeros(12 + n)
            state[0:3] = rng.uniform(-2, 2, 3)
            state[3:6] = rng.uniform(-5, 5, 3)
            state[6:9] = rng.uniform(-0.5, 0.5, 3)
            state[9:12] = rng.uniform(-2, 2, 3)
            state[12:] = rng.uniform(-1, 1, n)
            action = rng.uniform(-1, 1, n)

            want = np.asarray(sim.dynamics_func(state, action), dtype=float).ravel()
            got = dyn(
                torch.tensor(state, dtype=torch.float64).view(-1, 1),
                torch.tensor(action, dtype=torch.float64).view(-1, 1),
            ).numpy().ravel()
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_moment_arm_is_measured_from_the_centre_of_gravity() -> None:
    """Arms come from the CG, not the body origin.

    The scalar path used raw `loc[0]`, `loc[1]`. For a body whose CG is offset
    in-plane that is a live moment error even at zero cant, which is the whole
    EA search space.
    """
    props = _quad(0.0)
    props[0]["loc"] = [ARM + 0.05, ARM, 0.03]      # break the symmetry
    sim = DroneSimulator(propellers=props)
    cg = np.asarray(sim.config.cg, dtype=float)
    assert np.linalg.norm(cg[:2]) > 1e-4, "test body should have an in-plane CG offset"

    for prop, arm in zip(props, sim.params["rotor_arms"], strict=True):
        np.testing.assert_allclose(arm, np.asarray(prop["loc"], dtype=float) - cg,
                                   rtol=0, atol=1e-12)

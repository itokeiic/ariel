"""Backend-agnostic plant interface for articulated multirotors.

The point of this module is that everything above it — control allocation,
the pose/wrench controller, gate checking, fitness, the EA — is written once
against these types, and each physics engine contributes only a thin adapter.
MuJoCo (``LeeMujocoHoverBridge``) and Isaac Lab / PhysX (the
``spear_vua_upb`` wrench scripts) are the two intended adapters.

Frame convention
----------------
**Z-up, right-handed, gravity along -Z.** This matches MuJoCo, USD/Isaac Lab
and ``blueprint_to_mjspec`` / ``blueprint_to_usd``. It is deliberately NOT the
NED convention used by ``DroneSimulator`` and the Lee controller: NED lives
behind the Lee adapter and nowhere else, so a plant adapter never has to
reason about two frames at once.

Why ``rotor_frames()`` exists
-----------------------------
For a morphing airframe the rotor positions and thrust axes are functions of
the joint state, so a control-allocation matrix computed once at construction
(``DroneSimulator.get_params()["mixerFMinv"]``) is stale the moment a joint
moves. Adapters therefore report live geometry each step and
:func:`allocation_matrix` rebuilds ``B(q)`` from it. This also removes the
class of bug where a plant and its allocation disagree about thrust
direction: the geometry has exactly one source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


# --------------------------------------------------------------------------
# state / command
# --------------------------------------------------------------------------

@dataclass
class DroneState:
    """Full state of an articulated multirotor, in the Z-up world frame.

    Attributes:
        pos: world position of the base body, m, shape (3,).
        quat_wxyz: base orientation as (w, x, y, z), shape (4,).
        vel: world linear velocity of the base, m/s, shape (3,).
        omega_body: body-frame angular velocity, rad/s, shape (3,).
        joint_pos: actuated joint positions, rad, shape (M,). Empty for a
            rigid airframe.
        joint_vel: actuated joint velocities, rad/s, shape (M,).
        time: simulation time, s.
    """
    pos: Array
    quat_wxyz: Array
    vel: Array
    omega_body: Array
    joint_pos: Array = field(default_factory=lambda: np.zeros(0))
    joint_vel: Array = field(default_factory=lambda: np.zeros(0))
    time: float = 0.0


@dataclass
class DroneCommand:
    """Actuator command for one control step.

    Thrust is specified in newtons per rotor rather than as a motor speed or
    a normalised duty cycle: the mapping from thrust to whatever the engine
    wants (``ctrl = kf·w²/max_thrust`` in MuJoCo, an applied wrench in Isaac)
    is the adapter's business, and keeping newtons here makes the controller
    portable and unit-checkable.

    Attributes:
        rotor_thrusts: per-rotor thrust along that rotor's current axis, N,
            shape (N,). Non-negative for unidirectional propellers.
        joint_targets: target for each actuated joint, shape (M,) or None to
            hold the previous target. Units follow ``joint_target_type``.
        joint_target_type: "position" (rad), "velocity" (rad/s) or "effort"
            (N·m).
    """
    rotor_thrusts: Array
    joint_targets: Array | None = None
    joint_target_type: str = "position"


@dataclass
class RotorGeometry:
    """Live rotor geometry in the **body** frame, as reported by the plant.

    Attributes:
        positions: rotor positions relative to the base body origin, m,
            shape (N, 3).
        axes: unit thrust axes, shape (N, 3). Thrust acts along +axis.
        spins: +1 for a rotor spinning ccw about its own axis, -1 for cw,
            shape (N,).
    """
    positions: Array
    axes: Array
    spins: Array


@runtime_checkable
class DronePlant(Protocol):
    """What a physics backend must provide to be driven by ARIEL control.

    Deliberately small: no rendering, no task logic, no gate checking. An
    adapter owns exactly the translation between these calls and its engine.
    """

    @property
    def n_rotors(self) -> int: ...

    @property
    def n_joints(self) -> int:
        """Number of *actuated* joints (0 for a rigid airframe)."""

    @property
    def joint_names(self) -> list[str]:
        """Actuated joint names, ordered as in ``DroneState.joint_pos``."""

    def reset(self, state: DroneState) -> None:
        """Teleport to ``state`` and zero any internal integrator history."""

    def state(self) -> DroneState:
        """Current state, Z-up world frame."""

    def rotor_frames(self) -> RotorGeometry:
        """Rotor positions and thrust axes for the *current* joint state."""

    def mass_inertia(self) -> tuple[float, Array]:
        """Total mass (kg) and body-frame inertia tensor (3x3) about the CoM.

        Both are configuration-dependent for a morphing airframe; adapters
        should report the values for the current joint state rather than a
        cached construction-time value.
        """

    def step(self, cmd: DroneCommand, dt: float) -> DroneState:
        """Apply ``cmd`` and advance the simulation by ``dt``."""


# --------------------------------------------------------------------------
# control allocation
# --------------------------------------------------------------------------

def allocation_matrix(
    geom: RotorGeometry,
    *,
    torque_to_thrust_ratio: float,
) -> Array:
    """Build the wrench allocation matrix ``B(q)``: thrusts -> body wrench.

    ``B @ f = [Fx, Fy, Fz, Mx, My, Mz]`` for per-rotor thrusts ``f`` (N), all
    in the body frame. Each rotor contributes

    * force  ``f_i · n_i``
    * moment ``f_i · (r_i x n_i)``  (thrust acting at a lever arm), plus
      ``-spin_i · (km/kf) · f_i · n_i``  (aerodynamic reaction torque; a
      ccw-spinning rotor pushes the body cw)

    Args:
        geom: live rotor geometry in the body frame.
        torque_to_thrust_ratio: ``km/kf`` for the propeller, in metres. Both
            constants come from ``propeller_data``; only their ratio matters
            because both scale with W².

    Returns:
        Array of shape (6, N).

    Note:
        The reaction-torque sign convention (-1 for ccw) matches
        ``DroneConfiguration._compute_allocation_matrices``. It is the
        opposite of ``dynamics_params._spin_sign`` (+1 for ccw) because that
        module works in NED, where the vertical axis is flipped.
    """
    n = geom.positions.shape[0]
    if geom.axes.shape != (n, 3) or geom.spins.shape != (n,):
        raise ValueError(
            f"inconsistent RotorGeometry shapes: positions {geom.positions.shape}, "
            f"axes {geom.axes.shape}, spins {geom.spins.shape}"
        )

    B = np.zeros((6, n), dtype=float)
    for i in range(n):
        axis = np.asarray(geom.axes[i], dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError(f"rotor {i} has a zero-length thrust axis")
        axis = axis / norm
        r = np.asarray(geom.positions[i], dtype=float)

        B[0:3, i] = axis
        B[3:6, i] = np.cross(r, axis) - float(geom.spins[i]) * torque_to_thrust_ratio * axis
    return B


def moment_allocation(
    geom: RotorGeometry,
    *,
    torque_to_thrust_ratio: float,
    inertia: Array | None = None,
) -> Array:
    """The moment block ``Bm`` of :func:`allocation_matrix`, shape (3, N).

    Args:
        geom: live rotor geometry in the body frame.
        torque_to_thrust_ratio: ``km/kf`` in metres.
        inertia: optional 3x3 body inertia tensor. When given, returns
            ``I^-1 @ Bm`` — the *angular acceleration* each newton of thrust
            produces, rather than the torque. Prefer this for anything about
            agility: torque authority alone is misleading because inertia and
            authority move together. On a quad swept from a fore-aft to a
            lateral frame, roll torque rises 1.735 -> 3.896 N·m while roll
            angular acceleration *falls* 2921 -> 1325 rad/s^2. It also matches
            the plant, which works in angular acceleration
            (``dynamics_params`` bakes inertia into ``k_p_signed``).
    """
    B = allocation_matrix(geom, torque_to_thrust_ratio=torque_to_thrust_ratio)
    Bm = B[3:6, :]
    if inertia is None:
        return Bm
    return np.linalg.solve(np.asarray(inertia, dtype=float), Bm)


def maneuverability(
    geom: RotorGeometry,
    *,
    torque_to_thrust_ratio: float,
    inertia: Array | None = None,
) -> float:
    """Smallest eigenvalue of ``Bm @ Bm.T`` — moment authority on the weakest axis.

    Ported from airevolve's morphological descriptor
    (``inspection_tools/morphological_descriptors/hovering_info.py``), which
    computes ``min(eig(Bm @ Bm.T))`` from ``dronehover``'s allocation matrix.
    Here ``Bm`` comes from :func:`moment_allocation`, so one definition of the
    geometry serves the plant, the controller and this metric.

    Important caveat, measured rather than assumed: **for coplanar rotors the
    smallest eigenvalue is the yaw one**, and yaw authority comes from the
    propeller drag ratio ``km/kf``, which no in-plane geometry changes. On a
    SPEAR-matched quad the eigenvalues are ``[5.1e-04, 0.08, 0.08]`` with the
    small one's eigenvector exactly ``(0, 0, 1)``. So on ``Bm`` this metric is
    identical across a whole azimuth sweep *and* across a 2x change in arm
    length, while roll angular acceleration varies more than 2x over each. It
    reports "can this airframe yaw", not "is this airframe agile".

    Two ways to get a discriminating number:

    * pass ``inertia`` — ``I^-1 Bm`` divides by an inertia that does grow with
      arm length, so the metric varies (342 -> 16 as arms go 0.12 -> 0.25 m).
      It stays blind to pure rotations, since the inertia tensor rotates with
      the frame.
    * read per-axis rows of :func:`moment_allocation` when the task loads one
      axis in particular — a slalom loads roll.

    The metric comes into its own for non-coplanar (tilted) rotors, where
    geometry genuinely changes yaw authority.

    Args:
        geom: live rotor geometry in the body frame.
        torque_to_thrust_ratio: ``km/kf`` in metres.
        inertia: optional 3x3 inertia tensor; see :func:`moment_allocation`.

    Returns:
        Smallest eigenvalue, in (N·m)^2 per newton^2 (or (rad/s^2)^2 per
        newton^2 when ``inertia`` is given).
    """
    Bm = moment_allocation(
        geom, torque_to_thrust_ratio=torque_to_thrust_ratio, inertia=inertia
    )
    return float(np.min(np.linalg.eigvalsh(Bm @ Bm.T)))


def rank_controllability(
    geom: RotorGeometry,
    *,
    torque_to_thrust_ratio: float,
    tol: float | None = None,
) -> int:
    """Rank of ``Bm`` — how many moment axes the rotors can independently drive.

    airevolve's secondary maneuverability metric. 3 means full attitude
    authority; less means some axis cannot be commanded at all.
    """
    Bm = moment_allocation(geom, torque_to_thrust_ratio=torque_to_thrust_ratio)
    return int(np.linalg.matrix_rank(Bm, tol=tol))


def solve_thrusts(
    B: Array,
    wrench: Array,
    *,
    min_thrust: float = 0.0,
    max_thrust: float | None = None,
) -> tuple[Array, Array]:
    """Least-squares allocation of a desired body wrench onto rotor thrusts.

    Uses the pseudo-inverse, then clips to the thrust envelope. Clipping can
    make the achieved wrench differ from the demand — for an over-actuated or
    heavily canted airframe that is a real possibility, so the achieved
    wrench is returned for the caller to monitor rather than silently
    discarded.

    Args:
        B: allocation matrix from :func:`allocation_matrix`, shape (6, N).
        wrench: desired body wrench ``[Fx, Fy, Fz, Mx, My, Mz]``, shape (6,).
        min_thrust: lower clip, N (0 for unidirectional propellers).
        max_thrust: upper clip per rotor, N, or None for unbounded.

    Returns:
        ``(thrusts, achieved_wrench)``.
    """
    f = np.linalg.pinv(B) @ np.asarray(wrench, dtype=float)
    f = np.clip(f, min_thrust, max_thrust if max_thrust is not None else np.inf)
    return f, B @ f

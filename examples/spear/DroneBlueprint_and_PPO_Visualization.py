"""Load a DroneBlueprint + PPO hover policy and replay in MuJoCo.

 

Run:

    uv run examples/spear/17_hover_mujoco_viz.py \\

        --blueprint path/to/blueprint.json \\

        --policy path/to/hover_policy.zip

 

    uv run examples/spear/17_hover_mujoco_viz.py \\

        --blueprint path/to/blueprint.json \\

        --policy path/to/hover_policy.zip \\

        --view          # open interactive viewer instead of saving MP4

"""

from __future__ import annotations

 

import argparse

import math

import time as _time

from pathlib import Path

 

import mujoco

import numpy as np

import torch

from gymnasium import spaces

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import VecEnv

 

from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec, blueprint_to_propellers

from ariel.body_phenotypes.drone.blueprint import DroneBlueprint

from ariel.simulation.drone.drone_simulator import DroneSimulator

from ariel.simulation.environments import SimpleFlatWorld

from ariel.simulation.tasks.torch_drone_gate_env import _build_torch_dynamics

from ariel.utils.video_recorder import VideoRecorder

 

# ─────────────────────────────────────────────────────────────────────────────

# CLI

# ─────────────────────────────────────────────────────────────────────────────

 

parser = argparse.ArgumentParser(description="Replay hover policy in MuJoCo")

parser.add_argument("--blueprint", required=True, help="Path to DroneBlueprint JSON")

parser.add_argument("--policy",    required=True, help="Path to SB3 PPO ZIP")

parser.add_argument("--rollout-time", type=float, default=10.0,

                    help="Rollout duration in seconds (default 10)")

parser.add_argument("--dt",    type=float, default=0.01)

parser.add_argument("--device", default="cpu")

parser.add_argument("--out",

                    default="__data__/hover_mujoco/hover_viz.mp4",

                    help="Output MP4 path (ignored with --view)")

parser.add_argument("--view", action="store_true",

                    help="Open interactive MuJoCo viewer instead of writing MP4")

args = parser.parse_args()

 

HOVER_TARGET_NED = np.array([0.0, 0.0, -1.5], dtype=np.float32)

 

 

# ─────────────────────────────────────────────────────────────────────────────

# Hover env (single env, no resets during rollout)

# ─────────────────────────────────────────────────────────────────────────────

 

class TorchDroneHoverEnv(VecEnv):

    def __init__(self, propellers, num_envs, target_pos, device="cpu",

                 dt=0.01, max_steps=9999):

        self.drone_sim = DroneSimulator(propellers=propellers, dt=dt)

        self.num_motors = self.drone_sim.num_motors

        self.dt = dt

        self.max_steps = max_steps

        self.dev = torch.device(device)

        self.dtype = torch.float32

 

        self._dynamics = _build_torch_dynamics(

            self.drone_sim.params, self.num_motors,

            self.drone_sim.g, self.dev, self.dtype,

        )

        self.target = torch.tensor(target_pos, device=self.dev, dtype=self.dtype)

 

        n = self.num_motors

        self.state_dim = 12 + n

        VecEnv.__init__(

            self, num_envs,

            spaces.Box(-np.inf, np.inf, shape=(12 + n,), dtype=np.float64),

            spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float64),

        )

        self.states      = torch.zeros((num_envs, self.state_dim), device=self.dev, dtype=self.dtype)

        self.step_counts = torch.zeros(num_envs, device=self.dev, dtype=torch.long)

        self._acts       = torch.zeros((num_envs, n), device=self.dev, dtype=self.dtype)

        self._hover_w    = float(self._find_hover_equilibrium())

        self._action_authority = 0.4

 

    def _find_hover_equilibrium(self):

        probe = torch.zeros(self.state_dim, device=self.dev, dtype=self.dtype)

        probe[0:3] = self.target

        act_zero = torch.zeros(self.num_motors, device=self.dev, dtype=self.dtype)

        lo, hi = -1.0, 1.0

        for _ in range(40):

            mid = (lo + hi) / 2.0

            probe[12:] = mid

            sd = self._dynamics(probe.unsqueeze(1), act_zero.unsqueeze(1)).squeeze(1)

            if sd[5].item() > 0:

                lo = mid

            else:

                hi = mid

        return (lo + hi) / 2.0

 

    def _reset_envs(self, mask):

        n = int(mask.sum().item())

        if n == 0:

            return

        pos = self.target.unsqueeze(0).expand(n, -1).clone()

        pos += (torch.rand((n, 3), device=self.dev, dtype=self.dtype) - 0.5) * 0.4

        vel = (torch.rand((n, 3), device=self.dev, dtype=self.dtype) - 0.5) * 0.2

        att = torch.zeros((n, 3), device=self.dev, dtype=self.dtype)

        att[:, :2] = (torch.rand((n, 2), device=self.dev, dtype=self.dtype) - 0.5) * (math.pi / 9)

        att[:, 2]  = (torch.rand(n, device=self.dev, dtype=self.dtype) - 0.5) * math.pi

        rates  = torch.zeros((n, 3), device=self.dev, dtype=self.dtype)

        motors = torch.full((n, self.num_motors), self._hover_w, device=self.dev, dtype=self.dtype)

        self.states[mask]      = torch.cat([pos, vel, att, rates, motors], dim=1)

        self.step_counts[mask] = 0

 

    def _make_obs(self):

        obs = self.states.clone()

        obs[:, 0:3] = self.states[:, 0:3] - self.target.unsqueeze(0)

        return obs.cpu().numpy()

 

    def reset(self):

        mask = torch.ones(self.num_envs, device=self.dev, dtype=torch.bool)

        self._reset_envs(mask)

        return self._make_obs()

 

    def step_async(self, actions):

        raw = torch.as_tensor(actions, device=self.dev, dtype=self.dtype)

        self._acts = (self._hover_w + raw * self._action_authority).clamp(-1.0, 1.0)

 

    def step_wait(self):

        state_dot = self._dynamics(self.states.T, self._acts.T).T

        self.states = self.states + self.dt * state_dot

        self.step_counts += 1

        pos   = self.states[:, 0:3]

        rates = self.states[:, 9:12]

        dist  = torch.norm(pos - self.target.unsqueeze(0), dim=1)

        rews  = (-dist - 0.001 * torch.norm(rates, dim=1)) * 0.1

        dones = self.step_counts >= self.max_steps

        return self._make_obs(), rews.cpu().numpy().astype(np.float32), dones.cpu().numpy(), [{}]

 

    def close(self): pass

    def seed(self, s=None): pass

    def get_attr(self, attr_name, indices=None): raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None): pass

    def env_method(self, method_name, *args, indices=None, **kwargs): pass

    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

    def render(self, mode="human"): return {}

 

 

# ─────────────────────────────────────────────────────────────────────────────

# Load blueprint + policy

# ─────────────────────────────────────────────────────────────────────────────

 

bp = DroneBlueprint.load_json(Path(args.blueprint))

print(bp.summary())

 

propellers = blueprint_to_propellers(bp, convention="ned")

 

N = int(args.rollout_time / args.dt) + 1

env = TorchDroneHoverEnv(

    propellers=propellers, num_envs=1,

    target_pos=HOVER_TARGET_NED, device=args.device,

    dt=args.dt, max_steps=N + 1,

)

model = PPO.load(args.policy, env=None, device=args.device)

policy_obs_dim = model.observation_space.shape[0]

hover_obs_dim  = env.observation_space.shape[0]

pad = policy_obs_dim - hover_obs_dim

if pad < 0:

    raise ValueError(

        f"Policy obs dim ({policy_obs_dim}) < hover env obs dim ({hover_obs_dim}). "

        "Are you sure this policy matches the blueprint?"

    )

if pad > 0:

    print(

        f"Policy expects {policy_obs_dim}-dim obs, hover env provides {hover_obs_dim}. "

        f"Zero-padding {pad} gate dims (treats current position as target gate)."

    )

print(f"Policy loaded: {args.policy}")

 

# ─────────────────────────────────────────────────────────────────────────────

# Rollout

# ─────────────────────────────────────────────────────────────────────────────

 

pos_ned   = np.zeros((N, 3), dtype=np.float32)

euler_log = np.zeros((N, 3), dtype=np.float32)

 

obs = env.reset()

for i in range(N):

    pos_ned[i]   = env.states[0, 0:3].cpu().numpy()

    euler_log[i] = env.states[0, 6:9].cpu().numpy()

    if pad > 0:

        obs = np.concatenate([obs, np.zeros((obs.shape[0], pad), dtype=obs.dtype)], axis=1)

    action, _ = model.predict(obs, deterministic=True)

    obs, _, _, _ = env.step(action)

 

print(

    f"Rollout done: {N} steps ({N * args.dt:.1f}s)  "

    f"alt range (NED z): [{pos_ned[:, 2].min():.2f}, {pos_ned[:, 2].max():.2f}]"

)

 

# NED → ENU

pos_enu = pos_ned.copy()

pos_enu[:, 2] = -pos_enu[:, 2]

 

# Euler → quaternion (w, x, y, z) with NED→ENU y-flip

phi, theta, psi = euler_log[:, 0], euler_log[:, 1], euler_log[:, 2]

cy, sy = np.cos(psi / 2), np.sin(psi / 2)

cp, sp = np.cos(theta / 2), np.sin(theta / 2)

cr, sr = np.cos(phi / 2), np.sin(phi / 2)

quat_enu = np.stack([

    cr * cp * cy + sr * sp * sy,

    sr * cp * cy - cr * sp * sy,

    -(cr * sp * cy + sr * cp * sy),

    cr * cp * sy - sr * sp * cy,

], axis=-1)

 

# ─────────────────────────────────────────────────────────────────────────────

# Build MuJoCo scene

# ─────────────────────────────────────────────────────────────────────────────

 

arm_lengths = [bp.payload(a).length for a in bp.children(bp.root_id)]  # type: ignore[arg-type]

mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06

 

drone_spec = blueprint_to_mjspec(

    bp,

    motor_mass=0.01,

    arm_mass=0.034 * mean_arm_len,

    body_name="hover_quad",

)

world_mj = SimpleFlatWorld()

world_mj.spawn(

    drone_spec,

    position=(float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])),

    correct_collision_with_floor=False,

)

 

# Hover target marker (green sphere at ENU z=+1.5)

target_body = world_mj.spec.worldbody.add_body(

    name="hover_target",

    pos=[0.0, 0.0, float(-HOVER_TARGET_NED[2])],

)

target_body.add_geom(

    name="hover_target_geom",

    type=mujoco.mjtGeom.mjGEOM_SPHERE,

    size=[0.06, 0.0, 0.0],

    rgba=(0.1, 0.9, 0.1, 0.8),

    contype=0,

    conaffinity=0,

)

 

model_mj = world_mj.spec.compile()

data_mj  = mujoco.MjData(model_mj)

model_mj.opt.timestep = args.dt

 

 

def _set_pose(idx: int) -> None:

    p, q = pos_enu[idx], quat_enu[idx]

    data_mj.qpos[0:3] = [float(p[0]), float(p[1]), float(p[2])]

    data_mj.qpos[3:7] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]

    data_mj.qvel[:] = 0.0

    mujoco.mj_forward(model_mj, data_mj)

 

 

# ─────────────────────────────────────────────────────────────────────────────

# View or render

# ─────────────────────────────────────────────────────────────────────────────

 

if args.view:

    import mujoco.viewer

    print("Launching MuJoCo passive viewer …")

    _set_pose(0)

    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:

        idx = 0

        while viewer.is_running() and idx < len(pos_enu):

            t0 = _time.time()

            _set_pose(idx)

            viewer.sync()

            slack = args.dt - (_time.time() - t0)

            if slack > 0:

                _time.sleep(slack)

            idx += 1

else:

    mp4 = Path(args.out)

    mp4.parent.mkdir(parents=True, exist_ok=True)

    recorder = VideoRecorder(

        file_name=mp4.stem, output_folder=mp4.parent,

        width=720, height=540, fps=30,

    )

    steps_per_frame = max(1, int(round(1.0 / (recorder.fps * args.dt))))

    t_render = _time.time()

    with mujoco.Renderer(model_mj, width=recorder.width, height=recorder.height) as renderer:

        for idx in range(0, len(pos_enu), steps_per_frame):

            _set_pose(idx)

            renderer.update_scene(data_mj)

            recorder.write(frame=renderer.render())

    recorder.release()

    print(f"Rendered in {_time.time() - t_render:.2f}s")

    print(f"Video: {mp4}")

 

print("Done.")

 
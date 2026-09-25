#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F



class MorphingDroneWrenchModel:
    def __init__(
        self,
        allocation_json_path: str | Path,
        num_envs: int,
        device: str | torch.device = "cuda:0",
        use_full_wrench: bool = True,
        max_rotor_thrust: float = 15.0,
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.use_full_wrench = use_full_wrench
        self.max_rotor_thrust = max_rotor_thrust

        self._load_allocation(allocation_json_path)

    def _load_allocation(self, allocation_json_path: str | Path):
        allocation_json_path = Path(allocation_json_path).expanduser().resolve()
        data = json.loads(allocation_json_path.read_text())

        self.morphology_id = data["morphology_id"]
        self.rotor_names = data["rotor_names"]
        self.num_rotors = len(self.rotor_names)

        rotor_positions = torch.tensor(data["rotor_positions"], dtype=torch.float32, device=self.device)
        rotor_directions = torch.tensor(data["rotor_directions"], dtype=torch.float32, device=self.device)
        rotor_directions = F.normalize(rotor_directions, dim=-1)

        spin_signs_dict = data["spin_signs"]
        spin_signs = torch.tensor([spin_signs_dict[name] for name in self.rotor_names], dtype=torch.float32, device=self.device)

        self.rotor_pos_b = rotor_positions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.rotor_dir_b = rotor_directions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.spin_sign = spin_signs.unsqueeze(0).repeat(self.num_envs, 1)

        self.thrust_coeff = float(data.get("thrust_coeff", 1.0))
        self.moment_coeff = float(data.get("moment_coeff", 0.05))

        self.B_full = torch.tensor(data["B_full"], dtype=torch.float32, device=self.device)
        self.rank_full = int(data.get("rank_full", np.linalg.matrix_rank(self.B_full.cpu().numpy())))
        self.rank_rpyt = int(data.get("rank_rpyt", 0))
        self.cond_full = float(data.get("cond_full", 0.0))
        self.cond_rpyt = float(data.get("cond_rpyt", 0.0))

    def set_morphology(self, allocation_json_path: str | Path):
        self._load_allocation(allocation_json_path)

    def motor_commands_to_thrusts(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clamp(-1.0, 1.0)
        thrust_cmd = (actions + 1.0) / 2.0
        return self.max_rotor_thrust * thrust_cmd

    def compute_body_wrench_from_thrusts(self, rotor_thrusts: torch.Tensor):
        if rotor_thrusts.ndim != 2 or rotor_thrusts.shape[1] != self.num_rotors:
            raise ValueError(f"Expected rotor_thrusts shape (num_envs, {self.num_rotors}), got {tuple(rotor_thrusts.shape)}")

        rotor_thrusts = rotor_thrusts.to(self.device)
        f_i_b = rotor_thrusts.unsqueeze(-1) * self.rotor_dir_b
        tau_arm_b = torch.cross(self.rotor_pos_b, f_i_b, dim=-1)
        tau_drag_b = self.moment_coeff * rotor_thrusts.unsqueeze(-1) * self.spin_sign.unsqueeze(-1) * self.rotor_dir_b

        force_b = f_i_b.sum(dim=1)
        torque_b = (tau_arm_b + tau_drag_b).sum(dim=1)
        return force_b, torque_b

    def compute_body_wrench_from_allocation(self, rotor_thrusts: torch.Tensor):
        if rotor_thrusts.ndim != 2 or rotor_thrusts.shape[1] != self.num_rotors:
            raise ValueError(f"Expected rotor_thrusts shape (num_envs, {self.num_rotors}), got {tuple(rotor_thrusts.shape)}")

        B = self.B_full.unsqueeze(0).repeat(self.num_envs, 1, 1)
        wrench = torch.bmm(B, rotor_thrusts.unsqueeze(-1)).squeeze(-1)
        force_b = wrench[:, 0:3]
        torque_b = wrench[:, 3:6]
        return force_b, torque_b

    def compute_body_wrench(self, rotor_thrusts: torch.Tensor):
        if self.use_full_wrench:
            return self.compute_body_wrench_from_allocation(rotor_thrusts)
        return self.compute_body_wrench_from_thrusts(rotor_thrusts)

    def apply_to_buffers(
        self,
        thrust_buffer: torch.Tensor,
        moment_buffer: torch.Tensor,
        rotor_thrusts: torch.Tensor,
        body_slot: int = 0,
    ):
        force_b, torque_b = self.compute_body_wrench(rotor_thrusts)
        thrust_buffer[:, body_slot, :] = force_b
        moment_buffer[:, body_slot, :] = torque_b
        return force_b, torque_b



# Example Isaac Lab integration
class ExampleMorphingDroneControllerMixin:
    def setup_wrench_model(self, allocation_json_path: str):
        self.wrench_model = MorphingDroneWrenchModel(
            allocation_json_path=allocation_json_path,
            num_envs=self.num_envs,
            device=self.device,
            use_full_wrench=True,
            max_rotor_thrust=self.cfg.max_rotor_thrust,
        )

    def switch_morphology(self, allocation_json_path: str):
        self.wrench_model.set_morphology(allocation_json_path)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        rotor_actions = self._actions[:, : self.wrench_model.num_rotors]
        rotor_thrusts = self.wrench_model.motor_commands_to_thrusts(rotor_actions)
        self.wrench_model.apply_to_buffers(
            thrust_buffer=self._thrust,
            moment_buffer=self._moment,
            rotor_thrusts=rotor_thrusts,
            body_slot=0,
        )

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )
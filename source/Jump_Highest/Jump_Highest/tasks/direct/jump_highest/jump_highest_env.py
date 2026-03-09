# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .jump_highest_env_cfg import JumpHighestEnvCfg


class JumpHighestEnv(DirectRLEnv):
    cfg: JumpHighestEnvCfg

    def __init__(self, cfg: JumpHighestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # apply torques to ALL joints
        self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot.data.root_pos_w[:, 2:3],             # root height (1)
                self.robot.data.root_quat_w,                     # root orientation (4)
                self.robot.data.root_lin_vel_w,                  # root linear velocity (3)
                self.robot.data.root_ang_vel_w,                  # root angular velocity (3)
                self.robot.data.joint_pos,                       # all joint positions (21)
                self.robot.data.joint_vel,                       # all joint velocities (21)
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_jump_vel,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_termination,
            self.cfg.rew_scale_energy,
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.reset_terminated,
            self.actions,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # terminated if root height drops below threshold (fallen over)
        fallen = self.robot.data.root_pos_w[:, 2] < self.cfg.min_height
        return fallen, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_jump_vel: float,
    rew_scale_height: float,
    rew_scale_alive: float,
    rew_scale_upright: float,
    rew_scale_terminated: float,
    rew_scale_energy: float,
    root_lin_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
):
    # reward upward (z) velocity
    jump_vel = torch.clamp(root_lin_vel[:, 2], min=0.0)
    rew_jump = rew_scale_jump_vel * jump_vel

    # reward height above default standing (~1.34m for humanoid)
    height_above_default = torch.clamp(root_pos[:, 2] - 1.34, min=0.0)
    rew_height = rew_scale_height * height_above_default

    # alive bonus
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # upright reward
    upright = 1.0 - 2.0 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
    rew_upright = rew_scale_upright * torch.clamp(upright, min=0.0)

    # termination penalty
    rew_terminated = rew_scale_terminated * reset_terminated.float()

    # energy penalty - penalize large torques
    rew_energy = rew_scale_energy * torch.sum(torch.square(actions), dim=-1)

    total_reward = rew_jump + rew_height + rew_alive + rew_upright + rew_terminated + rew_energy
    return total_reward
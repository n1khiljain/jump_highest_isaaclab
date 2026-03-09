# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.humanoid import HUMANOID_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

print("LOADING JUMP HIGHEST CONFIG - action_space=21, obs=54")

@configclass
class JumpHighestEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 21
    observation_space = 54  # 1 + 4 + 3 + 3 + 21 + 21
    state_space = 0
    action_scale = 100.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # jump reward
    rew_scale_jump_vel = 0.5        # reward upward velocity
    rew_scale_height = 0.5          # reward height above ground
    rew_scale_alive = 5.0           # reward for not falling
    rew_scale_upright = 2.0         # reward for staying upright
    rew_scale_termination = -5.0    # penalty for falling

    # termination
    min_height = 0.5 # reset if root drops below this 
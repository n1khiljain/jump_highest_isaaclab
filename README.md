# Jump Highest — Humanoid Jumping Environment

This repository contains an environment designed to train a humanoid robot to jump as high as possible. The environment and training scripts are tailored for reinforcement learning experiments where the objective is maximizing peak jump height.

## Goal
- Train a humanoid (simulated) agent to perform vertical jumps and maximize the apex height reached during each episode.

## Environment Overview
- Observation space: joint angles, joint velocities, body orientation (quaternion or Euler), center-of-mass height and velocity, foot contact flags, and optionally sensor readings (IMU). Observations are stacked across a short window for better temporal context.
- Action space: continuous torques or target joint position velocities for the humanoid's actuators. Typical dimensionality matches the number of controllable joints.
- Reward: primary reward is proportional to the maximum center-of-mass (COM) height achieved in the episode. Shaping terms may include energy penalties, standing stability bonus, or time-to-apex incentives to encourage explosive but safe jumps.
- Episode termination: episode ends after a fixed time horizon (e.g., 2–4 seconds), on severe falls (torso hitting the ground), or when the agent has completed a jump and landed. Logged metric: peak COM height per episode.

## Files and Scripts
- Training entrypoint: `scripts/rsl_rl/train.py` — run experiments and configure training hyperparameters through the CLI.
- Evaluation / play: `scripts/rsl_rl/play.py` — load a trained model and run rollouts to visualize jumps.
- Agents: simple baselines are available at `scripts/random_agent.py` and `scripts/zero_agent.py`.
- Logs: training logs, TensorBoard events, and model checkpoints are saved under `logs/rsl_rl/cartpole_direct/` and `outputs/` subfolders (see the timestamped runs).

## Quickstart — Training
1. Install dependencies (recommended to use a virtualenv).

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run a training experiment (example):

```powershell
python scripts\rsl_rl\train.py --env jump_highest --algo ppo --total-timesteps 1000000 --log-dir logs/rsl_rl/jump_highest/exp1
```

Adjust `--env`, `--algo`, and hyperparameters as needed. See `scripts/rsl_rl/cli_args.py` for available CLI options.

## Recommended Reward & Hyperparameters
- Reward: `r = alpha * peak_height - beta * energy_penalty - gamma * fall_penalty` (tune alpha/beta/gamma).
- PPO starter hyperparameters: learning rate 3e-4, clip 0.2, n-steps 2048, batch size 64, epochs 10.

## Evaluation
- Use `scripts/rsl_rl/play.py` to load a checkpoint in `logs/.../exported/` and render jump episodes. Record peak COM height and average over multiple seeds.

## Tips
- Normalize observations (height, velocities) to stabilize training.
- Reward peak height at episode end to avoid noisy per-step incentives.
- Use curriculum learning: begin with smaller target heights or reduced gravity, then anneal to full difficulty.

## Contact
If you want help tuning or adding visualizations, open an issue or contact the maintainer.

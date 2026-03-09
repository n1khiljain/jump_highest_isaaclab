# Jump Highest (Isaac Lab / RSL-RL)

This repo is an **Isaac Lab extension** that adds a **Direct RL** humanoid task where the agent learns to jump upward (maximize height while staying upright).

## What you get

- **Gym task id**: `Template-Jump-Highest-Direct-v0`
- **Env type**: `DirectRLEnv` with torque control
- **Spaces**: action dim **21**, observation dim **54**
- **Episode length**: **5.0s**

## Prerequisites

You need an **Isaac Lab** Python environment (which includes Isaac Sim + the `isaaclab*` packages). These scripts are meant to be run with the Isaac Lab launcher (on Windows that’s typically `isaaclab.bat`).

If you run `scripts/rsl_rl/train.py` with the wrong Python, imports like `isaaclab.app` will fail.

If you are using the Isaac Lab launcher, you can usually replace `python <script>` with:

```powershell
.\isaaclab.bat -p <script> <args...>
```

## Install (editable)

From your Isaac Lab Python environment, install this extension in editable mode:

```powershell
python -m pip install -e .\source\Jump_Highest
```

## Verify the task is registered

This should print a table containing `Template-Jump-Highest-Direct-v0`:

```powershell
python .\scripts\list_envs.py --keyword Jump
```

## Train (PPO via RSL-RL)

Minimal training run:

```powershell
python .\scripts\rsl_rl\train.py --task Template-Jump-Highest-Direct-v0 --num_envs 1024 --headless
```

Useful flags:
- `--experiment_name jump_highest`: controls the log folder name (defaults come from the agent config).
- `--max_iterations 1000`: number of training iterations.
- `--video`: record training videos to the run folder.

Example with explicit log naming + videos:

```powershell
python .\scripts\rsl_rl\train.py `
  --task Template-Jump-Highest-Direct-v0 `
  --headless `
  --num_envs 1024 `
  --experiment_name jump_highest `
  --run_name ppo_baseline `
  --video --video_interval 2000 --video_length 200
```

## Play / evaluate a checkpoint

Play the latest checkpoint from a previous run (uses `--load_run` + `--checkpoint`), or pass an explicit path via `--checkpoint`.

```powershell
python .\scripts\rsl_rl\play.py --task Template-Jump-Highest-Direct-v0 --num_envs 1 --real-time
```

To record a short rollout video:

```powershell
python .\scripts\rsl_rl\play.py --task Template-Jump-Highest-Direct-v0 --num_envs 1 --video --video_length 400
```

## Outputs & where to look

Training creates a run directory under:

`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`

Inside each run folder you’ll typically find:
- `params/env.yaml`, `params/agent.yaml`: the exact configs used
- `videos/train/` and `videos/play/` (if enabled)
- `exported/policy.pt` and `exported/policy.onnx` (exported on play)

## Environment details (current implementation)

The task is implemented in:
- `source/Jump_Highest/Jump_Highest/tasks/direct/jump_highest/jump_highest_env.py`
- `source/Jump_Highest/Jump_Highest/tasks/direct/jump_highest/jump_highest_env_cfg.py`

Current reward terms (see `compute_rewards`):
- **Upward velocity** reward (`root_lin_vel_w[:,2]`)
- **Height above default** standing height (~1.34m)
- **Alive** bonus (not terminated)
- **Upright** bonus (penalizes tipping)
- **Energy** penalty (squared actions)
- **Termination** penalty (fall)

Termination / done conditions:
- **Fallen** when root height < `min_height` (default `0.5`)
- **Timeout** at 5 seconds

## Troubleshooting

- **It logs to `cartpole_direct`**: that’s coming from the agent config (`rsl_rl_ppo_cfg.py`). Override with `--experiment_name jump_highest`.
- **Out of memory / slow**: start with `--num_envs 256` or `1024` (the default environment config sets `4096` which can be heavy).
- **RSL-RL version error**: `train.py` checks for `rsl-rl-lib==3.0.1` and prints the exact install command it expects.

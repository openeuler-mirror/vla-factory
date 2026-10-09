# SimplerEnv Platform Integration

> 中文：[simplerenv.cn.md](./simplerenv.cn.md)

VLA Factory integrates with the [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
benchmark through a client/server boundary, following the same architecture as
the RoboTwin and RoboCasa integrations. The model and its dependencies run in
the VLA Factory environment, while the SimplerEnv simulator (ManiSkill /
SAPIEN) runs in a separate environment — SimplerEnv pins its own SAPIEN and
ManiSkill versions that conflict with the framework's torch line. The two
processes communicate through the same length-prefixed JSON-RPC protocol
RoboTwin and RoboCasa use. A dependency-free external connector reads the
camera frame and end-effector state from the observation dict, forwards them
together with the language instruction and step counter; dimension validation
and model preprocessing are performed by the VLA Factory server from the
checkpoint metadata.

## 1. Prepare the two environments

Install SimplerEnv in its own environment by following its
[official installation guide](https://github.com/simpler-env/SimplerEnv#installation)
(a Python ≥3.9 conda env, then `pip install -e .` plus its ManiSkill/SAPIEN
extra). Separate environments are required for VLA Factory and SimplerEnv so
that OpenPI, LeRobot, and SAPIEN dependencies do not conflict.

Install the required model dependencies in the VLA Factory environment (e.g.
the pi0 or openvla model env from `scripts/install.sh`). No `simpler_env`
package is needed there — the connector runs only on the simulator side and
raises an actionable install hint when absent.

## 2. Start the model server (VLA Factory environment)

```bash
vlafactory-cli deploy \
  --checkpoint outputs/<checkpoint> \
  --platform simplerenv \
  --host 0.0.0.0 \
  --port 9999
```

At startup, the server prints the camera list, `state_dim`, and `action_dim`
required by the checkpoint, then listens on the TCP port. The server uses the
`SimplerEnvAdapter` to translate the connector's observation into the
checkpoint's `DataSchema` keys. Use `--strategy receding_horizon` (the default
for this platform) to re-plan every few action steps.

## 3. Run the benchmark (SimplerEnv environment)

The closed-loop benchmark makes one env per task (`simpler_env.make(task)`),
runs `--trials` episodes on it, forwards the third-person camera frame and
`obs["agent"]["eef_pos"]` to the model server, executes the returned action
chunk via `env.step`, and collects the binary success flag into a per-task
success-rate report:

```bash
export VLA_FACTORY_PATH=/path/to/vla-factory

PYTHONPATH="$VLA_FACTORY_PATH${PYTHONPATH:+:$PYTHONPATH}" \
python -m vla_factory.inference.connectors.simplerenv \
  --port 9999 \
  --tasks google_robot_pick_coke_can widowx_put_eggplant_in_basket \
  --trials 25 \
  --report simplerenv_report.json
```

Useful flags: `--cameras` (single camera; by default inferred from the task
prefix — `overhead_camera` for `google_robot_*`, `3rd_view_camera` for
`widowx_*`), `--instruction` (override the per-episode
`get_language_instruction()`), `--gripper-mode` (see below), `--max-steps`
(by default the env's registered TimeLimit ends the episode), and `--seed`.

## Runtime contract

- **Cameras**: SimplerEnv exposes a single third-person camera per
  embodiment — `overhead_camera` (Google robot) / `3rd_view_camera` (WidowX)
  — and no wrist camera. Single-camera checkpoints are matched to it by
  name (with aliasing) or position; multi-camera checkpoints cannot be
  evaluated on SimplerEnv and the server reports this before inference.
  Images are forwarded at the env's native resolution; the checkpoint's
  saved transform pipeline resizes them server-side.
- **State**: `obs["agent"]["eef_pos"]`, 8-D = translation (3) + quaternion
  (4, wxyz) + gripper (1). The server validates it against the checkpoint's
  `state_dim` before inference, reporting required and available values on
  mismatch.
- **Actions**: both embodiments consume flat 7-D vectors
  `[dx, dy, dz, droll, dpitch, dyaw, gripper]`. 7-D checkpoints pass
  through; 4-D checkpoints (xyz + gripper) are expanded with zero
  rotations; anything else is rejected. Because gripper conventions vary
  across training pipelines, `--gripper-mode` adapts the last dimension:
  `raw` (default, passthrough), `google` ([0,1] → [-1,1] plus sticky-close
  repetition, matching the Google-robot tasks' gripper dynamics), `widowx`
  (binarise around 0.5). The result is clipped to [-1, 1].
- **Instruction**: `env.unwrapped.get_language_instruction()` per episode
  (e.g. "pick up the coke can"); `--instruction` overrides it. Models such
  as Diffusion Policy ignore it.
- **Success**: read from the terminal step — `info["success"]`, episode
  termination, or the wrapper's latched `success_once`. TimeLimit
  truncation alone is not success.
- The connector lives at `vla_factory.inference.connectors.simplerenv` and
  has no torch, Transformers, OpenPI, or LeRobot dependencies, so the
  SimplerEnv environment can import it directly.

## Protocol caveat

This runner resets with `env.reset(seed=seed + trial)` — the prepackaged
initial-state protocol. SimplerEnv's official benchmark additionally varies
robot/object initial states via `options` (variant aggregation) and supports
a visual-matching setting with RGB overlays. Success rates from this runner
are therefore useful for relative comparisons (ablations, checkpoints) but
are **not directly comparable to the published SimplerEnv benchmark
numbers**.

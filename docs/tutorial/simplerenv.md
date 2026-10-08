# SimplerEnv Platform Integration

> 中文：[simplerenv.cn.md](./simplerenv.cn.md)

VLA Factory integrates with the [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
benchmark through a client/server boundary, following the same architecture as
the RoboTwin and RoboCasa integrations. The model and its dependencies run in
the VLA Factory environment, while the SimplerEnv simulator (ManiSkill / SAPIEN)
runs in a separate environment — SimplerEnv pins its own SAPIEN, ManiSkill, and
dataset-server versions that conflict with the framework's torch line. The two
processes communicate through the same length-prefixed JSON-RPC protocol
RoboTwin and RoboCasa use. A dependency-free external connector reads the
cameras and robot state from the environment, forwards them together with the
language instruction and step counter; dimension validation and model
preprocessing are performed by the VLA Factory server from the checkpoint
metadata.

## 1. Prepare the two environments

Install SimplerEnv in its own environment by following its
[official installation guide](https://github.com/simpler-env/SimplerEnv#installation)
(a Python ≥3.9 conda env, then `pip install -e .` plus its ManiSkill/SAPIEN
extra). Separate environments are required for VLA Factory and SimplerEnv so
that OpenPI, LeRobot, and SAPIEN dependencies do not conflict.

Install the required model dependencies in the VLA Factory environment (e.g.
the pi0 or openvla model env from `scripts/install.sh`). No `simpler_env`
package is needed there — the deploy command probes it only on the simulator
side and the connector raises an actionable install hint when absent.

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
for this platform) to re-plan every 5 action steps.

## 3. Run the benchmark (SimplerEnv environment)

The closed-loop benchmark makes one SimplerEnv env per task × `--trials`
episodes, forwards the agentview/wrist camera frames and
`env.get_robot_state()` to the model server, executes the returned action
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

Useful flags: `--cameras` (default `agentview robot0_eye_in_hand`), 
`--image-size W H` (default 224 224 — must match the checkpoint's training
resolution), `--max-steps` (default: SimplerEnv's per-task horizon table,
280 for Google-robot tasks and 150 for WidowX tasks), and `--seed`.

## Runtime contract

- SimplerEnv consumes raw flat actions in the env's native action space:
  4-D for the Google robot (xyz delta + gripper) and 7-D for WidowX
  (end-effector delta + gripper). The connector validates the model's action
  width against `env.action_space` and clips to [-1, 1]; the checkpoint's
  action semantics must already match the embodiment.
- The robot state is `env.get_robot_state()`: 8-D per arm (gripper open
  amount + end-effector pos/quat). The server validates it against the
  checkpoint's `state_dim` before inference, reporting required and available
  values on mismatch.
- The runtime task must expose every camera recorded in the checkpoint schema.
  Checkpoint camera `wrist` maps to SimplerEnv's `robot0_eye_in_hand`.
- Language-conditioned models receive the task name as the instruction;
  models such as Diffusion Policy ignore it.
- Success is read from the episode's terminal step (termination, the
  `success` info key, or the wrapper's latched `success_once`).
- The connector lives at `vla_factory.inference.connectors.simplerenv` and has
  no torch, Transformers, OpenPI, or LeRobot dependencies, so the SimplerEnv
  environment can import it directly.

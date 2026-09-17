# RoboCasa365 Platform Integration

> 中文：[robocasa.cn.md](./robocasa.cn.md)

VLA Factory integrates with the RoboCasa365 benchmark through a client/server
boundary. The model and its dependencies run in the VLA Factory environment,
while the RoboCasa gymnasium simulator runs in a separate environment (robocasa
pulls in robosuite master, mujoco, ~10GB of kitchen 3D assets, and a numba pin
that conflicts with the framework's torch line). The two processes communicate
through the same length-prefixed JSON-RPC protocol RoboTwin uses. A
dependency-free external connector forwards the native gym observation, language
instruction, and step counter; camera selection, state-vector extraction,
dimension validation, and model preprocessing are performed by the VLA Factory
server from the checkpoint metadata.

RoboCasa365 ships LeRobot-format datasets (parquet + MP4 + v3-style `meta/`)
labeled `codebase_version: v2.1`. The `lerobot-v3` reader detects this by
structure (a parseable `meta/info.json` with a features map plus parquet
shards under `data/`), so no new reader is needed and no version string is
compared.

## 1. Prepare the two environments

Install RoboCasa in its own environment by following its
[official installation guide](https://robocasa.ai/docs/introduction/overview.html).
Separate Python/Conda environments are required for VLA Factory and RoboCasa so
that OpenPI, LeRobot, mujoco/robosuite, and numba dependencies do not conflict.

Install the required model dependencies in the VLA Factory environment (e.g. the
diffusion_policy or pi0 model env from `scripts/install.sh`). No `robocasa`
extra is needed there — it is a metadata-only extra; the download, deploy, and
evaluate commands probe the package at runtime and fall back with an actionable
install hint when it is absent.

## 2. Download the benchmark data

RoboCasa365 data is fetched from Box (hosted by UT Austin) and lands under the
robocasa install tree's `datasets/v1.0/`. To redirect it, set
`DATASET_BASE_PATH` in `robocasa/macros_private.py`. Use the upstream robocasa script
directly from the robocasa environment (VLA Factory does not wrap the
download — robocasa's asset/mujoco dependencies belong in that environment):

```bash
cd /path/to/robocasa && python -m robocasa.scripts.download_datasets \
  --split target --tasks PickPlaceCounterToCabinet

# Preview without fetching (lists every task that would be downloaded):
python -m robocasa.scripts.download_datasets --split target --dryrun

# Kitchen 3D assets (~10GB, needed the first time the gym env is built):
python -m robocasa.scripts.download_kitchen_assets
```

`--split` accepts multiple values; `--source human|mimicgen` filters the
pretrain collection. See the upstream
[robocasa docs](https://robocasa.ai/docs/introduction/overview.html) for the full flag
set. The downloaded LeRobot data is then read by VLA Factory's `lerobot-v3`
reader with no extra configuration.

## 3. Train / fine-tune on RoboCasa365 data

In a training recipe, point the data source at the downloaded LeRobot root and
bind the Panda+Omron body:

```yaml
data:
  path: /path/to/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
  format: lerobot-v3
  video_codec: pyav
robot:
  name: panda_omron
```

See `examples/robocasa365_diffusion_policy.yaml` and
`examples/robocasa365_pi0_lora.yaml` for full recipes. The action is a 12-D
EEF delta (OSC end-effector + gripper + base), so the robot profile declares
`native_action_type: eef_delta` and the camera semantic rules map
`robot0_eye_in_hand` → `wrist` and `robot0_agentview_left/right` →
`third_person_front`.

## 4. Start the model server (VLA Factory environment)

```bash
vlafactory-cli deploy \
  --checkpoint outputs/<checkpoint> \
  --platform robocasa \
  --host 0.0.0.0 \
  --port 9999
```

At startup, the server prints the camera list, `state_dim`, and `action_dim`
required by the checkpoint, then listens on the TCP port. The server uses the
`RoboCasaAdapter` to translate the gym observation into the checkpoint's
`DataSchema` keys.

## 5. Run the benchmark (RoboCasa environment)

The closed-loop benchmark drives `gym.make("robocasa/<task>", split=...)` × N
scenes per task, forwarding each observation to the model server and executing
the returned action chunk, then collecting the binary success flag. Run it from
the RoboCasa environment:

```bash
export VLA_FACTORY_PATH=/path/to/vla-factory

PYTHONPATH="$VLA_FACTORY_PATH${PYTHONPATH:+:$PYTHONPATH}" \
python -m vla_factory.inference.connectors.robocasa \
  --port 9999 \
  --tasks PickPlaceCounterToCabinet \
  --trials 50 \
  --report robocasa_report.json
```

The connector drives the gymnasium env, connects to the model server at
`--port`, and writes a JSON success-rate report. The client and server ports
must match. At inference time the server executes the action chunk via its
`PolicyExecutor`; use `--strategy receding_horizon --n-action-steps 5` on
`deploy` to match the upstream eval protocol (re-plan every 5 steps).

For offline evaluation (predicted vs recorded action L1, no simulator), use
the standard `evaluate` command — it reads the LeRobot dataset directly and is
format-agnostic:

```bash
vlafactory-cli evaluate \
  --checkpoint outputs/<checkpoint> \
  --dataset /path/to/robocasa/datasets/.../PickPlaceCounterToCabinet/20250811/lerobot
```

## Runtime contract

- Training and model output retain the dataset's 12-D `modality.json` order:
  base motion (4), control mode (1), EEF translation (3), EEF rotation (3),
  gripper (1). The connector names those segments when forming RoboCasa's
  `action.*` gym dictionary.
- The runtime task must expose every camera recorded in the checkpoint schema.
  Camera or state-dimension mismatches are reported by the server before
  inference, including required and available values.
- Language-conditioned models such as PI0 and PI0.5 receive the task
  instruction; models such as Diffusion Policy ignore it.
- The connector lives at `vla_factory.inference.connectors.robocasa` and has no
  torch, Transformers, OpenPI, or LeRobot dependencies, so the RoboCasa
  environment can import it directly.

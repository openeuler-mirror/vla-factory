# RoboTwin Platform Integration

> 中文：[robotwin.cn.md](./robotwin.cn.md)

VLA Factory integrates with RoboTwin through a client/server boundary. The
model and its dependencies run in the VLA Factory environment, while the
SAPIEN simulator runs in a separate RoboTwin environment. The two processes
communicate through RoboTwin's native TCP model protocol. A dependency-free
external connector forwards the native observation, language instruction, and
step counter; camera selection, joint-state extraction, dimension validation,
and model preprocessing are performed by the VLA Factory server from the
checkpoint metadata. No adapter code needs to be copied into the RoboTwin
repository, and there is no model-specific `CAMERAS` list to maintain.

## 1. Prepare the two environments

Install RoboTwin by following its
[official installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
Separate Python/Conda environments are recommended for VLA Factory and
RoboTwin so that OpenPI, LeRobot, and SAPIEN dependencies do not conflict.

Install the required model dependencies in the VLA Factory environment. Add
the `robotwin` extra when reading native RoboTwin HDF5 datasets:

```bash
cd /path/to/vla-factory
pip install -e ".[robotwin]"
```

## 2. Configure native RoboTwin training data

In a training recipe, point the data source at a RoboTwin task configuration
directory containing `data/episode*.hdf5` and
`instructions/episode*.json`:

```yaml
data:
  source:
    path: /path/to/dataset/<task>/<embodiment>_clean_50
    format: robotwin
    video_codec: hdf5_jpeg
```

## 3. Start the model server (VLA Factory environment)

```bash
vlafactory-cli serve \
  --checkpoint outputs/<checkpoint> \
  --platform robotwin \
  --host 0.0.0.0 \
  --port 9999
```

At startup, the server prints the camera list, `state_dim`, and `action_dim`
required by the checkpoint, then listens on the TCP port. Do not also start
RoboTwin's `policy_model_server.py`.

## 4. Start evaluation (RoboTwin environment)

Run the following from the RoboTwin repository root:

```bash
export VLA_FACTORY_PATH=/path/to/vla-factory

PYTHONPATH="$VLA_FACTORY_PATH${PYTHONPATH:+:$PYTHONPATH}" \
python script/eval_policy_client.py \
  --port 9999 \
  --config "$VLA_FACTORY_PATH/vla_factory/deploy/configs/robotwin.yml" \
  --overrides \
    task_name beat_block_hammer \
    task_config demo_randomized \
    ckpt_setting vla_factory \
    instruction_type unseen \
    seed 0
```

`robotwin.yml` is the minimal bootstrap configuration required by RoboTwin's
`eval_policy_client.py`; it only selects the external connector. The actual
model and checkpoint remain in the server process. The client and server ports
must match. RoboTwin writes results under its `eval_result/` directory.

## Runtime contract

- Aloha-AgileX actions are executed as qpos and are typically 14-D: six arm
  joints plus one gripper joint per side.
- The runtime task must expose every camera recorded in the checkpoint schema.
  Camera or state-dimension mismatches are reported by the server before
  inference, including required and available values.
- Language-conditioned models such as PI0 and PI0.5 receive the instruction
  returned by `TASK_ENV.get_instruction()`; models such as ACT ignore it.
- The connector lives at `vla_factory.deploy.robotwin_connector` and has no
  torch, Transformers, OpenPI, or LeRobot dependencies, so the RoboTwin
  environment can import it directly.

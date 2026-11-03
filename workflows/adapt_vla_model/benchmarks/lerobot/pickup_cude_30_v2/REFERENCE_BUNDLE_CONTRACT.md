# Reference Bundle Contract: __VERSION__

This contract is for the offline LeRobot benchmark
`leisaac/pickup_cude_30_v2`. It defines `policy_impl.py` and
`policy_spec.json` for native forward/loss parity. The Judge does not launch a
robot TCP client for this recorded-data benchmark.

Implement:

```python
from policy_contract import Policy

def build_policy(checkpoint: str) -> Policy:
    ...
```

`reset(task_info)` clears any action queue. `act(observation)` returns an
ordered `float[action_horizon, 6]` joint-position action chunk. Its horizon is
the selected checkpoint's configured action chunk size; never guess it from
this dataset.

Both `act()` and `compute_loss()` receive this complete mapping (the latter
also receives `training`):

```python
{
  "lerobot_observation": {
    "front": uint8[480, 640, 3],
    "wrist": uint8[480, 640, 3],
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "gripper.pos": float
  },
  "instruction": "Lift the red cube up.",
  "step": int,
  "training": {
    "actions": float32[action_horizon, 6],
    "action_is_pad": bool[action_horizon]
  },
  "_dataset_facts": {
    "statistics_source": "meta/stats.json",
    "statistics_scope": "all recorded frames",
    "action": {
      "field": "action",
      "shape": [6],
      "keys": [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
      ],
      "mean": [-0.7909990549087524, -32.578407287597656, 47.193084716796875, 51.047645568847656, 0.31894364953041077, 23.24579620361328],
      "std": [7.270109176635742, 46.29082489013672, 35.75340270996094, 8.826695442199707, 7.16658878326416, 22.15852928161621],
      "min": [-23.628379821777344, -99.99999237060547, -5.739906311035156, 11.278823852539062, -15.634666442871094, 8.669766202729079e-07],
      "max": [24.725677490234375, 32.26708984375, 100.0, 76.51991271972656, 26.47058868408203, 67.12748718261719],
      "q01": [-18.25310333251953, -99.3602523803711, -5.5605316162109375, 12.483022460937502, -13.209495544433594, 0.6912439465522766],
      "q99": [16.89832305908203, 32.027191162109375, 100.0, 69.51949890136714, 20.330238342285156, 63.35308944702148]
    }
  }
}
```

`_dataset_facts` is static contract metadata embedded in this example for the
Agent; Judge does not supply it in each runtime Policy Mapping. These are raw
dataset facts, not processor instructions: they do not prescribe a
normalization method, checkpoint profile, or coordinate conversion.

The ordered state and action dimensions are exactly:

```text
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos,
wrist_roll.pos, gripper.pos
```

Use the upstream checkpoint's official processor and normalisation statistics.
`compute_loss()` must use the official preprocessing and native model
forward/loss path, returning one finite Python `float`. It must not use future
observations, rewards, or files outside the provided mapping.

Create `policy_spec.json` with `input_schema` describing the `lerobot`
platform and `lerobot_observation` wrapper. Its `action_schema` must be:

```json
{"type": "joint_position", "action_horizon": 50, "action_dim": 6}
```

Here `50` is only an example: use the checkpoint's positive configured
`action_horizon`. `action_dim` is exactly `6`.

Judge owns runtime construction. It supplies a Judge-only Python 3.12 in
`JUDGE_BOOTSTRAP_PYTHON`, plus the role-isolated target
`JUDGE_RUNTIME_DIR` and its resulting `JUDGE_PYTHON`. The installer creates a
fresh venv at the target from the bootstrap Python, then installs and runs
only through `JUDGE_PYTHON`. Do not search `$HOME`, use `/home/opencode`, or
place a Python path in `policy_spec.json`; Judge uses that role runtime for
smoke and native-loss capture.

Judge is CPU-only. The upstream checkout currently maps Linux `torch` and
`torchvision` to a CUDA index through `[tool.uv.sources]`; merely setting
`CUDA_VISIBLE_DEVICES` does not change that dependency resolution. An
`uv pip`-based installer must therefore use `--no-sources --torch-backend cpu`
when installing the upstream project, so it resolves CPU wheels rather than
downloading CUDA/Triton packages.

Also create `runtime.json` as:

```json
{"protocol": "offline_parity", "command": []}
```

It records that this benchmark has no live serving endpoint; the Judge only
runs the installer, smoke test, and offline native-loss parity capture.

# Reference Bundle Contract: __VERSION__

This document applies only to the Reference Agent. It defines the upstream
Policy Endpoint and its artifacts: `policy_impl.py`, `policy_spec.json`, and
`runtime.json`. The Adapter Agent receives a separate Factory Bundle Contract
and must not create these files.

Implement `policy_impl.py` with:

```python
from policy_contract import Policy

def build_policy(checkpoint: str) -> Policy:
    ...
```

The returned policy implements:

```python
reset(task_info: Mapping[str, Any]) -> None
act(observation: Mapping[str, Any]) -> Sequence[Sequence[float]]
compute_loss(observation: Mapping[str, Any]) -> float
```

## `reset(task_info)` input

The TCP reset command calls `reset({})`.  It must clear episode-local action
queues and any other per-episode state; `task_info` is otherwise empty in this
benchmark.

## `act(observation)` input and output

For `beat_block_hammer`, these are the complete contract inputs to `act()`
(with no `training` key):

```python
{
  "robotwin_observation": {
    "observation": {
      "head_camera": {"rgb": uint8[240, 320, 3], ...},
      "left_camera": {"rgb": uint8[240, 320, 3], ...},
      "right_camera": {"rgb": uint8[240, 320, 3], ...}
    },
    "joint_action": {
      "vector": float32[14]
    }
  },
  "instruction": str,
  "step": int,
  "_dataset_facts": {
    "statistics_source": "data/episode*.hdf5:joint_action/vector",
    "statistics_scope": "50 episodes, 5717 recorded frames",
    "action": {
      "field": "joint_action/vector",
      "shape": [14],
      "keys": [
        "left_arm[0]", "left_arm[1]", "left_arm[2]", "left_arm[3]", "left_arm[4]", "left_arm[5]", "left_gripper",
        "right_arm[0]", "right_arm[1]", "right_arm[2]", "right_arm[3]", "right_arm[4]", "right_arm[5]", "right_gripper"
      ],
      "mean": [-0.27275509175862245, 0.861867857934216, 0.6968026618341768, -0.4090683807987731, -0.04584107040991897, -0.2572719082239548, 0.77334899088762, 0.3020947015535908, 1.0071248251166751, 0.8049906351616111, -0.44699134799337886, 0.049298609421954284, 0.3094720085120053, 0.7372396352938384],
      "std": [0.34283463100725087, 1.0409667060526038, 0.8408547717101394, 0.5010275333535283, 0.06393669520050868, 0.32426781636977914, 0.4079281493915179, 0.33277748622735237, 1.0617443429964888, 0.8471054068122889, 0.4722225295748437, 0.0686690115220298, 0.33789348760996846, 0.42813424419229246],
      "min": [-0.8091220855712891, 0.0, 0.0, -1.6359981298446655, -0.22974619269371033, -0.7683258652687073, 0.0, 0.0, 0.0, 0.0, -1.4826239347457886, -0.11985785514116287, 0.0, 0.0],
      "max": [0.0, 2.473158359527588, 2.483600616455078, 0.0, 0.07235725224018097, 0.0, 1.0, 0.7679885625839233, 2.659459352493286, 2.7118945121765137, 0.0, 0.14906178414821625, 0.7770721316337585, 1.0],
      "q01": [-0.8083319664001465, 0.0, 0.0, -1.4389171600341797, -0.1469709873199463, -0.7675059676170349, 0.0, 0.0, 0.0, 0.0, -1.2700529098510742, -0.1050169214606285, 0.0, 0.0],
      "q99": [0.0, 2.414711208343506, 2.2247293090820315, 0.0, 0.030249341651797303, 0.0, 1.0, 0.767224771976471, 2.466321983337403, 2.2570340061187744, 0.0, 0.14565223455429077, 0.7765146732330322, 1.0]
    }
  }
}
```

`_dataset_facts` is static contract metadata embedded in this example for the
Agent; Judge does not supply it in each runtime Policy Mapping. These are raw
dataset facts, not processor instructions: they do not prescribe a
normalization method, checkpoint profile, or coordinate conversion.

The three RGB frames and `joint_action.vector` are the model inputs.
`joint_action.vector` is exactly 14 floats in this order:
`left_arm[6] + left_gripper[1] + right_arm[6] + right_gripper[1]`. There is no
28-dimensional action or state input. RoboTwin may transport additional raw
metadata, including separately named joint segments, but it is not a contract
input and the policy must neither read nor concatenate it.

Return an ordered qpos action chunk with shape
`float[action_horizon, action_dim]`, where `action_dim = 14` and
`action_horizon` is the positive, checkpoint-defined native action chunk
length. Read that exact integer from the selected checkpoint/model
configuration; do not invent or hard-code a horizon from this benchmark.
Do not read RoboTwin directories, reward, success, future state, or benchmark
files.

## `compute_loss(observation)` input and output

`compute_loss()` receives the same Mapping as `act()`, plus this required
Judge-owned `training` Mapping:

```python
"training": {
  "actions": float32[action_horizon, action_dim],
  "action_is_pad": bool[action_horizon]
}
```

Here `action_horizon` is the same checkpoint-defined value used by `act()`,
and `action_dim = 14`. `act()` ignores `training`; `compute_loss()` requires
it and returns one finite Python `float` after official upstream preprocessing
and the native model forward/loss path.

Create `policy_spec.json`:

```json
{
  "contract_version": "__VERSION__",
  "implementation": "policy_impl.py",
  "install_script": "install_model.sh",
  "smoke_test": "smoke_test_model.py",
  "input_schema": {"platform": "robotwin", "wrapper": "robotwin_observation"},
  "action_schema": {"type": "qpos", "action_horizon": 50, "action_dim": 14}
}
```

`action_horizon` above is an example only: write the exact positive integer
resolved from the Judge-provided checkpoint. `action_dim` must be exactly `14`.
Judge owns runtime construction. It provides Judge-only Python 3.12 in
`JUDGE_BOOTSTRAP_PYTHON`, the role-isolated target `JUDGE_RUNTIME_DIR`, and
the resulting `JUDGE_PYTHON`. The installer creates a fresh venv at that
target from the bootstrap Python, then installs and runs only through
`JUDGE_PYTHON`. Do not search `$HOME`, use `/home/opencode`, or place a Python
path in `policy_spec.json`.

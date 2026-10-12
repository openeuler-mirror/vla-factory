"""Judge-owned serialization and scalar comparison for one policy sample."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _array(value: Any):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def write_sample(observation: Mapping[str, Any], metadata: dict[str, Any], output: Path) -> None:
    """Persist one Judge-selected Policy Mapping and its training targets."""
    import numpy as np

    required = {"instruction", "step", "training"}
    missing = sorted(required - set(observation))
    if missing:
        raise ValueError(f"policy sample misses: {', '.join(missing)}")
    training = observation["training"]
    wrappers = {key for key in ("robotwin_observation", "lerobot_observation") if key in observation}
    if len(wrappers) != 1:
        raise ValueError("policy sample requires exactly one platform observation wrapper")

    if "robotwin_observation" in wrappers:
        platform = "robotwin"
        robotwin = observation["robotwin_observation"]
        cameras = robotwin["observation"]
        if not isinstance(cameras, Mapping) or not cameras:
            raise ValueError("policy sample requires cameras")
        arrays = {f"images.{name}": _array(value["rgb"]) for name, value in cameras.items()}
        arrays["state"] = _array(robotwin["joint_action"]["vector"])
    else:
        platform = "lerobot"
        host = observation["lerobot_observation"]
        if not isinstance(host, Mapping):
            raise ValueError("LeRobot policy sample requires a host observation mapping")
        arrays = {}
        for key, value in host.items():
            array = _array(value)
            # The LeRobot host interface contains RGB arrays and scalar joint
            # values.  Store their names separately so capture can reproduce
            # the exact host mapping, including benchmark-defined state keys.
            if getattr(array, "ndim", 0) >= 2:
                arrays[f"images.{key}"] = array
            elif getattr(array, "ndim", 0) == 0:
                arrays[f"state.{key}"] = array
            else:
                raise ValueError(f"LeRobot host field {key!r} is neither an image nor a scalar")
        if not any(key.startswith("images.") for key in arrays):
            raise ValueError("policy sample requires cameras")
    arrays.update({
        "actions": _array(training["actions"]),
        "action_is_pad": _array(training["action_is_pad"]),
        "instruction": np.asarray(str(observation["instruction"])),
        "step": np.asarray(int(observation["step"])),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    sidecar = {
        **metadata,
        "format": "judge-policy-sample/v1",
        "platform": platform,
        "fields": sorted(arrays),
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_sample(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hydrate one Policy Mapping and its training targets for a Judge runner."""
    import numpy as np

    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("format") != "judge-policy-sample/v1":
        raise ValueError(f"unsupported policy sample format: {metadata.get('format')!r}")
    with np.load(path, allow_pickle=False) as arrays:
        cameras = {
            key.removeprefix("images."): {"rgb": arrays[key]}
            for key in arrays.files if key.startswith("images.")
        }
        common = {
            "instruction": str(arrays["instruction"].item()),
            "step": int(arrays["step"].item()),
            "training": {"actions": arrays["actions"], "action_is_pad": arrays["action_is_pad"]},
        }
        platform = metadata.get("platform", "robotwin")
        if platform == "robotwin":
            state = arrays["state"]
            observation = {
                "robotwin_observation": {
                    "observation": cameras,
                    "joint_action": {
                        "left_arm": state[:6], "left_gripper": state[6],
                        "right_arm": state[7:13], "right_gripper": state[13], "vector": state,
                    },
                    "pointcloud": [], "endpose": {},
                },
                **common,
            }
        elif platform == "lerobot":
            host = {name: value["rgb"] for name, value in cameras.items()}
            host.update({
                key.removeprefix("state."): float(arrays[key].item())
                for key in arrays.files if key.startswith("state.")
            })
            observation = {"lerobot_observation": host, **common}
        else:
            raise ValueError(f"unsupported policy sample platform: {platform!r}")
    if not cameras:
        raise ValueError("policy sample has no cameras")
    return observation, metadata


def write_loss(loss: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"loss": float(loss)}) + "\n", encoding="utf-8")


def compare_loss(reference: Path, factory: Path, report: Path, *, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    import math

    reference_loss = float(json.loads(reference.read_text(encoding="utf-8"))["loss"])
    factory_loss = float(json.loads(factory.read_text(encoding="utf-8"))["loss"])
    error = abs(reference_loss - factory_loss)
    passed = math.isfinite(reference_loss) and math.isfinite(factory_loss) and math.isclose(
        reference_loss, factory_loss, abs_tol=atol, rel_tol=rtol
    )
    result = {"status": "PASS" if passed else "FAIL", "reference_loss": reference_loss,
              "factory_loss": factory_loss, "abs_error": error, "atol": atol, "rtol": rtol}
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return passed

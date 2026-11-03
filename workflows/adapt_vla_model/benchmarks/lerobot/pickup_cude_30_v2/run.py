"""Judge metadata for the offline LeRobot pickup_cude_30_v2 benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adapt_vla_model.benchmarks import parity_dataset


SPEC = json.loads(Path(__file__).with_name("spec.json").read_text(encoding="utf-8"))


def _dataset_info() -> dict[str, object]:
    dataset, _ = parity_dataset(SPEC["name"])
    path = dataset / "meta" / "info.json"
    if not path.is_file():
        raise ValueError(f"LeRobot benchmark metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def revision() -> str:
    """Pin the dataset schema, not a mutable cache-directory timestamp."""
    info = _dataset_info()
    features = info.get("features", {})
    expected = SPEC["state_keys"]
    for key in ("action", "observation.state"):
        feature = features.get(key, {}) if isinstance(features, dict) else {}
        if feature.get("shape") != [6] or feature.get("names") != expected:
            raise ValueError(f"LeRobot benchmark has incompatible {key} schema")
    for camera in SPEC["camera_keys"]:
        if f"observation.images.{camera}" not in features:
            raise ValueError(f"LeRobot benchmark is missing {camera} camera")
    canonical = json.dumps(
        {"codebase_version": info.get("codebase_version"), "robot_type": info.get("robot_type"), "features": features},
        sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"leisaac/pickup_cude_30_v2@{info.get('codebase_version', 'unknown')}+{digest}"


def run(*_args, **_kwargs) -> dict[str, object]:
    raise RuntimeError("lerobot-leisaac-pickup-cude-30-v2 is an offline parity benchmark")

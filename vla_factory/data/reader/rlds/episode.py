"""Convert parsed RLDS Example features into per-step values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .spec import RldsSpec
from .tf_example import FeatureList


@dataclass(frozen=True)
class EpisodeEntry:
    shard: Path
    offset: int
    length: int
    num_steps: int


def episode_steps(
    features: dict[str, FeatureList],
    spec: RldsSpec,
    action_dim: int | None,
    shard: Path,
    offset: int,
) -> tuple[int, int | None]:
    """Return episode length and infer a dynamic action width when needed."""
    action = features.get(spec.action_path)

    if spec.is_first_path and spec.is_first_path in features:
        first_flags = features[spec.is_first_path]
        if first_flags.kind != "ints" or not first_flags.int_values:
            raise ValueError(
                f"{shard}@{offset}: feature {spec.is_first_path!r} is not "
                "a non-empty bool/int list"
            )
        steps = len(first_flags)
        if action is None:
            raise ValueError(
                f"{shard}@{offset}: missing required feature {spec.action_path!r}"
            )
        return steps, _infer_width(action, action_dim, steps, spec.action_path)

    if action is None or action.kind != "floats" or not action.float_values:
        raise ValueError(
            f"{shard}@{offset}: cannot infer episode length; neither "
            f"{spec.is_first_path!r} nor {spec.action_path!r} is present"
        )
    if action_dim is None:
        raise ValueError(
            f"{shard}@{offset}: {spec.action_path!r} has no declared width "
            "and the record has no is_first flags"
        )
    if len(action.float_values) % action_dim:
        raise ValueError(
            f"{shard}@{offset}: {spec.action_path!r} has "
            f"{len(action.float_values)} values, not a multiple of {action_dim}"
        )
    return len(action.float_values) // action_dim, action_dim


def vector_rows(
    features: dict[str, FeatureList],
    path: str | None,
    steps: int,
    width: int | None,
) -> np.ndarray | None:
    if path is None:
        return None
    values = features.get(path)
    if values is None or values.kind != "floats" or not values.float_values:
        return None
    width = _infer_width(values, width, steps, path)
    return np.asarray(values.float_values, dtype=np.float32).reshape(steps, width)


def flag_values(
    features: dict[str, FeatureList], path: str | None, default: list[bool]
) -> list[bool]:
    if path is not None and path in features:
        feature = features[path]
        if feature.kind != "ints" or len(feature.int_values) != len(default):
            raise ValueError(
                f"feature {path!r} must contain {len(default)} bool/int values"
            )
        return [bool(value) for value in feature.int_values]
    return default


def step_bytes(
    features: dict[str, FeatureList], path: str, step: int, entry: EpisodeEntry
) -> bytes | None:
    values = features.get(path)
    if values is None:
        return None
    if values.kind != "bytes":
        raise ValueError(
            f"{entry.shard}@{entry.offset}: image feature {path!r} is not "
            "byte-encoded"
        )
    if step >= len(values.bytes_values):
        raise ValueError(
            f"{entry.shard}@{entry.offset}: image feature {path!r} has "
            f"{len(values.bytes_values)} frames for {entry.num_steps} steps"
        )
    return values.bytes_values[step]


def step_language(
    features: dict[str, FeatureList], paths: tuple[str, ...], step: int
) -> tuple[str, ...]:
    values: list[str] = []
    for path in paths:
        feature = features.get(path)
        if feature is None or feature.kind != "bytes":
            continue
        if step < len(feature.bytes_values):
            text = feature.bytes_values[step].decode("utf-8", errors="replace").strip()
            if text:
                values.append(text)
    return tuple(values)


def context_language(
    features: dict[str, FeatureList], paths: tuple[str, ...]
) -> tuple[str, ...]:
    values: list[str] = []
    for path in paths:
        feature = features.get(path)
        if feature is None or feature.kind != "bytes" or not feature.bytes_values:
            continue
        text = feature.bytes_values[0].decode("utf-8", errors="replace").strip()
        if text:
            values.append(text)
    return tuple(values)


def _infer_width(
    values: FeatureList, width: int | None, steps: int, path: str
) -> int:
    if values.kind != "floats":
        raise ValueError(f"feature {path!r} is not a float vector")
    count = len(values.float_values)
    if steps <= 0 or count % steps:
        raise ValueError(
            f"feature {path!r} has {count} values that do not divide into "
            f"{steps} steps"
        )
    inferred = count // steps
    if width is not None and inferred != width:
        raise ValueError(
            f"feature {path!r} has width {inferred}; expected {width}"
        )
    return inferred

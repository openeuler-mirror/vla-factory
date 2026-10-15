"""Map a TFDS ``features.json`` file to the fields consumed by RLDS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_STATE_NAMES = ("state", "qpos", "robot_state", "proprio", "joint_position")
_LANGUAGE_NAMES = (
    "language_instruction*",
    "natural_language*",
    "language",
    "task",
    "instruction",
)


@dataclass(frozen=True)
class CameraSpec:
    path: str
    key: str
    height: int | None = None
    width: int | None = None
    channels: int = 3
    encoding: str | None = None


@dataclass(frozen=True)
class RldsSpec:
    root: Path
    cameras: tuple[CameraSpec, ...] = ()
    state_path: str | None = None
    state_dim: int | None = None
    action_path: str = "steps/action"
    action_dim: int | None = None
    language_paths: tuple[str, ...] = ()
    context_language_paths: tuple[str, ...] = ()
    is_first_path: str | None = None
    is_last_path: str | None = None


@dataclass(frozen=True)
class _Leaf:
    path: str
    dtype: str
    shape: tuple[int | None, ...]
    kind: str
    encoding: str | None
    in_sequence: bool

    @property
    def is_scalar(self) -> bool:
        return not self.shape

    @property
    def is_vector(self) -> bool:
        return len(self.shape) == 1


def load_rlds_spec(root: Path) -> RldsSpec | None:
    """Load the RLDS ``steps`` schema from a generated TFDS directory.

    TFDS serializes ``features.json`` as the protobuf-JSON representation of
    ``tensorflow_datasets.Feature``. A small compatibility branch also accepts
    the older ``_type`` representation used by some third-party exporters.
    """
    path = root / "features.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None

    features = _feature_mapping(raw)
    if features is None or "steps" not in features:
        return None

    leaves: list[_Leaf] = []
    for name, node in features.items():
        if isinstance(node, dict):
            leaves.extend(_flatten(node, str(name), in_sequence=False))

    step_leaves = [leaf for leaf in leaves if leaf.path.startswith("steps/")]
    if not step_leaves:
        return None

    action = next(
        (leaf for leaf in step_leaves if leaf.path == "steps/action" and leaf.is_vector),
        None,
    )
    if action is None:
        names = ", ".join(leaf.path for leaf in step_leaves[:20])
        raise ValueError(
            f"{root}: expected a vector feature at 'steps/action'; found: {names}"
        )

    state = _pick_state(step_leaves)
    language = tuple(
        leaf.path
        for leaf in step_leaves
        if _is_text(leaf) and _matches_language(_leaf_name(leaf.path))
    )
    context_language = tuple(
        leaf.path
        for leaf in leaves
        if not leaf.in_sequence
        and _is_text(leaf)
        and _matches_language(_leaf_name(leaf.path))
    )

    return RldsSpec(
        root=root,
        cameras=_pick_cameras(step_leaves),
        state_path=state.path if state else None,
        state_dim=_vector_dim(state),
        action_path=action.path,
        action_dim=_vector_dim(action),
        language_paths=language,
        context_language_paths=context_language,
        is_first_path=_scalar_path(step_leaves, "steps/is_first"),
        is_last_path=_scalar_path(step_leaves, "steps/is_last"),
    )


def _feature_mapping(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return children from a TFDS FeaturesDict protobuf-JSON node."""
    wrapper = node.get("featuresDict")
    if isinstance(wrapper, dict) and isinstance(wrapper.get("features"), dict):
        return wrapper["features"]

    # Older connector JSON uses a bare mapping at the root and ``feature`` for
    # nested Dict/Sequence nodes.
    if "steps" in node:
        return node
    children = node.get("feature")
    return children if isinstance(children, dict) else None


def _flatten(
    node: dict[str, Any], prefix: str, *, in_sequence: bool
) -> list[_Leaf]:
    sequence = node.get("sequence")
    if isinstance(sequence, dict) and isinstance(sequence.get("feature"), dict):
        return _flatten(sequence["feature"], prefix, in_sequence=True)

    children = _feature_mapping(node)
    if children is not None:
        legacy_sequence = node.get("_type") == "Sequence"
        leaves: list[_Leaf] = []
        for name, child in children.items():
            if isinstance(child, dict):
                leaves.extend(
                    _flatten(
                        child,
                        f"{prefix}/{name}",
                        in_sequence=in_sequence or legacy_sequence,
                    )
                )
        return leaves

    content, kind = _leaf_content(node)
    if content is None:
        return []
    return [
        _Leaf(
            path=prefix,
            dtype=str(content.get("dtype", "string" if kind == "text" else "")),
            shape=_shape(content.get("shape")),
            kind=kind,
            encoding=content.get("encodingFormat")
            or content.get("encoding_format"),
            in_sequence=in_sequence,
        )
    ]


def _leaf_content(node: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    for key in ("image", "tensor", "text"):
        content = node.get(key)
        if isinstance(content, dict):
            return content, key
    if "dtype" in node or "shape" in node:
        marker = str(node.get("_type", "")).lower()
        return node, marker or "tensor"
    return None, ""


def _shape(value: Any) -> tuple[int | None, ...]:
    if isinstance(value, dict):
        value = value.get("dimensions", [])
    if not isinstance(value, list):
        return ()
    result: list[int | None] = []
    for dim in value:
        number = int(dim)
        result.append(None if number < 0 else number)
    return tuple(result)


def _pick_state(leaves: list[_Leaf]) -> _Leaf | None:
    vectors = {_leaf_name(leaf.path): leaf for leaf in leaves if leaf.is_vector}
    return next((vectors[name] for name in _STATE_NAMES if name in vectors), None)


def _pick_cameras(leaves: list[_Leaf]) -> tuple[CameraSpec, ...]:
    used: set[str] = set()
    cameras: list[CameraSpec] = []
    for leaf in (item for item in leaves if _is_image(item)):
        key = _unique_camera_key(leaf.path, used)
        used.add(key)
        height, width = (leaf.shape + (None, None))[:2]
        channels = leaf.shape[2] if len(leaf.shape) > 2 else 3
        cameras.append(
            CameraSpec(
                path=leaf.path,
                key=key,
                height=height,
                width=width,
                channels=channels or 3,
                encoding=leaf.encoding,
            )
        )
    return tuple(cameras)


def _unique_camera_key(path: str, used: set[str]) -> str:
    parts = path.split("/")
    key = parts[-1]
    for parent in reversed(parts[1:-1]):
        if key not in used:
            return key
        key = f"{parent}_{key}"
    if key not in used:
        return key
    suffix = 2
    while f"{key}_{suffix}" in used:
        suffix += 1
    return f"{key}_{suffix}"


def _scalar_path(leaves: list[_Leaf], path: str) -> str | None:
    return next(
        (leaf.path for leaf in leaves if leaf.path == path and leaf.is_scalar),
        None,
    )


def _vector_dim(leaf: _Leaf | None) -> int | None:
    return leaf.shape[0] if leaf and leaf.is_vector else None


def _leaf_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_image(leaf: _Leaf) -> bool:
    return leaf.kind == "image"


def _is_text(leaf: _Leaf) -> bool:
    return leaf.kind == "text" or leaf.dtype == "string"


def _matches_language(name: str) -> bool:
    name = name.lower()
    return any(
        name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern
        for pattern in _LANGUAGE_NAMES
    )

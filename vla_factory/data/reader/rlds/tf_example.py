"""``tf.train.Example`` parsing via TensorFlow's official protobuf bindings.

Each RLDS episode record is one serialised ``tf.train.Example`` whose
nested feature tree is flattened to ``/``-joined keys — per-step values
under ``steps/...`` (``steps/observation/image``, ``steps/action``,
``steps/is_first``), episode-level context at the top level. One key's
values are a single flat list: a ``[T, D]`` float tensor becomes ``T * D``
floats, an image stream becomes ``T`` encoded JPEG/PNG byte strings.

The binary format is owned by TensorFlow's own proto definitions — the
same surface forge and the wider RLDS ecosystem parse through — so real
TFDS output needs no reimplementation here. Only the framing (record
offsets for random access, see :mod:`.tfrecord`) stays local because
``tf.data.TFRecordDataset`` streams without exposing offsets.

``tensorflow`` is an optional (``[rlds]``) dependency imported lazily at
the first parse with an actionable error. Only its protobuf message API is
used; record iteration and random access stay in :mod:`.tfrecord`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FeatureList:
    """One feature key's values inside a parsed example.

    Exactly one of the three lists is populated, mirroring the proto's
    ``oneof kind``. Floats are IEEE-754 float32 widened to Python floats;
    ints are signed 64-bit.
    """

    kind: str = "bytes"  # "bytes" | "floats" | "ints"
    bytes_values: list[bytes] = field(default_factory=list)
    float_values: list[float] = field(default_factory=list)
    int_values: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(
            self.bytes_values or self.float_values or self.int_values
        )


def _load_tf() -> Any:
    """Import TensorFlow lazily with an actionable error if it is missing."""
    try:
        # Forge-style silencing; harmless if the user already configured TF.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow  # noqa: PLC0415 - lazy by design
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Reading RLDS datasets requires TensorFlow's tf.train.Example "
            'protobuf bindings. Install the RLDS extra: pip install -e ".[rlds]" '
            "(or pip install tensorflow-cpu)."
        ) from exc
    return tensorflow


def parse_example(
    data: bytes,
    keys: Callable[[str], bool] | None = None,
) -> dict[str, FeatureList]:
    """Parse a serialised ``tf.train.Example`` into ``{key: FeatureList}``.

    ``keys`` optionally filters which feature keys are converted; the proto
    itself is always parsed in full (protobuf has no partial-parse API),
    so the filter saves only the conversion, not the parse.
    """
    example = _load_tf().train.Example()
    try:
        example.ParseFromString(data)
    except Exception as exc:  # protobuf raises DecodeError subclasses
        raise ValueError(
            f"Cannot parse a tf.train.Example from {len(data)} bytes "
            f"({type(exc).__name__}: {exc}). The record is corrupt or not "
            "RLDS/TFDS data."
        ) from exc

    result: dict[str, FeatureList] = {}
    for key, feature in example.features.feature.items():
        if keys is not None and not keys(key):
            continue
        kind = feature.WhichOneof("kind")
        if kind == "bytes_list":
            result[key] = FeatureList(
                "bytes", bytes_values=list(feature.bytes_list.value)
            )
        elif kind == "float_list":
            result[key] = FeatureList(
                "floats", float_values=list(feature.float_list.value)
            )
        elif kind == "int64_list":
            result[key] = FeatureList(
                "ints", int_values=list(feature.int64_list.value)
            )
        # kind None: the feature key exists but holds no list — treat as
        # absent, matching how TFDS omits optional features entirely.
    return result

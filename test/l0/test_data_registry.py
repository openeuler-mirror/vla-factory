"""Reader and codec registration, selection, and plugin discovery."""

from pathlib import Path

import numpy as np
import pytest

from vla_factory.data.codec.registry import CodecRegistry
from vla_factory.data.data_schema import DataSchema, Episode, NormStats, VideoRef
from vla_factory.data.reader.registry import ReaderRegistry


class _Reader:
    def can_read(self, path: Path) -> bool:
        return path.name == "recognised"

    def get_schema(self, path: Path) -> DataSchema:
        return DataSchema(identity_name=path.name)

    def get_norm_stats(self, path: Path) -> NormStats:
        return NormStats()

    def get_episode_lengths(self, path: Path) -> dict[int, int]:
        return {}

    def get_episode_ranges(self, path: Path) -> dict[int, tuple[int, int]]:
        return {}

    def read_episode(self, path: Path, episode_index: int, codec) -> Episode:
        return Episode(str(episode_index), episode_index, 0)


class _Codec:
    @property
    def name(self) -> str:
        return "test"

    def decode_frame(self, ref: VideoRef) -> np.ndarray:
        return np.empty((ref.height, ref.width, ref.channels), dtype=np.uint8)


def test_reader_decorator_registers_factory_and_alias():
    ReaderRegistry.register("_test-reader", aliases=("_test_reader",))(_Reader)

    assert isinstance(ReaderRegistry.create("_test-reader"), _Reader)
    assert isinstance(ReaderRegistry.create("_test_reader"), _Reader)


def test_codec_decorator_registers_factory():
    CodecRegistry.register("_test-codec")(_Codec)

    assert isinstance(CodecRegistry.create("_test-codec"), _Codec)


def test_duplicate_registration_is_rejected():
    ReaderRegistry.register("_duplicate-reader")(_Reader)

    with pytest.raises(ValueError, match="already registered"):
        ReaderRegistry.register("_duplicate-reader")(_Reader)


def test_unknown_names_are_not_silently_defaulted():
    with pytest.raises(ValueError, match="Unknown dataset format"):
        ReaderRegistry.create("_missing-reader")
    with pytest.raises(ValueError, match="Unknown video codec"):
        CodecRegistry.create("_missing-codec")


def test_reader_is_discovered_from_external_entry_point(monkeypatch):
    from vla_factory.data.reader import registry

    class EntryPoint:
        name = "_plugin-reader"

        @staticmethod
        def load():
            return _Reader

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda *, group: (EntryPoint(),)
        if group == ReaderRegistry.ENTRY_POINT_GROUP else (),
    )

    assert isinstance(ReaderRegistry.create("_plugin-reader"), _Reader)


def test_codec_is_discovered_from_external_entry_point(monkeypatch):
    from vla_factory.data.codec import registry

    class EntryPoint:
        name = "_plugin-codec"

        @staticmethod
        def load():
            return _Codec

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda *, group: (EntryPoint(),)
        if group == CodecRegistry.ENTRY_POINT_GROUP else (),
    )

    assert isinstance(CodecRegistry.create("_plugin-codec"), _Codec)
def test_feature_names_normalisation():
    # Regression for the real-data finding: lerobot/utokyo_xarm_pick_and_place
    # (a real HF lerobot-v3 dataset) declares feature names as a NESTED dict
    # {"motors": ["motor_0", ...]}, while synthetic fixtures use a flat list.
    # _feature_names must accept both; anything else yields [] (dimensions get
    # name=None and resolution fails loudly on canonical-name validation).
    from vla_factory.data.reader.lerobot_v3 import _feature_names

    # Flat list (synthetic fixtures / simple datasets).
    assert _feature_names(["dx", "dy", "dz"]) == ["dx", "dy", "dz"]

    # Nested dict (real HF lerobot-v3 layout).
    assert _feature_names(
        {"motors": ["motor_0", "motor_1", "motor_2"]}
    ) == ["motor_0", "motor_1", "motor_2"]

    # Empty / unknown shapes -> [], never crash.
    assert _feature_names(None) == []
    assert _feature_names({}) == []
    assert _feature_names({"motors": "not-a-list"}) == []


def test_modality_dim_names_preserve_dataset_order(tmp_path):
    import json

    from vla_factory.data.reader.lerobot_v3 import _modality_dim_names

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "modality.json").write_text(json.dumps({
        "state": {
            "base_position": {"original_key": "observation.state", "start": 0, "end": 3},
            "base_rotation": {"original_key": "observation.state", "start": 3, "end": 7},
            "end_effector_position_relative": {"original_key": "observation.state", "start": 7, "end": 10},
            "end_effector_rotation_relative": {"original_key": "observation.state", "start": 10, "end": 14},
            "gripper_qpos": {"original_key": "observation.state", "start": 14, "end": 16},
        },
        "action": {
            "base_motion": {"original_key": "action", "start": 0, "end": 4},
            "control_mode": {"original_key": "action", "start": 4, "end": 5},
            "end_effector_position": {"original_key": "action", "start": 5, "end": 8},
            "end_effector_rotation": {"original_key": "action", "start": 8, "end": 11},
            "gripper_close": {"original_key": "action", "start": 11, "end": 12},
        },
    }))

    assert _modality_dim_names(tmp_path, "observation.state", 16) == [
        *(f"base_position.{i}" for i in range(3)),
        *(f"base_rotation.{i}" for i in range(4)),
        *(f"end_effector_position_relative.{i}" for i in range(3)),
        *(f"end_effector_rotation_relative.{i}" for i in range(4)),
        *(f"gripper_qpos.{i}" for i in range(2)),
    ]
    assert _modality_dim_names(tmp_path, "action", 12) == [
        *(f"base_motion.{i}" for i in range(4)),
        "control_mode.0",
        *(f"end_effector_position.{i}" for i in range(3)),
        *(f"end_effector_rotation.{i}" for i in range(3)),
        "gripper_close.0",
    ]


def test_can_read_is_structural_not_version_gated(tmp_path):
    """A full lerobot layout qualifies regardless of the version label.

    RoboCasa365 ships the v3 layout (episodic parquet + jsonl tables) labeled
    ``codebase_version: v2.1``, so detection probes the structure this reader
    consumes — a parseable ``meta/info.json`` with a features map plus parquet
    shards under ``data/`` — instead of comparing version strings.
    """
    import json

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vla_factory.data.reader.registry import ReaderRegistry

    def _make(root: Path, version: str) -> Path:
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        info = {
            "codebase_version": version,
            "total_frames": 1,
            "total_episodes": 1,
            "fps": 1,
            "features": {"action": {"dtype": "float32", "shape": [1]}},
        }
        (root / "meta" / "info.json").write_text(json.dumps(info))
        df = pd.DataFrame({"action": [[0.0]], "episode_index": [0]})
        pq.write_table(
            pa.Table.from_pandas(df),
            root / "data" / "chunk-000" / "file-000.parquet",
        )
        return root

    reader = ReaderRegistry.create("lerobot-v3")
    # RoboCasa365's label and the canonical one both qualify.
    assert reader.can_read(_make(tmp_path / "v21", "v2.1"))
    assert reader.can_read(_make(tmp_path / "v30", "v3.0"))

    # meta only (no parquet shards) → not consumable, rejected.
    meta_only = tmp_path / "meta_only"
    (meta_only / "meta").mkdir(parents=True)
    (meta_only / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": {}})
    )
    assert not reader.can_read(meta_only)

    # info.json without a features map → rejected.
    no_features = tmp_path / "no_features"
    (no_features / "meta").mkdir(parents=True)
    (no_features / "data").mkdir(parents=True)
    (no_features / "meta" / "info.json").write_text(json.dumps({"a": 1}))
    assert not reader.can_read(no_features)


def test_modality_json_names_unnamed_dims(tmp_path):
    """``meta/modality.json`` supplies canonical dim names when features lack them.

    RoboCasa365 ships no per-feature ``names``; its GR00T-style modality map
    segments each vector into named slices. Coverage must be complete — a
    partial map never mixes named and unnamed dims.
    """
    import json

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vla_factory.data.reader.registry import ReaderRegistry

    (tmp_path / "meta").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v2.1",
        "total_frames": 1,
        "total_episodes": 1,
        "fps": 20,
        "features": {
            "observation.state": {"dtype": "float64", "shape": [3]},
            "action": {"dtype": "float64", "shape": [3]},
        },
    }
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    modality = {
        "state": {
            "base_position": {
                "original_key": "observation.state", "start": 0, "end": 2,
            },
            "gripper_qpos": {
                "original_key": "observation.state", "start": 2, "end": 3,
            },
        },
        "action": {
            "base_motion": {"original_key": "action", "start": 0, "end": 3},
        },
    }
    (tmp_path / "meta" / "modality.json").write_text(json.dumps(modality))
    df = pd.DataFrame(
        {"observation.state": [[0.0, 1.0, 2.0]], "action": [[0.0, 1.0, 2.0]],
         "episode_index": [0]}
    )
    pq.write_table(
        pa.Table.from_pandas(df), tmp_path / "data" / "chunk-000" / "file-000.parquet"
    )

    reader = ReaderRegistry.create("lerobot-v3")
    schema = reader.get_schema(tmp_path)
    assert [d.name for d in schema.state_dims] == [
        "base_position.0", "base_position.1", "gripper_qpos.0",
    ]
    assert [d.name for d in schema.action_dims] == [
        "base_motion.0", "base_motion.1", "base_motion.2",
    ]
    # Numeric segment suffixes carry no mode evidence → mode stays undeclared.
    assert all(d.mode is None for d in schema.action_dims)

    # A partial map (one state dim uncovered) must not half-name the vector.
    partial = tmp_path / "partial"
    (partial / "meta").mkdir(parents=True)
    (partial / "meta" / "info.json").write_text(json.dumps(info))
    (partial / "meta" / "modality.json").write_text(json.dumps({
        "state": {"base_position": {
            "original_key": "observation.state", "start": 0, "end": 2,
        }},
        "action": modality["action"],
    }))
    partial_schema = reader.get_schema(partial)
    assert all(d.name is None for d in partial_schema.state_dims)
    assert [d.name for d in partial_schema.action_dims] == [
        "base_motion.0", "base_motion.1", "base_motion.2",
    ]


def test_get_schema_accepts_nested_dict_names(tmp_path):
    """End-to-end: a real-style info.json with nested names resolves dims."""
    import json
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vla_factory.data.reader.registry import ReaderRegistry

    (tmp_path / "meta").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "xarm",
        "total_frames": 10,
        "total_episodes": 1,
        "fps": 10,
        "features": {
            "observation.images.image": {
                "dtype": "video", "shape": [10, 224, 224, 3],
                "video_info": {"video.height": 224, "video.width": 224,
                               "video.channels": 3},
            },
            "observation.state": {
                "dtype": "float32", "shape": [8],
                "names": {"motors": [f"motor_{i}" for i in range(8)]},
            },
            "action": {
                "dtype": "float32", "shape": [7],
                "names": {"motors": [f"motor_{i}" for i in range(7)]},
            },
        },
    }
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))

    df = pd.DataFrame({
        "observation.state": [list(range(8))],
        "action": [list(range(7))],
        "episode_index": [0],
        "frame_index": [0],
        "timestamp": [0.0],
        "index": [0],
        "task_index": [0],
    })
    pq.write_table(
        pa.Table.from_pandas(df),
        tmp_path / "data" / "chunk-000" / "file-000.parquet",
    )

    reader = ReaderRegistry.create("lerobot-v3")
    schema = reader.get_schema(tmp_path)
    assert [d.name for d in schema.state_dims] == [f"motor_{i}" for i in range(8)]
    assert [d.name for d in schema.action_dims] == [f"motor_{i}" for i in range(7)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

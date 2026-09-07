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

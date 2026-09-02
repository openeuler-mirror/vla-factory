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

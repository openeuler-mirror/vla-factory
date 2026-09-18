"""File-level open-video LRU contract shared by all per-file codecs.

Every entry in a codec's ``_decoders`` registry holds an open handle — an fd
plus a decoder/h5py context. The registry used to be an unbounded ``dict``,
so datasets with more distinct video files than the fd limit exhausted fds
partway through training (the DataLoader workers crashed after ~100+ steps
once random sampling had touched enough distinct files).

Contract (identical for PyAVCodec, TorchCodec, Hdf5JpegCodec):

1. The open set is bounded by ``max_open_videos``.
2. Eviction is LRU: re-touching an entry protects it.
3. An evicted entry is closed, releasing its fd.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from vla_factory.data.data_schema import VideoRef

DATASET_PATH = Path(_project_root) / "test/data" / "lerobot_train_data_3_episodes"
VIDEO_PATH = (
    DATASET_PATH
    / "videos"
    / "observation.images.front"
    / "chunk-000"
    / "file-000.mp4"
)


class VideoDecoderCacheContract:
    """Codec-agnostic bookkeeping tests; subclasses provide ``make_codec``."""

    def make_codec(self, **kwargs):
        raise NotImplementedError

    def test_open_video_set_is_bounded(self):
        codec = self.make_codec(max_open_videos=2)
        paths = [Path(f"/nonexistent/video_{i}.mp4") for i in range(3)]
        for p in paths:
            codec._decoder_for(p)
        self.assertEqual(len(codec._decoders), 2)
        self.assertNotIn(paths[0], codec._decoders, "oldest entry must be evicted")
        self.assertIn(paths[1], codec._decoders)
        self.assertIn(paths[2], codec._decoders)

    def test_recent_touch_is_not_evicted(self):
        codec = self.make_codec(max_open_videos=2)
        a, b, c = (Path(f"/nonexistent/touch_{i}.mp4") for i in range(3))
        codec._decoder_for(a)
        codec._decoder_for(b)
        codec._decoder_for(a)  # re-touch: a becomes most-recent again
        codec._decoder_for(c)  # must evict b, not a
        self.assertIn(a, codec._decoders)
        self.assertNotIn(b, codec._decoders)
        self.assertIn(c, codec._decoders)


class TestPyAVDecoderCache(VideoDecoderCacheContract, unittest.TestCase):
    def make_codec(self, **kwargs):
        from vla_factory.data.codec.pyav import PyAVCodec

        return PyAVCodec(**kwargs)

    def test_eviction_closes_open_container(self):
        """Evicting an entry must close its real av container (release the fd)."""
        if not VIDEO_PATH.exists():
            self.skipTest("test video not found")
        codec = self.make_codec(max_open_videos=1)
        cache = codec._decoder_for(VIDEO_PATH)
        codec.decode_frame(VideoRef(VIDEO_PATH, 0, 480, 640, 3))
        self.assertIsNotNone(cache["container"])
        codec._decoder_for(VIDEO_PATH.with_name("other.mp4"))
        self.assertIsNone(
            cache["container"], "evicted cache must be closed (fd released)"
        )


class TestTorchCodecDecoderCache(VideoDecoderCacheContract, unittest.TestCase):
    def make_codec(self, **kwargs):
        from vla_factory.data.codec.torchcodec import TorchCodec

        return TorchCodec(**kwargs)

    def test_eviction_closes_open_decoder(self):
        """Evicting an entry must drop the real torchcodec decoder handle."""
        try:
            import torchcodec  # noqa: F401 - importable check (incl. ABI)
        except (ImportError, OSError, RuntimeError) as exc:
            self.skipTest(f"torchcodec not importable: {exc}")
        if not VIDEO_PATH.exists():
            self.skipTest("test video not found")
        codec = self.make_codec(max_open_videos=1)
        cache = codec._decoder_for(VIDEO_PATH)
        codec.decode_frame(VideoRef(VIDEO_PATH, 0, 480, 640, 3))
        self.assertIsNotNone(cache["decoder"])
        codec._decoder_for(VIDEO_PATH.with_name("other.mp4"))
        self.assertIsNone(
            cache["decoder"], "evicted cache must be closed (fd released)"
        )


class TestHdf5JpegDecoderCache(VideoDecoderCacheContract, unittest.TestCase):
    def make_codec(self, **kwargs):
        from vla_factory.data.codec.hdf5_jpeg import Hdf5JpegCodec

        return Hdf5JpegCodec(**kwargs)


class TestVideoDecoderCache(unittest.TestCase):
    """Direct tests for the shared registry (complement the codec contracts)."""

    def make_cache(self, max_size=2):
        from vla_factory.data.codec.base import VideoDecoderCache

        class Handle:
            def __init__(self, name):
                self.name = name
                self.closed = False

            def close(self):
                self.closed = True

        return VideoDecoderCache(max_size), Handle

    def test_put_past_capacity_evicts_and_closes_lru(self):
        lru, Handle = self.make_cache(max_size=2)
        h0, h1, h2 = Handle("a"), Handle("b"), Handle("c")
        lru.put("v0", h0)
        lru.put("v1", h1)
        lru.put("v2", h2)
        self.assertTrue(h0.closed, "evicted handle must be closed")
        self.assertNotIn("v0", lru)
        self.assertEqual(len(lru), 2)
        self.assertFalse(h1.closed or h2.closed)

    def test_get_touches_recency_peek_does_not(self):
        lru, Handle = self.make_cache(max_size=2)
        h0, h1, h2 = Handle("a"), Handle("b"), Handle("c")
        lru.put("v0", h0)
        lru.put("v1", h1)
        self.assertIs(lru.get("v0"), h0)  # touch: v0 becomes MRU
        lru.put("v2", h2)  # evicts v1, not v0
        self.assertIs(lru["v0"], h0)
        self.assertNotIn("v1", lru)
        with self.assertRaises(KeyError):
            lru["missing"]

    def test_close_all_closes_and_empties(self):
        lru, Handle = self.make_cache(max_size=4)
        handles = [Handle(f"h{i}") for i in range(3)]
        for i, h in enumerate(handles):
            lru.put(f"v{i}", h)
        lru.close()
        self.assertTrue(all(h.closed for h in handles))
        self.assertEqual(len(lru), 0)
        self.assertEqual(lru.values(), [])

    def test_values_snapshot_inspection(self):
        lru, Handle = self.make_cache(max_size=4)
        lru.put("v0", Handle("a"))
        lru.put("v1", Handle("b"))
        self.assertEqual([h.name for h in lru.values()], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

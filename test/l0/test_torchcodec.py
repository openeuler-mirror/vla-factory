"""Tests for the optional torchcodec video codec.

Two layers:

1. ``test_resolve_torchcodec_lazy`` — never touches the torchcodec package
   (import is deferred to first decode), so it runs even without the
   ``[torchcodec]`` extra installed.
2. ``TestTorchCodec`` — real decode against the built-in test dataset, skipped
   when torchcodec is not importable. The guard catches a missing install
   (``ImportError``) and a load failure from an ABI-mismatched wheel
   (``RuntimeError``, or ``OSError`` on some Python versions).

Run:
    python test/test_torchcodec.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

DATASET_PATH = Path(_project_root) / "test/data" / "lerobot_train_data_3_episodes"
IMAGE_H = 480
IMAGE_W = 640


def test_resolve_torchcodec_lazy():
    """resolve_codec('torchcodec') must not import torchcodec at construction."""
    from vla_factory.data.codec import resolve_codec

    codec = resolve_codec("torchcodec")
    assert codec.name == "torchcodec"
    assert codec._caches == {}  # no decoder opened yet


def test_resolve_auto_format_aware(monkeypatch):
    """auto codec selection is format-aware and falls back safely."""
    import vla_factory.data.codec as codec_mod
    from vla_factory.data.codec import resolve_codec

    # Lerobot + torchcodec healthy -> torchcodec
    monkeypatch.setattr(codec_mod, "_TORCHCODEC_AVAILABLE", True)
    assert resolve_codec("auto", "lerobot-v3").name == "torchcodec"
    assert resolve_codec("auto", "lerobot_v3").name == "torchcodec"

    # Lerobot + torchcodec unavailable/broken -> PyAV
    monkeypatch.setattr(codec_mod, "_TORCHCODEC_AVAILABLE", False)
    assert resolve_codec("auto", "lerobot-v3").name == "pyav"

    # RoboTwin -> hdf5_jpeg
    assert resolve_codec("auto", "robotwin").name == "hdf5_jpeg"

    # Unknown/None -> PyAV
    assert resolve_codec("auto", None).name == "pyav"
    assert resolve_codec("auto", "unknown").name == "pyav"

    # Explicit codec always wins
    assert resolve_codec("pyav", "lerobot-v3").name == "pyav"


class TestTorchCodec(unittest.TestCase):
    """Test torchcodec video decoding → numpy HWC uint8."""

    @classmethod
    def setUpClass(cls):
        try:
            import torchcodec  # noqa: F401 - importable check (incl. ABI)
        except (ImportError, OSError, RuntimeError) as exc:
            raise unittest.SkipTest(f"torchcodec not importable: {exc}")
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Dataset not found")

        from vla_factory.data.codec import resolve_codec

        video_path = (
            DATASET_PATH
            / "videos"
            / "observation.images.front"
            / "chunk-000"
            / "file-000.mp4"
        )
        if not video_path.exists():
            raise unittest.SkipTest("Video file not found")
        cls.codec = resolve_codec("torchcodec")
        cls.video_path = video_path

    def test_decode_frame_shape(self):
        from vla_factory.data.data_schema import VideoRef

        ref = VideoRef(
            video_path=self.video_path,
            frame_index=0,
            height=IMAGE_H,
            width=IMAGE_W,
            channels=3,
        )
        img = self.codec.decode_frame(ref)
        self.assertEqual(img.shape, (IMAGE_H, IMAGE_W, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_decode_multiple_frames(self):
        from vla_factory.data.data_schema import VideoRef

        for idx in [0, 100, 200]:
            ref = VideoRef(
                video_path=self.video_path,
                frame_index=idx,
                height=IMAGE_H,
                width=IMAGE_W,
                channels=3,
            )
            img = self.codec.decode_frame(ref)
            self.assertEqual(img.shape, (IMAGE_H, IMAGE_W, 3))

    def test_lru_reuses_decoded_frame(self):
        """Re-decoding a cached frame must return the same pixels."""
        from vla_factory.data.codec.torchcodec import TorchCodec
        from vla_factory.data.data_schema import VideoRef

        codec = TorchCodec(disk_cache=False)
        ref = VideoRef(
            video_path=self.video_path,
            frame_index=50,
            height=IMAGE_H,
            width=IMAGE_W,
            channels=3,
        )
        first = codec.decode_frame(ref)
        second = codec.decode_frame(ref)  # served from the in-memory LRU
        self.assertTrue(np.array_equal(first, second))

    def test_lru_eviction(self):
        """The per-video frame LRU must evict the least-recently-used frame."""
        from vla_factory.data.codec.torchcodec import TorchCodec
        from vla_factory.data.data_schema import VideoRef

        codec = TorchCodec(max_cached_per_video=3, disk_cache=False)
        cache = codec._get_cache(self.video_path)
        for idx in range(4):
            codec.decode_frame(
                VideoRef(
                    video_path=self.video_path,
                    frame_index=idx,
                    height=IMAGE_H,
                    width=IMAGE_W,
                    channels=3,
                )
            )
        # Only the last 3 distinct frames remain; frame 0 was evicted first.
        self.assertEqual(len(cache._cache), 3)
        self.assertNotIn(0, cache._cache)
        self.assertIn(1, cache._cache)
        self.assertIn(3, cache._cache)

    def test_disk_cache_serves_frame_without_decoder(self):
        """A decoded frame must be served from the shared .npy disk cache."""
        from vla_factory.data.codec.torchcodec import TorchCodec
        from vla_factory.data.data_schema import VideoRef

        ref = VideoRef(
            video_path=self.video_path,
            frame_index=77,
            height=IMAGE_H,
            width=IMAGE_W,
            channels=3,
        )
        codec = TorchCodec(disk_cache=True)
        expected = codec.decode_frame(ref)  # decodes + writes <video>.frame_cache/000077.npy

        fresh = TorchCodec(disk_cache=True)
        img = fresh.decode_frame(ref)  # served from disk, no decoder opened
        self.assertTrue(np.array_equal(img, expected))
        self.assertEqual(fresh._caches, {}, "disk hit must not open a decoder")

    def test_pixel_parity_with_pyav(self):
        """torchcodec and pyav must decode the same pixels (codec parity)."""
        from vla_factory.data.codec.pyav import PyAVCodec
        from vla_factory.data.codec.torchcodec import TorchCodec
        from vla_factory.data.data_schema import VideoRef

        tc = TorchCodec(disk_cache=False)
        pv = PyAVCodec(disk_cache=False)
        for idx in [0, 1, 50, 100, 200, 300, 413]:
            ref = VideoRef(
                video_path=self.video_path,
                frame_index=idx,
                height=IMAGE_H,
                width=IMAGE_W,
                channels=3,
            )
            a = tc.decode_frame(ref)
            b = pv.decode_frame(ref)
            self.assertEqual(
                np.abs(a.astype(np.int16) - b.astype(np.int16)).max(), 0,
                f"frame {idx}: torchcodec and pyav differ",
            )


if __name__ == "__main__":
    unittest.main()

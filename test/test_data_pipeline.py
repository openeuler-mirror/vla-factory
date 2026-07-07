"""End-to-end tests for the VLA-Factory data pipeline (Phase 2).

Uses the built-in test dataset at ``test/data/lerobot_train_data_3_episodes/``.
If that directory is missing, all tests are skipped with a clear message.

Run:
    python test/test_data_pipeline.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch

DATASET_PATH = Path(_project_root) / "test/data" / "lerobot_train_data_3_episodes"

# Dataset constants (3 episodes × 414 frames each)
NUM_EPISODES = 3
FRAMES_PER_EPISODE = 414
TOTAL_FRAMES = NUM_EPISODES * FRAMES_PER_EPISODE  # 1242
STATE_DIM = 6
ACTION_DIM = 8
ACTION_HORIZON = 100
IMAGE_H = 480
IMAGE_W = 640
# Per-frame sampling: 414 frames, n_obs_steps=1 → 414 locators per episode
LOCATORS_PER_EPISODE = FRAMES_PER_EPISODE  # 414
TOTAL_LOCATORS = NUM_EPISODES * LOCATORS_PER_EPISODE  # 1242
# 3 episodes, train_ratio=0.9, seed=42 → 2 train, 1 val
TRAIN_EP_COUNT = 2
VAL_EP_COUNT = 1
TRAIN_LOCATORS = TRAIN_EP_COUNT * LOCATORS_PER_EPISODE  # 828


class TestLeRobotV3Reader(unittest.TestCase):
    """Test LeRobotV3Reader: schema, stats, episode ranges."""

    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Dataset not found")
        from vla_factory.data.formats.lerobot_v3 import LeRobotV3Reader
        cls.reader = LeRobotV3Reader()

    def test_can_read(self):
        self.assertTrue(self.reader.can_read(DATASET_PATH))

    def test_schema(self):
        schema = self.reader.get_schema(DATASET_PATH)
        self.assertEqual(schema.state_dim, STATE_DIM)
        self.assertEqual(schema.action_dim, ACTION_DIM)
        self.assertIn("front", schema.cameras)
        self.assertIn("wrist", schema.cameras)
        self.assertEqual(schema.fps, 30)
        self.assertEqual(schema.total_episodes, NUM_EPISODES)
        self.assertEqual(schema.total_frames, TOTAL_FRAMES)
        self.assertEqual(schema.robot_type, "so101_follower")
        self.assertEqual(schema.image_sizes["front"], (IMAGE_H, IMAGE_W))
        self.assertEqual(schema.image_sizes["wrist"], (IMAGE_H, IMAGE_W))

    def test_norm_stats(self):
        stats = self.reader.get_norm_stats(DATASET_PATH)
        self.assertIsNotNone(stats.state)
        self.assertIsNotNone(stats.action)
        self.assertEqual(len(stats.action.mean), ACTION_DIM)
        self.assertEqual(len(stats.action.std), ACTION_DIM)
        self.assertEqual(len(stats.state.mean), STATE_DIM)
        self.assertEqual(len(stats.state.std), STATE_DIM)
        self.assertEqual(stats.method, "zscore")

    def test_episode_ranges(self):
        ranges = self.reader.get_episode_ranges(DATASET_PATH)
        self.assertEqual(len(ranges), NUM_EPISODES)
        # Episode 0 should start at index 0
        self.assertEqual(ranges[0][0], 0)
        for ep_idx, (start, end) in ranges.items():
            self.assertEqual(
                end - start + 1, FRAMES_PER_EPISODE,
                f"Episode {ep_idx} length mismatch",
            )

    def test_episode_lengths(self):
        lengths = self.reader.get_episode_lengths(DATASET_PATH)
        self.assertEqual(len(lengths), NUM_EPISODES)
        for ep_idx, length in lengths.items():
            self.assertEqual(
                length, FRAMES_PER_EPISODE,
                f"Episode {ep_idx} length mismatch",
            )


class TestPyAVCodec(unittest.TestCase):
    """Test PyAVCodec video decoding → numpy HWC uint8."""

    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Dataset not found")
        from vla_factory.data.codec.pyav import PyAVCodec
        from vla_factory.data.formats.base import VideoRef

        video_path = (
            DATASET_PATH
            / "videos"
            / "observation.images.front"
            / "chunk-000"
            / "file-000.mp4"
        )
        if not video_path.exists():
            raise unittest.SkipTest("Video file not found")
        cls.codec = PyAVCodec()
        cls.video_path = video_path

    def test_decode_frame_shape(self):
        from vla_factory.data.formats.base import VideoRef
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
        from vla_factory.data.formats.base import VideoRef
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


class TestEpisodeLoad(unittest.TestCase):
    """Test read_episode → Episode → load_frames → Frame list."""

    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Dataset not found")
        from vla_factory.data.formats.lerobot_v3 import LeRobotV3Reader
        from vla_factory.data.codec.pyav import PyAVCodec

        cls.reader = LeRobotV3Reader()
        cls.codec = PyAVCodec()

    def test_episode_structure(self):
        episode = self.reader.read_episode(DATASET_PATH, 0, self.codec)
        self.assertEqual(episode.episode_index, 0)
        self.assertEqual(episode.num_frames, FRAMES_PER_EPISODE)

    def test_load_frames(self):
        episode = self.reader.read_episode(DATASET_PATH, 0, self.codec)
        frames = episode.load_frames()
        self.assertEqual(len(frames), FRAMES_PER_EPISODE)

        frame = frames[0]
        self.assertEqual(frame.index, 0)
        self.assertTrue(frame.is_first)
        self.assertIsNotNone(frame.state)
        self.assertEqual(len(frame.state), STATE_DIM)
        self.assertIsNotNone(frame.action)
        self.assertEqual(len(frame.action), ACTION_DIM)
        self.assertEqual(len(frame.images), 2)

        last_frame = frames[-1]
        self.assertTrue(last_frame.is_last)

    def test_decode_episode_image(self):
        """Verify images from Episode frames can be decoded via codec."""
        episode = self.reader.read_episode(DATASET_PATH, 0, self.codec)
        frames = episode.load_frames()
        frame = frames[0]
        for cam_name, ref in frame.images.items():
            img = self.codec.decode_frame(ref)
            self.assertEqual(img.shape, (IMAGE_H, IMAGE_W, 3))
            self.assertEqual(img.dtype, np.uint8)


class TestSlidingWindowSampler(unittest.TestCase):
    """Test SlidingWindowSampler logic."""

    def test_per_frame_sampling(self):
        from vla_factory.data.sampling.sampler import SlidingWindowSampler
        sampler = SlidingWindowSampler(n_obs_steps=1, action_horizon=ACTION_HORIZON)
        locators = sampler.sample_episode(0, FRAMES_PER_EPISODE)
        self.assertEqual(len(locators), FRAMES_PER_EPISODE)
        for i, loc in enumerate(locators):
            self.assertEqual(loc.start_frame_index, i)

    def test_n_obs_steps_2(self):
        from vla_factory.data.sampling.sampler import SlidingWindowSampler
        sampler = SlidingWindowSampler(n_obs_steps=2, action_horizon=ACTION_HORIZON)
        locators = sampler.sample_episode(0, FRAMES_PER_EPISODE)
        self.assertEqual(len(locators), FRAMES_PER_EPISODE - 1)

    def test_short_episode(self):
        from vla_factory.data.sampling.sampler import SlidingWindowSampler
        sampler = SlidingWindowSampler(n_obs_steps=1, action_horizon=ACTION_HORIZON)
        locators = sampler.sample_episode(0, 50)
        self.assertEqual(len(locators), 50)

    def test_empty_episode(self):
        from vla_factory.data.sampling.sampler import SlidingWindowSampler
        sampler = SlidingWindowSampler(n_obs_steps=1, action_horizon=ACTION_HORIZON)
        locators = sampler.sample_episode(0, 0)
        self.assertEqual(len(locators), 0)

    def test_single_frame_episode(self):
        from vla_factory.data.sampling.sampler import SlidingWindowSampler
        sampler = SlidingWindowSampler(n_obs_steps=1, action_horizon=ACTION_HORIZON)
        locators = sampler.sample_episode(0, 1)
        self.assertEqual(len(locators), 1)


class TestManifestBuild(unittest.TestCase):
    """Test manifest construction and train/val split."""

    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Dataset not found")
        from vla_factory.data.formats.lerobot_v3 import LeRobotV3Reader
        from vla_factory.data.manifest import build_manifest

        cls.reader = LeRobotV3Reader()
        cls.schema = cls.reader.get_schema(DATASET_PATH)
        cls.norm_stats = cls.reader.get_norm_stats(DATASET_PATH)
        cls.episode_ranges = cls.reader.get_episode_ranges(DATASET_PATH)
        cls.episode_lengths = cls.reader.get_episode_lengths(DATASET_PATH)
        cls.manifest = build_manifest(
            schema=cls.schema,
            norm_stats=cls.norm_stats,
            episode_ranges=cls.episode_ranges,
            episode_lengths=cls.episode_lengths,
            n_obs_steps=1,
            action_horizon=ACTION_HORIZON,
            train_ratio=0.9,
            seed=42,
        )

    def test_total_locators(self):
        self.assertEqual(len(self.manifest.locators), TOTAL_LOCATORS)

    def test_train_val_split(self):
        train_eps = set(self.manifest.splits["train"])
        val_eps = set(self.manifest.splits["val"])
        self.assertEqual(len(train_eps & val_eps), 0)
        self.assertEqual(len(train_eps | val_eps), NUM_EPISODES)
        self.assertEqual(len(train_eps), TRAIN_EP_COUNT)
        self.assertEqual(len(val_eps), VAL_EP_COUNT)

    def test_no_data_leakage(self):
        train_locators = self.manifest.train_locators
        val_locators = self.manifest.val_locators
        train_eps = {loc.episode_index for loc in train_locators}
        val_eps = {loc.episode_index for loc in val_locators}
        self.assertEqual(len(train_eps & val_eps), 0)

    def test_deterministic_split(self):
        from vla_factory.data.manifest import build_manifest
        manifest2 = build_manifest(
            schema=self.schema,
            norm_stats=self.norm_stats,
            episode_ranges=self.episode_ranges,
            episode_lengths=self.episode_lengths,
            n_obs_steps=1,
            action_horizon=ACTION_HORIZON,
            train_ratio=0.9,
            seed=42,
        )
        self.assertEqual(
            self.manifest.splits["train"],
            manifest2.splits["train"],
        )


class TestTransforms(unittest.TestCase):
    """Test individual transforms (numpy-based)."""

    def test_normalize(self):
        from vla_factory.data.transforms.normalize import Normalize
        from vla_factory.data.manifest import NormStats, FeatureStats
        stats = NormStats(
            state=FeatureStats(mean=[1.0, 2.0], std=[0.5, 0.5]),
            action=FeatureStats(mean=[3.0, 4.0], std=[1.0, 1.0]),
        )
        norm = Normalize(stats)
        sample = {
            "state": np.array([1.5, 2.5], dtype=np.float32),
            "actions": np.array([[4.0, 5.0], [2.0, 3.0]], dtype=np.float32),
        }
        result = norm(sample)
        # State IS normalized (z-score, matching lerobot)
        np.testing.assert_allclose(result["state"], [1.0, 1.0])
        # Actions ARE normalized (z-score, matching lerobot)
        np.testing.assert_allclose(result["actions"][0], [1.0, 1.0])

    def test_resize_images_noop(self):
        from vla_factory.data.transforms.resize_images import ResizeImages
        resize = ResizeImages(0, 0)  # no-op
        img = np.random.rand(3, IMAGE_H, IMAGE_W).astype(np.float32)
        sample = {
            "images.front": img,
            "image_masks.front": np.ones(3, dtype=bool),
            "actions": np.random.rand(10, ACTION_DIM).astype(np.float32),
        }
        result = resize(sample)
        self.assertEqual(result["images.front"].shape, (3, IMAGE_H, IMAGE_W))

    def test_resize_images_without_size_is_noop(self):
        from vla_factory.data.transforms import TransformContext
        from vla_factory.data.transforms.resize_images import ResizeImages

        self.assertIsNone(ResizeImages.from_config({"type": "resize_images"}, TransformContext(model_config={})))

    def test_resize_images_requires_height_and_width_together(self):
        from vla_factory.data.transforms import TransformContext
        from vla_factory.data.transforms.resize_images import ResizeImages

        with self.assertRaisesRegex(ValueError, "height.*width"):
            ResizeImages.from_config({"type": "resize_images", "height": 224}, TransformContext(model_config={}))

    def test_pad_dimensions(self):
        from vla_factory.data.transforms.pad_dimensions import PadDimensions
        pad = PadDimensions(target_dim=32)
        actions = np.random.rand(10, ACTION_DIM).astype(np.float32)
        sample = {"actions": actions}
        result = pad(sample)
        self.assertEqual(result["actions"].shape, (10, 32))
        np.testing.assert_allclose(result["actions"][:, :ACTION_DIM], actions)
        np.testing.assert_array_equal(result["actions"][:, ACTION_DIM:], 0)

    def test_pad_dimensions_noop(self):
        from vla_factory.data.transforms.pad_dimensions import PadDimensions
        pad = PadDimensions(target_dim=ACTION_DIM)  # already 8, no-op
        actions = np.random.rand(10, ACTION_DIM).astype(np.float32)
        sample = {"actions": actions}
        result = pad(sample)
        self.assertEqual(result["actions"].shape, (10, ACTION_DIM))
        np.testing.assert_allclose(result["actions"], actions)


class TestVLADataset(unittest.TestCase):
    """Test VLADataset __getitem__ output (numpy dict)."""

    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Dataset not found")
        from vla_factory.data.formats.lerobot_v3 import LeRobotV3Reader
        from vla_factory.data.codec.pyav import PyAVCodec
        from vla_factory.data.manifest import build_manifest
        from vla_factory.data.dataset import VLADataset

        cls.reader = LeRobotV3Reader()
        cls.codec = PyAVCodec()
        schema = cls.reader.get_schema(DATASET_PATH)
        norm_stats = cls.reader.get_norm_stats(DATASET_PATH)
        episode_ranges = cls.reader.get_episode_ranges(DATASET_PATH)
        episode_lengths = cls.reader.get_episode_lengths(DATASET_PATH)
        cls.manifest = build_manifest(
            schema=schema,
            norm_stats=norm_stats,
            episode_ranges=episode_ranges,
            episode_lengths=episode_lengths,
            n_obs_steps=1,
            action_horizon=ACTION_HORIZON,
            train_ratio=0.9,
            seed=42,
        )
        cls.dataset = VLADataset(
            cls.manifest, cls.reader, cls.codec, DATASET_PATH,
            transforms=[], split="train",
        )

    def test_len(self):
        self.assertGreater(len(self.dataset), 0)

    def test_getitem_structure(self):
        sample = self.dataset[0]
        self.assertIn("actions", sample)
        self.assertIn("action_is_pad", sample)
        self.assertIn("state", sample)
        self.assertIn("images.front", sample)
        self.assertIn("images.wrist", sample)
        self.assertIn("image_masks.front", sample)
        self.assertIn("image_masks.wrist", sample)

        actions = sample["actions"]
        self.assertEqual(actions.shape[0], ACTION_HORIZON)
        self.assertEqual(actions.shape[1], ACTION_DIM)

    def test_getitem_is_numpy(self):
        """Verify outputs are numpy arrays (not torch tensors)."""
        sample = self.dataset[0]
        self.assertIsInstance(sample["actions"], np.ndarray)
        self.assertIsInstance(sample["state"], np.ndarray)
        self.assertIsInstance(sample["images.front"], np.ndarray)

    def test_image_format(self):
        """Dataset images should stay raw HWC uint8; transforms own model formatting."""
        sample = self.dataset[0]
        for key in ("images.front", "images.wrist"):
            img = sample[key]
            self.assertEqual(img.ndim, 3)
            self.assertEqual(img.shape, (IMAGE_H, IMAGE_W, 3))
            self.assertEqual(img.dtype, np.uint8)

    def test_state_format(self):
        sample = self.dataset[0]
        state = sample["state"]
        self.assertIsNotNone(state)
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(state.dtype, np.float32)

    def test_actions_format(self):
        sample = self.dataset[0]
        actions = sample["actions"]
        self.assertEqual(actions.shape, (ACTION_HORIZON, ACTION_DIM))
        self.assertEqual(actions.dtype, np.float32)

    def test_action_is_pad_format(self):
        sample = self.dataset[0]
        is_pad = sample["action_is_pad"]
        self.assertEqual(is_pad.shape, (ACTION_HORIZON,))
        self.assertEqual(is_pad.dtype, bool)
        # First sample (start_frame_index=0) has all real actions
        self.assertFalse(is_pad.any())

    def test_action_is_pad_tail(self):
        """Last frame in episode should have mostly padded actions."""
        # Get the last locator of the first training episode
        train_locs = self.manifest.train_locators
        ep_idx = train_locs[-1].episode_index
        # Find the last locator for this episode
        ep_locs = [loc for loc in train_locs if loc.episode_index == ep_idx]
        last_loc = ep_locs[-1]
        # Find its index in the manifest
        idx = self.manifest.locators.index(last_loc)
        real_idx = self.dataset._indices.index(idx)
        sample = self.dataset[real_idx]
        is_pad = sample["action_is_pad"]
        # The last frame should have action_horizon - 1 padded steps
        self.assertTrue(is_pad[-1])


class TestDataLoaderBatching(unittest.TestCase):
    """Test DataLoader batch dimensions with collate_fn (numpy → Tensor)."""

    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Dataset not found")
        from vla_factory.data.formats.lerobot_v3 import LeRobotV3Reader
        from vla_factory.data.codec.pyav import PyAVCodec
        from vla_factory.data.manifest import build_manifest
        from vla_factory.data.dataset import VLADataset, collate_fn

        cls.reader = LeRobotV3Reader()
        cls.codec = PyAVCodec()
        schema = cls.reader.get_schema(DATASET_PATH)
        norm_stats = cls.reader.get_norm_stats(DATASET_PATH)
        episode_ranges = cls.reader.get_episode_ranges(DATASET_PATH)
        episode_lengths = cls.reader.get_episode_lengths(DATASET_PATH)
        cls.manifest = build_manifest(
            schema=schema,
            norm_stats=norm_stats,
            episode_ranges=episode_ranges,
            episode_lengths=episode_lengths,
            n_obs_steps=1,
            action_horizon=ACTION_HORIZON,
            train_ratio=0.9,
            seed=42,
        )
        cls.dataset = VLADataset(
            cls.manifest, cls.reader, cls.codec, DATASET_PATH,
            transforms=[], split="train",
        )
        cls._collate_fn = staticmethod(collate_fn)

    def test_batch_dimensions(self):
        # Train split has 828 samples, batch_size=2 is safe
        batch_size = 2
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=self._collate_fn,
        )
        batch = next(iter(loader))
        obs = batch["observation"]
        actions = batch["actions"]

        self.assertEqual(actions.shape, (batch_size, ACTION_HORIZON, ACTION_DIM))
        self.assertIsNotNone(obs.state)
        self.assertEqual(obs.state.shape, (batch_size, STATE_DIM))
        for cam_name in obs.images:
            self.assertEqual(obs.images[cam_name].shape, (batch_size, IMAGE_H, IMAGE_W, 3))

    def test_batch_is_tensor(self):
        """Verify collate_fn converts numpy → torch.Tensor."""
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=2, shuffle=False, num_workers=0,
            collate_fn=self._collate_fn,
        )
        batch = next(iter(loader))
        self.assertIsInstance(batch["actions"], torch.Tensor)
        self.assertIsInstance(batch["observation"].state, torch.Tensor)

    def test_batch_action_is_pad(self):
        """Verify action_is_pad is in batch and is a bool tensor."""
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=2, shuffle=False, num_workers=0,
            collate_fn=self._collate_fn,
        )
        batch = next(iter(loader))
        self.assertIn("action_is_pad", batch)
        is_pad = batch["action_is_pad"]
        self.assertIsInstance(is_pad, torch.Tensor)
        self.assertEqual(is_pad.shape, (2, ACTION_HORIZON))
        self.assertEqual(is_pad.dtype, torch.bool)


class TestBuildTransforms(unittest.TestCase):
    """Test _build_transforms logic in loader.py."""

    def test_empty(self):
        from vla_factory.data.loader import _build_transforms
        with self.assertRaises(ValueError):
            _build_transforms()

    def test_normalize_only(self):
        from vla_factory.data.loader import _build_transforms
        from vla_factory.data.manifest import NormStats, FeatureStats
        norm_stats = NormStats(
            action=FeatureStats(mean=[0.0] * ACTION_DIM, std=[1.0] * ACTION_DIM),
        )
        with self.assertRaises(ValueError):
            _build_transforms(norm_stats=norm_stats)

        transforms = _build_transforms(
            norm_stats=norm_stats,
            model_metadata={
                "transform_inputs": [
                    {"type": "normalize_vector", "fields": ["actions"]},
                ],
            },
        )
        self.assertEqual(len(transforms), 1)
        from vla_factory.data.transforms.normalize import NormalizeVector
        self.assertIsInstance(transforms[0], NormalizeVector)

    def test_normalize_and_pad(self):
        from vla_factory.data.loader import _build_transforms
        from vla_factory.data.manifest import NormStats, FeatureStats
        norm_stats = NormStats(
            action=FeatureStats(mean=[0.0] * ACTION_DIM, std=[1.0] * ACTION_DIM),
        )
        transforms = _build_transforms(
            norm_stats=norm_stats,
            model_metadata={
                "action_dim": 32,
                "transform_inputs": [
                    {"type": "normalize_vector", "fields": ["actions"]},
                    {"type": "pad_dimensions", "fields": ["actions"]},
                ],
            },
            action_dim=ACTION_DIM,
        )
        self.assertEqual(len(transforms), 2)

    def test_act_profile_image_pipeline(self):
        from vla_factory.config.parser import parse_recipe_from_string
        from vla_factory.data.loader import _build_transforms
        from vla_factory.data.manifest import DataSchema, NormStats, FeatureStats
        from vla_factory.config.defaults import resolve_recipe

        recipe = parse_recipe_from_string("model:\n  name: act\n")
        recipe = resolve_recipe(recipe)
        transforms = _build_transforms(
            norm_stats=NormStats(
                state=FeatureStats(mean=[0.0] * STATE_DIM, std=[1.0] * STATE_DIM),
                action=FeatureStats(mean=[0.0] * ACTION_DIM, std=[1.0] * ACTION_DIM),
            ),
            model_metadata={
                "action_dim": ACTION_DIM,
                "profile": recipe.model_config,
                "transform_inputs": recipe.model_config["transforms"]["inputs"],
            },
            action_dim=ACTION_DIM,
            recipe=recipe,
            schema=DataSchema(state_dim=STATE_DIM, action_dim=ACTION_DIM, cameras=("front",)),
        )
        sample = {
            "images.front": np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
            "state": np.zeros(STATE_DIM, dtype=np.float32),
            "actions": np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32),
        }
        out = transforms(sample)
        self.assertEqual(out["images.front"].shape, (3, IMAGE_H, IMAGE_W))
        self.assertEqual(out["images.front"].dtype, np.float32)


class TestEndToEnd(unittest.TestCase):
    """Full pipeline: YAML → DataLoaders → batch."""

    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists():
            raise unittest.SkipTest("Dataset not found")
        from vla_factory.config.parser import parse_recipe
        from vla_factory.config.defaults import resolve_recipe
        from vla_factory.data.loader import create_dataloaders

        yaml_path = Path(_project_root) / "vla_factory" / "examples" / "act_aloha.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest("YAML config not found")
        cls.recipe = parse_recipe(yaml_path)
        # Override dataset path and batch_size for the small test dataset
        cls.recipe.data.source.path = str(DATASET_PATH)
        cls.recipe.batch_size = 2
        cls.recipe = resolve_recipe(cls.recipe)
        cls.train_loader, cls.val_loader = create_dataloaders(cls.recipe)

    def test_train_loader_nonempty(self):
        self.assertGreater(len(self.train_loader), 0)

    def test_val_loader_nonempty(self):
        self.assertGreater(len(self.val_loader), 0)

    def test_batch_output(self):
        batch = next(iter(self.train_loader))
        self.assertIn("observation", batch)
        self.assertIn("actions", batch)
        self.assertIn("action_is_pad", batch)

        obs = batch["observation"]
        actions = batch["actions"]

        batch_size = self.recipe.batch_size
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(actions.shape[1], ACTION_HORIZON)
        self.assertEqual(actions.shape[2], ACTION_DIM)


if __name__ == "__main__":
    unittest.main(verbosity=2)

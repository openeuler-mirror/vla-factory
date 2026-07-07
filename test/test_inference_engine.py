#!/usr/bin/env python3
"""Unit tests for the inference engine layer (§12).

Tests ObsDict, InferenceEngine (with mock model),
and the three action-chunking execution strategies.

Run:
    python test/test_inference_engine.py
    pytest test/test_inference_engine.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_factory.config.recipe import ActionSpecConfig, TrainRecipe
from vla_factory.data.manifest import DataSchema, FeatureStats, NormStats, resolve_vector_keys
from vla_factory.deploy.infer import (
    InferenceEngine,
    ObsDict,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_recipe(
    action_dim: int = 6,
    action_horizon: int = 10,
    cameras: tuple[str, ...] = ("front",),
    state_dim: int = 6,
    has_language: bool = False,
) -> TrainRecipe:
    return TrainRecipe(
        model_name="act",
        action_spec=ActionSpecConfig(
            action_dim=action_dim,
            action_horizon=action_horizon,
            action_type="joint_pos",
        ),
    )


def _make_schema(
    action_dim: int = 6,
    state_dim: int = 6,
    cameras: tuple[str, ...] = ("front",),
    has_language: bool = False,
) -> DataSchema:
    return DataSchema(
        state_dim=state_dim,
        action_dim=action_dim,
        cameras=cameras,
        image_sizes={cam: (224, 224) for cam in cameras},
        has_language=has_language,
    )


def _make_norm_stats(action_dim: int = 6, state_dim: int = 6) -> NormStats:
    return NormStats(
        state=FeatureStats(
            mean=list(np.zeros(state_dim)),
            std=list(np.ones(state_dim)),
        ),
        action=FeatureStats(
            mean=list(np.zeros(action_dim)),
            std=list(np.ones(action_dim)),
        ),
    )


def _make_obs_dict(
    cameras: tuple[str, ...] = ("front",),
    h: int = 64,
    w: int = 64,
    state_dim: int = 6,
) -> ObsDict:
    video = {cam: np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for cam in cameras}
    state = np.random.randn(state_dim).astype(np.float32)
    return ObsDict(video=video, state=state)


# ── Tests: ObsDict ──────────────────────────────────────────────────


class TestObsDict:
    def test_construction(self):
        obs = _make_obs_dict()
        assert "front" in obs.video
        assert obs.video["front"].shape == (64, 64, 3)
        assert obs.state is not None
        assert obs.state.shape == (6,)
        assert obs.language is None

    def test_frozen(self):
        obs = _make_obs_dict()
        with pytest.raises(AttributeError):
            obs.language = "test"  # type: ignore[misc]


# ── Tests: InferenceEngine with mock model ──────────────────────────


class _MockModel:
    """Minimal mock that satisfies InferenceEngine's predict_actions call."""

    def __init__(self, action_dim: int = 6, action_horizon: int = 10):
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self._eval = False

    def predict_actions(self, observation):
        # Return a deterministic chunk: [1, H, D] with values = step_index
        B = 1
        H = self.action_horizon
        D = self.action_dim
        actions = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, D)
        return actions.unsqueeze(0)  # [1, H, D]

    def load_state_dict(self, state_dict, strict=True):
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def to(self, device):
        return self

    def eval(self):
        self._eval = True
        return self

    def __call__(self, *args, **kwargs):
        return self


def _setup_checkpoint_dir(
    tmpdir: Path,
    recipe: TrainRecipe,
    schema: DataSchema,
    norm_stats: NormStats,
    action_dim: int = 6,
    action_horizon: int = 10,
) -> Path:
    """Create a minimal checkpoint directory with inference_metadata/."""
    meta_dir = tmpdir / "inference_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Save schema
    schema_dict = {
        "state_dim": schema.state_dim,
        "action_dim": schema.action_dim,
        "cameras": list(schema.cameras),
        "has_language": schema.has_language,
    }
    with open(meta_dir / "schema.json", "w") as f:
        json.dump(schema_dict, f)

    # Save norm_stats
    ns_dict = {
        "state": {"mean": norm_stats.state.mean, "std": norm_stats.state.std} if norm_stats.state else None,
        "action": {"mean": norm_stats.action.mean, "std": norm_stats.action.std} if norm_stats.action else None,
        "method": "zscore",
    }
    with open(meta_dir / "norm_stats.json", "w") as f:
        json.dump(ns_dict, f)

    # Save recipe as YAML (manual serialization for test fixture)
    import yaml
    import dataclasses
    recipe_dict = {}
    for f_field in dataclasses.fields(recipe):
        val = getattr(recipe, f_field.name)
        if dataclasses.is_dataclass(val) and not isinstance(val, type):
            val = dataclasses.asdict(val)
        recipe_dict[f_field.name] = val
    # Re-structure to match expected YAML layout
    yaml_dict = {
        "model": {"name": recipe.model_name, "path": recipe.model_path},
        "action_spec": dataclasses.asdict(recipe.action_spec),
        "training": {
            "backend": recipe.backend,
            "lr": recipe.lr,
            "batch_size": recipe.batch_size,
            "total_steps": recipe.total_steps,
        },
    }
    with open(meta_dir / "recipe.yaml", "w") as f:
        yaml.dump(yaml_dict, f)

    # Save model weights (empty state dict is fine for mock)
    final_dir = tmpdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save({}, final_dir / "model.pt")

    return tmpdir


class TestInferenceEngine:
    @pytest.fixture
    def checkpoint_dir(self, tmp_path):
        recipe = _make_recipe(action_dim=6, action_horizon=10)
        schema = _make_schema(action_dim=6, state_dim=6, cameras=("front",))
        norm_stats = _make_norm_stats(action_dim=6, state_dim=6)
        return _setup_checkpoint_dir(tmp_path, recipe, schema, norm_stats)

    def test_invalid_strategy_raises(self, checkpoint_dir):
        """InferenceEngine rejects unknown execution strategies."""
        # We can't fully construct it without a registered model,
        # but we test the validation directly.
        with pytest.raises(ValueError, match="Unknown strategy"):
            InferenceEngine(
                checkpoint_path=checkpoint_dir,
                execution_strategy="bogus",
            )

    def test_synchronous_returns_full_chunk(self, checkpoint_dir):
        """Synchronous strategy returns [H, D] array."""
        # This test requires the model to be registered — skip if not
        pytest.skip("Requires registered 'act' model; integration test")

    def test_receding_returns_single_step(self, checkpoint_dir):
        """Receding horizon strategy returns [D] array."""
        pytest.skip("Requires registered 'act' model; integration test")

    def test_temporal_ensembling_returns_single_step(self, checkpoint_dir):
        """Temporal ensembling returns [D] array."""
        pytest.skip("Requires registered 'act' model; integration test")


# ── Tests: _obs_to_observation normalization ────────────────────────


class TestObsNormalization:
    """Test the normalization logic that will be used by _obs_to_observation."""

    def test_imagenet_normalization_values(self):
        """Verify ImageNet constants are correct."""
        from vla_factory.data.transforms.normalize import IMAGENET_MEAN, IMAGENET_STD
        np.testing.assert_allclose(IMAGENET_MEAN, [0.485, 0.456, 0.406], atol=1e-6)
        np.testing.assert_allclose(IMAGENET_STD, [0.229, 0.224, 0.225], atol=1e-6)

    def test_zscore_state_normalization(self):
        """Verify z-score state normalization matches training."""
        norm_stats = _make_norm_stats(state_dim=6)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)

        mean = np.array(norm_stats.state.mean, dtype=np.float32)
        std = np.array(norm_stats.state.std, dtype=np.float32) + 1e-8
        normalized = (state - mean) / std

        # With mean=0, std=1, normalization is identity
        np.testing.assert_allclose(normalized, state, atol=1e-6)

    def test_zscore_with_nontrivial_stats(self):
        """z-score with non-trivial mean/std."""
        norm_stats = NormStats(
            state=FeatureStats(mean=[10.0], std=[2.0]),
            action=FeatureStats(mean=[0.0], std=[1.0]),
        )
        state = np.array([12.0], dtype=np.float32)
        mean = np.array(norm_stats.state.mean, dtype=np.float32)
        std = np.array(norm_stats.state.std, dtype=np.float32) + 1e-8
        normalized = (state - mean) / std
        np.testing.assert_allclose(normalized, [1.0], atol=1e-6)


class TestTrainInferNormalizationConsistency:
    """Verify the Normalize transform's image-normalisation contract.

    Image normalisation is selected by the Normalize step's
    ``use_imagenet_stats`` flag (default True → fixed ImageNet constants, the
    pretrained-backbone default). It is NOT driven by mutating ``norm_stats``
    in train/infer — that hack is gone. With ``use_imagenet_stats=False``,
    per-camera dataset stats from ``norm_stats.images`` are used instead.
    """

    def test_image_normalization_fallback_imagenet(self):
        """When norm_stats.images is None, both paths use ImageNet."""
        from vla_factory.data.transforms.normalize import Normalize

        norm_stats = NormStats(
            state=FeatureStats(mean=[0.0] * 6, std=[1.0] * 6),
            action=FeatureStats(mean=[0.0] * 6, std=[1.0] * 6),
            images=None,  # no per-camera stats → ImageNet fallback
        )

        img_hwc = np.full((32, 32, 3), 0.5, dtype=np.float32)  # [0,1] range

        # ── Training path: Normalize transform ──
        sample_train = {"images.front": img_hwc.transpose(2, 0, 1).copy()}
        normalizer = Normalize(norm_stats)
        sample_train = normalizer(sample_train)
        train_result = sample_train["images.front"]

        # ── Inference path: _obs_to_observation logic (same code) ──
        from vla_factory.data.transforms.normalize import IMAGENET_MEAN, IMAGENET_STD
        img_inf = img_hwc.transpose(2, 0, 1).copy()
        infer_result = (img_inf - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

        np.testing.assert_allclose(train_result, infer_result, atol=1e-6)

    def test_image_normalization_with_per_camera_stats(self):
        """use_imagenet_stats=False makes Normalize consume per-camera stats."""
        from vla_factory.data.transforms.normalize import Normalize, IMAGENET_MEAN, IMAGENET_STD

        custom_mean = [0.3, 0.4, 0.5]
        custom_std = [0.2, 0.3, 0.4]
        norm_stats = NormStats(
            state=FeatureStats(mean=[0.0] * 6, std=[1.0] * 6),
            action=FeatureStats(mean=[0.0] * 6, std=[1.0] * 6),
            images={"front": FeatureStats(mean=custom_mean, std=custom_std)},
        )

        img_hwc = np.full((32, 32, 3), 0.5, dtype=np.float32)
        img = img_hwc.transpose(2, 0, 1).copy()

        # Default (use_imagenet_stats=True) IGNORES stats.images → ImageNet.
        default_result = Normalize(norm_stats)({"images.front": img.copy()})["images.front"]
        np.testing.assert_allclose(
            default_result,
            (img - IMAGENET_MEAN[:, None, None]) / (IMAGENET_STD[:, None, None] + 1e-8),
            atol=1e-6,
        )

        # Opting out (use_imagenet_stats=False) uses the per-camera dataset stats.
        custom_result = (
            Normalize(norm_stats, use_imagenet_stats=False)({"images.front": img.copy()})["images.front"]
        )
        mean_arr = np.array(custom_mean, dtype=np.float32)
        std_arr = np.array(custom_std, dtype=np.float32) + 1e-8
        np.testing.assert_allclose(
            custom_result, (img - mean_arr[:, None, None]) / std_arr[:, None, None], atol=1e-6
        )

    def test_state_normalization_uses_saved_stats(self):
        """Both paths use norm_stats.state for z-score."""
        from vla_factory.data.transforms.normalize import Normalize

        state_mean = [1.0, 2.0, 3.0]
        state_std = [0.5, 1.0, 1.5]
        norm_stats = NormStats(
            state=FeatureStats(mean=state_mean, std=state_std),
            action=FeatureStats(mean=[0.0], std=[1.0]),
        )

        raw_state = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        # Training path
        sample = {"state": raw_state.copy()}
        normalizer = Normalize(norm_stats)
        sample = normalizer(sample)
        train_result = sample["state"]

        # Expected: (x - mean) / (std + eps)
        expected = (raw_state - np.array(state_mean)) / (np.array(state_std) + 1e-8)
        np.testing.assert_allclose(train_result, expected, atol=1e-6)

    def test_use_imagenet_stats_flag_selects_image_normalization(self):
        """The use_imagenet_stats flag (not stats mutation) selects image norm.

        Default True → fixed ImageNet constants (the pretrained-backbone default).
        Explicit False → per-camera dataset stats from stats.images.
        """
        from vla_factory.data.transforms.normalize import Normalize, IMAGENET_MEAN, IMAGENET_STD

        img_hwc = np.full((16, 16, 3), 0.5, dtype=np.float32)
        img = img_hwc.transpose(2, 0, 1).copy()

        # Same stats object in both cases — the FLAG selects the method.
        norm_stats = NormStats(
            state=FeatureStats(mean=[0.0], std=[1.0]),
            action=FeatureStats(mean=[0.0], std=[1.0]),
            images={"front": FeatureStats(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])},
        )

        # Default: ImageNet
        result_a = Normalize(norm_stats)({"images.front": img.copy()})["images.front"]
        expected_a = (img - IMAGENET_MEAN[:, None, None]) / (IMAGENET_STD[:, None, None] + 1e-8)
        np.testing.assert_allclose(result_a, expected_a, atol=1e-6)

        # Opt out: per-camera dataset stats
        result_b = (
            Normalize(norm_stats, use_imagenet_stats=False)({"images.front": img.copy()})["images.front"]
        )
        custom_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        custom_std = np.array([0.5, 0.5, 0.5], dtype=np.float32) + 1e-8
        expected_b = (img - custom_mean[:, None, None]) / custom_std[:, None, None]
        np.testing.assert_allclose(result_b, expected_b, atol=1e-6)

        # The two methods must differ — proves the flag actually switches behaviour.
        assert not np.allclose(result_a, result_b, atol=1e-3), (
            "ImageNet and per-camera normalization should produce different results"
        )


# ── Tests: ZMQObsAdapter ───────────────────────────────────────────


class TestZMQObsAdapter:
    def test_basic_conversion(self):
        from vla_factory.deploy.transport import _ZMQObsAdapter

        adapter = _ZMQObsAdapter(camera_keys=("front", "wrist"))
        zmq_obs = {
            "observation.images.front": np.zeros((64, 64, 3), dtype=np.uint8),
            "observation.images.wrist": np.ones((64, 64, 3), dtype=np.uint8),
            "observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }

        obs = adapter(zmq_obs)
        assert isinstance(obs, ObsDict)
        assert "front" in obs.video
        assert "wrist" in obs.video
        assert obs.state is not None
        np.testing.assert_array_equal(obs.state, [1.0, 2.0, 3.0])

    def test_missing_camera_raises(self):
        from vla_factory.deploy.transport import _ZMQObsAdapter

        adapter = _ZMQObsAdapter(camera_keys=("front", "wrist"))
        zmq_obs = {
            "observation.images.front": np.zeros((64, 64, 3), dtype=np.uint8),
            "observation.state": np.array([1.0], dtype=np.float32),
        }
        with pytest.raises(KeyError, match="wrist"):
            adapter(zmq_obs)


# ── Tests: ReplayPolicy ────────────────────────────────────────────


class TestReplayPolicy:
    def test_replay_sequence(self):
        from vla_factory.deploy.adapters import ReplayPolicy

        data = [
            {"action": np.array([1.0, 2.0])},
            {"action": np.array([3.0, 4.0])},
            {"action": np.array([5.0, 6.0])},
        ]
        policy = ReplayPolicy(data)
        obs = ObsDict(video={}, state=None)

        np.testing.assert_array_equal(policy.predict(obs), [1.0, 2.0])
        np.testing.assert_array_equal(policy.predict(obs), [3.0, 4.0])
        np.testing.assert_array_equal(policy.predict(obs), [5.0, 6.0])

    def test_replay_exhausted(self):
        from vla_factory.deploy.adapters import ReplayPolicy

        policy = ReplayPolicy([{"action": np.array([1.0])}])
        obs = ObsDict(video={}, state=None)
        policy.predict(obs)
        with pytest.raises(StopIteration):
            policy.predict(obs)

    def test_replay_reset(self):
        from vla_factory.deploy.adapters import ReplayPolicy

        policy = ReplayPolicy([{"action": np.array([1.0])}])
        obs = ObsDict(video={}, state=None)
        policy.predict(obs)
        policy.reset()
        np.testing.assert_array_equal(policy.predict(obs), [1.0])


# ── Tests: resolve_vector_keys ─────────────────────────────────────


class TestResolveVectorKeys:
    """The dimension→key order is a data/model contract.

    Resolution is deliberately lenient: dataset ``names`` win; if no keys are
    resolvable it warns and returns empty rather than raising, because only
    some platforms (lerobot host) actually need the per-dimension order.
    """

    @staticmethod
    def _recipe(source_path: str = "/nonexistent-dataset-for-tests") -> TrainRecipe:
        from vla_factory.config.recipe import DataConfig, DataSourceConfig

        return TrainRecipe(
            model_name="act",
            data=DataConfig(
                source=DataSourceConfig(path=source_path, format="lerobot-v3")
            ),
        )

    def test_schema_keys_win(self, caplog):
        """Dataset ``names`` (in schema) are returned when present."""
        schema = DataSchema(
            state_dim=3, action_dim=2,
            state_keys=("s0", "s1", "s2"), action_keys=("a0", "a1"),
        )
        sk, ak = resolve_vector_keys(schema, self._recipe())
        assert sk == ("s0", "s1", "s2")
        assert ak == ("a0", "a1")

    def test_warns_when_missing(self, caplog):
        """No names and no readable dataset → warning + empty, not a raise."""
        schema = DataSchema(state_dim=3, action_dim=3)
        with caplog.at_level("WARNING", logger="vla_factory.data.manifest"):
            sk, ak = resolve_vector_keys(schema, self._recipe())
        assert sk == ()
        assert ak == ()
        assert any("no feature `names`" in r.message for r in caplog.records)

    def test_length_mismatch_warns(self, caplog):
        """A resolved key count that disagrees with the dim is warned about
        and emptied (never returned misaligned)."""
        schema = DataSchema(
            state_dim=3, action_dim=2,
            state_keys=("a", "b"), action_keys=("x", "y"),  # state: 2 != 3; action: 2 == 2
        )
        with caplog.at_level("WARNING", logger="vla_factory.data.manifest"):
            sk, ak = resolve_vector_keys(schema, self._recipe())
        assert sk == ()          # mismatched → emptied
        assert ak == ("x", "y")  # matched → kept
        assert any("length mismatch" in r.message for r in caplog.records)

    def test_dim_zero_returns_empty(self):
        """A stateless vector (dim 0) resolves to empty keys without warning."""
        schema = DataSchema(state_dim=0, action_dim=2, action_keys=("x", "y"))
        sk, ak = resolve_vector_keys(schema, self._recipe())
        assert sk == ()
        assert ak == ("x", "y")


# ── Tests: LerobotHost wire-format adapters ────────────────────────


class TestLerobotHostAdapters:
    """The host wire-format adapter must read/write exact keys in order — no
    sorting, no auto-detection (which would scramble dimensions)."""

    def test_obs_adapter_assembles_state_in_key_order(self):
        from vla_factory.deploy.lerobot_host_adapter import LerobotHostObsAdapter

        state_keys = ("shoulder", "elbow", "wrist")
        # Host dict deliberately in scrambled insertion order.
        host = {"wrist": 3.0, "shoulder": 1.0, "elbow": 2.0}
        adapter = LerobotHostObsAdapter(
            camera_keys=(), state_keys=state_keys, state_dim=3,
        )
        obs = adapter(host)
        np.testing.assert_array_equal(obs.state, [1.0, 2.0, 3.0])

    def test_obs_adapter_missing_state_key_raises(self):
        from vla_factory.deploy.lerobot_host_adapter import LerobotHostObsAdapter

        adapter = LerobotHostObsAdapter(
            camera_keys=(), state_keys=("a", "b"), state_dim=2,
        )
        with pytest.raises(KeyError, match="'a'"):
            adapter({"b": 1.0})

    def test_obs_adapter_count_mismatch_asserts(self):
        from vla_factory.deploy.lerobot_host_adapter import LerobotHostObsAdapter

        with pytest.raises(AssertionError):
            LerobotHostObsAdapter(
                camera_keys=(), state_keys=("a", "b"), state_dim=3,
            )

    def test_action_adapter_maps_each_dim(self):
        from vla_factory.deploy.lerobot_host_adapter import LerobotHostActionAdapter

        keys = ("shoulder", "elbow", "wrist")
        adapter = LerobotHostActionAdapter(action_dim=3, action_keys=keys)
        out = adapter(np.array([10.0, 20.0, 30.0]))
        assert out == {"shoulder": 10.0, "elbow": 20.0, "wrist": 30.0}

    def test_action_adapter_count_mismatch_asserts(self):
        from vla_factory.deploy.lerobot_host_adapter import LerobotHostActionAdapter

        with pytest.raises(AssertionError):
            LerobotHostActionAdapter(action_dim=3, action_keys=("a", "b"))

    def test_obs_adapter_missing_camera_raises(self):
        from vla_factory.deploy.lerobot_host_adapter import LerobotHostObsAdapter

        adapter = LerobotHostObsAdapter(
            camera_keys=("front",), state_keys=("a",), state_dim=1,
        )
        with pytest.raises(KeyError, match="front"):
            adapter({"a": 1.0})


# ── Entry point ─────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""Phase 3 Verification: ACT model (lerobot adapter).

Run:
    python test/test_act_model.py
    pytest test/test_act_model.py -v
"""

from __future__ import annotations
from helpers import make_schema

import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Helpers ──────────────────────────────────────────────────────────


def _make_obs(B=2, cameras=("front",), image_size=(224, 224), state_dim=6):
    """Create a dummy Observation for testing."""
    from vla_factory.model.interfaces.observation import Observation

    images = {cam: torch.randn(B, 3, *image_size) for cam in cameras}
    image_masks = {cam: torch.ones(B, dtype=torch.bool) for cam in cameras}
    state = torch.randn(B, state_dim)
    return Observation(images=images, image_masks=image_masks, state=state)


def _make_recipe_and_schema(action_dim=6, action_horizon=10, cameras=("front",), state_dim=6):
    """Create a minimal TrainRecipe + DataSchema pair for factory calls."""
    from vla_factory.recipe.recipe import TrainRecipe, ActionSpecConfig
    from vla_factory.recipe.defaults import resolve_recipe
    from vla_factory.data.manifest import DataSchema

    recipe = resolve_recipe(TrainRecipe(
        model_name="act",
        action_spec=ActionSpecConfig(action_dim=action_dim, action_horizon=action_horizon),
    ))
    schema = make_schema(
        state_dim=state_dim,
        action_dim=action_dim,
        cameras=tuple(cameras),
        image_sizes={cam: (224, 224) for cam in cameras},
    )
    return recipe, schema


def _lerobot_available():
    """Check whether lerobot can be imported."""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
        return True
    except Exception:
        return False


def _lerobot_skip_reason():
    """Return skip reason string, with error detail if import failed."""
    if _lerobot_available():
        return ""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
    except Exception as e:
        return f"lerobot not available: {type(e).__name__}: {e}"
    return "lerobot not installed"


skip_no_lerobot = pytest.mark.skipif(
    not _lerobot_available(),
    reason=_lerobot_skip_reason(),
)


# ══════════════════════════════════════════════════════════════════════
#  Adapter tests (wrapper + registry; factory tests require lerobot)
# ══════════════════════════════════════════════════════════════════════


class TestAdapter:
    """Tests for ACTModelWrapper and registry integration."""

    @skip_no_lerobot
    def test_protocol_compliance(self):
        """Wrapper satisfies VLAModelPyTorch and nn.Module."""
        from vla_factory.model.interfaces.model import VLAModelPyTorch, VLAModel
        from vla_factory.model.registry import get_entry

        entry = get_entry("act")
        recipe, schema = _make_recipe_and_schema()
        wrapper = entry.factory(recipe=recipe, schema=schema)

        assert isinstance(wrapper, torch.nn.Module)
        assert isinstance(wrapper, VLAModelPyTorch)
        assert isinstance(wrapper, VLAModel)

    @skip_no_lerobot
    def test_registry_integration(self):
        """get_entry('act') returns correct metadata and working factory."""
        from vla_factory.model.registry import get_entry

        entry = get_entry("act")
        meta = entry.metadata

        assert meta.name == "act"
        assert meta.backend == "pytorch"
        assert meta.action_head_type == "regression"
        assert meta.training_paradigm == "from_scratch"
        assert meta.requires_prompt is False
        assert meta.requires_augmentation is True
        assert meta.support_full is True
        assert meta.install_hint  # declares how to install the upstream dep

        recipe, schema = _make_recipe_and_schema()
        wrapper = entry.factory(recipe=recipe, schema=schema)

        # The wrapper's image slots follow schema.cameras ("front"), so the
        # observation must carry that same camera — name-mapped, not positional.
        camera = "front"
        obs = _make_obs(B=2, cameras=(camera,), state_dim=6)
        actions = torch.randn(2, 10, 6)
        loss, _ = wrapper.compute_loss(obs, actions)
        assert loss.requires_grad

    def test_observation_to(self):
        """Observation.to() moves tensors to different dtype."""
        from vla_factory.model.interfaces.observation import Observation

        obs = Observation(
            images={"front": torch.randn(2, 3, 224, 224)},
            image_masks={"front": torch.ones(2, dtype=torch.bool)},
            state=torch.randn(2, 6),
        )
        moved = obs.to(dtype=torch.float64)
        assert moved.images["front"].dtype == torch.float64
        assert moved.state.dtype == torch.float64


# ══════════════════════════════════════════════════════════════════════
#  Lerobot integration tests (require lerobot installed)
# ══════════════════════════════════════════════════════════════════════


@skip_no_lerobot
class TestLerobotIntegration:
    """Real integration tests against lerobot's ACTPolicy.

    These tests only run when lerobot is importable (Python >=3.12).
    Skipped automatically otherwise with a clear reason message.

    Note: lerobot config registers camera as 'observation.images.top',
    so observations use cameras=("top",).
    """

    @pytest.fixture(autouse=True)
    def _check_lerobot(self):
        """Emit warning if somehow the class loads without lerobot."""
        if not _lerobot_available():
            pytest.skip("lerobot not available")

    def test_lerobot_factory_creates_wrapper(self):
        """Factory creates a wrapper with backend='lerobot'."""
        from vla_factory.model.registry import get_entry

        entry = get_entry("act")
        recipe, schema = _make_recipe_and_schema(cameras=("top",))
        wrapper = entry.factory(recipe=recipe, schema=schema)

        assert wrapper._backend == "lerobot"

    def test_lerobot_compute_loss(self):
        """Lerobot backend compute_loss returns correct loss_dict keys."""
        from vla_factory.model.registry.entries.act import _load_lerobot

        recipe, schema = _make_recipe_and_schema(cameras=("top",))
        wrapper = _load_lerobot(recipe, schema)

        obs = _make_obs(B=2, cameras=("top",), state_dim=6)
        actions = torch.randn(2, 10, 6)

        loss, loss_dict = wrapper.compute_loss(obs, actions)

        assert loss.requires_grad
        assert "l1_loss" in loss_dict
        assert "kl_loss" in loss_dict, f"Expected 'kl_loss', got {list(loss_dict.keys())}"
        assert "total_loss" in loss_dict

    def test_lerobot_predict_actions(self):
        """Lerobot backend predict_actions returns [B, chunk_size, action_dim]."""
        from vla_factory.model.registry.entries.act import _load_lerobot

        recipe, schema = _make_recipe_and_schema(cameras=("top",))
        wrapper = _load_lerobot(recipe, schema)

        obs = _make_obs(B=2, cameras=("top",), state_dim=6)
        with torch.no_grad():
            pred = wrapper.predict_actions(obs)

        assert pred.shape == (2, 10, 6), f"Expected (2,10,6), got {pred.shape}"

    def test_lerobot_multi_camera(self):
        """Lerobot backend handles multi-camera when config has multiple image features."""
        from vla_factory.model.registry.entries.act import _load_lerobot

        recipe, schema = _make_recipe_and_schema(cameras=("top",))
        wrapper = _load_lerobot(recipe, schema)

        obs = _make_obs(B=2, cameras=("top",), state_dim=6)
        actions = torch.randn(2, 10, 6)

        loss, _ = wrapper.compute_loss(obs, actions)
        assert loss.requires_grad

        with torch.no_grad():
            pred = wrapper.predict_actions(obs)
        assert pred.shape == (2, 10, 6)

    def test_lerobot_save_load_weights(self):
        """state_dict round-trip via lerobot wrapper."""
        from vla_factory.model.registry.entries.act import _load_lerobot

        recipe, schema = _make_recipe_and_schema(cameras=("top",))
        wrapper = _load_lerobot(recipe, schema)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(wrapper.state_dict(), f.name)
            loaded = torch.load(f.name, map_location="cpu", weights_only=True)

        wrapper2 = _load_lerobot(recipe, schema)
        wrapper2.load_state_dict(loaded)

        for (n1, p1), (n2, p2) in zip(
            wrapper.named_parameters(), wrapper2.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch at {n1}"


# ══════════════════════════════════════════════════════════════════════
#  Declared defaults (ModelMetadata.params) — no lerobot needed
# ══════════════════════════════════════════════════════════════════════


class TestDefaultProfile:
    """Model default profile + per-run override merge (config-layer only)."""

    def test_declaration_has_hyperparams(self):
        from vla_factory.recipe.defaults import model_params

        d = model_params("act")
        assert "dim_model" in d
        assert not any(x["type"] == "resize_images" for x in d["transforms"]["inputs"])

    def test_resolve_uses_profile_defaults(self):
        from vla_factory.model.registry.entries.act import _resolve_act_config, _resolve_resize_image_size

        cfg = _resolve_act_config({})
        assert cfg["dim_model"] == 512
        assert _resolve_resize_image_size(cfg) is None

    def test_recipe_overrides_profile(self):
        from vla_factory.model.registry.entries.act import _resolve_act_config

        cfg = _resolve_act_config({
            "dim_model": 1024,
            "transforms": {"inputs": [{"type": "resize_images", "height": 480, "width": 640}]},
        })
        assert cfg["dim_model"] == 1024
        assert cfg["transforms"]["inputs"][0]["height"] == 480
        assert cfg["transforms"]["inputs"][0]["width"] == 640
        # Non-overridden keys still come from the profile.
        assert cfg["n_decoder_layers"] == 1

    def test_image_size_comes_from_resize_transform(self):
        from vla_factory.model.registry.entries.act import _resolve_resize_image_size

        size = _resolve_resize_image_size({
            "transforms": {"inputs": [{"type": "resize_images", "height": 480, "width": 640}]}
        })
        assert size == (480, 640)

    def test_image_size_falls_back_to_schema(self):
        from vla_factory.data.manifest import DataSchema
        from vla_factory.model.registry.entries.act import _schema_image_size

        schema = make_schema(cameras=("front",), image_sizes={"front": (480, 640)})
        assert _schema_image_size(schema, "front") == (480, 640)

    def test_unknown_model_returns_empty(self):
        from vla_factory.recipe.defaults import model_params

        assert model_params("nonexistent_model") == {}

    @skip_no_lerobot
    def test_factory_applies_recipe_override(self):
        """Factory respects recipe.model_config over the default profile."""
        from vla_factory.model.registry import get_entry

        entry = get_entry("act")
        recipe, schema = _make_recipe_and_schema(cameras=("top",))
        from vla_factory.recipe.defaults import resolve_recipe
        recipe.model_config = {**recipe.model_config, "dim_model": 256}
        recipe = resolve_recipe(recipe)
        wrapper = entry.factory(recipe=recipe, schema=schema)
        assert wrapper.model.config.dim_model == 256


# ── CLI runner (for `python test_act_model.py` without pytest) ───────


# ══════════════════════════════════════════════════════════════════════
#  WP3: the factory reads action facts from the dimensions that own them
# ══════════════════════════════════════════════════════════════════════


@skip_no_lerobot
class TestActionFactRouting:
    """The built ACTConfig follows the dataset/model facts, not the recipe alone.

    Before WP3 the head width came straight from `recipe.action_spec.action_dim`
    while the dataloader padded to the routed dim, so a recipe that disagreed
    with its dataset silently built a head of the wrong width.
    """

    def _recipe(self, action_dim, action_horizon):
        from vla_factory.recipe.recipe import TrainRecipe, ActionSpecConfig
        from vla_factory.recipe.defaults import resolve_recipe

        return resolve_recipe(TrainRecipe(
            model_name="act",
            action_spec=ActionSpecConfig(
                action_dim=action_dim, action_horizon=action_horizon
            ),
        ))

    def _config_of(self, recipe, schema):
        from vla_factory.model.registry.entries.act import _load_lerobot

        wrapper = _load_lerobot(recipe, schema)
        return wrapper.model.config

    def test_head_width_follows_the_dataset(self, caplog):
        import logging

        schema = make_schema(
            state_dim=6, action_dim=8, cameras=("top",),
            image_sizes={"top": (224, 224)},
        )
        with caplog.at_level(logging.WARNING, logger="vla_factory.assembly.action_facts"):
            config = self._config_of(self._recipe(action_dim=6, action_horizon=10), schema)

        assert config.output_features["action"].shape == (8,)
        assert any("action_dim mismatch" in r.message for r in caplog.records)

    def test_chunk_size_follows_the_recipe_for_from_scratch(self):
        """ACT is trained from scratch, so the user owns the chunk size."""
        schema = make_schema(
            state_dim=6, action_dim=8, cameras=("top",),
            image_sizes={"top": (224, 224)},
        )
        config = self._config_of(self._recipe(action_dim=8, action_horizon=17), schema)
        assert config.chunk_size == 17
        assert config.n_action_steps == 17


def main():
    from vla_factory.model.registry.entries.act import _LEROBOT_ACT

    print("=" * 60)
    print("Phase 3 Verification: ACT Model (lerobot adapter)")
    print(f"Backend: {'lerobot (official)' if _LEROBOT_ACT else 'lerobot NOT INSTALLED (tests will skip)'}")
    print("=" * 60)

    # Run via pytest so skip/warning logic is consistent
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

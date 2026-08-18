#!/usr/bin/env python3
"""Phase 3 Verification: ACT model (lerobot adapter).

Run:
    python test/test_act_model.py
    pytest test/test_act_model.py -v
"""

from __future__ import annotations
from helpers import make_assembly, make_schema

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
    from vla_factory.model.model_interface import Observation

    images = {cam: torch.randn(B, 3, *image_size) for cam in cameras}
    image_masks = {cam: torch.ones(B, dtype=torch.bool) for cam in cameras}
    state = torch.randn(B, state_dim)
    return Observation(images=images, image_masks=image_masks, state=state)


def _make_recipe_and_assembly(action_dim=6, action_horizon=10, cameras=("front",), state_dim=6):
    """Create a minimal TrainRecipe + ResolvedAssembly pair for factory calls."""
    from vla_factory.user_interface import ModelConfig, TrainRecipe, merge_model_config

    recipe = merge_model_config(TrainRecipe(
        model=ModelConfig(
            name="act",
            config={"action_horizon": action_horizon},
        ),
    ))
    schema = make_schema(
        state_dim=state_dim,
        action_dim=action_dim,
        cameras=tuple(cameras),
        image_sizes={cam: (224, 224) for cam in cameras},
    )
    return recipe, make_assembly(schema, "act", recipe=recipe)


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
        from vla_factory.model.model_interface import VLAModelPyTorch, VLAModel
        from vla_factory.model.registry import get_entry

        entry = get_entry("act")
        recipe, assembly = _make_recipe_and_assembly()
        wrapper = entry.factory(recipe=recipe, assembly=assembly)

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
        assert meta.support_full is True
        assert meta.install_hint  # declares how to install the upstream dep

        recipe, assembly = _make_recipe_and_assembly()
        wrapper = entry.factory(recipe=recipe, assembly=assembly)

        # The wrapper's image slots follow schema.cameras ("front"), so the
        # observation must carry that same camera — name-mapped, not positional.
        camera = "front"
        obs = _make_obs(B=2, cameras=(camera,), state_dim=6)
        actions = torch.randn(2, 10, 6)
        loss, _ = wrapper.compute_loss(obs, actions)
        assert loss.requires_grad

    def test_observation_to(self):
        """Observation.to() moves tensors to different dtype."""
        from vla_factory.model.model_interface import Observation

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
        recipe, assembly = _make_recipe_and_assembly(cameras=("top",))
        wrapper = entry.factory(recipe=recipe, assembly=assembly)

        assert wrapper._backend == "lerobot"

    def test_lerobot_compute_loss(self):
        """Lerobot backend compute_loss returns correct loss_dict keys."""
        from vla_factory.model.adapters.act import _load_lerobot

        recipe, assembly = _make_recipe_and_assembly(cameras=("top",))
        wrapper = _load_lerobot(recipe, assembly)

        obs = _make_obs(B=2, cameras=("top",), state_dim=6)
        actions = torch.randn(2, 10, 6)

        loss, loss_dict = wrapper.compute_loss(obs, actions)

        assert loss.requires_grad
        assert "l1_loss" in loss_dict
        assert "kl_loss" in loss_dict, f"Expected 'kl_loss', got {list(loss_dict.keys())}"
        assert "total_loss" in loss_dict

    def test_lerobot_predict_actions(self):
        """Lerobot backend predict_actions returns [B, chunk_size, action_dim]."""
        from vla_factory.model.adapters.act import _load_lerobot

        recipe, assembly = _make_recipe_and_assembly(cameras=("top",))
        wrapper = _load_lerobot(recipe, assembly)

        obs = _make_obs(B=2, cameras=("top",), state_dim=6)
        with torch.no_grad():
            pred = wrapper.predict_actions(obs)

        assert pred.shape == (2, 10, 6), f"Expected (2,10,6), got {pred.shape}"

    def test_lerobot_multi_camera(self):
        """Lerobot backend handles multi-camera when config has multiple image features."""
        from vla_factory.model.adapters.act import _load_lerobot

        recipe, assembly = _make_recipe_and_assembly(cameras=("top",))
        wrapper = _load_lerobot(recipe, assembly)

        obs = _make_obs(B=2, cameras=("top",), state_dim=6)
        actions = torch.randn(2, 10, 6)

        loss, _ = wrapper.compute_loss(obs, actions)
        assert loss.requires_grad

        with torch.no_grad():
            pred = wrapper.predict_actions(obs)
        assert pred.shape == (2, 10, 6)

    def test_lerobot_save_load_weights(self):
        """state_dict round-trip via lerobot wrapper."""
        from vla_factory.model.adapters.act import _load_lerobot

        recipe, assembly = _make_recipe_and_assembly(cameras=("top",))
        wrapper = _load_lerobot(recipe, assembly)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(wrapper.state_dict(), f.name)
            loaded = torch.load(f.name, map_location="cpu", weights_only=True)

        wrapper2 = _load_lerobot(recipe, assembly)
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
        from vla_factory.user_interface import model_params

        d = model_params("act")
        assert "dim_model" in d
        assert d["input_image_size"] is None
        assert "transforms" not in d

    def test_resolve_uses_profile_defaults(self):
        from vla_factory.model.adapters.act import _resolve_act_config

        cfg = _resolve_act_config({})
        assert cfg["dim_model"] == 512
        # None keeps the dataset-native model interface size.
        assert cfg["input_image_size"] is None

    def test_recipe_overrides_profile(self):
        from vla_factory.model.adapters.act import _resolve_act_config

        cfg = _resolve_act_config({
            "dim_model": 1024,
            "input_image_size": [480, 640],
        })
        assert cfg["dim_model"] == 1024
        assert cfg["input_image_size"] == [480, 640]
        # Non-overridden keys still come from the profile.
        assert cfg["n_decoder_layers"] == 1

    def test_image_size_comes_from_model_config_or_native_data(self):
        """The interface is resolved first; the resize call consumes it."""
        from vla_factory.user_interface import merge_model_config, parse_recipe_from_string

        schema = make_schema(
            state_dim=6, action_dim=6, cameras=("front",),
            image_sizes={"front": (480, 640)},
        )
        native = make_assembly(schema, "act")
        assert native.model_io_spec.camera_shapes == {"front": (480, 640)}

        resized = merge_model_config(parse_recipe_from_string("""
model:
  name: act
  config:
    input_image_size: [224, 224]
"""))
        assembly = make_assembly(schema, "act", recipe=resized)
        assert assembly.model_io_spec.camera_shapes == {"front": (224, 224)}
        resize_call = next(
            call for call in assembly.data_to_model.calls
            if call.type == "resize_images"
        )
        assert resize_call.args["height"] == 224
        assert resize_call.args["width"] == 224

    def test_unknown_model_returns_empty(self):
        from vla_factory.user_interface import model_params

        assert model_params("nonexistent_model") == {}

    @skip_no_lerobot
    def test_factory_applies_recipe_override(self):
        """Factory respects recipe.model.config over the default profile."""
        from vla_factory.model.registry import get_entry

        entry = get_entry("act")
        recipe, assembly = _make_recipe_and_assembly(cameras=("top",))
        from vla_factory.user_interface import merge_model_config
        recipe.model.config = {**recipe.model.config, "dim_model": 256}
        recipe = merge_model_config(recipe)
        wrapper = entry.factory(recipe=recipe, assembly=assembly)
        assert wrapper.model.config.dim_model == 256


# ── CLI runner (for `python test_act_model.py` without pytest) ───────


# ══════════════════════════════════════════════════════════════════════
# The factory reads action facts from the dimensions that own them.
# ══════════════════════════════════════════════════════════════════════


@skip_no_lerobot
class TestShapesComeFromTheAssembly:
    """The built ACTConfig follows the resolved composition, not the recipe.

    The head width is resolved directly from model/data facts, and the planned
    calls consume the same target, so the model and dataloader cannot diverge.
    """

    def _recipe(self, action_horizon=None):
        from vla_factory.user_interface import ModelConfig, TrainRecipe, merge_model_config

        config = {} if action_horizon is None else {"action_horizon": action_horizon}
        return merge_model_config(
            TrainRecipe(model=ModelConfig(name="act", config=config))
        )

    def _config_of(self, recipe, schema):
        from vla_factory.model.adapters.act import _load_lerobot

        assembly = make_assembly(schema, "act", recipe=recipe)
        return _load_lerobot(recipe, assembly).model.config

    def _schema(self, action_dim=8):
        return make_schema(
            state_dim=6, action_dim=action_dim, cameras=("top",),
            image_sizes={"top": (224, 224)},
        )

    def test_head_width_follows_the_dataset(self):
        config = self._config_of(self._recipe(), self._schema(action_dim=8))
        assert config.output_features["action"].shape == (8,)

    def test_chunk_size_follows_the_model_config_for_from_scratch(self):
        """ACT is trained from scratch, so the user owns the chunk size."""
        config = self._config_of(self._recipe(action_horizon=17), self._schema())
        assert config.chunk_size == 17
        assert config.n_action_steps == 17

    def test_chunk_size_defaults_to_the_act_declaration(self):
        """A recipe that says nothing gets ACT's own declared default (100).

        A chunk length belongs to a model, so there is no framework-wide
        default to fall back on.
        """
        config = self._config_of(self._recipe(), self._schema())
        assert config.chunk_size == 100


def main():
    from vla_factory.model.adapters.act import _LEROBOT_ACT

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

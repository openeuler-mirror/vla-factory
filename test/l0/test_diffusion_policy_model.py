#!/usr/bin/env python3
"""Diffusion Policy adapter, declared facts, and factory integration.

Mirrors ``test_act_model.py``: config-layer tests run everywhere, upstream
tests are guarded by an import check so the default suite stays green
without the ``[diffusion_policy]`` extra.

The shapes under test carry the D8 time-axis contract: this family declares
``history_frames=2``, so every observation tensor is one rank taller than
ACT's — images ``(B, To, 3, H, W)``, state ``(B, To, D)`` — while actions
stay window-level ``(B, horizon, Da)``.
"""

from __future__ import annotations

from helpers import make_assembly, make_schema

import importlib.util
import tempfile

import pytest
import torch


# ── Helpers ──────────────────────────────────────────────────────────


def _upstream_available() -> bool:
    """Check whether the real-stanford upstream is importable."""
    return (
        importlib.util.find_spec("diffusion_policy") is not None
        and importlib.util.find_spec("robomimic") is not None
        and importlib.util.find_spec("diffusers") is not None
    )


skip_no_upstream = pytest.mark.skipif(
    not _upstream_available(),
    reason="diffusion_policy upstream not installed "
           "(bash scripts/install.sh --model diffusion_policy)",
)

STATE_DIM = 6
ACTION_DIM = 6
CAMERAS = ("front",)
IMAGE_SIZE = (96, 96)


def _make_recipe_and_assembly(
    *, action_horizon=16, cameras=CAMERAS, image_size=IMAGE_SIZE,
):
    from vla_factory.user_interface import ModelConfig, TrainRecipe, merge_model_config

    recipe = merge_model_config(TrainRecipe(
        model=ModelConfig(
            name="diffusion_policy",
            config={"action_horizon": action_horizon},
        ),
    ))
    schema = make_schema(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        cameras=tuple(cameras),
        image_sizes={cam: image_size for cam in cameras},
    )
    return recipe, make_assembly(schema, "diffusion_policy", recipe=recipe)


def _make_obs(*, batch=2, n_obs_steps=2, cameras=CAMERAS,
              image_size=IMAGE_SIZE, state_dim=STATE_DIM):
    """Observation with the D8 time axis (this family is To=2)."""
    from vla_factory.model.model_interface import Observation

    images = {
        cam: torch.rand(batch, n_obs_steps, 3, *image_size) for cam in cameras
    }
    image_masks = {
        cam: torch.ones(batch, n_obs_steps, dtype=torch.bool) for cam in cameras
    }
    state = torch.randn(batch, n_obs_steps, state_dim)
    return Observation(images=images, image_masks=image_masks, state=state)


# ══════════════════════════════════════════════════════════════════════
#  Declared facts and resolution (no upstream needed)
# ══════════════════════════════════════════════════════════════════════


class TestDeclaration:
    """ModelMetadata facts + the composition they produce."""

    def test_declared_facts(self):
        from vla_factory.model.registry import list_entries

        metadata = list_entries()["diffusion_policy"]
        assert metadata.action_head_type == "diffusion"
        assert metadata.training_paradigm == "from_scratch"
        assert metadata.requires_prompt is False
        assert metadata.support_lora is False
        assert metadata.history_frames == 2
        assert metadata.vector_normalization == "min_max"
        assert metadata.image_input_range == (0.0, 1.0)
        assert metadata.image_normalize_mode is None
        assert metadata.image_layout == "CHW"
        assert metadata.install_hint

    def test_history_frames_reaches_the_io_spec(self):
        """The four-layer n_obs_steps plumbing starts here: declaration →
        ModelIOSpec, the single translation point."""
        recipe, assembly = _make_recipe_and_assembly()
        assert assembly.model_io_spec.n_obs_steps == 2

    def test_resolution_plans_min_max_normalization(self):
        """vector_normalization=min_max yields the min_max pipeline with the
        declared eps, and the planned inverse matches the forward method."""
        recipe, assembly = _make_recipe_and_assembly()
        normalize_call = next(
            call for call in assembly.data_to_model.calls
            if call.type == "normalize_vector"
        )
        assert normalize_call.args["method"] == "min_max"
        assert normalize_call.args["eps"] == 1e-4

        inverse = next(
            call for call in assembly.model_to_robot.calls
            if call.type.startswith("unnormalize_action")
        )
        assert inverse.type == "unnormalize_action_min_max"

    def test_resolution_plans_raw_images(self):
        """[0,1] float CHW with no ImageNet step — the upstream contract."""
        recipe, assembly = _make_recipe_and_assembly()
        types = [call.type for call in assembly.data_to_model.calls]
        assert "image_to_float" in types
        assert "image_layout" in types
        assert "image_normalize" not in types

    def test_image_size_tunable(self):
        """input_image_size overrides the native resolution via the resolver,
        and the resize call consumes the same target (ACT precedent)."""
        from vla_factory.user_interface import ModelConfig, TrainRecipe, merge_model_config

        schema = make_schema(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, cameras=CAMERAS,
            image_sizes={"front": (480, 640)},
        )
        native = make_assembly(schema, "diffusion_policy")
        assert native.model_io_spec.camera_shapes == {"front": (480, 640)}

        recipe = merge_model_config(TrainRecipe(model=ModelConfig(
            name="diffusion_policy",
            config={"input_image_size": [96, 96]},
        )))
        assembly = make_assembly(schema, "diffusion_policy", recipe=recipe)
        assert assembly.model_io_spec.camera_shapes == {"front": (96, 96)}

    def test_config_defaults_and_override(self):
        from vla_factory.model.adapters.diffusion_policy import _resolve_config

        cfg = _resolve_config({})
        assert cfg["action_horizon"] == 16
        assert cfg["num_train_timesteps"] == 100
        assert cfg["crop_shape"] is None

        cfg = _resolve_config({"down_dims": [128, 256]})
        assert cfg["down_dims"] == [128, 256]
        assert cfg["kernel_size"] == 5


# ══════════════════════════════════════════════════════════════════════
#  Upstream integration (require the diffusion_policy extra)
# ══════════════════════════════════════════════════════════════════════


@skip_no_upstream
class TestUpstreamIntegration:
    """Real integration against DiffusionUnetHybridImagePolicy."""

    def test_factory_creates_wrapper(self):
        from vla_factory.model.registry import get_entry
        from vla_factory.model.model_interface import VLAModelPyTorch

        recipe, assembly = _make_recipe_and_assembly()
        wrapper = get_entry("diffusion_policy").factory(
            recipe=recipe, assembly=assembly,
        )

        assert isinstance(wrapper, torch.nn.Module)
        assert isinstance(wrapper, VLAModelPyTorch)
        assert wrapper._backend == "diffusion_policy"
        # Identity normalizer installed (D1): upstream normalize is a
        # pass-through, statistics live in the assembly.
        normalized = wrapper.policy.normalizer.normalize(
            {"state": torch.tensor([[1.0, -2.0, 3.0, 0.5, 0.0, 9.0]])}
        )
        assert torch.equal(normalized["state"], torch.tensor([[1.0, -2.0, 3.0, 0.5, 0.0, 9.0]]))

    def test_compute_loss_returns_trainer_tuple(self):
        from vla_factory.model.registry import get_entry

        recipe, assembly = _make_recipe_and_assembly(action_horizon=8)
        wrapper = get_entry("diffusion_policy").factory(
            recipe=recipe, assembly=assembly,
        )

        obs = _make_obs()
        actions = torch.randn(2, 8, ACTION_DIM)
        loss, loss_dict = wrapper.compute_loss(obs, actions)

        assert loss.ndim == 0 and loss.requires_grad
        assert "diffusion_loss" in loss_dict
        loss.backward()

    def test_predict_actions_returns_full_horizon(self):
        """The wrapper surfaces the whole trajectory; receding-horizon
        slicing belongs to the execution layer (decision D3)."""
        from vla_factory.model.registry import get_entry

        recipe, assembly = _make_recipe_and_assembly(action_horizon=8)
        wrapper = get_entry("diffusion_policy").factory(
            recipe=recipe, assembly=assembly,
        )
        wrapper.eval()

        with torch.no_grad():
            pred = wrapper.predict_actions(_make_obs(), num_steps=4)
        assert pred.shape == (2, 8, ACTION_DIM)

    def test_stateless_dataset_fails_at_the_factory(self):
        """The hybrid variant needs proprioception: a stateless composition
        is rejected at construction, not deep inside the encoder."""
        from vla_factory.model.registry import get_entry
        from vla_factory.user_interface import ModelConfig, TrainRecipe, merge_model_config

        recipe = merge_model_config(TrainRecipe(model=ModelConfig(
            name="diffusion_policy", config={"action_horizon": 8},
        )))
        schema = make_schema(
            state_dim=0, action_dim=ACTION_DIM, cameras=CAMERAS,
            image_sizes={"front": IMAGE_SIZE},
        )
        with pytest.raises(ValueError, match="state_dim=0"):
            get_entry("diffusion_policy").factory(
                recipe=recipe, assembly=make_assembly(schema, "diffusion_policy", recipe=recipe),
            )

    def test_recipe_checkpoint_path_round_trip(self):
        """model.path loads a previous run's wrapper checkpoint exactly."""
        from vla_factory.user_interface import ModelConfig, TrainRecipe, merge_model_config
        from vla_factory.model.adapters.diffusion_policy import _load_upstream

        recipe, assembly = _make_recipe_and_assembly(action_horizon=8)
        wrapper = _load_upstream(recipe, assembly)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(wrapper.state_dict(), f.name)
            checkpoint = f.name

        resume = merge_model_config(TrainRecipe(model=ModelConfig(
            name="diffusion_policy",
            config={"action_horizon": 8},
            path=checkpoint,
        )))
        reloaded = _load_upstream(
            resume, make_assembly(
                make_schema(
                    state_dim=STATE_DIM, action_dim=ACTION_DIM,
                    cameras=CAMERAS, image_sizes={"front": IMAGE_SIZE},
                ),
                "diffusion_policy", recipe=resume,
            ),
        )
        for (n1, p1), (n2, p2) in zip(
            wrapper.named_parameters(), reloaded.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Mismatch at {n1}"

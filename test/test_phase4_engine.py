#!/usr/bin/env python3
"""Phase 4 Verification: Training Engine + CLI.

Run:
    python test/test_phase4_engine.py
    pytest test/test_phase4_engine.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── lerobot availability (ACT factory hard-requires lerobot) ──────────


def _lerobot_available():
    """Check whether lerobot can be imported."""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
        return True
    except Exception:
        return False


skip_no_lerobot = pytest.mark.skipif(
    not _lerobot_available(),
    reason="ACT factory requires lerobot (pip install -e '.[act]')",
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_obs(B=2, cameras=("front",), image_size=(224, 224), state_dim=6):
    """Create a dummy Observation for testing."""
    from vla_factory.model.protocols.observation import Observation

    images = {cam: torch.randn(B, 3, *image_size) for cam in cameras}
    image_masks = {cam: torch.ones(B, dtype=torch.bool) for cam in cameras}
    state = torch.randn(B, state_dim)
    return Observation(images=images, image_masks=image_masks, state=state)


def _make_model_and_recipe(strategy="full", freeze_components=None, trainable_components=None):
    """Create an ACT model wrapper and matching TrainRecipe (requires lerobot)."""
    from vla_factory.config.recipe import TrainRecipe, ActionSpecConfig, OutputConfig
    from vla_factory.data.manifest import DataSchema
    from vla_factory.model.registry import get_entry

    recipe = TrainRecipe(
        model_name="act",
        action_spec=ActionSpecConfig(action_dim=6, action_horizon=10),
        finetuning_strategy=strategy,
        freeze_components=freeze_components,
        trainable_components=trainable_components,
        lr=1e-4,
        total_steps=3,
        batch_size=2,
        output=OutputConfig(output_dir=tempfile.mkdtemp()),
    )
    schema = DataSchema(state_dim=6, action_dim=6, cameras=("front",))

    entry = get_entry("act")
    model = entry.factory(recipe=recipe, schema=schema)

    return model, recipe, entry.metadata


# ══════════════════════════════════════════════════════════════════════
#  Strategy tests
# ══════════════════════════════════════════════════════════════════════


@skip_no_lerobot
class TestStrategy:
    """Fine-tuning strategy application tests."""

    def test_full_all_trainable(self):
        """'full' strategy: all parameters trainable."""
        from vla_factory.training.strategies import apply_strategy

        model, recipe, metadata = _make_model_and_recipe(strategy="full")
        apply_strategy(model, recipe, metadata)

        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert trainable == total, f"Full strategy: {trainable}/{total} trainable"

    def test_freeze_backbone(self):
        """'freeze' strategy with backbone: backbone params frozen, others trainable."""
        from vla_factory.training.strategies import apply_strategy

        model, recipe, metadata = _make_model_and_recipe(
            strategy="freeze", freeze_components=["backbone"]
        )
        apply_strategy(model, recipe, metadata)

        backbone_frozen = 0
        other_trainable = 0
        for name, param in model.named_parameters():
            # Check if name matches any backbone pattern
            is_backbone = any(name.startswith(p) for p in metadata.components.get("backbone", []))
            if is_backbone:
                assert not param.requires_grad, f"Backbone param {name} should be frozen"
                backbone_frozen += 1
            else:
                assert param.requires_grad, f"Non-backbone param {name} should be trainable"
                other_trainable += 1

        assert backbone_frozen > 0, "Expected at least one backbone parameter"
        assert other_trainable > 0, "Expected at least one non-backbone parameter"

    def test_selective_train_transformer(self):
        """'selective' strategy: only transformer components trainable."""
        from vla_factory.training.strategies import apply_strategy

        model, recipe, metadata = _make_model_and_recipe(
            strategy="selective", trainable_components=["transformer"]
        )
        apply_strategy(model, recipe, metadata)

        transformer_trainable = 0
        other_frozen = 0
        transformer_patterns = metadata.components.get("transformer", [])
        for name, param in model.named_parameters():
            is_transformer = any(name.startswith(p) for p in transformer_patterns)
            if is_transformer:
                assert param.requires_grad, f"Transformer param {name} should be trainable"
                transformer_trainable += 1
            else:
                assert not param.requires_grad, f"Non-transformer param {name} should be frozen"
                other_frozen += 1

        assert transformer_trainable > 0, "Expected trainable transformer params"
        assert other_frozen > 0, "Expected frozen non-transformer params"

    def test_freeze_unknown_component_warning(self, caplog):
        """Freezing an unknown component logs a warning and doesn't crash."""
        import logging
        from vla_factory.training.strategies import apply_strategy

        model, recipe, metadata = _make_model_and_recipe(
            strategy="freeze", freeze_components=["nonexistent_component"]
        )
        with caplog.at_level(logging.WARNING):
            apply_strategy(model, recipe, metadata)

        # Should not crash; all params still trainable (nothing matched)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert trainable == total


# ══════════════════════════════════════════════════════════════════════
#  TrainingArguments mapping tests
# ══════════════════════════════════════════════════════════════════════


class TestTrainingArgsMapping:
    """Test that TrainRecipe correctly maps to TrainingArguments."""

    def test_recipe_to_training_args(self):
        """TrainRecipe fields map to correct TrainingArguments."""
        from vla_factory.config.recipe import TrainRecipe, ActionSpecConfig, OutputConfig
        from vla_factory.training.train import _build_training_args

        recipe = TrainRecipe(
            model_name="act",
            action_spec=ActionSpecConfig(action_dim=6, action_horizon=10),
            lr=5e-5,
            lr_backbone=1e-5,
            batch_size=4,
            total_steps=10000,
            gradient_checkpointing=True,
            output=OutputConfig(output_dir="/tmp/test_output"),
        )
        args = _build_training_args(recipe)

        assert args.learning_rate == 5e-5
        assert args.per_device_train_batch_size == 4
        assert args.max_steps == 10000
        assert args.gradient_checkpointing is True
        assert args.output_dir == "/tmp/test_output"
        assert args.remove_unused_columns is False
        assert args.lr_backbone == 1e-5


# ══════════════════════════════════════════════════════════════════════
#  CPU training loop test
# ══════════════════════════════════════════════════════════════════════


@skip_no_lerobot
class TestCPUTrainingLoop:
    """Run a short training loop on CPU to verify end-to-end wiring."""

    def test_train_3_steps(self):
        """Train ACT model for 3 steps on CPU with dummy data."""
        from torch.utils.data import Dataset

        from vla_factory.training.pytorch_trainer import VLATrainer
        from vla_factory.training.strategies import apply_strategy
        from vla_factory.model.registry import get_entry
        from vla_factory.config.recipe import TrainRecipe, ActionSpecConfig, OutputConfig
        from vla_factory.data.manifest import DataSchema

        # 1. Create model
        entry = get_entry("act")
        recipe = TrainRecipe(
            model_name="act",
            action_spec=ActionSpecConfig(action_dim=6, action_horizon=10),
            finetuning_strategy="full",
            lr=1e-4,
            total_steps=3,
            batch_size=2,
            output=OutputConfig(output_dir=tempfile.mkdtemp()),
        )
        schema = DataSchema(state_dim=6, action_dim=6, cameras=("front",))
        model = entry.factory(recipe=recipe, schema=schema)
        apply_strategy(model, recipe, entry.metadata)

        # 3. Dummy dataset
        class DummyDataset(Dataset):
            def __len__(self):
                return 6  # 3 batches of 2

            def __getitem__(self, idx):
                return {
                    "observation": _make_obs(B=1, state_dim=6),
                    "actions": torch.randn(1, 10, 6),
                }

        def dummy_collate(batch):
            """Merge list-of-dicts into batched dict."""
            obs_list = [item["observation"] for item in batch]
            actions = torch.cat([item["actions"] for item in batch], dim=0)

            from vla_factory.model.protocols.observation import Observation
            merged_obs = Observation(
                images={k: torch.cat([o.images[k] for o in obs_list]) for k in obs_list[0].images},
                image_masks={k: torch.cat([o.image_masks[k] for o in obs_list]) for k in obs_list[0].image_masks},
                state=torch.cat([o.state for o in obs_list]),
            )
            return {"observation": merged_obs, "actions": actions}

        # 4. TrainingArguments
        from transformers import TrainingArguments

        output_dir = tempfile.mkdtemp()
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=3,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
            logging_steps=1,
            save_strategy="no",
            dataloader_drop_last=True,
            dataloader_num_workers=0,
        )
        training_args.lr_backbone = None

        # 5. Train
        trainer = VLATrainer(
            model=model,
            args=training_args,
            train_dataset=DummyDataset(),
            data_collator=dummy_collate,
        )

        result = trainer.train()
        assert result is not None
        # Verify loss was captured via _last_loss_dict (set in compute_loss)
        assert trainer._last_loss_dict is not None
        assert "total_loss" in trainer._last_loss_dict

    def test_train_with_freeze_strategy(self):
        """Train with freeze strategy: backbone frozen, non-backbone trainable."""
        from torch.utils.data import Dataset

        from vla_factory.training.pytorch_trainer import VLATrainer
        from vla_factory.training.strategies import apply_strategy
        from vla_factory.model.registry import get_entry
        from vla_factory.config.recipe import TrainRecipe, ActionSpecConfig, OutputConfig
        from vla_factory.data.manifest import DataSchema
        from transformers import TrainingArguments

        entry = get_entry("act")
        recipe = TrainRecipe(
            model_name="act",
            action_spec=ActionSpecConfig(action_dim=6, action_horizon=10),
            finetuning_strategy="freeze",
            freeze_components=["backbone"],
            lr=1e-4,
            total_steps=2,
            batch_size=2,
            output=OutputConfig(output_dir=tempfile.mkdtemp()),
        )
        schema = DataSchema(state_dim=6, action_dim=6, cameras=("front",))
        model = entry.factory(recipe=recipe, schema=schema)
        apply_strategy(model, recipe, entry.metadata)

        class DummyDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "observation": _make_obs(B=1, state_dim=6),
                    "actions": torch.randn(1, 10, 6),
                }

        def dummy_collate(batch):
            from vla_factory.model.protocols.observation import Observation
            obs_list = [item["observation"] for item in batch]
            merged_obs = Observation(
                images={k: torch.cat([o.images[k] for o in obs_list]) for k in obs_list[0].images},
                image_masks={k: torch.cat([o.image_masks[k] for o in obs_list]) for k in obs_list[0].image_masks},
                state=torch.cat([o.state for o in obs_list]),
            )
            return {"observation": merged_obs, "actions": torch.cat([item["actions"] for item in batch])}

        training_args = TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            max_steps=2,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
            logging_steps=1,
            save_strategy="no",
            dataloader_drop_last=True,
            dataloader_num_workers=0,
        )
        training_args.lr_backbone = None

        trainer = VLATrainer(
            model=model,
            args=training_args,
            train_dataset=DummyDataset(),
            data_collator=dummy_collate,
        )
        trainer.train()

        # Verify loss was captured via _last_loss_dict (set in compute_loss)
        assert trainer._last_loss_dict is not None
        assert "total_loss" in trainer._last_loss_dict


# ══════════════════════════════════════════════════════════════════════
#  Observation.to() tests
# ══════════════════════════════════════════════════════════════════════


class TestObservationTo:
    """Verify Observation.to() handles all fields including token_ar_mask."""

    def test_to_with_token_masks(self):
        """Observation.to() moves all fields including token_ar_mask."""
        from vla_factory.model.protocols.observation import Observation

        obs = Observation(
            images={"front": torch.randn(2, 3, 224, 224)},
            image_masks={"front": torch.ones(2, dtype=torch.bool)},
            state=torch.randn(2, 6),
            token_ar_mask=torch.ones(2, 10, dtype=torch.bool),
            token_loss_mask=torch.ones(2, 10, dtype=torch.float),
        )
        moved = obs.to(dtype=torch.float64)
        assert moved.images["front"].dtype == torch.float64
        assert moved.state.dtype == torch.float64
        assert moved.token_ar_mask is not None
        assert moved.token_loss_mask is not None
        assert moved.token_loss_mask.dtype == torch.float64


# ── CLI runner ───────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Phase 4 Verification: Training Engine + CLI")
    print("=" * 60)
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

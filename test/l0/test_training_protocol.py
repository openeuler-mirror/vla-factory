"""Coverage for the OpenPI-aligned training protocol controls."""

import json
from pathlib import Path

import pytest
import torch

from vla_factory.data.reader.robotwin import _load_instructions
from vla_factory.user_interface import (
    merge_model_config,
    parse_recipe,
    parse_recipe_from_string,
)


def test_training_protocol_fields_parse():
    recipe = parse_recipe_from_string(
        """
model: {name: pi0}
training:
  gradient_accumulation_steps: 8
  lr_scheduler_type: cosine
  warmup_steps: 1000
  max_grad_norm: 1.0
"""
    )
    training = recipe.training
    assert training.gradient_accumulation_steps == 8
    assert training.lr_scheduler_type == "cosine"
    assert training.warmup_steps == 1000
    assert training.max_grad_norm == pytest.approx(1.0)


def test_removed_optimizer_and_ema_fields_are_rejected():
    with pytest.raises(ValueError):
        parse_recipe_from_string(
            """
model: {name: pi0}
training:
  adam_beta2: 0.95
"""
        )
    with pytest.raises(ValueError):
        parse_recipe_from_string(
            """
model: {name: pi0}
training:
  ema_decay: 0.99
"""
        )
    with pytest.raises(ValueError):
        parse_recipe_from_string(
            """
model: {name: pi0}
training:
  min_lr_ratio: 0.1
"""
        )


def test_official_robotwin_lora_recipe_contract():
    recipe = merge_model_config(
        parse_recipe("examples/pi0_robotwin_dump_bin_bigbin_lora.yaml")
    )
    training = recipe.training
    assert recipe.finetuning.strategy == "lora"
    assert training.batch_size * training.gradient_accumulation_steps == 32
    assert training.lr_scheduler_type == "cosine"
    assert training.warmup_steps == 1000
    assert training.max_grad_norm == pytest.approx(1.0)
    assert recipe.output.save_steps == 5000


def test_official_robotwin_full_recipe_contract():
    recipe_path = Path("examples/pi0_robotwin_dump_bin_bigbin_full.yaml")
    if not recipe_path.exists():
        pytest.skip("official full RobotWin recipe is not present in this checkout")
    recipe = merge_model_config(
        parse_recipe(recipe_path)
    )
    training = recipe.training
    assert recipe.finetuning.strategy == "full"
    assert not recipe.finetuning.config
    assert training.batch_size * training.gradient_accumulation_steps == 32
    assert training.lr_scheduler_type == "cosine"
    assert training.warmup_steps == 1000


def test_robotwin_loads_all_seen_instructions(tmp_path):
    instructions = tmp_path / "instructions"
    instructions.mkdir()
    (instructions / "episode3.json").write_text(
        json.dumps({"seen": ["prompt one", "prompt two"], "unseen": ["ignored"]})
    )
    assert _load_instructions(tmp_path, 3) == ("prompt one", "prompt two")


def test_inference_boundary_resolves_first_instruction():
    """Multi-instruction episodes must never stringify the container."""
    from vla_factory.inference.inference_engine import resolve_inference_language

    assert resolve_inference_language("pick up the bin") == "pick up the bin"
    assert (
        resolve_inference_language(("first", "second", "third")) == "first"
    )
    assert resolve_inference_language(()) is None
    assert resolve_inference_language(None) is None


def test_training_arguments_receive_protocol_values():
    pytest.importorskip("transformers")
    from vla_factory.training.trainer import build_training_args

    recipe = parse_recipe("examples/pi0_robotwin_dump_bin_bigbin_lora.yaml")
    args = build_training_args(recipe)
    assert args.gradient_accumulation_steps == 1
    assert args.warmup_steps == 1000
    assert args.max_grad_norm == pytest.approx(1.0)
    assert args.lr_scheduler_type.value == "cosine"
    # Optimizer knobs are no longer configurable: HF Trainer defaults apply.
    assert args.weight_decay == pytest.approx(1e-4)
    assert args.adam_beta1 == pytest.approx(0.9)
    assert args.adam_beta2 == pytest.approx(0.999)
    assert args.adam_epsilon == pytest.approx(1e-8)


def test_raw_trainer_checkpoint_is_available_for_resume_and_inference(tmp_path):
    pytest.importorskip("transformers")
    from vla_factory.inference.checkpoint import resolve_checkpoint_path

    model = torch.nn.Linear(1, 1, bias=False)
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    torch.save(model.state_dict(), checkpoint / "pytorch_model.bin")
    assert resolve_checkpoint_path(tmp_path) == checkpoint / "pytorch_model.bin"

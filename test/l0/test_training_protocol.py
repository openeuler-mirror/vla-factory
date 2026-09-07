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


def test_delta_only_checkpointing_is_opt_in():
    recipe = parse_recipe_from_string(
        """
model: {name: pi0}
output:
  save_delta_only: true
"""
    )
    assert recipe.output.save_delta_only is True


def test_delta_only_requires_a_base_checkpoint():
    from vla_factory.model.model_interface import ModelMetadata
    from vla_factory.training.train import _validate_training_request

    recipe = parse_recipe_from_string(
        """
model: {name: act}
finetuning: {strategy: lora}
output: {save_delta_only: true}
"""
    )
    with pytest.raises(ValueError, match="requires model.path"):
        _validate_training_request(recipe, ModelMetadata(name="act", training_paradigm="from_scratch"))


def test_delta_only_trainer_saves_only_trainable_parameters(tmp_path):
    pytest.importorskip("transformers")
    from transformers import TrainingArguments

    from vla_factory.training.trainer import VLATrainer

    model = torch.nn.Linear(2, 1)
    model.bias.requires_grad = False
    trainer = VLATrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        save_delta_only=True,
        checkpoint_format="lora_delta",
    )
    trainer._save(str(tmp_path))

    assert json.loads((tmp_path / "weights.json").read_text()) == {"format": "lora_delta"}
    assert (tmp_path / "model.safetensors").is_file()
    from safetensors.torch import load_file
    assert set(load_file(tmp_path / "model.safetensors")) == {"weight"}


def test_delta_only_trainer_keeps_persistent_buffers(tmp_path):
    pytest.importorskip("transformers")
    from transformers import TrainingArguments
    from safetensors.torch import load_file

    from vla_factory.training.trainer import VLATrainer

    model = torch.nn.BatchNorm1d(2)
    model.register_buffer("nonpersistent", torch.zeros(2), persistent=False)
    trainer = VLATrainer(
        model=model, args=TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        save_delta_only=True, checkpoint_format="lora_delta",
    )
    trainer._save(str(tmp_path))
    saved = load_file(tmp_path / "model.safetensors")
    assert set(saved) == set(model.state_dict())
    assert "nonpersistent" not in saved


def test_delta_load_rejects_missing_trainable_parameters_and_buffers():
    from vla_factory.inference.checkpoint import validate_delta_load

    model = torch.nn.BatchNorm1d(2)
    result = model.load_state_dict({}, strict=False)
    with pytest.raises(ValueError, match="missing trainable state"):
        validate_delta_load(model, result)


def test_delta_load_allows_missing_frozen_parameters(tmp_path):
    from vla_factory.inference.checkpoint import validate_delta_load

    model = torch.nn.Linear(2, 1)
    model.bias.requires_grad = False
    result = model.load_state_dict({"weight": model.weight.detach()}, strict=False)
    validate_delta_load(model, result)


def test_weight_format_is_independent_of_directory_name(tmp_path):
    from vla_factory.inference.checkpoint import checkpoint_format

    weights = tmp_path / "moved-anywhere" / "model.safetensors"
    weights.parent.mkdir()
    (weights.parent / "weights.json").write_text('{"format": "lora_wrapped_full"}')
    assert checkpoint_format(weights) == "lora_wrapped_full"


def test_transformers_checkpoint_private_api_signature_is_pinned():
    import inspect
    from transformers import Trainer

    assert list(inspect.signature(Trainer._save_checkpoint).parameters) == [
        "self", "model", "trial",
    ]


def test_checkpoint_save_reports_completed_location(monkeypatch, tmp_path, caplog):
    import logging
    pytest.importorskip("transformers")
    from transformers import Trainer, TrainingArguments

    from vla_factory.training.trainer import VLATrainer

    def fake_save_checkpoint(self, model, trial):
        (tmp_path / f"checkpoint-{self.state.global_step}").mkdir()

    monkeypatch.setattr(Trainer, "_save_checkpoint", fake_save_checkpoint)
    trainer = VLATrainer(
        model=torch.nn.Linear(2, 1),
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[]),
    )
    trainer.state.global_step = 7
    caplog.set_level(logging.INFO)
    trainer._save_checkpoint(trainer.model, trial=None)

    assert "Checkpoint complete" in caplog.text


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
    (checkpoint / "trainer_state.json").write_text("{}")
    assert resolve_checkpoint_path(tmp_path) == checkpoint / "pytorch_model.bin"


def test_incomplete_newer_checkpoint_is_skipped(tmp_path):
    from vla_factory.inference.checkpoint import resolve_checkpoint_path

    complete = tmp_path / "checkpoint-7"
    complete.mkdir()
    torch.save({}, complete / "pytorch_model.bin")
    (complete / "trainer_state.json").write_text("{}")
    incomplete = tmp_path / "checkpoint-9"
    incomplete.mkdir()
    torch.save({}, incomplete / "pytorch_model.bin")
    assert resolve_checkpoint_path(tmp_path) == complete / "pytorch_model.bin"

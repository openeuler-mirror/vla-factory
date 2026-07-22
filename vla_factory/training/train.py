"""High-level training orchestration: recipe → trained model.

Wires together:
  1. YAML config parsing (TrainRecipe)
  2. Model creation via registry
  3. Fine-tuning strategy application (freeze/selective/full)
  4. DataLoader creation via data pipeline
  5. TrainingArguments mapping
  6. VLATrainer invocation

Usage::

    from vla_factory.training.train import train
    train("configs/act_aloha.yaml")
    # or
    train(recipe_object)
"""

from __future__ import annotations

import inspect
import json
import logging
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import yaml
from transformers import TrainingArguments

from vla_factory.config.parser import parse_recipe
from vla_factory.config.defaults import resolve_recipe
from vla_factory.config.recipe import TrainRecipe
from vla_factory.data.loader import create_dataloaders
from vla_factory.data.formats import get_reader
from vla_factory.data.manifest import resolve_vector_keys
from vla_factory.training.pytorch_trainer import VLATrainer
from vla_factory.training.strategies import apply_strategy
from vla_factory.utils.constants import (
    INFERENCE_META_DIR, RECIPE_FILE, SCHEMA_FILE, NORM_STATS_FILE,
    FINAL_DIR, MODEL_WEIGHTS_FILE,
)
from vla_factory.model.registry import get_entry

logger = logging.getLogger(__name__)


def train(
    config: Union[str, Path, TrainRecipe],
    *,
    override_steps: int | None = None,
    override_batch_size: int | None = None,
    override_output_dir: str | None = None,
) -> dict:
    """Run a training session from a YAML config or TrainRecipe.

    Parameters
    ----------
    config : str | Path | TrainRecipe
        Path to a YAML recipe file, or an already-parsed TrainRecipe.
    override_steps : int, optional
        Override ``total_steps`` from the recipe (useful for quick tests).
    override_batch_size : int, optional
        Override ``batch_size`` from the recipe.
    override_output_dir : str, optional
        Override ``output_dir`` from the recipe.

    Returns
    -------
    dict
        Training metrics from the final logging callback.
    """
    # 1. Parse recipe
    if isinstance(config, (str, Path)):
        recipe = parse_recipe(config)
    else:
        recipe = config

    if override_steps is not None:
        recipe.total_steps = override_steps
    if override_batch_size is not None:
        recipe.batch_size = override_batch_size
    if override_output_dir is not None:
        recipe.output.output_dir = override_output_dir

    recipe = resolve_recipe(recipe)

    logger.info("Recipe: model=%s, strategy=%s, lr=%s, steps=%d",
                recipe.model_name, recipe.finetuning_strategy,
                recipe.lr, recipe.total_steps)

    # 1.5 Handle output_dir
    output_path = Path(recipe.output.output_dir)
    if recipe.output.overwrite_output_dir and output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Read dataset schema + norm_stats
    data_cfg = recipe.data
    data_path = Path(data_cfg.source.path)
    reader = get_reader(data_cfg.source.format, path=data_path)
    schema = reader.get_schema(data_path)
    norm_stats = reader.get_norm_stats(data_path)

    # ImageNet image normalisation lives in the Normalize transform
    # (use_imagenet_stats=True default); norm_stats is no longer mutated here.

    logger.info("Dataset schema: cameras=%s, action_dim=%d, state_dim=%d",
                list(schema.cameras), schema.action_dim, schema.state_dim)

    # 2.0 Resolve canonical state/action key order (data/model contract).
    # Writes the resolved keys back onto ``schema`` so they land in
    # inference_metadata/schema.json via ``_save_inference_metadata`` (asdict).
    # This is what lets a checkpoint be served on the real robot without
    # re-sorting motor keys. Lenient: warns (does not raise) if the dataset has
    # no ``names`` — only some platforms (lerobot host) actually need the keys.
    resolved_state_keys, resolved_action_keys = resolve_vector_keys(schema, recipe)
    schema = replace(
        schema,
        state_keys=resolved_state_keys,
        action_keys=resolved_action_keys,
    )
    logger.info(
        "Resolved vector keys — state=%s action=%s",
        list(schema.state_keys), list(schema.action_keys),
    )

    # 2.1 Video frame cache: PyAVCodec.decode_frame() reads from the .npy
    # disk cache (``<video>.frame_cache/``) before falling back to live MP4
    # decoding. Pre-populate it with ``python -m vla_factory preprocess
    # --config ...``; otherwise frames are decoded on first access and cached
    # for subsequent epochs. No action needed here at training time.

    # 2.5 Save inference metadata BEFORE training starts.
    # Any intermediate checkpoint can then be used for inference.
    _save_inference_metadata(output_path, recipe, schema, norm_stats)

    # 3. Create model — adapter extracts what it needs from recipe + schema
    entry = get_entry(recipe.model_name)
    metadata = entry.metadata

    # 3.1 Finetune-only models (pi0, pi05, ...) require a base checkpoint.
    # ACT (from_scratch) allows model.path=None; pretrained_finetune does not.
    # Inference bypasses this — InferenceEngine sets model_path=None and loads
    # weights via load_state_dict — so the check belongs in the train() entry.
    if metadata.training_paradigm == "pretrained_finetune" and not recipe.model_path:
        raise ValueError(
            f"Model {recipe.model_name!r} is finetune-only "
            f"(training_paradigm=pretrained_finetune): model.path must point to "
            f"a HF base checkpoint (e.g. lerobot/pi0_base). from_scratch training "
            f"is not supported for this model."
        )

    model = entry.factory(recipe=recipe, schema=schema)

    # 4. Apply fine-tuning strategy (freeze/selective/full/lora). LoRA may
    # re-wrap subtrees in place, so use the returned model.
    model = apply_strategy(model, recipe, metadata)

    # 5. Create DataLoaders
    model_metadata_dict = {
        "action_dim": metadata.action_dim or recipe.action_spec.action_dim,
    }
    train_loader, val_loader = create_dataloaders(recipe, model_metadata=model_metadata_dict)

    # 6. Build TrainingArguments
    training_args = _build_training_args(recipe)

    # 7. Create trainer and run
    trainer = VLATrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset if len(val_loader.dataset) > 0 else None,
        data_collator=train_loader.collate_fn,
    )

    logger.info("Starting training: %d steps, batch_size=%d", recipe.total_steps, recipe.batch_size)
    trainer.train()

    # 8. Save final model. For LoRA, merge adapters into the base weights and
    # save the full state_dict so inference loads a single file (no peft merge
    # needed at deploy). peft's merged state_dict still carries a residual
    # `base_model.model.` wrapper on wrapped subtrees; strip it so the saved
    # keys match the upstream model's bare structure (loaded strict by InferenceEngine).
    final_dir = Path(recipe.output.output_dir) / FINAL_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    save_model = model
    try:
        if recipe.finetuning_strategy == "lora":
            save_model = _merge_lora_adapters(model)
    except Exception as e:  # noqa: BLE001 - merge failure shouldn't lose the run
        logger.warning("LoRA merge failed (%s); saving unmerged state_dict.", e)

    state_dict = save_model.state_dict()
    if recipe.finetuning_strategy == "lora":
        state_dict = _strip_peft_prefix(state_dict)
    torch.save(state_dict, final_dir / MODEL_WEIGHTS_FILE)
    logger.info("Model saved to %s", final_dir)

    # Return final metrics
    return trainer.state.log_history[-1] if trainer.state.log_history else {}


def _merge_lora_adapters(model: nn.Module) -> nn.Module:
    """Merge peft LoRA adapters into base weights and return the merged model.

    Two wrap shapes exist (``strategies/lora.py:_wrap_subtree``):

    * **Subtree-LoRA** (single ``target_components`` path): a *child* module —
      e.g. only the VLM/paligemma subtree — is a ``PeftModel`` while the
      top-level model is not. Walk the tree, ``merge_and_unload()`` each
      ``PeftModel`` child, and re-attach the merged base to its parent.
    * **Whole-model LoRA** (multiple targets): ``get_peft_model`` wraps the
      top-level model itself, so no *child* is a ``PeftModel`` and the tree
      walk alone would be a no-op — the checkpoint would keep ``lora_A`` /
      ``lora_B`` keys that break the InferenceEngine's ``strict=True`` load.
      Merge the top-level ``PeftModel`` directly and return its base.

    Idempotent — a no-op when nothing is peft-wrapped (non-LoRA strategies).
    """
    from peft import PeftModel
    if isinstance(model, PeftModel):
        _merge_lora_layers_inplace(model)
        merged = model.merge_and_unload()
        _merge_peft_subtrees(merged)
        return merged
    _merge_peft_subtrees(model)
    return model


def _merge_peft_subtrees(module: nn.Module) -> None:
    """Recursively replace ``PeftModel`` children with their merged base modules.

    ``list(named_children())`` snapshots the children before mutation so the
    ``setattr`` re-attach doesn't invalidate iteration. Recursing into the
    merged base is defensive — our wrap paths don't nest peft, but it catches
    a nested wrap for free and can't loop (the ``isinstance`` gate ends the
    branch once a ``PeftModel`` is merged away).
    """
    from peft import PeftModel
    for name, child in list(module.named_children()):
        if isinstance(child, PeftModel):
            # Merge adapters in place *before* merge_and_unload: peft's
            # get_delta_weight materialises the full B@A delta on the
            # submodule's device per layer, which OOMs a 12 GB GPU on the
            # PaliGemma VLM projections when the trained model still holds
            # the run's residency. Chunked in-place merge keeps the delta at
            # one row-chunk; the subsequent merge_and_unload then finds every
            # adapter already in merged_adapters and only strips the wrapper
            # (no second materialisation).
            _merge_lora_layers_inplace(child)
            merged = child.merge_and_unload()
            setattr(module, name, merged)
            _merge_peft_subtrees(merged)
        else:
            _merge_peft_subtrees(child)


def _merge_lora_layers_inplace(peft_model: nn.Module) -> None:
    """Merge every LoRA layer's delta into its base weight, chunked, on-device.

    Iterates all ``LoraLayer``s under ``peft_model`` and folds each adapter's
    ``B @ A * scaling`` into ``base.weight`` row-chunk by row-chunk so the
    transient delta never exceeds ``[chunk_rows, in_features]`` — vs peft's
    default which materialises the full ``[out, in]`` delta at once. Adapters
    are appended to ``merged_adapters`` so a later ``merge_and_unload`` skips
    re-merging and only unloads the wrapper. Falls back to peft's own
    ``layer.merge`` for non-vanilla cases (lora_variant / fan_in_fan_out) it
    can't chunk safely.
    """
    from peft.tuners.lora import LoraLayer
    for _, layer in peft_model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        for adapter in list(layer.lora_A.keys()):
            if adapter in layer.merged_adapters:
                continue
            if not _merge_lora_layer_chunked(layer, adapter):
                layer.merge(safe_merge=False, adapter_names=[adapter])


def _merge_lora_layer_chunked(layer, adapter: str, chunk_rows: int = 256) -> bool:
    """Fold one adapter's LoRA delta into its base weight in row-chunks.

    Returns ``True`` if merged here, ``False`` for non-vanilla variants
    (``lora_variant`` set, or ``fan_in_fan_out`` — those need peft's
    transpose/variant logic, not a plain ``B @ A``) so the caller can fall
    back to peft's merge.
    """
    if adapter in getattr(layer, "lora_variant", {}):
        return False
    if getattr(layer, "fan_in_fan_out", False):
        return False

    base = layer.get_base_layer()
    weight_a = layer.lora_A[adapter].weight  # [r, in]
    weight_b = layer.lora_B[adapter].weight  # [out, r]
    scaling = layer.scaling[adapter]

    with torch.no_grad():
        out_dim = weight_b.shape[0]
        for i in range(0, out_dim, chunk_rows):
            j = min(i + chunk_rows, out_dim)
            # partial delta for rows [i:j] only: [chunk, in]
            delta = torch.mm(weight_b[i:j], weight_a) * scaling
            base.weight.data[i:j] += delta.to(base.weight.dtype)

        if layer.lora_bias[adapter]:
            if getattr(base, "bias", None) is None:
                raise RuntimeError(
                    "lora_bias=True but base layer has no bias; cannot merge."
                )
            base.bias.data += (
                layer.lora_B[adapter].bias * scaling
            ).to(base.bias.dtype)

    layer.merged_adapters.append(adapter)
    return True


def _strip_peft_prefix(state_dict: dict) -> dict:
    """Strip peft's residual ``base_model.model.`` wrapper from merged LoRA keys.

    After ``merge_and_unload``, a fully-merged subtree has clean keys, but peft
    can leave a residual ``base_model.model.`` wrapper on partially-merged or
    whole-model-wrapped subtrees (e.g. ``...paligemma.base_model.model.model.X``
    → ``...paligemma.model.X``). InferenceEngine loads strict against the
    upstream's bare structure, so strip the wrapper for a clean match. Keys
    without the wrapper pass through unchanged — a no-op when merge succeeded
    cleanly.
    """
    return {
        (k.replace(".base_model.model.", ".") if ".base_model.model." in k else k): v
        for k, v in state_dict.items()
    }


def _build_training_args(recipe: TrainRecipe):
    """Map TrainRecipe → transformers.TrainingArguments."""
    # Resolve report_to: "none" → [], "tensorboard" → ["tensorboard"], etc.
    report_to = _resolve_report_to(recipe.output.report_to)

    ta_kwargs = dict(
        output_dir=recipe.output.output_dir,
        # Step-based training (not epoch-based)
        num_train_epochs=1,
        max_steps=recipe.total_steps,
        per_device_train_batch_size=recipe.batch_size,
        # Optimizer
        learning_rate=recipe.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=1e-4,
        # Gradient management
        gradient_accumulation_steps=1,
        max_grad_norm=10.0,
        gradient_checkpointing=recipe.gradient_checkpointing,
        # Logging & saving
        logging_steps=recipe.output.logging_steps,
        save_steps=recipe.output.save_steps,
        save_total_limit=recipe.output.save_total_limit,
        # Eval
        eval_strategy="no",  # MVP: no eval during training
        # DataLoader
        dataloader_drop_last=True,
        dataloader_num_workers=recipe.num_workers,
        # Critical: keep custom columns (Observation, etc.)
        remove_unused_columns=False,
        # Reporting
        report_to=report_to,
        logging_nan_inf_filter=False,
    )
    # pi0-base ships tied weights (lm_head == embed_tokens); safetensors refuses
    # to save shared-storage tensors, so pi0 needs torch.save (.bin). The flag
    # was removed in transformers >=5 (safetensors is always used there); pass it
    # only when the installed version still accepts it, so both the pi0 (4.x) and
    # ACT (5.x, via lerobot) environments construct TrainingArguments cleanly.
    if "save_safetensors" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["save_safetensors"] = False

    args = TrainingArguments(**ta_kwargs)

    # Pass lr_backbone to VLATrainer via custom attribute
    args.lr_backbone = recipe.lr_backbone

    return args


def _resolve_report_to(value: str) -> list[str]:
    """Resolve YAML report_to field to a list for TrainingArguments.

    Accepts: "none", "tensorboard", "wandb", or a string like "tensorboard,wandb".
    Validates that the requested backend is actually installed.
    """
    if value == "none" or not value:
        return []

    backends = [b.strip() for b in value.split(",") if b.strip()]
    available = []
    for name in backends:
        try:
            __import__(name)
            available.append(name)
        except ImportError:
            logger.warning(
                "report_to=%r requested but %s is not installed, skipping.",
                name, name,
            )
    return available


def _save_inference_metadata(
    output_path: Path,
    recipe: TrainRecipe,
    schema: object,
    norm_stats: object,
) -> None:
    """Save recipe, schema, and norm_stats to output_dir for inference.

    These files are immutable during training — written once before training
    starts so that any intermediate checkpoint can be used for inference.

    Saves:
      - ``recipe.yaml``: resolved recipe used by this run
      - ``schema.json``: dataset feature schema (cameras, dims, etc.)
      - ``norm_stats.json``: normalisation mean/std for state and action
    """
    meta_dir = output_path / INFERENCE_META_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolved recipe: this is the single deployment config source.
    with open(meta_dir / RECIPE_FILE, "w") as f:
        yaml.safe_dump(_recipe_to_yaml_dict(recipe), f, sort_keys=False)

    # 2. Schema
    with open(meta_dir / SCHEMA_FILE, "w") as f:
        json.dump(asdict(schema), f, indent=2)

    # 3. Norm stats
    with open(meta_dir / NORM_STATS_FILE, "w") as f:
        json.dump(asdict(norm_stats), f, indent=2)

    logger.info("Inference metadata saved to %s", meta_dir)


def _recipe_to_yaml_dict(recipe: TrainRecipe) -> dict:
    """Serialize a fully expanded TrainRecipe using the public YAML structure."""
    return {
        "transforms": {
            "imports": list(recipe.transform_imports),
        },
        "model": {
            "name": recipe.model_name,
            "path": recipe.model_path,
            "config": recipe.model_config,
        },
        "action_spec": asdict(recipe.action_spec),
        "data": {
            "source": asdict(recipe.data.source),
            "sampler": asdict(recipe.data.sampler),
            "split": asdict(recipe.data.split),
        },
        "finetuning": {
            "strategy": recipe.finetuning_strategy,
            "lora": asdict(recipe.lora_config) if recipe.lora_config is not None else None,
            "freeze_components": recipe.freeze_components,
            "trainable_components": recipe.trainable_components,
        },
        "training": {
            "backend": recipe.backend,
            "lr": recipe.lr,
            "lr_backbone": recipe.lr_backbone,
            "batch_size": recipe.batch_size,
            "total_steps": recipe.total_steps,
            "gradient_checkpointing": recipe.gradient_checkpointing,
            "inference_steps": recipe.inference_steps,
            "num_workers": recipe.num_workers,
            "augmentation": asdict(recipe.augmentation),
        },
        "output": asdict(recipe.output),
    }

"""HuggingFace Trainer adaptation and training-argument construction.

Bridges the batch dict {"observation": Observation, "actions": Tensor} format
produced by the data pipeline to the model.compute_loss(obs, actions) interface.

Inherits for free: mixed precision, DDP/FSDP/DeepSpeed, gradient accumulation,
LR scheduling, checkpointing, wandb/tensorboard logging, progress bar.
"""

from __future__ import annotations

import inspect
import logging

import torch
from transformers import Trainer, TrainingArguments

from vla_factory.user_interface import TrainRecipe


logger = logging.getLogger(__name__)


class VLATrainer(Trainer):
    """VLA-Factory Trainer, bridges batch dict → model.compute_loss(obs, actions).

    Inherited capabilities:
      - Mixed precision (fp16/bf16)
      - DDP / FSDP / DeepSpeed
      - Gradient accumulation & clipping
      - LR scheduling (cosine, linear, etc.)
      - Checkpointing & resume
      - Wandb / TensorBoard logging
      - Progress bar & ETA
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_loss_dict: dict | None = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        obs = inputs["observation"]
        actions = inputs["actions"]
        action_is_pad = inputs.get("action_is_pad")

        # Trainer._prepare_inputs only handles Tensor/dict/list.
        # Observation is a dataclass — move manually.
        device = next(model.parameters()).device
        if not isinstance(obs, torch.Tensor):
            obs = obs.to(device)
            actions = actions.to(device)
            if action_is_pad is not None:
                action_is_pad = action_is_pad.to(device)

        loss, loss_dict = model.compute_loss(obs, actions, action_is_pad=action_is_pad)

        # Record loss_dict for logging — detach to prevent autograd graph leak.
        # Storing tensors with grad_fn keeps the entire backward computation graph
        # alive (backbone features, encoder/decoder activations, VAE intermediates),
        # which leaks ~hundreds of MB per logged step and causes OOM kill.
        if self.state.is_world_process_zero:
            self._last_loss_dict = {
                k: v.detach().item() if hasattr(v, "detach") else v
                for k, v in loss_dict.items()
            }

        return (loss, loss_dict) if return_outputs else loss

    def log(self, logs: dict, start_time: float | None = None):
        """Merge auxiliary loss metrics into the log dict."""
        if hasattr(self, "_last_loss_dict") and self._last_loss_dict:
            logs.update(self._last_loss_dict)
        super().log(logs, start_time=start_time)

    def create_optimizer(self):
        """Support lr_backbone: backbone parameters use a separate (lower) LR."""
        if self.optimizer is not None:
            return self.optimizer

        lr_backbone = getattr(self.args, "lr_backbone", None)
        if lr_backbone is not None:
            backbone_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "backbone." in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr_backbone},
                    {"params": other_params, "lr": self.args.learning_rate},
                ],
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer

        return super().create_optimizer()


def build_training_args(recipe: TrainRecipe) -> TrainingArguments:
    """Map the framework training recipe onto HuggingFace arguments."""
    training = recipe.training
    ta_kwargs = dict(
        output_dir=recipe.output.output_dir,
        num_train_epochs=1,
        max_steps=training.total_steps,
        per_device_train_batch_size=training.batch_size,
        learning_rate=training.lr,
        lr_scheduler_type=training.lr_scheduler_type,
        warmup_steps=training.warmup_steps,
        weight_decay=1e-4,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_grad_norm=training.max_grad_norm,
        gradient_checkpointing=training.gradient_checkpointing,
        logging_steps=recipe.output.logging_steps,
        save_steps=recipe.output.save_steps,
        save_total_limit=recipe.output.save_total_limit,
        eval_strategy="no",
        dataloader_drop_last=True,
        dataloader_num_workers=training.num_workers,
        remove_unused_columns=False,
        report_to=_resolve_report_to(recipe.output.report_to),
        logging_nan_inf_filter=False,
    )
    if "save_safetensors" in inspect.signature(
        TrainingArguments.__init__
    ).parameters:
        ta_kwargs["save_safetensors"] = False

    args = TrainingArguments(**ta_kwargs)
    args.lr_backbone = training.lr_backbone
    return args


def _resolve_report_to(value: str) -> list[str]:
    """Keep only requested logging backends that are installed."""
    if value == "none" or not value:
        return []

    available = []
    for name in (part.strip() for part in value.split(",")):
        if not name:
            continue
        try:
            __import__(name)
            available.append(name)
        except ImportError:
            logger.warning(
                "report_to=%r requested but %s is not installed, skipping.",
                name, name,
            )
    return available

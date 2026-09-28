"""HuggingFace Trainer adaptation and training-argument construction.

Bridges the batch dict {"observation": Observation, "actions": Tensor} format
produced by the data pipeline to the model.compute_loss(obs, actions) interface.

Inherits for free: mixed precision, DDP/FSDP/DeepSpeed, gradient accumulation,
LR scheduling, checkpointing, wandb/tensorboard logging, progress bar.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from vla_factory.user_interface import TrainRecipe
from vla_factory.utils.constants import WEIGHTS_META_FILE
from vla_factory.training.profiler import StepProfiler


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
        self.save_delta_only = kwargs.pop("save_delta_only", False)
        self.checkpoint_format = kwargs.pop("checkpoint_format", "bare_full")
        # Opt-in step profiler: --profile on the CLI enables it. When disabled
        # the StepProfiler is a complete no-op (every hook short-circuits on
        # an enabled flag), so unprofiled runs pay nothing but one attribute
        # read per hook point.
        profile = kwargs.pop("profile", False)
        super().__init__(*args, **kwargs)
        self._last_loss_dict: dict | None = None
        self.profiler = StepProfiler(enabled=profile)

    def _save(self, output_dir=None, state_dict=None):
        """Let LoRA runs persist only parameters changed by training."""
        if self.save_delta_only:
            parameters = dict(self.model.named_parameters())
            trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
            state_dict = {
                key: value.detach().cpu().contiguous()
                for key, value in self.model.state_dict().items()
                if key in trainable or key not in parameters
            }
        super()._save(output_dir, state_dict)
        path = Path(output_dir or self.args.output_dir) / WEIGHTS_META_FILE
        path.write_text(json.dumps({"format": self.checkpoint_format}) + "\n")

    def get_train_dataloader(self):
        """Wrap the train dataloader so each batch fetch is timed."""
        loader = super().get_train_dataloader()
        return self.profiler.wrap_dataloader(loader)

    def _save_checkpoint(self, model, trial):
        """Log the checkpoint only after Trainer has finished writing it."""
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            logger.info(
                "Checkpoint complete: %s",
                Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}",
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        obs = inputs["observation"]
        actions = inputs["actions"]
        action_is_pad = inputs.get("action_is_pad")

        # Trainer._prepare_inputs only handles Tensor/dict/list.
        # Observation is a dataclass — move manually.
        device = next(model.parameters()).device
        if not isinstance(obs, torch.Tensor):
            with self.profiler.stage("h2d"):
                obs = obs.to(device)
                actions = actions.to(device)
                if action_is_pad is not None:
                    action_is_pad = action_is_pad.to(device)

        with self.profiler.stage("forward"):
            loss, loss_dict = model.compute_loss(obs, actions, action_is_pad=action_is_pad)
        # Forward→backward boundary: backward runs inside training_step
        # (accelerator.backward), so timestamp here and close the span on
        # training_step's return (record_backward_from).
        self.profiler.mark_forward_end()

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

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Forward + backward, with the backward span timed by delta.

        HuggingFace's ``training_step`` runs ``compute_loss`` (forward) then
        ``accelerator.backward`` (backward). We inherit that wholesale and only
        close the backward timer opened in ``compute_loss`` on return.
        ``num_items_in_batch`` is forwarded when the installed transformers
        accepts it (4.41+); 4.40 does not, so it is dropped via *args probing.
        """
        sig = inspect.signature(super().training_step)
        if "num_items_in_batch" in sig.parameters and num_items_in_batch is not None:
            result = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        else:
            result = super().training_step(model, inputs)
        self.profiler.record_backward_from()
        self.profiler.commit_step()
        return result

    def log(self, logs: dict, start_time: float | None = None):
        """Merge auxiliary loss metrics into the log dict and emit the profile."""
        if self.state.is_world_process_zero:
            report = self.profiler.drain_and_report(
                self.state.global_step,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            )
            if report:
                # The bottleneck breakdown prints just before the loss/metrics
                # line Trainer is about to emit, so they read as one block.
                logger.info("\n%s", report)
        if hasattr(self, "_last_loss_dict") and self._last_loss_dict:
            logs.update(self._last_loss_dict)
        # Trainer.log(start_time=...) was added after transformers 4.40 (the
        # OpenVLA venv); probe the signature like build_training_args.
        if "start_time" in inspect.signature(super().log).parameters:
            super().log(logs, start_time=start_time)
        else:
            super().log(logs)

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
            self.profiler.wrap_optimizer(self.optimizer)
            return self.optimizer

        opt = super().create_optimizer()
        self.profiler.wrap_optimizer(opt)
        return opt

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Wrap the LR scheduler's step into the optimizer timing stage."""
        sched = super().create_scheduler(num_training_steps, optimizer)
        self.profiler.wrap_lr_scheduler(sched)
        return sched


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
        # eval_strategy is set conditionally below (signature probe for
        # transformers <4.41 compatibility).
        dataloader_drop_last=True,
        dataloader_num_workers=training.num_workers,
        remove_unused_columns=False,
        report_to=_resolve_report_to(recipe.output.report_to),
        logging_nan_inf_filter=False,
    )
    # eval_strategy was renamed from evaluation_strategy in transformers 4.41;
    # the OpenVLA venv (transformers==4.40.1) predates the rename. Same
    # signature-probe pattern as save_safetensors below.
    if "eval_strategy" in inspect.signature(
        TrainingArguments.__init__
    ).parameters:
        ta_kwargs["eval_strategy"] = "no"
    else:
        ta_kwargs["evaluation_strategy"] = "no"
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

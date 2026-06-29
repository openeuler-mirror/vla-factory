"""VLATrainer: thin subclass of transformers.Trainer for VLA models.

Bridges the batch dict {"observation": Observation, "actions": Tensor} format
produced by the data pipeline to the model.compute_loss(obs, actions) interface.

Inherits for free: mixed precision, DDP/FSDP/DeepSpeed, gradient accumulation,
LR scheduling, checkpointing, wandb/tensorboard logging, progress bar.
"""

from __future__ import annotations

import torch
from transformers import Trainer


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

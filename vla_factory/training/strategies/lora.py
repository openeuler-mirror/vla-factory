"""LoRA fine-tuning strategy via peft.

Injects PEFT LoRA adapters into the model's trainable components. Unlike
freeze/selective (which toggle ``requires_grad``), LoRA wraps linear layers
with low-rank adapters; the base weights stay frozen.

Design
------
``finetuning.config.target_components`` (e.g. ``["llm"]``) names components
declared in ``ModelMetadata.components`` (e.g. ``"llm" ->
["paligemma_with_expert.paligemma."]``). Each prefix locates a subtree to wrap;
only those subtrees get adapters, so LoRA can target just the VLM (openpi's
convention) or the action expert too.

Freeze semantics (matches openpi's ``get_freeze_filter``): peft freezes the
base weights INSIDE the wrapped subtree; parameters OUTSIDE it (for pi0:
action expert, state/action/time projections) keep ``requires_grad=True`` and
are fully fine-tuned. So ``target_components: ["llm"]`` means "LoRA adapters
on the VLM + full FT of everything else", exactly like openpi's
``paligemma_variant="gemma_2b_lora"`` with a non-lora action expert — NOT
adapter-only training. The stats log below splits the two so the numbers
aren't misread as adapter size.

``target_modules`` (the linear-layer names peft matches, e.g. ``q_proj``) are
not per-model config — they're the standard attention/MLP projections shared
across Gemma/PaliGemma-style backbones. Verified against RLinf's openpi LoRA.

Adapter storage: the strategy finalizes by merging adapters into the base model
and returns a clean inference state dict. Checkpoint persistence itself remains
in ``training/checkpoint.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from vla_factory.model.model_interface import ModelMetadata
from vla_factory.training.strategies.base import FinetuningStrategy
from vla_factory.training.strategies.registry import register_strategy
from vla_factory.utils.format import human_count

logger = logging.getLogger(__name__)

# Linear-layer names peft matches inside the wrapped subtree. Standard
# Gemma/PaliGemma attention + MLP projections (matches RLinf's openpi set).
# lm_head is deliberately NOT here: pi0/pi05's flow-matching forward never
# calls it (no logits), so its adapter would get no gradients and only waste
# optimizer state — and it is weight-tied to embed_tokens, which makes peft
# emit tied-weight warnings at wrap/merge time. Add it per-model when a
# variant actually trains the token head (e.g. a future pi-fast).
_DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "proj", "qkv", "fc1", "fc2", "fc3", "out_proj",
]


@dataclass(frozen=True)
class LoraConfig:
    """Strict LoRA configuration owned by :class:`LoRAStrategy`."""

    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = False
    init_lora_weights: object = "gaussian"
    target_components: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.r, bool) or not isinstance(self.r, int) or self.r <= 0:
            raise ValueError("finetuning.config.r must be a positive integer")
        if (
            isinstance(self.lora_alpha, bool)
            or not isinstance(self.lora_alpha, int)
            or self.lora_alpha <= 0
        ):
            raise ValueError(
                "finetuning.config.lora_alpha must be a positive integer"
            )
        if not isinstance(self.lora_dropout, (int, float)) or not (
            0.0 <= self.lora_dropout < 1.0
        ):
            raise ValueError(
                "finetuning.config.lora_dropout must be in [0, 1)"
            )
        if not isinstance(self.use_rslora, bool):
            raise TypeError("finetuning.config.use_rslora must be a boolean")
        if not isinstance(self.target_components, list) or any(
            not isinstance(name, str) or not name
            for name in self.target_components
        ):
            raise TypeError(
                "finetuning.config.target_components must be a list of names"
            )


@register_strategy("lora")
class LoRAStrategy(FinetuningStrategy[LoraConfig]):
    config_type = LoraConfig

    def prepare_model(self, model, config, metadata):
        return apply_lora(model, config, metadata)

    def finalize_model(self, model):
        return merge_lora_adapters(model)

    def state_dict(self, model):
        return _strip_peft_prefix(model.state_dict())


def apply_lora(
    model: nn.Module,
    config: LoraConfig,
    metadata: ModelMetadata,
) -> nn.Module:
    """Inject LoRA adapters into the model's target components.

    Returns the (possibly re-wrapped) model — peft may replace subtrees in
    place, so callers should use the returned object.

    Raises ``ValueError`` if the model doesn't support LoRA or no target
    component resolves to a real subtree.
    """
    from peft import LoraConfig as PeftLoraConfig

    if not metadata.support_lora:
        raise ValueError(
            f"Model {metadata.name!r} does not support LoRA "
            f"(support_lora=False). Use a different finetuning strategy."
        )

    targets = config.target_components or []
    if not targets:
        raise ValueError(
            "LoRA: finetuning.config.target_components is empty. "
            f"Known components for {metadata.name!r}: {list(metadata.components)}."
        )

    # Forward peft-aligned fields to peft.LoraConfig as-is; vla only adds the
    # target_components → subtree → target_modules mapping below.
    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_rslora=config.use_rslora,
        init_lora_weights=config.init_lora_weights,
        target_modules=_DEFAULT_TARGET_MODULES,
    )

    # Resolve target_components → subtree paths via metadata.components.
    subtree_paths: list[str] = []
    for comp in targets:
        if comp not in metadata.components:
            logger.warning(
                "LoRA target_components %r not in metadata.components %s; skipping.",
                comp, list(metadata.components),
            )
            continue
        subtree_paths.extend(metadata.components[comp])

    if not subtree_paths:
        raise ValueError(
            f"LoRA: target_components {targets} resolved to no subtree paths. "
            f"metadata.components={metadata.components}"
        )

    # If a single subtree is the whole target, wrap it in place (openpi: only
    # the VLM/paligemma subtree gets adapters; its base weights are frozen by
    # peft, while params outside the subtree — action expert, projections —
    # stay requires_grad=True and train fully, same as openpi's freeze filter).
    # For multiple subtrees or a whole-model target, wrap the whole model.
    wrapped = _wrap_subtree(model, subtree_paths, peft_config)
    _log_lora_stats(
        wrapped, label=f"lora(r={config.r}, α={config.lora_alpha}, {targets})",
    )
    return wrapped


def _wrap_subtree(model: nn.Module, subtree_paths: list[str], peft_config) -> nn.Module:
    """Apply get_peft_model to the resolved subtrees.

    For a single subtree path like ``paligemma_with_expert.paligemma.``, walk
    to that submodule, wrap it, and re-attach so the parent sees the peft-wrapped
    version. For multiple/whole-model targets, wrap the whole model.
    """
    from peft import get_peft_model

    if len(subtree_paths) == 1:
        path = subtree_paths[0].rstrip(".")
        parent, leaf = _resolve_parent(model, path)
        if parent is None or leaf is None:
            logger.warning(
                "LoRA: subtree %r not found on model; falling back to whole-model wrap.", path
            )
            return get_peft_model(model, peft_config)
        subtree = getattr(parent, leaf)
        wrapped = get_peft_model(subtree, peft_config)
        setattr(parent, leaf, wrapped)
        return model

    # Multiple subtrees: wrap the whole model (peft matches target_modules
    # across all of them; base weights outside target_modules stay frozen).
    return get_peft_model(model, peft_config)


def _resolve_parent(model: nn.Module, dotted_path: str):
    """Walk a.b.c -> return (obj_at_a_b, "c") or (None, None) if not found.

    Tries the path on *model* first, then on ``model.model`` if present —
    vla-factory wrappers (PI0ModelWrapper) hold the upstream model at
    ``self.model``, while metadata.components paths are written against the
    upstream's own structure (e.g. ``paligemma_with_expert.paligemma.``).
    """
    parent, leaf = _walk(model, dotted_path)
    if parent is not None:
        return parent, leaf
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return _walk(inner, dotted_path)
    return None, None


def _walk(root: nn.Module, dotted_path: str):
    parts = dotted_path.split(".")
    obj = root
    for p in parts[:-1]:
        if not hasattr(obj, p):
            return None, None
        obj = getattr(obj, p)
    leaf = parts[-1]
    return obj, leaf if hasattr(obj, leaf) else None


def _log_lora_stats(model: nn.Module, label: str) -> None:
    """Log trainable vs total params, splitting adapters from full-FT params.

    The split matters: with subtree LoRA, most trainable params are usually
    the fully fine-tuned modules OUTSIDE the wrapped subtree (pi0: the action
    expert), not the adapters — a single "trainable" number reads like
    adapter size and misleads memory/lr decisions.
    """
    total = sum(p.numel() for p in model.parameters())
    adapter = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    )
    full_ft = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" not in n
    )
    trainable = adapter + full_ft
    logger.info(
        "Strategy [%s]: %s (%s) trainable / %s (%s) total (%.2f%%) — "
        "%s (%s) LoRA adapters + %s (%s) full-FT (modules outside the wrapped "
        "subtree, trained like openpi's freeze filter)",
        label,
        f"{trainable:,}", human_count(trainable),
        f"{total:,}", human_count(total),
        100.0 * trainable / max(total, 1),
        f"{adapter:,}", human_count(adapter),
        f"{full_ft:,}", human_count(full_ft),
    )


def merge_lora_adapters(model: nn.Module) -> nn.Module:
    """Merge whole-model or subtree PEFT wrappers into their base weights."""
    from peft import PeftModel

    if isinstance(model, PeftModel):
        _merge_lora_layers_inplace(model)
        merged = model.merge_and_unload()
        _merge_peft_subtrees(merged)
        return merged
    _merge_peft_subtrees(model)
    return model


def _merge_peft_subtrees(module: nn.Module) -> None:
    """Recursively replace PEFT children with their merged base modules."""
    from peft import PeftModel

    for name, child in list(module.named_children()):
        if isinstance(child, PeftModel):
            _merge_lora_layers_inplace(child)
            merged = child.merge_and_unload()
            setattr(module, name, merged)
            _merge_peft_subtrees(merged)
        else:
            _merge_peft_subtrees(child)


def _merge_lora_layers_inplace(peft_model: nn.Module) -> None:
    """Fold LoRA deltas into base weights in bounded row chunks."""
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
    """Merge one vanilla LoRA adapter without materializing its full delta."""
    if adapter in getattr(layer, "lora_variant", {}):
        return False
    if getattr(layer, "fan_in_fan_out", False):
        return False

    base = layer.get_base_layer()
    weight_a = layer.lora_A[adapter].weight
    weight_b = layer.lora_B[adapter].weight
    scaling = layer.scaling[adapter]

    with torch.no_grad():
        out_dim = weight_b.shape[0]
        for start in range(0, out_dim, chunk_rows):
            end = min(start + chunk_rows, out_dim)
            delta = torch.mm(weight_b[start:end], weight_a) * scaling
            base.weight.data[start:end] += delta.to(base.weight.dtype)

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
    """Remove PEFT's residual wrapper segment from finalized state keys."""
    return {
        (key.replace(".base_model.model.", ".")
         if ".base_model.model." in key else key): value
        for key, value in state_dict.items()
    }

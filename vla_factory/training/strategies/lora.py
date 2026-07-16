"""LoRA fine-tuning strategy via peft.

Injects PEFT LoRA adapters into the model's trainable components. Unlike
freeze/selective (which toggle ``requires_grad``), LoRA wraps linear layers
with low-rank adapters; the base weights stay frozen.

Design
------
``recipe.lora_config.target_components`` (e.g. ``["llm"]``) names components
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

Adapter storage: peft saves adapter-only by default. The training entry saves
the full state_dict (base + merged adapter) for inference simplicity; peft also
writes an ``adapter_model.safetensors`` alongside. See ``train.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch.nn as nn

from vla_factory.utils.format import human_count

if TYPE_CHECKING:
    from vla_factory.config.recipe import LoraConfig, TrainRecipe
    from vla_factory.model.protocols.model import ModelMetadata

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


def apply_lora(
    model: nn.Module,
    recipe: TrainRecipe,
    metadata: ModelMetadata,
) -> nn.Module:
    """Inject LoRA adapters into the model's target components.

    Returns the (possibly re-wrapped) model — peft may replace subtrees in
    place, so callers should use the returned object.

    Raises ``ValueError`` if the model doesn't support LoRA or no target
    component resolves to a real subtree.
    """
    from peft import LoraConfig, get_peft_model

    if not metadata.support_lora:
        raise ValueError(
            f"Model {metadata.name!r} does not support LoRA "
            f"(support_lora=False). Use a different finetuning strategy."
        )

    lora_cfg = recipe.lora_config
    if lora_cfg is None:
        raise ValueError(
            "finetuning.strategy='lora' but no finetuning.lora block in the recipe. "
            "Set finetuning.lora: {r, lora_alpha, target_components}."
        )

    targets = lora_cfg.target_components or []
    if not targets:
        raise ValueError(
            "LoRA: finetuning.lora.target_components is empty. "
            f"Known components for {metadata.name!r}: {list(metadata.components)}."
        )

    # Forward peft-aligned fields to peft.LoraConfig as-is; vla only adds the
    # target_components → subtree → target_modules mapping below.
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        use_rslora=lora_cfg.use_rslora,
        init_lora_weights=lora_cfg.init_lora_weights,
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
    _log_lora_stats(wrapped, label=f"lora(r={lora_cfg.r}, α={lora_cfg.lora_alpha}, {targets})")
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

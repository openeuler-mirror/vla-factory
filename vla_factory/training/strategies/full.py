"""Full / freeze / selective fine-tuning strategies.

Applies parameter freezing to a model based on the recipe's
``finetuning_strategy`` and the component name patterns declared in
``ModelMetadata.components``.

LoRA is a separate future strategy — it requires ``peft`` injection
rather than simple ``requires_grad_(False)``.

Usage::

    apply_strategy(model, recipe, metadata)
    # Now only the desired parameters have requires_grad=True.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch.nn as nn

from vla_factory.utils.format import human_count

if TYPE_CHECKING:
    from vla_factory.config.recipe import TrainRecipe
    from vla_factory.model.protocols.model import ModelMetadata

logger = logging.getLogger(__name__)


def apply_strategy(
    model: nn.Module,
    recipe: TrainRecipe,
    metadata: ModelMetadata,
) -> nn.Module:
    """Apply parameter freezing / adapter injection per the recipe's strategy.

    Strategies
    ----------
    ``full``
        All parameters trainable (no-op).
    ``freeze``
        Freeze components listed in ``recipe.freeze_components``.
    ``selective``
        Freeze everything *except* components in ``recipe.trainable_components``.
    ``lora``
        Inject PEFT LoRA adapters into ``recipe.lora_config.target_components``.

    Returns the model (LoRA may re-wrap subtrees in place; callers should use
    the returned object). For freeze/selective the model is mutated in place.
    """
    strategy = recipe.finetuning_strategy

    if strategy == "full":
        # All parameters already have requires_grad=True by default.
        _log_param_stats(model, "full (all trainable)")
        return model

    if strategy == "freeze":
        _freeze_components(model, recipe.freeze_components or [], metadata)
        _log_param_stats(model, f"freeze({recipe.freeze_components})")
        return model

    if strategy == "selective":
        _selective_train(model, recipe.trainable_components or [], metadata)
        _log_param_stats(model, f"selective({recipe.trainable_components})")
        return model

    if strategy == "lora":
        # Delegated to the peft-based strategy (separate module).
        from vla_factory.training.strategies.lora import apply_lora
        return apply_lora(model, recipe, metadata)

    raise ValueError(f"Unknown finetuning strategy: {strategy!r}")


# ── Internal helpers ────────────────────────────────────────────────


def _get_component_patterns(
    component_names: list[str],
    metadata: ModelMetadata,
) -> list[str]:
    """Resolve component names to parameter-name prefix patterns.

    ``metadata.components`` maps ``"backbone"`` → ``["model.backbone."]``.
    This function expands a list of component names into the union of their
    prefix patterns.
    """
    patterns: list[str] = []
    for name in component_names:
        if name not in metadata.components:
            logger.warning(
                "Component %r not found in metadata.components (%s). Skipping.",
                name,
                list(metadata.components.keys()),
            )
            continue
        patterns.extend(metadata.components[name])
    return patterns


def _match_prefix(name: str, patterns: list[str]) -> bool:
    """Return True if *name* starts with any of the *patterns*."""
    return any(name.startswith(p) for p in patterns)


def _freeze_components(
    model: nn.Module,
    freeze_names: list[str],
    metadata: ModelMetadata,
) -> None:
    """Freeze parameters belonging to the named components."""
    patterns = _get_component_patterns(freeze_names, metadata)
    if not patterns:
        logger.warning("freeze: no matching component patterns found, nothing frozen.")
        return

    for name, param in model.named_parameters():
        if _match_prefix(name, patterns):
            param.requires_grad_(False)


def _selective_train(
    model: nn.Module,
    trainable_names: list[str],
    metadata: ModelMetadata,
) -> None:
    """Freeze everything, then un-freeze only the named components."""
    # Step 1: freeze all
    for param in model.parameters():
        param.requires_grad_(False)

    # Step 2: un-freeze selected components
    patterns = _get_component_patterns(trainable_names, metadata)
    if not patterns:
        logger.warning(
            "selective: no matching component patterns found. "
            "All parameters are frozen — training will not update anything."
        )
        return

    for name, param in model.named_parameters():
        if _match_prefix(name, patterns):
            param.requires_grad_(True)


def _log_param_stats(model: nn.Module, label: str) -> None:
    """Log how many parameters are trainable vs frozen."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    logger.info(
        "Strategy [%s]: %s (%s) trainable, %s (%s) frozen, %s (%s) total",
        label,
        f"{trainable:,}", human_count(trainable),
        f"{frozen:,}", human_count(frozen),
        f"{total:,}", human_count(total),
    )

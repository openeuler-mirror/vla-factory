"""Built-in full, freeze, and selective parameter strategies.

Applies parameter freezing to a model based on the recipe's
``finetuning_strategy`` and the component name patterns declared in
``ModelMetadata.components``.

LoRA is a separate future strategy — it requires ``peft`` injection
rather than simple ``requires_grad_(False)``.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch.nn as nn

from vla_factory.model.model_interface import ModelMetadata
from vla_factory.training.strategies.base import FinetuningStrategy
from vla_factory.training.strategies.registry import register_strategy
from vla_factory.utils.format import human_count

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FullConfig:
    """The full strategy has no strategy-specific parameters."""


@dataclass(frozen=True)
class ComponentConfig:
    """ModelMetadata component names selected by a basic strategy."""

    components: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.components, list) or any(
            not isinstance(name, str) or not name for name in self.components
        ):
            raise TypeError("finetuning.config.components must be a list of names")


@register_strategy("full")
class FullStrategy(FinetuningStrategy[FullConfig]):
    config_type = FullConfig

    def prepare_model(self, model, config, metadata):
        _log_param_stats(model, "full (all trainable)")
        return model


@register_strategy("freeze")
class FreezeStrategy(FinetuningStrategy[ComponentConfig]):
    config_type = ComponentConfig

    def prepare_model(self, model, config, metadata):
        _freeze_components(model, config.components, metadata)
        _log_param_stats(model, f"freeze({config.components})")
        return model


@register_strategy("selective")
class SelectiveStrategy(FinetuningStrategy[ComponentConfig]):
    config_type = ComponentConfig

    def prepare_model(self, model, config, metadata):
        _selective_train(model, config.components, metadata)
        _log_param_stats(model, f"selective({config.components})")
        return model


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

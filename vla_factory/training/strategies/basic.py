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
    context: str,
) -> list[str]:
    """Resolve component names to parameter-name prefix patterns.

    ``metadata.components`` maps ``"backbone"`` → ``["model.backbone."]``.
    This function expands a list of component names into the union of their
    prefix patterns.

    Raises ``ValueError`` on an unknown component name. Skipping it instead
    would silently change which parameters train — a typo in
    ``freeze_components`` would freeze nothing and train the whole model, with
    only a log line to show for it. That is the "runs fine but trains wrong"
    failure mode the framework must surface, not log.
    """
    patterns: list[str] = []
    for name in component_names:
        if name not in metadata.components:
            raise ValueError(
                f"{context}: component {name!r} is not declared in "
                f"ModelMetadata.components for model {metadata.name!r}. "
                f"Available components: {list(metadata.components.keys())}."
            )
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
    if not freeze_names:
        raise ValueError(
            "freeze: finetuning.strategy='freeze' but freeze_components is empty. "
            "Nothing would be frozen and the run would train the whole model — "
            "use strategy='full' if that is the intent. Available components: "
            f"{list(metadata.components.keys())}."
        )
    patterns = _get_component_patterns(freeze_names, metadata, context="freeze")
    if not patterns:
        raise ValueError(
            f"freeze: components {freeze_names} declare no parameter-name prefixes "
            f"in ModelMetadata.components for model {metadata.name!r} — nothing "
            "would be frozen."
        )

    for name, param in model.named_parameters():
        if _match_prefix(name, patterns):
            param.requires_grad_(False)


def _selective_train(
    model: nn.Module,
    trainable_names: list[str],
    metadata: ModelMetadata,
) -> None:
    """Freeze everything, then un-freeze only the named components."""
    # Step 1: resolve before mutating, so a bad component name leaves the model
    # untouched rather than fully frozen.
    if not trainable_names:
        raise ValueError(
            "selective: finetuning.strategy='selective' but trainable_components "
            "is empty. Every parameter would stay frozen and training would "
            "update nothing. Available components: "
            f"{list(metadata.components.keys())}."
        )
    patterns = _get_component_patterns(trainable_names, metadata, context="selective")
    if not patterns:
        raise ValueError(
            f"selective: components {trainable_names} declare no parameter-name "
            f"prefixes in ModelMetadata.components for model {metadata.name!r} — "
            "every parameter would stay frozen."
        )

    # Step 2: freeze all
    for param in model.parameters():
        param.requires_grad_(False)

    # Step 3: un-freeze selected components
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

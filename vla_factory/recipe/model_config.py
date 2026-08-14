"""Validate and merge per-run model tunables with declared defaults."""

from __future__ import annotations

import difflib
from dataclasses import replace

from omegaconf import OmegaConf

from vla_factory.model.registry import list_entries
from vla_factory.recipe.train_recipe import TrainRecipe


def model_params(model_name: str) -> dict:
    """Return a registered model's declared tunable defaults."""
    metadata = list_entries().get(model_name)
    return dict(metadata.params) if metadata is not None else {}


def merge_model_config(recipe: TrainRecipe) -> TrainRecipe:
    """Return ``recipe`` with declared model defaults merged under overrides.

    Unknown models pass through so assembly resolution can report its
    structured ``UNKNOWN_MODEL`` error. A registered model with no declared
    params, however, accepts no arbitrary config keys.
    """
    entries = list_entries()
    metadata = entries.get(recipe.model.name)
    overrides = recipe.model.config or {}
    if metadata is None:
        return recipe

    params = dict(metadata.params)
    _validate_override_keys(recipe.model.name, params, overrides)
    merged = OmegaConf.merge(params, overrides)
    model = replace(
        recipe.model,
        config=OmegaConf.to_container(merged, resolve=True),
    )
    return replace(recipe, model=model)


def _validate_override_keys(
    model_name: str,
    params: dict,
    overrides: dict,
) -> None:
    if "transforms" in overrides:
        raise ValueError(
            "model.config.transforms is not supported: transform operations are "
            "derived by the assembly resolver and cannot be overridden per run."
        )

    unknown = sorted(set(overrides) - set(params))
    if not unknown:
        return

    known = sorted(params)
    lines = []
    for key in unknown:
        close = difflib.get_close_matches(key, known, n=3, cutoff=0.6)
        hint = f" — did you mean {', '.join(close)}?" if close else ""
        lines.append(f"  {key}{hint}")
    raise ValueError(
        f"model.config for {model_name!r} sets key(s) the model does not "
        f"declare:\n" + "\n".join(lines) + "\n"
        f"Tunable keys for {model_name!r}: {known}\n"
        "Values the composition resolver consumes (image range, normalization, "
        "dimension policy, ...) are facts on ModelMetadata and are deliberately "
        "not overridable from a recipe."
    )


__all__ = ["merge_model_config", "model_params"]

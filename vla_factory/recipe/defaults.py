"""Recipe resolution — fold a model's declared defaults under the recipe.

A model ships one declaration: ``ModelMetadata``. Its named fields are *facts*
the composition resolver reads and a recipe can never override; its ``params``
dict holds that model's own tunable defaults (upstream hyperparameters plus the
default ``transforms`` step list). The container is the attribute, so a model
author never classifies anything — framework facts have names and types,
everything else goes in ``params``.

``resolve_recipe()`` is the single merge point: the declared ``params`` sit
underneath the recipe's per-run ``model.config`` and the recipe wins. It is also
where the tunable allow-list is enforced — a ``model.config`` key the model
never declared is a typo or a stale knob, and silently accepting it is how
"I changed it and nothing happened" bugs get made.

There is exactly one vocabulary to check against: a field the composition
resolver derives was removed from the recipe rather than aliased, so no legacy
spelling reaches here.
"""

from __future__ import annotations

import difflib
from dataclasses import replace

from omegaconf import OmegaConf

from vla_factory.model.registry import list_entries
from vla_factory.recipe.recipe import TrainRecipe


def model_params(model_name: str) -> dict:
    """Return a model's declared tunable defaults (``{}`` if unregistered).

    Reads the registry only — no model factory, no heavy optional deps, so this
    stays callable from ``list`` / ``resolve`` / ``inspect`` on a bare install.
    """
    metadata = list_entries().get(model_name)
    return dict(metadata.params) if metadata is not None else {}


def _check_override_keys(model_name: str, params: dict, overrides: dict) -> None:
    """Reject ``model.config`` keys the model never declared.

    Skipped when the model declares no params at all — a newly added entry
    should not be unable to accept any config before its declaration is filled
    in.
    """
    if not params:
        return
    unknown = [key for key in overrides if key not in params]
    if not unknown:
        return

    known = sorted(params)
    lines = []
    for key in sorted(unknown):
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


def resolve_recipe(recipe: TrainRecipe) -> TrainRecipe:
    """Return a fully resolved recipe.

    An entrypoint operation for authoring recipes: the model's declared
    ``params`` are deep-merged under ``model_config`` and user recipe values
    win. Transforms are part of that same tree under
    ``model_config["transforms"]``.
    """
    params = model_params(recipe.model_name)
    overrides = recipe.model_config or {}
    _check_override_keys(recipe.model_name, params, overrides)
    merged = OmegaConf.merge(params, overrides)
    return replace(
        recipe,
        model_config=OmegaConf.to_container(merged, resolve=True),
    )

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
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import replace

from omegaconf import OmegaConf

from vla_factory.model.registry import list_entries
from vla_factory.recipe.recipe import TrainRecipe

logger = logging.getLogger(__name__)

# Keys accepted in ``model.config`` even though they are not model params:
# both are ``assembly:`` block fields (architecture §3.1) still honoured in
# their legacy ``model.config`` location during the migration, with a
# deprecation warning raised by their own readers.
_ASSEMBLY_LEGACY_KEYS = frozenset({"camera_mapping", "default_task"})


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
    unknown = [
        key for key in overrides
        if key not in params and key not in _ASSEMBLY_LEGACY_KEYS
    ]
    if not unknown:
        return

    known = sorted(set(params) | _ASSEMBLY_LEGACY_KEYS)
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


def _forward_legacy_action_horizon(
    recipe: TrainRecipe, params: dict, overrides: dict,
) -> dict:
    """Move a legacy ``action_spec.action_horizon`` into ``model.config``.

    The horizon's home is the model declaration (a family fact for pretrained
    models, a tunable for from-scratch ones), so the recipe field is deprecated.
    Forwarding happens here rather than in the parser because only this module
    can see ``ModelMetadata.params`` — i.e. whether *this* model accepts the
    tunable at all. Forwarding it to a model that does not (pi0) would trip the
    allow-list below and stop an otherwise fine legacy recipe from running, so
    that case warns and drops instead.
    """
    legacy = recipe.action_spec.action_horizon
    if legacy is None or "action_horizon" in overrides:
        return overrides
    if "action_horizon" not in params:
        logger.warning(
            "action_spec.action_horizon is deprecated and ignored for %r: this "
            "model declares its own action horizon (a pretrained chunk length "
            "cannot be changed per run). Remove the field from the recipe.",
            recipe.model_name,
        )
        return overrides
    logger.warning(
        "action_spec.action_horizon is deprecated — set model.config."
        "action_horizon instead. Forwarding %s for this run.", legacy,
    )
    return {**overrides, "action_horizon": int(legacy)}


def resolve_recipe(recipe: TrainRecipe) -> TrainRecipe:
    """Return a fully resolved recipe.

    An entrypoint operation for authoring recipes: the model's declared
    ``params`` are deep-merged under ``model_config`` and user recipe values
    win. Transforms are part of that same tree under
    ``model_config["transforms"]``.
    """
    params = model_params(recipe.model_name)
    overrides = recipe.model_config or {}
    overrides = _forward_legacy_action_horizon(recipe, params, overrides)
    _check_override_keys(recipe.model_name, params, overrides)
    merged = OmegaConf.merge(params, overrides)
    return replace(
        recipe,
        model_config=OmegaConf.to_container(merged, resolve=True),
    )

"""Model default-profile loader.

Each model may ship a baseline profile under ``vla_factory/config/model/<name>.yaml``.
The factory loads it and deep-merges the recipe's per-run ``model.config``
on top (recipe wins). This keeps model defaults external, user-editable, and
diff-friendly without coupling the framework to Hydra's runtime.

The ``vla_factory/config/model/`` directory follows Hydra's group convention
(``conf/<group>/<name>.yaml``), so these profiles can be adopted by Hydra's
``defaults: [model/<name>]`` mechanism later with no file changes.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import yaml
from omegaconf import OmegaConf

from vla_factory.utils.constants import MODEL_CONFIG_DIR
from vla_factory.config.recipe import TrainRecipe

# vla_factory/config/model/ — this file lives in vla_factory/config/.
_MODEL_CONFIG_DIR = Path(__file__).resolve().parent / MODEL_CONFIG_DIR


def model_defaults_path(model_name: str) -> Path:
    """Return the path to a model's default profile (may not exist)."""
    return _MODEL_CONFIG_DIR / f"{model_name}.yaml"


def load_model_defaults(model_name: str) -> dict:
    """Load a model's default profile as a plain dict.

    Returns ``{}`` if no profile exists for *model_name* — models are not
    required to ship one. The dict is deep-merged with the recipe's
    ``model.config`` by the factory.
    """
    path = model_defaults_path(model_name)
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Model default profile {path} must be a YAML mapping, "
            f"got {type(data).__name__}"
        )
    return data


def resolve_recipe(recipe: TrainRecipe) -> TrainRecipe:
    """Return a fully resolved recipe.

    This is an entrypoint operation for authoring recipes: model default
    profile values are deep-merged into ``model_config`` and user recipe values
    win. Transforms are part of that same model config tree under
    ``model_config["transforms"]``.
    """
    merged = OmegaConf.merge(load_model_defaults(recipe.model_name), recipe.model_config or {})
    return replace(
        recipe,
        model_config=OmegaConf.to_container(merged, resolve=True),
    )

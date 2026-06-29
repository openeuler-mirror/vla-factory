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

import yaml

from vla_factory.utils.constants import MODEL_CONFIG_DIR

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

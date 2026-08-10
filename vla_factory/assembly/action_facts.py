"""Action-fact routing (WP3): resolve ``action_spec`` facts from the three
descriptions with recipe fallback.

Architecture §7.4 phase-1 task 1: ``action_spec`` fields are sourced from the
dimension that owns them (data / model / robot), with the recipe as fallback.
On mismatch the dimension fact wins and a warning is emitted (strict failure is
deferred to phase 2). For the default path — where the recipe already agrees
with the dimensions — the resolved values are bit-identical to the legacy
behaviour ("行为不变").

Homes (established in WP1/WP2):
- ``action_dim``    → data ``schema.action_dim`` (model ``dim_policy`` is the cap)
- ``action_horizon``→ model (BaseContract > ModelMetadata) for finetune, recipe
                      for from-scratch (R4: ACT-from-scratch legitimately lets
                      the user pick the chunk size; pi0's comes from the checkpoint)
- ``action_type``   → data ``action.dims[].mode`` / robot ``native_action_type``
- ``bounds``        → robot ``safety_bounds``

This module only routes + warns; it adds no pair-compatibility checks (phase 2).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_action_dim(
    *,
    schema: Any,
    metadata: Any,
    recipe_action_dim: int,
) -> int:
    """Resolve the action dimension used to build the model/dataloader.

    Data fact (``schema.action_dim``) is authoritative when present; the recipe
    value is the fallback. A mismatch warns (and the data fact wins).
    """
    data_dim = int(getattr(schema, "action_dim", 0) or 0)
    recipe_dim = int(recipe_action_dim or 0)
    if data_dim and recipe_dim and data_dim != recipe_dim:
        logger.warning(
            "action_dim mismatch: dataset has %d dims but recipe.action_spec.action_dim=%d. "
            "Using the dataset fact (%d); update the recipe or override to silence "
            "(strict check arrives in phase 2).",
            data_dim, recipe_dim, data_dim,
        )
    return data_dim or recipe_dim


def resolve_action_horizon(
    *,
    metadata: Any,
    base_contract: Any,
    recipe_action_horizon: int,
) -> int:
    """Resolve the action horizon (chunk size), R4-aware by training paradigm.

    - ``from_scratch`` (e.g. ACT): the recipe chooses the chunk size → recipe
      is authoritative.
    - ``pretrained_finetune`` (e.g. pi0): the checkpoint knows →
      BaseContract > ModelMetadata > recipe.
    """
    paradigm = getattr(metadata, "training_paradigm", "pretrained_finetune")
    recipe_h = int(recipe_action_horizon or 0)

    if paradigm == "from_scratch":
        return recipe_h

    # finetune: prefer instance facts.
    contract_h = _horizon_from_contract(base_contract)
    if contract_h:
        if recipe_h and contract_h != recipe_h:
            logger.warning(
                "action_horizon mismatch: base checkpoint has %d but recipe.action_horizon=%d. "
                "Using the checkpoint fact (%d).",
                contract_h, recipe_h, contract_h,
            )
        return contract_h
    metadata_h = int(getattr(metadata, "action_horizon", 0) or 0)
    return metadata_h or recipe_h


def _horizon_from_contract(base_contract: Any) -> int:
    """Best-effort action-horizon from a checkpoint contract (not yet carried
    as a first-class field on BaseContract today; returns 0 when unknown)."""
    if base_contract is None:
        return 0
    # BaseContract currently exposes dims/resolution but not horizon. Hook kept
    # so phase 2 can read it without changing call sites.
    return int(getattr(base_contract, "action_horizon", 0) or 0)

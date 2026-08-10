"""Tests for action-fact routing (WP3)."""

from __future__ import annotations

import logging

from helpers import make_schema

from vla_factory.assembly.action_facts import resolve_action_dim, resolve_action_horizon
from vla_factory.model.interfaces.model import ModelMetadata


def test_action_dim_data_authoritative_when_present():
    schema = make_schema(action_dim=8)
    meta = ModelMetadata(name="act")  # flexible, action_dim=0
    # recipe agrees → identical, no warning.
    assert resolve_action_dim(schema=schema, metadata=meta, recipe_action_dim=8) == 8


def test_action_dim_recipe_fallback_when_data_absent():
    schema = make_schema(action_dim=0)
    meta = ModelMetadata(name="act")
    assert resolve_action_dim(schema=schema, metadata=meta, recipe_action_dim=6) == 6


def test_action_dim_mismatch_warns_and_uses_data_fact(caplog):
    schema = make_schema(action_dim=8)
    meta = ModelMetadata(name="act")
    with caplog.at_level(logging.WARNING, logger="vla_factory.assembly.action_facts"):
        chosen = resolve_action_dim(schema=schema, metadata=meta, recipe_action_dim=6)
    assert chosen == 8  # data fact wins
    assert any("action_dim mismatch" in r.message for r in caplog.records)


def test_action_horizon_from_scratch_uses_recipe():
    # ACT-from-scratch: the user picks the chunk size.
    meta = ModelMetadata(name="act", training_paradigm="from_scratch")
    assert resolve_action_horizon(
        metadata=meta, base_contract=None, recipe_action_horizon=100
    ) == 100


def test_action_horizon_finetune_prefers_metadata_then_recipe():
    meta = ModelMetadata(name="pi0", training_paradigm="pretrained_finetune", action_horizon=50)
    # No contract horizon yet → metadata (50) wins over recipe (40).
    assert resolve_action_horizon(
        metadata=meta, base_contract=None, recipe_action_horizon=40
    ) == 50


def test_action_horizon_finetune_falls_back_to_recipe():
    meta = ModelMetadata(name="x", training_paradigm="pretrained_finetune", action_horizon=0)
    assert resolve_action_horizon(
        metadata=meta, base_contract=None, recipe_action_horizon=40
    ) == 40

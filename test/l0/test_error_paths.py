"""L0 error-path tests for the config / registry / format-lookup boundaries.

Everything here asserts the framework's "unknown keys surface as an error, no
silent typo failures" rule (CLAUDE.md) on the paths a user actually mistypes:
a transform ``type`` in a recipe, a dataset ``format``, a missing ``model.name``.
These are the cheapest possible tests — no model, no dataset, no extras — and
they cover branches that had no coverage at all before Issue #7.
"""

from __future__ import annotations

import pytest

from vla_factory.assembly.transform.base import TransformStep
from vla_factory.assembly.transform.plan import TransformStepCall
from vla_factory.assembly.transform.registry import TransformRegistry
from vla_factory.data.reader import get_reader
from vla_factory.user_interface import parse_recipe_from_string


# ── Transform registry ───────────────────────────────────────────────


def test_unknown_transform_type_raises_and_lists_available():
    """A mistyped transform name in a recipe must name the valid choices."""
    with pytest.raises(KeyError) as exc:
        TransformRegistry.get("normalise")  # British spelling of "normalize"

    message = str(exc.value)
    assert "normalise" in message
    assert "normalize" in message, "the error must enumerate registered steps"


def test_transform_config_without_type_raises():
    """A transform entry missing its 'type' key is a recipe typo, not a default."""
    with pytest.raises(ValueError, match="type"):
        TransformStepCall.from_dict({"args": {"height": 224, "width": 224}})


def test_register_rejects_non_transform_step():
    """@register must reject anything that is not a TransformStep subclass.

    Registering a plain function or unrelated class would fail much later, at
    pipeline-build time, with an unrelated AttributeError.
    """
    with pytest.raises(TypeError, match="TransformStep subclass"):
        @TransformRegistry.register("__not_a_step")
        def not_a_class(sample):
            return sample

    assert "__not_a_step" not in TransformRegistry.names()


def test_register_accepts_transform_step_subclass():
    """Happy path, and it must not leak into the global registry afterwards."""
    try:
        @TransformRegistry.register("__probe_step")
        class _ProbeStep(TransformStep):
            def __call__(self, sample):
                return sample

        assert TransformRegistry.get("__probe_step") is _ProbeStep
        assert TransformRegistry.name_of(_ProbeStep()) == "__probe_step"
    finally:
        TransformRegistry._steps.pop("__probe_step", None)


# ── Format reader lookup ─────────────────────────────────────────────


def test_unknown_format_raises_and_lists_available():
    with pytest.raises(ValueError) as exc:
        get_reader("lerobot-v2")

    message = str(exc.value)
    assert "lerobot-v2" in message
    assert "lerobot-v3" in message


def test_auto_format_with_unreadable_path_raises(tmp_path):
    """'auto' probes every reader; none matching is an error, not a None return."""
    empty = tmp_path / "not_a_dataset"
    empty.mkdir()
    with pytest.raises(ValueError, match="No reader found"):
        get_reader("auto", path=empty)


def test_format_name_is_case_insensitive():
    assert type(get_reader("LeRobot-V3")) is type(get_reader("lerobot-v3"))


# ── Recipe parsing ───────────────────────────────────────────────────


def test_missing_model_name_raises():
    """model.name drives the whole registry lookup — it cannot default."""
    with pytest.raises(ValueError, match="model is required"):
        parse_recipe_from_string("training:\n  lr: 1e-4")


def test_empty_model_name_raises():
    with pytest.raises(ValueError, match="model.name is required"):
        parse_recipe_from_string("model:\n  name: ''")


# ── Registry load failures ───────────────────────────────────────────


def test_broken_entry_surfaces_as_registry_load_error(monkeypatch):
    """A broken entry module must raise RegistryLoadError, never be masked.

    Masking it would surface as ``KeyError: Model 'act' not registered``, which
    sends the reader hunting for a registration bug instead of the real import
    error. CLAUDE.md makes this an explicit framework guarantee; before Issue #7
    nothing tested it.
    """
    import importlib

    from vla_factory.model import registry as registry_mod

    real_import_module = importlib.import_module

    def _explode(name, *args, **kwargs):
        if name.startswith("vla_factory.model.adapters."):
            raise ImportError("simulated broken entry module")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _explode)
    monkeypatch.setattr(registry_mod.ModelRegistry, "_builtins_loaded", False)
    monkeypatch.setattr(registry_mod.ModelRegistry, "_builtin_error", None)
    monkeypatch.setattr(registry_mod.ModelRegistry, "_entries", {})

    with pytest.raises(registry_mod.RegistryLoadError) as exc:
        registry_mod.get_entry("act")

    message = str(exc.value)
    assert "simulated broken entry module" in message
    assert "ImportError" in message, "the original exception type must survive"


def test_registry_load_is_restored_after_the_broken_entry_test():
    """Guard the guard: the monkeypatched test above must not poison the registry."""
    from vla_factory.model.registry import list_entries

    assert "act" in list_entries()


# ── Removed recipe facts ─────────────────────────────────────────────


def test_old_action_spec_block_is_rejected_by_recipe_parsing():
    """Action-space facts now come from data/model/robot descriptions."""
    with pytest.raises(ValueError, match="action_spec"):
        parse_recipe_from_string(
            "model: {name: act}\naction_spec: {action_type: eef_delta}\n"
        )

"""The model config surface: what a recipe may override, and what it may not.

A model ships one declaration (``ModelMetadata``). Its named fields are facts the
composition resolver reads; ``params`` holds that model's tunable defaults. Three
guards keep the boundary honest, and each has a real bug behind it:

* an undeclared ``model.config`` key is a typo or a stale knob — pi0 used to drop
  it silently because its factory reads keys one by one;
* a declared key nothing reads is a silent no-op — ``num_inference_steps`` once
  shipped that way;
* a fact set per run wins silently and corrupts the run — a recipe could put pi0's
  images in ``[0, 1]`` while SigLIP expects ``[-1, 1]``.

Everything here runs without model extras, GPU or a dataset.
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from vla_factory.assembly.transform.images import ImageNormalize, ImageToFloat
from vla_factory.assembly.transform.normalize import NormalizeVector
from vla_factory.assembly.transform.pad_dimensions import PadDimensions
from vla_factory.assembly.transform.base import PlanContext
from vla_factory.data.data_schema import FeatureStats, NormStats
from vla_factory.model.model_interface import ModelMetadata
from vla_factory.user_interface import (
    merge_model_config,
    model_params,
    parse_recipe_from_string,
)
from vla_factory.utils.tracked_config import TrackedConfig


# ── Gate 1: model.config may only set declared params ────────────────


class TestDeclaredKeyAllowList(unittest.TestCase):

    def test_declared_key_is_accepted_and_wins(self):
        recipe = parse_recipe_from_string(
            "model:\n  name: act\n  config:\n    dim_model: 1024\n"
        )
        resolved = merge_model_config(recipe)
        self.assertEqual(resolved.model.config["dim_model"], 1024)
        # Untouched declarations still come through.
        self.assertEqual(resolved.model.config["n_decoder_layers"], 1)

    def test_undeclared_key_is_rejected_with_candidates(self):
        recipe = parse_recipe_from_string(
            "model:\n  name: act\n  config:\n    dim_modell: 1024\n"
        )
        with self.assertRaises(ValueError) as ctx:
            merge_model_config(recipe)
        message = str(ctx.exception)
        self.assertIn("dim_modell", message)
        # The near-miss is offered rather than leaving the user to guess.
        self.assertIn("dim_model", message)

    def test_unregistered_model_skips_the_gate(self):
        """A model with no declaration must not be unable to take any config."""
        recipe = parse_recipe_from_string(
            "model:\n  name: not_a_registered_model\n  config:\n    whatever: 1\n"
        )
        self.assertEqual(merge_model_config(recipe).model.config["whatever"], 1)


# ── Gate 2: a declared key nothing reads is an error ──────────────────


class TestUnreadKeyGuard(unittest.TestCase):

    def test_reading_marks_consumed(self):
        cfg = TrackedConfig({"a": 1, "b": 2}, framework_keys=())
        cfg.get("a")
        cfg["b"]
        self.assertEqual(cfg.unread(), [])
        cfg.assert_all_consumed("stub")  # does not raise

    def test_unread_key_raises_and_names_it(self):
        cfg = TrackedConfig({"read_me": 1, "forgotten": 2}, framework_keys=())
        cfg.get("read_me")
        self.assertEqual(cfg.unread(), ["forgotten"])
        with self.assertRaises(ValueError) as ctx:
            cfg.assert_all_consumed("stub")
        self.assertIn("forgotten", str(ctx.exception))

    def test_star_expansion_counts_as_a_read(self):
        """`ACTConfig(**cfg)` must mark every forwarded key consumed.

        A plain ``dict`` subclass would not: CPython merges it through the
        concrete fast path and never calls ``__getitem__``.
        """
        cfg = TrackedConfig({"a": 1, "b": 2}, framework_keys=())

        def sink(**kwargs):
            return kwargs

        sink(**cfg)
        self.assertEqual(cfg.unread(), [])

    def test_framework_consumed_keys_are_not_false_alarms(self):
        """Keys read outside the factory (engine, loader) are pre-marked."""
        cfg = TrackedConfig({"num_inference_steps": 10})
        cfg.assert_all_consumed("stub")  # does not raise

    def test_pop_counts_as_a_read(self):
        cfg = TrackedConfig({"framework_managed": 1}, framework_keys=())
        cfg.pop("framework_managed", None)
        cfg.assert_all_consumed("stub")  # does not raise


# ── Gate 3: facts cannot be overridden per run ───────────────────────


class TestFactOverrideRejected(unittest.TestCase):

    def _ctx(self, **metadata_kwargs) -> PlanContext:
        """The facts a step compiles its call against.

        The gate lives in ``compile_call``, which is where the composition
        resolver asks each step for its arguments — so this is the real entry
        point a bad recipe would come through, not a test-only shortcut.
        """
        return PlanContext(
            metadata=ModelMetadata(name="stub", **metadata_kwargs),
            target_action_dim=32,
            source_action_dim=8,
            has_norm_stats=True,
            has_action_stats=True,
        )

    def test_image_range_override_rejected(self):
        ctx = self._ctx(image_input_range=(-1.0, 1.0))
        with self.assertRaises(ValueError) as err:
            ImageToFloat.compile_call({"range": [0.0, 1.0]}, ctx)
        self.assertIn("image_input_range", str(err.exception))
        # Without the override the declared fact is used.
        self.assertEqual(ImageToFloat.compile_call({}, ctx)["range"], [-1.0, 1.0])

    def test_image_normalize_mode_override_rejected(self):
        ctx = self._ctx(image_normalize_mode="imagenet")
        with self.assertRaises(ValueError):
            ImageNormalize.compile_call({"mode": "none"}, ctx)
        self.assertEqual(ImageNormalize.compile_call({}, ctx)["mode"], "imagenet")

    def test_normalize_method_override_rejected(self):
        ctx = self._ctx(vector_normalization="quantile")
        with self.assertRaises(ValueError) as err:
            NormalizeVector.compile_call({"method": "zscore"}, ctx)
        self.assertIn("vector_normalization", str(err.exception))
        call = NormalizeVector.compile_call({"fields": ["actions"]}, ctx)
        self.assertEqual(call["method"], "quantile")

    def test_pad_target_override_rejected(self):
        ctx = self._ctx()
        with self.assertRaises(ValueError) as err:
            PadDimensions.compile_call({"target_dim": 64, "fields": ["actions"]}, ctx)
        self.assertIn("dim_policy_max", str(err.exception))
        call = PadDimensions.compile_call({"fields": ["actions"]}, ctx)
        self.assertEqual(call["target_dim"], 32)

    def test_missing_fact_is_an_error_not_a_default(self):
        """Neither config nor declaration carries it → refuse to guess."""
        ctx = self._ctx()  # no image_input_range declared
        with self.assertRaises(ValueError):
            ImageToFloat.compile_call({}, ctx)

    def test_non_fact_step_keys_stay_configurable(self):
        ctx = self._ctx(vector_normalization="mean_std")
        call = NormalizeVector.compile_call({"fields": ["state"]}, ctx)
        self.assertEqual(call["fields"], ["state"])


# ── Inference steps: one entry, one effective value ──────────────────


class TestInferenceStepsPriority(unittest.TestCase):

    def test_declaration_supplies_the_default(self):
        recipe = merge_model_config(parse_recipe_from_string("model:\n  name: pi0\n"))
        self.assertEqual(recipe.model.config["num_inference_steps"], 10)
        self.assertEqual(model_params("act")["num_inference_steps"], 1)

    def test_recipe_overrides_the_default(self):
        recipe = merge_model_config(parse_recipe_from_string(
            "model:\n  name: pi0\n  config:\n    num_inference_steps: 2\n"
        ))
        self.assertEqual(recipe.model.config["num_inference_steps"], 2)


# ── Visibility: "what may I change, and did it take effect" ──────────


class TestTunablesView(unittest.TestCase):

    def test_source_column_separates_recipe_from_declaration(self):
        from vla_factory.user_interface.cli import _tunables_view

        view = _tunables_view(
            {"dim_model": 512, "dropout": 0.1},
            {"dim_model": 1024},
        )
        self.assertEqual(view["dim_model"], {"value": 1024, "source": "recipe"})
        self.assertEqual(view["dropout"], {"value": 0.1, "source": "model default"})

    def test_transforms_are_not_model_tunables(self):
        recipe = parse_recipe_from_string(
            "model:\n  name: act\n  config:\n    transforms:\n      inputs: []\n"
        )
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            merge_model_config(recipe)

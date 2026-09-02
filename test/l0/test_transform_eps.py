"""L0 plumbing for the per-model normalisation epsilon.

The *value* of each epsilon is an upstream contract asserted in
``test/l1/test_normalize_parity.py``. This file covers our side of it: that
a value declared in a model profile survives config parsing, reaches the built
transform call, and is inherited by the inverse step — because an eps that is
declared but silently dropped looks exactly like one that was never declared.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_factory.assembly.transform.base import PlanContext
from vla_factory.assembly.transform.pipeline import TransformContext
from vla_factory.assembly.transform.normalize import (
    DEFAULT_QUANTILE_EPS,
    DEFAULT_ZSCORE_EPS,
    NormalizeVector,
    UnnormalizeActionQuantileStep,
    UnnormalizeActionStep,
)
from vla_factory.data.data_schema import FeatureStats, NormStats
from vla_factory.model.registry import list_entries

# A constant dimension (std = 0) makes the epsilon the whole denominator, so
# any drop or mismatch shows up as an order-of-magnitude difference.
_CONSTANT_STATS = NormStats(
    state=FeatureStats(mean=[0.0], std=[0.0]),
    action=FeatureStats(mean=[0.0], std=[0.0], q01=[0.0], q99=[0.0]),
)


# ── config → step ────────────────────────────────────────────────────


def test_from_call_reads_eps():
    step = NormalizeVector.from_call(
        {"fields": ["state"], "method": "zscore", "eps": 1.0e-6},
        ctx=TransformContext(norm_stats=_CONSTANT_STATS),
    )
    assert step.eps == 1.0e-6


def test_from_call_without_eps_falls_back_per_method():
    ctx = TransformContext(norm_stats=_CONSTANT_STATS)

    zscore = NormalizeVector.from_call({}, ctx=ctx)
    quantile = NormalizeVector.from_call({"method": "quantile"}, ctx=ctx)

    assert zscore.eps == DEFAULT_ZSCORE_EPS
    assert quantile.eps == DEFAULT_QUANTILE_EPS


@pytest.mark.parametrize("model_name,expected", [
    ("act", 1.0e-8),
    ("pi0", 1.0e-6),
    ("pi05", 1.0e-6),
])
def test_profile_eps_reaches_the_built_transform(model_name, expected):
    """ModelMetadata → compiled transform call → runtime step."""
    metadata = list_entries()[model_name]
    plan_ctx = PlanContext(
        metadata=metadata,
        source_state_dim=1,
        source_action_dim=1,
        has_norm_stats=True,
        has_action_stats=True,
    )
    call = NormalizeVector.compile_call({"fields": ["state", "actions"]}, plan_ctx)
    assert call is not None
    step = NormalizeVector.from_call(call, TransformContext(norm_stats=_CONSTANT_STATS))

    assert step.eps == expected


# ── forward → inverse must share the eps ─────────────────────────────


def test_zscore_inverse_inherits_eps():
    """A mismatched inverse eps silently biases every action sent to the robot."""
    step = NormalizeVector(
        _CONSTANT_STATS, fields=("actions",), method="zscore", eps=1.0e-6
    )
    call = {"fields": ["actions"], "method": step.method, "eps": step.eps}
    inverse_name, inverse_args = NormalizeVector.inverse_call(
        call, PlanContext(has_action_stats=True),
    )
    assert inverse_name == "unnormalize_action"
    inverse = UnnormalizeActionStep.from_call(
        inverse_args, TransformContext(norm_stats=_CONSTANT_STATS),
    )

    assert isinstance(inverse, UnnormalizeActionStep)
    assert inverse.eps == 1.0e-6


def test_quantile_inverse_inherits_eps():
    step = NormalizeVector(
        _CONSTANT_STATS, fields=("actions",), method="quantile", eps=1.0e-6
    )
    call = {"fields": ["actions"], "method": step.method, "eps": step.eps}
    inverse_name, inverse_args = NormalizeVector.inverse_call(
        call, PlanContext(has_action_stats=True),
    )
    assert inverse_name == "unnormalize_action_quantile"
    inverse = UnnormalizeActionQuantileStep.from_call(
        inverse_args, TransformContext(norm_stats=_CONSTANT_STATS),
    )

    assert isinstance(inverse, UnnormalizeActionQuantileStep)
    assert inverse.eps == 1.0e-6


@pytest.mark.parametrize("method", ["zscore", "quantile"])
@pytest.mark.parametrize("eps", [1.0e-6, 1.0e-8])
def test_round_trip_is_exact_on_a_constant_dimension(method, eps):
    """normalize → unnormalize must return the original even at std = 0.

    This is where a forward/inverse eps mismatch is loudest: the round trip
    scales by ``eps_inverse / eps_forward`` instead of 1.
    """
    step = NormalizeVector(_CONSTANT_STATS, fields=("actions",), method=method, eps=eps)
    call = {"fields": ["actions"], "method": step.method, "eps": step.eps}
    inverse_name, inverse_args = NormalizeVector.inverse_call(
        call, PlanContext(has_action_stats=True),
    )
    inverse_cls = (
        UnnormalizeActionQuantileStep
        if inverse_name == "unnormalize_action_quantile"
        else UnnormalizeActionStep
    )
    inverse = inverse_cls.from_call(
        inverse_args, TransformContext(norm_stats=_CONSTANT_STATS),
    )

    actions = np.array([[0.25], [-0.75]], dtype=np.float32)
    normalized = step({"actions": actions.copy()})["actions"]
    restored = inverse({"actions": normalized})["actions"]

    np.testing.assert_allclose(restored, actions, rtol=1e-4, atol=1e-6)

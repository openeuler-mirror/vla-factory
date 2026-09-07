"""L0 plumbing for min-max normalisation (diffusion_policy's contract).

The *value-level* parity against upstream ``LinearNormalizer(mode='limits')``
is asserted in ``test/l1`` (needs the diffusion_policy extra); this file
covers our side of the contract:

- forward numerics: observed range maps to [-1, 1], midpoint to 0;
- a missing min/max fails loudly (a silent identity would train on garbage);
- the inverse is selected by method and shares the forward's eps;
- round trips are exact, including the degenerate constant dimension.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_factory.assembly.transform.base import PlanContext
from vla_factory.assembly.transform.normalize import (
    DEFAULT_MINMAX_EPS,
    NormalizeVector,
    UnnormalizeActionMinMaxStep,
)
from vla_factory.assembly.transform.pipeline import TransformContext
from vla_factory.data.data_schema import FeatureStats, NormStats

# min=-2, max=+2 per dim: the [-1, 1] mapping is legible by hand.
_RANGE_STATS = NormStats(
    state=FeatureStats(mean=[0.0], std=[1.0], min=[-2.0], max=[2.0]),
    action=FeatureStats(mean=[0.0], std=[1.0], min=[-2.0], max=[2.0]),
)

# A constant dimension (min == max) is where the eps is the whole denominator.
_CONSTANT_STATS = NormStats(
    state=FeatureStats(mean=[0.5], std=[0.0], min=[0.5], max=[0.5]),
    action=FeatureStats(mean=[0.5], std=[0.0], min=[0.5], max=[0.5]),
)


def _min_max_step(stats: NormStats, eps: float | None = None) -> NormalizeVector:
    return NormalizeVector(
        stats, fields=("state", "actions"), method="min_max", eps=eps
    )


# ── vocabulary & compile ─────────────────────────────────────────────
# Plan-level min_max selection (compile_call → method/eps/inverse) is asserted
# end-to-end by test_diffusion_policy_model's resolution tests — no separate
# unit copy here.


def test_eps_falls_back_to_upstream_range_eps():
    step = NormalizeVector.from_call(
        {"method": "min_max"},
        ctx=TransformContext(norm_stats=_RANGE_STATS),
    )
    assert step.eps == DEFAULT_MINMAX_EPS == 1.0e-4


# ── forward numerics ─────────────────────────────────────────────────


def test_observed_range_maps_to_minus_one_one():
    step = _min_max_step(_RANGE_STATS)
    sample = step({
        "state": np.array([-2.0, 0.0, 2.0], dtype=np.float32),
        "actions": np.array([[-2.0], [0.0], [2.0]], dtype=np.float32),
    })
    # atol is the eps scale: max maps to 1 - eps/range by design (~2.5e-5 here).
    np.testing.assert_allclose(
        sample["state"], [-1.0, 0.0, 1.0], atol=1e-4,
    )
    np.testing.assert_allclose(sample["actions"][:, 0], [-1.0, 0.0, 1.0], atol=1e-4)


def test_out_of_range_values_extrapolate_linearly():
    """min-max normalisation has no clamp: stats from training, values from
    deployment, and the two can legitimately cross (receding-horizon rollouts)."""
    step = _min_max_step(_RANGE_STATS)
    out = step({"actions": np.array([[4.0]], dtype=np.float32)})["actions"]
    np.testing.assert_allclose(out, [[2.0]], atol=1e-3)


def test_missing_min_max_fails_loudly():
    """Stats without min/max must not silently pass through unnormalised."""
    broken = NormStats(
        action=FeatureStats(mean=[0.0], std=[1.0]),  # no min/max
    )
    step = _min_max_step(broken)
    with pytest.raises(ValueError, match="min/max"):
        step({"actions": np.zeros((1, 1), dtype=np.float32)})


def test_constant_dimension_maps_to_minus_one_and_survives_round_trip():
    """min == max: forward lands on -1 (additive-eps convention — upstream
    ``range_eps`` semantics would land on 0; both restore the same constant),
    and the round trip must return the original value exactly."""
    step = _min_max_step(_CONSTANT_STATS)
    normalized = step({"actions": np.array([[0.5]], dtype=np.float32)})["actions"]
    np.testing.assert_allclose(normalized, [[-1.0]], atol=1e-6)

    inverse = UnnormalizeActionMinMaxStep(_CONSTANT_STATS)
    restored = inverse({"actions": normalized})["actions"]
    np.testing.assert_allclose(restored, [[0.5]], atol=1e-6)


# ── inverse selection & eps sharing ──────────────────────────────────


def test_inverse_call_selects_min_max_step_and_shares_eps():
    step = _min_max_step(_RANGE_STATS, eps=5.0e-4)
    name, args = NormalizeVector.inverse_call(
        {"fields": ["actions"], "method": step.method, "eps": step.eps},
        PlanContext(has_action_stats=True),
    )
    assert name == "unnormalize_action_min_max"

    inverse = UnnormalizeActionMinMaxStep.from_call(
        args, TransformContext(norm_stats=_RANGE_STATS),
    )
    assert isinstance(inverse, UnnormalizeActionMinMaxStep)
    assert inverse.eps == 5.0e-4


@pytest.mark.parametrize("eps", [1.0e-4, 1.0e-2])
def test_round_trip_is_exact(eps):
    """normalize → unnormalize returns the original for in-range actions."""
    step = _min_max_step(_RANGE_STATS, eps=eps)
    inverse = UnnormalizeActionMinMaxStep(_RANGE_STATS, eps=eps)

    actions = np.array([[-2.0], [-1.3], [0.0], [1.7], [2.0]], dtype=np.float32)
    normalized = step({"actions": actions.copy()})["actions"]
    restored = inverse({"actions": normalized})["actions"]

    np.testing.assert_allclose(restored, actions, rtol=1e-5, atol=1e-6)

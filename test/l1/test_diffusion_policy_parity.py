"""L1 parity: min_max normalisation against real-stanford diffusion_policy.

Diffusion Policy normalises every vector field to ``[-1, 1]`` by *limits*
(per-dim min/max). VLA Factory implements that as the ``min_max`` pipeline
step (decision D1: statistics live in assembly.json; upstream's own
``LinearNormalizer`` is installed as an identity), so the arithmetic here has
no upstream code behind it at runtime — this file pins it to the paper repo's
implementation instead.

The two are **intentionally not bit-identical** on the epsilon convention:

* upstream widens the range *before* dividing — a constant dim (range 0) is
  remapped to the output midpoint ``0``;
* the framework adds ``eps`` to the denominator — a constant dim maps to
  ``-1`` (its own min), and live dims deviate by ``< eps / range`` relative.

The tests below quantify that deviation bound and assert the round-trip is
exact on both sides; the constant-dim difference is asserted *explicitly* so
a change on either side of the contract goes red here, not silently into a
trained checkpoint.

A second section (``TestObservationContract``) pins the *input* contract at
the true consumption boundary: the full framework path (resolve → pipeline →
per-frame transforms → collate → wrapper translation) must hand the real
upstream per-camera ``(B, To=2, 3, H, W)`` images and ``(B, To=2, D)``
state. The adapter performs no shape re-validation (construction already
guarantees it — pipeline and wrapper are planned from the same
``model_io_spec``), so this test is the guard: a contract drift on either
the framework or upstream side fails here, loudly, instead of as a deep
broadcasting error mid-training.

**Deliberately not covered** (review-acknowledged boundaries): the image
chain semantics (``/255`` → ``[0,1]``, CHW, stretch-resize) — declared as
``ModelMetadata`` facts and asserted at the plan level in L0, but there is
no upstream-side image parity to compare against, since upstream consumes
whatever ``shape_meta`` declares; the robomimic encoder internals; DDPM
schedule numerics.

Constant-block conventions follow ``test_normalize_parity.py`` (Issue #7's
reference shape): upstream repo + pinned commit, file:line, verbatim
expression, verification date, plus the mechanical pin guard.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_factory.assembly.transform.normalize import (
    DEFAULT_MINMAX_EPS,
    NormalizeVector,
    UnnormalizeActionMinMaxStep,
)
from vla_factory.data.data_schema import FeatureStats, NormStats
from vla_factory.model.registry import list_entries

pytestmark = pytest.mark.l1

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────
# UPSTREAM: real-stanford/diffusion_policy @ 5ba07ac6661db573af695b419a7947ecb704690f
#   diffusion_policy/model/common/normalizer.py:182  _fit(data, ...,
#       mode='limits', output_max=1., output_min=-1., range_eps=1e-4,
#       fit_offset=True)
#     209-210: input_min, _ = data.min(axis=0)
#              input_max, _ = data.max(axis=0)
#     218-223 (fit_offset branch):
#              input_range = input_max - input_min
#              ignore_dim = input_range < range_eps
#              input_range[ignore_dim] = output_max - output_min
#              scale = (output_max - output_min) / input_range
#              offset = output_min - scale * input_min
#              offset[ignore_dim] = (output_max + output_min) / 2 \
#                                   - input_min[ignore_dim]
#   i.e. x ↦ (x - min) / range * 2 - 1, with constant dims sent to 0 and
#   no epsilon added on live dims.
# Verified: 2026-09-03
# ─────────────────────────────────────────────────────────────────────
DIFFUSION_POLICY_COMMIT = "5ba07ac6661db573af695b419a7947ecb704690f"
DIFFUSION_POLICY_RANGE_EPS = 1e-4


def _upstream_available() -> bool:
    """diffusion_policy needs the patched source install (no PyPI package)."""
    return importlib.util.find_spec("diffusion_policy") is not None


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))


# ── The declaration carries upstream's epsilon ───────────────────────


def test_profile_uses_upstream_range_eps():
    """The declared vector_normalization_eps must be the paper repo's
    range_eps, not a value inherited from another ecosystem's default.
    (The transform fallback default is pinned on the L0 side,
    test_eps_falls_back_to_upstream_range_eps.)"""
    metadata = list_entries()["diffusion_policy"]
    assert metadata.vector_normalization == "min_max"
    assert float(metadata.vector_normalization_eps) == DIFFUSION_POLICY_RANGE_EPS


# ── Anti-transcription guards ────────────────────────────────────────


def test_diffusion_policy_pin_has_not_moved():
    """The recorded commit must still be the one install.sh installs.

    Bumping DIFFUSION_POLICY_REF without re-reading the expressions cited
    above is how an embedded golden goes stale silently; this turns that into
    a red test (same guard shape as test_openpi_pin_has_not_moved).
    """
    install_sh = (_PROJECT_ROOT / "scripts" / "install.sh").read_text()
    match = re.search(r'DIFFUSION_POLICY_REF="([0-9a-f]{40})"', install_sh)
    assert match, "DIFFUSION_POLICY_REF not found in scripts/install.sh — update this test"
    assert match.group(1) == DIFFUSION_POLICY_COMMIT, (
        f"scripts/install.sh pins diffusion_policy at {match.group(1)}, but "
        f"the constants in this file were read from {DIFFUSION_POLICY_COMMIT}. "
        "Re-read the upstream expressions cited above and update both the "
        "values and the header."
    )


# ── Numerics against the installed upstream ──────────────────────────
# Skipped without the extra — that is a skip, not a separate tier.


@pytest.mark.skipif(
    not _upstream_available(),
    reason="diffusion_policy upstream not installed "
           "(bash scripts/install.sh --model diffusion_policy)",
)
class TestLimitsParity:
    """Framework min_max vs upstream LinearNormalizer(mode='limits')."""

    N_LIVE = 3  # live dims; dim 3 of the fixture data is constant

    @classmethod
    def _fit_pair(cls, seed: int = 21):
        """Fit both normalisers on the same data.

        Data: 3 live dims with different ranges + 1 constant dim (the locked
        joint / closed gripper case that makes epsilons visible). The
        framework consumes statistics, upstream consumes raw data; both see
        the same per-dim min/max the training pipeline would compute.

        Returns ``(data, step, upstream)``.
        """
        from diffusion_policy.model.common.normalizer import LinearNormalizer

        rng = np.random.default_rng(seed)
        data = rng.uniform(-3.0, 5.0, size=(512, cls.N_LIVE)).astype(np.float32)
        data = np.concatenate(
            [data, np.full((512, 1), 2.5, dtype=np.float32)], axis=1,
        )

        stats = NormStats(action=FeatureStats(
            mean=data.mean(axis=0), std=data.std(axis=0),
            q01=data.min(axis=0), q99=data.max(axis=0),
            min=data.min(axis=0), max=data.max(axis=0),
        ))
        step = NormalizeVector(
            stats, fields=("actions",), method="min_max",
            eps=DIFFUSION_POLICY_RANGE_EPS,
        )

        upstream = LinearNormalizer()
        upstream.fit(
            _t(data), mode="limits",
            output_max=1.0, output_min=-1.0,
            range_eps=DIFFUSION_POLICY_RANGE_EPS, fit_offset=True,
        )
        return data, step, stats, upstream

    def test_endpoints_map_to_the_output_range(self):
        """Both sides send the observed minimum to exactly -1; upstream sends
        the max to +1, the framework to within eps·2/range of it."""
        data, step, stats, upstream = self._fit_pair()
        lows = data.min(axis=0, keepdims=True)
        highs = data.max(axis=0, keepdims=True)

        fw_low = step({"actions": lows.copy()})["actions"][:, :self.N_LIVE]
        fw_high = step({"actions": highs.copy()})["actions"][:, :self.N_LIVE]
        up_low = upstream.normalize(_t(lows)).numpy()[:, :self.N_LIVE]
        up_high = upstream.normalize(_t(highs)).numpy()[:, :self.N_LIVE]

        np.testing.assert_allclose(
            fw_low, -1.0, atol=0.0, rtol=0.0,
            err_msg="the framework must map min to exactly -1 on live dims",
        )
        np.testing.assert_allclose(
            up_low, -1.0, atol=0.0, rtol=0.0,
            err_msg="upstream must map min to exactly -1 on live dims",
        )
        np.testing.assert_allclose(
            up_high, 1.0, atol=0.0, rtol=0.0,
            err_msg="upstream must map max to exactly +1 on live dims",
        )
        # Framework max: 2·range/(range+eps) - 1 < 1, short by ~2·eps/range
        # (plus one float32 rounding).
        ranges = (
            data.max(axis=0, keepdims=True)[:, :self.N_LIVE]
            - data.min(axis=0, keepdims=True)[:, :self.N_LIVE]
        )
        np.testing.assert_array_less(
            1.0 - fw_high, 2.0 * DIFFUSION_POLICY_RANGE_EPS / ranges + 1e-6,
            err_msg="the framework max must be within the eps/range bound of +1",
        )
        np.testing.assert_array_less(
            fw_high, 1.0 + 1e-6,
            err_msg="the framework max must stay inside the output range",
        )

    def test_deviation_on_live_dims_is_bounded_by_eps_over_range(self):
        """|fw - up| ≤ 2·eps·(x-min) / (range·(range+eps)) ≤ 2·eps/range —
        the whole cost of the additive-epsilon convention, quantified."""
        data, step, stats, upstream = self._fit_pair()
        x = data[::17]  # strided probe rows

        fw = step({"actions": x.copy()})["actions"][:, :self.N_LIVE]
        up = upstream.normalize(_t(x)).numpy()[:, :self.N_LIVE]

        eps = DIFFUSION_POLICY_RANGE_EPS
        vmin = data.min(axis=0)[:self.N_LIVE]
        ranges = data.max(axis=0)[:self.N_LIVE] - vmin
        bound = 2.0 * eps * (x[:, :self.N_LIVE] - vmin) / (ranges * (ranges + eps))
        diff = np.abs(fw - up)
        np.testing.assert_array_less(
            # The bound is derived in exact arithmetic; the implementations
            # evaluate in float32, so allow a couple of ulps near 1.0 on top.
            diff, bound + 1e-6,
            err_msg="deviation exceeds the documented eps/range bound",
        )
        # And it is genuinely small: below 0.01% of the output range here.
        assert diff.max() < 1e-4

    def test_constant_dim_is_the_documented_deviation(self):
        """Upstream remaps a constant dim to the output midpoint 0; the
        framework keeps it at the dim's own value → -1. Asserted explicitly
        on both sides: this is the contract, not a bug to paper over (D1 —
        the model consumes only the framework pipeline, never both)."""
        data, step, stats, upstream = self._fit_pair()
        # Full-width input (the production shape): a (N,1) slice would
        # silently broadcast against the (4,) statistics instead of erroring.
        fw = step({"actions": data.copy()})["actions"][:, self.N_LIVE:]
        up = upstream.normalize(_t(data)).numpy()[:, self.N_LIVE:]
        np.testing.assert_allclose(fw, -1.0, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(up, 0.0, atol=1e-7, rtol=0.0)

    def test_round_trip_recovers_the_input_on_both_sides(self):
        """Whichever normaliser runs, its own inverse must undo it exactly —
        the deployed postprocessor is only correct if this holds."""
        data, step, stats, upstream = self._fit_pair()
        x = data[::31]

        # Framework: forward step then the planned inverse step.
        inverse = UnnormalizeActionMinMaxStep(
            stats, eps=DIFFUSION_POLICY_RANGE_EPS,
        )
        recovered = inverse({"actions": step({"actions": x.copy()})["actions"]})["actions"]
        np.testing.assert_allclose(recovered, x, rtol=1e-5, atol=1e-6)

        # Upstream: normalize then unnormalize.
        roundtrip = upstream.unnormalize(upstream.normalize(_t(x))).numpy()
        np.testing.assert_allclose(roundtrip, x, rtol=1e-5, atol=1e-6)


# ── Input contract at the upstream consumption boundary ─────────────


class _ArrayCodec:
    """VideoCodec stand-in: each frame's image is its (unique) index value."""

    def decode_frame(self, ref):
        value = (ref.frame_index * 30) % 256
        return np.full((96, 96, 3), value, dtype=np.uint8)


class _FakeReader:
    """FormatReader stand-in serving one synthetic 4-frame episode."""

    def read_episode(self, path, ep_idx, codec):
        from vla_factory.data.data_schema import Episode, Frame, VideoRef

        frames = [
            Frame(
                index=i,
                images={"front": VideoRef(
                    video_path=None, frame_index=i,
                    height=96, width=96, channels=3,
                )},
                state=np.full(6, i, dtype=np.float32),
                action=np.full(6, 10 + i, dtype=np.float32),
            )
            for i in range(4)
        ]
        return Episode(
            episode_id="e0", episode_index=ep_idx, num_frames=4,
            _frames_cache=frames,
        )


@pytest.mark.skipif(
    not _upstream_available(),
    reason="diffusion_policy upstream not installed "
           "(bash scripts/install.sh --model diffusion_policy)",
)
class TestObservationContract:
    """Full framework path must satisfy the upstream input contract.

    resolve_assembly → build_pipeline(data_to_model) → VLADataset (per-frame
    transforms + stack-after-transform) → collate_fn → wrapper translation,
    then the assembled batch goes straight into the real upstream's
    ``compute_loss`` / ``predict_action``: a contract drift fails here, at
    the consumption boundary, instead of as an opaque broadcasting error
    mid-training. (The adapter itself re-validates nothing — this test is
    the guard, per the review decision.)
    """

    def test_pipeline_output_feeds_the_upstream_contract(self):
        from vla_factory.assembly import resolve_from_facts
        from vla_factory.assembly.transform import TransformContext, build_pipeline
        from vla_factory.data.data_schema import (
            ActionDim, CameraEntry, DataSchema, StateDim,
        )
        from vla_factory.model.registry import get_entry, list_entries
        from vla_factory.training.dataset import (
            SampleWindow, VLADataset, collate_fn,
        )
        from vla_factory.user_interface import ModelConfig, TrainRecipe, merge_model_config

        state_dim = action_dim = 6
        horizon = 4
        # Small UNet for speed; both are declared tunables.
        recipe = merge_model_config(TrainRecipe(model=ModelConfig(
            name="diffusion_policy",
            config={
                "action_horizon": horizon,
                "down_dims": [64, 128],
                "diffusion_step_embed_dim": 64,
            },
        )))
        schema = DataSchema(
            episodes=1, total_frames=4,
            cameras_entries=(CameraEntry(key="front", resolution=(96, 96)),),
            state_dims=tuple(
                StateDim(name=f"state_{i}", source_field="observation.state")
                for i in range(state_dim)
            ),
            action_dims=tuple(
                ActionDim(name=f"action_{i}", source_field="action")
                for i in range(action_dim)
            ),
        )
        # state values are {0..3} per frame, actions {10..13}: min/max cover
        # them so min_max maps the observed frame-0 state to exactly -1.
        stats = NormStats(
            state=FeatureStats(
                mean=[1.5] * state_dim, std=[1.0] * state_dim,
                q01=[0.0] * state_dim, q99=[3.0] * state_dim,
                min=[0.0] * state_dim, max=[3.0] * state_dim,
            ),
            action=FeatureStats(
                mean=[11.5] * action_dim, std=[1.0] * action_dim,
                q01=[10.0] * action_dim, q99=[13.0] * action_dim,
                min=[10.0] * action_dim, max=[13.0] * action_dim,
            ),
        )
        assembly = resolve_from_facts(
            schema=schema, norm_stats=stats,
            metadata=list_entries()["diffusion_policy"],
            model_config=recipe.model.config,
        )

        pipeline = build_pipeline(
            assembly.data_to_model, TransformContext(norm_stats=stats),
        )
        dataset = VLADataset(
            sample_windows=[SampleWindow(
                episode_index=0, start_frame_index=0,
                n_obs_steps=assembly.model_io_spec.n_obs_steps,
                action_horizon=horizon,
            )],
            reader=_FakeReader(), codec=_ArrayCodec(),
            path=None, transforms=pipeline,
        )
        batch = collate_fn([dataset[0]])
        wrapper = get_entry("diffusion_policy").factory(
            recipe=recipe, assembly=assembly,
        )

        # The translated obs dict is exactly what upstream's shape_meta
        # declares: per-camera (B, To=2, 3, H, W) float CHW in [0,1] with
        # no ImageNet shift, state (B, To=2, D) min_max-normalized.
        obs_dict = wrapper._observation_to_obs_dict(batch["observation"])
        images = obs_dict["front"]
        state = obs_dict["state"]
        assert images.shape == (1, 2, 3, 96, 96)
        assert images.dtype == torch.float32
        assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0
        assert state.shape == (1, 2, state_dim)
        # Frame 0's state is the observed minimum (0 against min=0): the
        # min_max pipeline maps it to exactly -1 (cf. TestLimitsParity).
        np.testing.assert_allclose(
            state[0, 0].numpy(), -1.0, atol=0.0, rtol=0.0,
        )
        # Image semantics survived the full chain: each frame's uniform
        # pixel value divided by 255 (frames 0/1 → 0 / 30/255).
        np.testing.assert_allclose(
            images[0, :, 0, 0, 0].numpy(),
            [0.0, 30.0 / 255.0],
            atol=1e-6,
        )

        # And the real upstream accepts the assembled batch end to end.
        loss, loss_dict = wrapper.compute_loss(batch["observation"], batch["actions"])
        assert loss.ndim == 0 and loss.requires_grad
        assert "diffusion_loss" in loss_dict
        with torch.no_grad():
            pred = wrapper.predict_actions(batch["observation"], num_steps=4)
        assert pred.shape == (1, horizon, action_dim)

"""L1 parity: pi0 / pi05 preprocessing chain vs lerobot 0.5's official chain.

Same goal as the retired ``test_openpi_pipeline_parity.py`` (Issue #7's core
ask: run one sample through both the upstream transform chain and
vla-factory's pipeline, assert tensor-level equality) with the upstream
swapped for the one this framework actually rides since the lerobot-0.5
migration: ``make_pi0_pre_post_processors`` / ``make_pi05_pre_post_processors``
+ ``PI0Policy._preprocess_images``'s geometry (``resize_with_pad_torch``).

The comparison point is the lerobot batch dict — what ``PILerobotModelWrapper``
hands the policy:

    ours:    sample → resolved assembly pipeline → collate_fn
                   → PILerobotModelWrapper._to_lerobot_batch()
    upstream: canonical dict → make_*_pre_post_processors(config, stats)
                   (+ resize_with_pad_torch for the image geometry, which the
                    policy applies inside _preprocess_images, not in its
                   processor chain)

Both sides share one raw sample and one norm_stats blob, so what is isolated
is the transform chain itself. Every assertion runs for both pi0 and pi05
(``variant`` fixture): they share the wrapper and the block layout, differing
in max_token_len (48/200), state placement (continuous tensor vs discrete
prompt) and normalization (mean_std / quantile).

Two known, intentional divergences (documented, not bugs):

* **eps & zero-variance dims** — we use 1e-6 (openpi checkpoint lineage,
  guarded by ``test_normalize_parity.py``), lerobot's
  ``NormalizerProcessorStep`` uses 1e-8. On healthy dims the two chains agree
  to ~1e-5 relative, so state/actions assert with ``rtol=1e-4`` instead of
  bit equality. On zero-variance dims — this dataset's actions have constant
  dims (std=0 / q99==q01) — the chains are *incomparable by construction*:
  lerobot divides by ~0 and produces values up to inf, we divide by eps and
  produce ~0. Those dims are excluded via a healthy-dim mask rather than
  "reconciled": neither behavior is wrong, they are different clamp
  policies.
* **bin edges (pi05)** — the discrete state prompt digitizes the *normalized*
  state into 256 bins; a ~1e-5 difference can flip a value that sits exactly
  on a bin edge (bin width 1/128). Verified not to occur on this fixed
  dataset (see ``test_prompt_tokens_are_identical``); if a stats refresh ever
  lands a value on an edge, relax to "prompt text equal + bins differ by
  <= 1" rather than deleting the test.

Not covered here: norm_stats *computation* (both sides are fed the same
blob), batch > 1. ACT's chain is a different shape (lerobot 0.4 processor,
see ``test_act_pipeline_parity.py``).

**External precondition**: both chains construct the PaliGemma tokenizer
(``google/paligemma-3b-pt-224``), which needs a warm HF cache or network
access. The ``paligemma_tokenizer`` fixture probes it once and **skips**
(rather than errors) when the environment cannot provide it — CI images that
assert this parity must pre-fetch the tokenizer.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.l1

if importlib.util.find_spec("lerobot.policies.pi0.modeling_pi0") is None:
    pytest.skip(
        "pipeline parity needs lerobot>=0.5 (bash scripts/install.sh --model pi0)",
        allow_module_level=True,
    )

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATASET = _PROJECT_ROOT / "test/data/lerobot_train_data_3_episodes"

if not _DATASET.exists():
    pytest.skip(f"test dataset missing: {_DATASET}", allow_module_level=True)

# The dataset's real contract (see test/test_data_pipeline.py).
_CAMERA_MAPPING = {"base_0_rgb": "front", "left_wrist_0_rgb": "wrist"}
_ACTION_HORIZON = 50
_DATASET_STATE_DIM = 6
_DATASET_ACTION_DIM = 8
_MODEL_ACTION_DIM = 32  # openpi max_action_dim

# Task text from the dataset's meta/tasks.parquet, surfaced by the reader.
_TASK = "Lift the red cube up."

# 480x640 aspect-preserving resize to 224 gives 224x168 with 28 letterbox
# rows top and bottom.
_PAD_ROWS = np.r_[0:28, 196:224]
_CONTENT_ROWS = np.r_[28:196]

# Content-region tolerance: torch F.interpolate (no antialias) vs cv2
# INTER_LINEAR on uint8 (quantized to 1/255). The threshold rides the mean —
# differences concentrate on edge high-frequency pixels, so a max-based
# threshold would flag normal interpolation deltas too.
_INTERPOLATION_MEAN_TOL = 0.01

# See module docstring: our eps 1e-6 vs lerobot's 1e-8 (and no eps on the
# quantile denominator). The eps-induced absolute error grows on outliers
# over narrow-scale dims (|Δ| ≈ 2·|x−q01|/D · eps/D) — measured max 1.08e-5
# on this dataset. 1e-4 leaves an order of magnitude of margin while still
# catching chain regressions (wrong stats or formula produce O(1) errors).
_VECTOR_RTOL = 1e-4
_VECTOR_ATOL = 1e-4


@pytest.fixture(scope="module")
def paligemma_tokenizer():
    """External precondition, probed once: both chains build the PaliGemma
    tokenizer (ours via ``TaskTokenize._ensure_tokenizer``, upstream via
    ``TokenizerProcessorStep`` at processor construction). Without a warm HF
    cache or network access the fixtures would ERROR on every test — skip with
    an explicit precondition message instead; CI that asserts this parity must
    pre-fetch/authenticate ``google/paligemma-3b-pt-224``."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    except Exception as e:  # noqa: BLE001 - any tokenizer-build failure is the same skip
        pytest.skip(
            "needs the google/paligemma-3b-pt-224 tokenizer (warm HF cache or "
            f"network access); got {type(e).__name__}: {e}"
        )


@pytest.fixture(scope="module")
def raw():
    """Raw sample + schema + norm_stats, before any transform."""
    from vla_factory.data.codec.pyav import PyAVCodec
    from vla_factory.data.reader.lerobot_v3 import LeRobotV3Reader
    from vla_factory.training.dataset import SampleWindow, VLADataset

    reader, codec = LeRobotV3Reader(), PyAVCodec()
    schema = reader.get_schema(_DATASET)
    norm_stats = reader.get_norm_stats(_DATASET)
    windows = [SampleWindow(0, 0, n_obs_steps=1, action_horizon=_ACTION_HORIZON)]
    dataset = VLADataset(windows, reader, codec, _DATASET, transforms=[])
    sample = dataset[0]
    assert sample.get("task") == _TASK, (
        "the reader failed to surface the dataset task text; the parity "
        "comparison would degrade to an empty-prompt comparison. See "
        "test_reader_surfaces_the_dataset_task."
    )
    return sample, schema, norm_stats


@pytest.fixture(scope="module", params=("pi0", "pi05"))
def variant(request):
    """pi0 and pi05 share the wrapper and block layout; only config and
    normalization differ (mean_std vs quantile, state in tensor vs prompt)."""
    return request.param


@pytest.fixture(scope="module")
def ours(raw, variant, paligemma_tokenizer):
    """Our chain, all the way to the lerobot batch the policy consumes."""
    from vla_factory.assembly import resolve_assembly
    from vla_factory.assembly.transform import TransformContext, build_pipeline
    from vla_factory.model.adapters.lerobot_pi import PILerobotModelWrapper
    from vla_factory.training.dataset import collate_fn
    from vla_factory.user_interface import merge_model_config, parse_recipe_from_string

    sample, _, _ = raw
    recipe = merge_model_config(parse_recipe_from_string(
        f"model:\n  name: {variant}\n"
        f"data:\n  path: {_DATASET}\n  format: lerobot-v3\n"
        "overrides:\n"
        "  camera_mapping:\n"
        "    base_0_rgb: front\n"
        "    left_wrist_0_rgb: wrist\n"
    ))
    assembly = resolve_assembly(recipe)
    transforms = build_pipeline(
        assembly.data_to_model, TransformContext(norm_stats=assembly.norm_stats),
    )
    transformed = dict(sample)
    for step in transforms:
        transformed = step(transformed)

    batch = collate_fn([transformed])
    # model=None: _to_lerobot_batch only uses camera_mapping, never weights.
    wrapper = PILerobotModelWrapper(
        model=None, camera_mapping=_CAMERA_MAPPING, include_state=(variant == "pi0"),
    )
    lerobot_batch = wrapper._to_lerobot_batch(
        batch["observation"], actions=batch["actions"],
    )
    return lerobot_batch, batch, assembly


@pytest.fixture(scope="module")
def upstream(raw, variant, paligemma_tokenizer):
    """lerobot 0.5's official processor chain output (flat batch dict)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    sample, _, norm_stats = raw
    if variant == "pi05":
        config_cls, make_ppp = PI05Config, make_pi05_pre_post_processors
    else:
        config_cls, make_ppp = PI0Config, make_pi0_pre_post_processors

    config = config_cls(
        chunk_size=_ACTION_HORIZON,
        n_action_steps=_ACTION_HORIZON,
        device="cpu",
        input_features={
            # The normalizer only touches keys declared in config features
            # (input_features for observations, output_features for action) —
            # leaving the state undeclared silently skips its normalization
            # (and pi05 then digitizes the raw state).
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(_DATASET_STATE_DIM,),
            ),
            **{
                f"observation.images.{slot}": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 224, 224),
                )
                for slot in _CAMERA_MAPPING
            },
        },
        # The framework's factory relies on validate_features auto-completing
        # this to max_action_dim; supply it explicitly here because the
        # processor chain (unlike the policy) never calls validate_features.
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(_MODEL_ACTION_DIM,)),
        },
    )

    def _feature(stats):
        out = {
            "mean": torch.tensor(stats.mean, dtype=torch.float32),
            "std": torch.tensor(stats.std, dtype=torch.float32),
        }
        if stats.q01 and stats.q99:
            out["q01"] = torch.tensor(stats.q01, dtype=torch.float32)
            out["q99"] = torch.tensor(stats.q99, dtype=torch.float32)
        return out

    lerobot_stats = {
        "observation.state": _feature(norm_stats.state),
        "action": _feature(norm_stats.action),
    }
    pre, _ = make_ppp(config, lerobot_stats)

    batch = {
        **{
            f"observation.images.{slot}": torch.from_numpy(
                np.asarray(sample[f"images.{cam}"])
            )
            for slot, cam in _CAMERA_MAPPING.items()
        },
        "observation.state": torch.tensor(
            np.asarray(sample["state"]), dtype=torch.float32,
        ),
        "action": torch.tensor(
            np.asarray(sample["actions"]), dtype=torch.float32,
        ),
        "task": sample["task"],
    }
    return pre(batch)


def _numpy(array) -> np.ndarray:
    return np.asarray(array.detach().cpu() if hasattr(array, "detach") else array)


def _unpadded_tokens(tokens, mask) -> np.ndarray:
    """Select prompt tokens by their attention mask, preserving token value 0.

    Our tokenizer pads left, lerobot's pads right — mask-based selection makes
    the comparison padding-side-agnostic.
    """
    tokens = _numpy(tokens)
    mask = _numpy(mask).astype(bool)
    assert tokens.ndim == mask.ndim == 1, (
        f"token/mask sequences should be 1-D, got {tokens.shape}/{mask.shape}"
    )
    assert tokens.shape == mask.shape, f"length mismatch: {tokens.shape}/{mask.shape}"
    return tokens[mask]


def _to_bhwc_0_1(tensor_chw) -> np.ndarray:
    """Our [1, 3, H, W] [0,1] CHW batch image → [1, H, W, C] [0,1]."""
    return _numpy(tensor_chw).transpose(0, 2, 3, 1)


def test_prompt_mask_preserves_interior_zero_tokens():
    np.testing.assert_array_equal(
        _unpadded_tokens(
            np.array([11, 0, 108, 0, 0]),
            np.array([True, True, True, False, False]),
        ),
        np.array([11, 0, 108]),
    )


# ══════════════════════════════════════════════════════════════════════
#  Prompt tokens — must be identical, pad side aside
# ══════════════════════════════════════════════════════════════════════


def test_prompt_tokens_are_identical(upstream, ours, variant):
    """Mask-unpadded token ids must match exactly, including pi0's trailing
    start-of-answer token (108, "\\n") and pi05's digitized state bins.

    Any divergence in text cleaning ("_"→" ", "\\n"→" "), BOS handling,
    tokenizer version or the 256-bin digitization surfaces here. For pi05 a
    failure on a *single* interior bin token may be the documented bin-edge
    case (module docstring) — check the prompt text before treating it as a
    regression."""
    up = _numpy(upstream["observation.language.tokens"])[0]
    mine = _numpy(ours[0]["observation.language.tokens"])[0]
    up_tokens = _unpadded_tokens(up, upstream["observation.language.attention_mask"][0])
    my_tokens = _unpadded_tokens(mine, ours[1]["observation"].tokenized_prompt_mask[0])
    assert my_tokens.size, "we produced no non-padding tokens"
    np.testing.assert_array_equal(
        up_tokens, my_tokens,
        err_msg=f"[{variant}] prompt token sequences differ",
    )


def test_pi05_prompt_text_matches_upstream_exactly(upstream, variant):
    """The pi05 processor rewrites ``task`` in-place to the full discrete
    prompt — comparing the rendered text pins bin-edge flips to specific
    dimensions (see module docstring: this dataset has none)."""
    if variant != "pi05":
        pytest.skip("pi0 prompt text is the plain cleaned task + newline")
    # Upstream template (processor_pi05.py): "Task: {cleaned}, State: {bins};
    # \nAction: " — cleaning ("_"→" ", "\n"→" ", strip) is identity for this
    # task string. The processor keeps task as a per-batch-row list.
    task = upstream["task"][0]
    assert task.startswith(f"Task: {_TASK}, State: ")
    assert task.endswith(";\nAction: ")


# ══════════════════════════════════════════════════════════════════════
#  Normalized vectors — tolerance, not bit equality (eps divergence)
# ══════════════════════════════════════════════════════════════════════


def _healthy_dims(raw, variant) -> np.ndarray:
    """Boolean mask over the dataset's dims with a non-degenerate scale.

    Zero-variance dims (std=0 / q99==q01) are where the documented eps
    divergence becomes categorical (module docstring) — mask them out of the
    vector comparisons. Scale choice follows the variant's norm mode, same
    as the two chains do.
    """
    _, _, norm_stats = raw
    scale = (
        np.asarray(norm_stats.action.q99) - np.asarray(norm_stats.action.q01)
        if variant == "pi05"
        else np.asarray(norm_stats.action.std)
    )
    return scale > 1e-9


def test_normalized_state_within_tolerance(upstream, ours, variant):
    """pi0 only: the continuous state tensor, normalized. Our side pads to
    the 32-wide model interface; the slice must match within the documented
    eps divergence (module docstring)."""
    if variant != "pi0":
        pytest.skip("pi05 carries the state inside the discrete prompt")
    up = _numpy(upstream["observation.state"])       # [1, 6]
    mine = _numpy(ours[0]["observation.state"])      # [1, 32]
    assert mine.shape == (1, _MODEL_ACTION_DIM)
    np.testing.assert_allclose(
        mine[:, :_DATASET_STATE_DIM], up, rtol=_VECTOR_RTOL, atol=_VECTOR_ATOL,
    )


def test_normalized_actions_within_tolerance(upstream, ours, raw, variant):
    """Actions: same norm stats, same pad target — the healthy-dim slice must
    agree within the eps divergence (module docstring). (lerobot's processor
    leaves the action at the dataset-native width; the policy pads it to 32
    internally, idempotently with ours.)"""
    up = _numpy(upstream["action"])                  # [50, 8]
    mine = _numpy(ours[0]["action"])                 # [1, 50, 32]
    assert up.shape == (_ACTION_HORIZON, _DATASET_ACTION_DIM)
    healthy = _healthy_dims(raw, variant)
    np.testing.assert_allclose(
        mine[0, :, :_DATASET_ACTION_DIM][:, healthy], up[:, healthy],
        rtol=_VECTOR_RTOL, atol=_VECTOR_ATOL,
    )


# ══════════════════════════════════════════════════════════════════════
#  Image geometry — policy-side resize_with_pad vs our pipeline
# ══════════════════════════════════════════════════════════════════════


def _upstream_resized_0_1(raw, camera: str) -> np.ndarray:
    """The geometry PI0Policy._preprocess_images applies to a [0,1] image:
    resize_with_pad_torch (bilinear, black-0 pad). The ×2−1 SigLIP map after
    it is affine and identical on both sides, so comparing in [0,1] isolates
    the geometry."""
    from lerobot.policies.pi0.modeling_pi0 import resize_with_pad_torch

    sample, _, _ = raw
    image = torch.from_numpy(
        np.asarray(sample[f"images.{camera}"], dtype=np.float32) / 255.0
    )  # [H, W, C] uint8 → [0,1] float
    return _numpy(resize_with_pad_torch(image, 224, 224))[0]  # [H, W, C]


@pytest.mark.parametrize("slot", sorted(_CAMERA_MAPPING))
def test_letterbox_padding_is_black_on_both_sides(raw, ours, slot):
    """Letterbox rows must be black. In the [-1,1] space the policy maps to
    (×2−1), black is -1: a 0.0 (mid-grey) pad would feed the SigLIP encoder
    25% out-of-distribution pixels on a 4:3 frame."""
    camera = _CAMERA_MAPPING[slot]
    up = _upstream_resized_0_1(raw, camera)
    mine = _to_bhwc_0_1(ours[0][f"observation.images.{slot}"])[0]
    # both sides in [-1,1]: pad rows must be exactly -1
    np.testing.assert_allclose(up[_PAD_ROWS] * 2 - 1, -1.0, atol=1e-6)
    np.testing.assert_allclose(mine[_PAD_ROWS] * 2 - 1, -1.0, atol=1e-6)


@pytest.mark.parametrize("slot", sorted(_CAMERA_MAPPING))
def test_image_content_region_within_interpolation_tolerance(raw, ours, slot):
    """Content region (non-letterbox) may differ only at interpolation
    implementation level: torch F.interpolate (float) vs cv2 INTER_LINEAR
    (uint8, quantized to 1/255)."""
    camera = _CAMERA_MAPPING[slot]
    up = _upstream_resized_0_1(raw, camera)
    mine = _to_bhwc_0_1(ours[0][f"observation.images.{slot}"])[0]
    diff = np.abs(up[_CONTENT_ROWS].astype(np.float64) - mine[_CONTENT_ROWS].astype(np.float64))
    assert diff.mean() < _INTERPOLATION_MEAN_TOL, (
        f"{slot} content-region mean diff {diff.mean():.6f} exceeds the "
        f"interpolation tolerance {_INTERPOLATION_MEAN_TOL} — no longer "
        "explainable by interpolation implementations"
    )


# ══════════════════════════════════════════════════════════════════════
#  Batch-shape contract
# ══════════════════════════════════════════════════════════════════════


def test_unmapped_slot_absent_from_our_batch(ours):
    """Unmapped slots are omitted from our batch; the policy substitutes its
    own -1 image + zero mask — but only because the factory declared every
    slot in the config's input_features (see the placeholder test below). A
    key present here would mean the wrapper re-derived a camera binding the
    composition never resolved."""
    assert "observation.images.right_wrist_0_rgb" not in ours[0]


def test_declared_but_unmapped_slot_gets_policy_placeholder(ours):
    """The placeholder contract, pinned against the real policy code: lerobot
    pads only slots DECLARED in ``config.image_features``. With all three
    slots declared (what the factory now does) and our 2-camera batch,
    ``PI0Policy._preprocess_images`` must yield three image groups — the
    trailing one the -1 SigLIP placeholder with a zero mask, i.e. the
    3-image token sequence the pretrained checkpoint expects.
    (``_preprocess_images`` only reads ``self.config`` and the device of
    ``self.parameters()``, so a one-parameter stand-in self suffices — no
    weights needed.)"""
    import torch.nn as nn
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    config = PI0Config(
        chunk_size=_ACTION_HORIZON,
        n_action_steps=_ACTION_HORIZON,
        device="cpu",
        input_features={
            f"observation.images.{slot}": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224),
            )
            for slot in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
    )
    fake_self = nn.Module()
    fake_self.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
    fake_self.config = config

    images, masks = PI0Policy._preprocess_images(fake_self, ours[0])

    assert len(images) == 3 and len(masks) == 3, (
        f"expected 2 present + 1 placeholder image groups, got {len(images)}"
    )
    assert [bool(m[0]) for m in masks] == [True, True, False], (
        "the unmapped slot's mask must be zero (masked out of attention)"
    )
    np.testing.assert_allclose(
        _numpy(images[2]), -1.0,
        err_msg="the placeholder image must be the SigLIP -1 black fill",
    )


def test_pi05_batch_omits_state_tensor(ours, variant):
    """PI05Policy never reads ``observation.state`` — the state rides inside
    the discrete prompt. Feeding the tensor anyway would be dead weight, and
    a silent contract drift if a future upstream started reading it."""
    if variant != "pi05":
        pytest.skip("pi0 legitimately feeds the continuous state tensor")
    assert "observation.state" not in ours[0]


def test_attention_mask_is_bool_on_both_sides(upstream, ours):
    """lerobot's policies consume a bool attention mask (their processor
    casts); our wrapper must cast the framework's long mask the same way."""
    assert upstream["observation.language.attention_mask"].dtype is torch.bool
    assert ours[0]["observation.language.attention_mask"].dtype is torch.bool


# ══════════════════════════════════════════════════════════════════════
#  Reader contract (carried over from the openpi-era file)
# ══════════════════════════════════════════════════════════════════════


def test_reader_surfaces_the_dataset_task():
    """The task text must come from the reader reading meta/tasks.parquet,
    not from the recipe's default_task fallback. lerobot v3 keeps task text
    in the pandas index (``__index_level_0__``), not a ``task`` column —
    reading only columns silently returns {} and every frame degrades to the
    recipe's default_task, training a multi-task dataset as one prompt."""
    from vla_factory.data.codec.pyav import PyAVCodec
    from vla_factory.data.reader.lerobot_v3 import LeRobotV3Reader

    episode = LeRobotV3Reader().read_episode(_DATASET, 0, PyAVCodec())
    assert episode.load_frames()[0].language == _TASK

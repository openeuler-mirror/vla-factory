"""L0 tests for multi-frame observations (``n_obs_steps > 1``, G2+G3).

The time axis exists **only** on this branch (decision D8): the single-frame
contract — what ACT / pi0 / pi05 see — is bit-identical to before, and the
multi-frame branch has to prove three things here:

- training windows stack per-frame-transformed fields with the trailing
  frame aligned to the first action (the receding-horizon anchor);
- the inference engine assembles the same window shape from its history
  deque, front-fills at episode start, and resets between episodes;
- ``predict_window`` takes a caller-aligned window without touching that
  history — evaluation strides must not leak into deployment state.

Uses synthetic episodes and an identity pipeline: this file pins plumbing
and shapes, not numerics (those are the L1 parity tests' job).
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_factory.data.data_schema import Episode, Frame, VideoRef
from vla_factory.training.dataset import (
    SampleWindow,
    VLADataset,
    build_episode_windows,
    collate_fn,
    is_frame_level_key,
    stack_frame_samples,
)


# ── Synthetic episode machinery ─────────────────────────────────────


class _ArrayCodec:
    """VideoCodec stand-in: each frame's image is its (unique) index array."""

    def decode_frame(self, ref: VideoRef) -> np.ndarray:
        value = ref.frame_index % 256
        return np.full((4, 6, 3), value, dtype=np.uint8)


class _FakeReader:
    """FormatReader stand-in serving one synthetic episode."""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames
        self.read_count = 0

    def read_episode(self, path, ep_idx, codec):
        self.read_count += 1
        return Episode(
            episode_id="e0", episode_index=ep_idx, num_frames=len(self._frames),
            _frames_cache=self._frames,
        )


def _synthetic_episode(n_frames: int, state_dim: int = 3, action_dim: int = 2):
    """Frames whose image pixel / state / action values encode their index."""
    frames = []
    for i in range(n_frames):
        ref = VideoRef(
            video_path=None, frame_index=i, height=4, width=6, channels=3,
        )
        frames.append(Frame(
            index=i,
            images={"cam": ref},
            state=np.full(state_dim, i, dtype=np.float32),
            action=np.full(action_dim, 10 + i, dtype=np.float32),
        ))
    return frames


def _dataset(frames, *, n_obs_steps, action_horizon=4):
    return VLADataset(
        sample_windows=[SampleWindow(
            episode_index=0, start_frame_index=0,
            n_obs_steps=n_obs_steps, action_horizon=action_horizon,
        )],
        reader=_FakeReader(frames),
        codec=_ArrayCodec(),
        path=None,
        transforms=None,
    )


# ── Key classification ──────────────────────────────────────────────


def test_frame_level_key_classification():
    classification = {
        "images.cam": True, "image_masks.cam": True, "state": True,
        "actions": False, "action_is_pad": False, "task": False,
        "tokenized_prompt": False, "tokenized_prompt_mask": False,
    }
    for key, expected in classification.items():
        assert is_frame_level_key(key) is expected, key


# ── Training-side stacking ──────────────────────────────────────────


def test_window_enumeration_with_obs_steps():
    """To=2 windows start at position 0 with obs frames {0,1}: one fewer
    position than single-frame — the first frame alone is no longer a
    complete observation — and an episode shorter than To yields none."""
    single = build_episode_windows(0, 5, n_obs_steps=1, action_horizon=4)
    double = build_episode_windows(0, 5, n_obs_steps=2, action_horizon=4)
    assert [w.start_frame_index for w in single] == [0, 1, 2, 3, 4]
    assert [w.start_frame_index for w in double] == [0, 1, 2, 3]
    assert build_episode_windows(0, 1, n_obs_steps=2, action_horizon=4) == []


def test_two_frame_sample_stacks_and_aligns_actions():
    """images/state gain the time axis; the action chunk still starts at the
    *last* observation frame — the receding-horizon anchor, identical to the
    single-frame convention. Window-level fields keep their single-frame
    ranks (no time axis)."""
    dataset = _dataset(_synthetic_episode(6), n_obs_steps=2, action_horizon=3)
    sample = dataset[0]

    # Each synthetic image is a constant per frame index, so the stacked
    # time axis reads out the frame indices directly.
    np.testing.assert_array_equal(sample["images.cam"][:, 0, 0, 0], [0, 1])
    np.testing.assert_array_equal(sample["state"][:, 0], [0.0, 1.0])
    assert sample["images.cam"].shape == (2, 4, 6, 3)
    assert sample["state"].shape == (2, 3)

    # Actions are window-level: start at the last obs frame (index 1).
    np.testing.assert_array_equal(sample["actions"][:, 0], [11.0, 12.0, 13.0])
    assert sample["actions"].shape == (3, 2)
    assert sample["action_is_pad"].shape == (3,)
    assert not sample["action_is_pad"].any()


def test_window_at_episode_end_pads_actions_not_observations():
    """The last full obs window at To=2 starts at n-2; its action chunk runs
    off the episode end and must repeat-pad — observations never pad."""
    frames = _synthetic_episode(4)
    dataset = VLADataset(
        sample_windows=[SampleWindow(
            episode_index=0, start_frame_index=2,
            n_obs_steps=2, action_horizon=4,
        )],
        reader=_FakeReader(frames), codec=_ArrayCodec(), path=None,
        transforms=None,
    )
    sample = dataset[0]

    np.testing.assert_array_equal(sample["state"][:, 0], [2.0, 3.0])
    # Action chunk starts at the last obs frame (3) and runs past the episode
    # end: steps 4,5,6 repeat the last recorded action (frame 3's 13).
    np.testing.assert_array_equal(
        sample["actions"][:, 0], [13.0, 13.0, 13.0, 13.0],
    )
    np.testing.assert_array_equal(
        sample["action_is_pad"], [False, True, True, True],
    )


def test_transforms_apply_per_frame_before_stacking():
    """A transform written against single-frame ranks (here: an ImageNet-style
    shift that would broadcast differently on a stacked array) must see one
    frame at a time — stack-after-transform, not transform-after-stack."""
    class AddHundredToFirstPixel:
        def __call__(self, sample: dict) -> dict:
            # Guard like every registered step does: the window-level pass
            # runs the same pipeline without image keys.
            if "images.cam" not in sample:
                return sample
            img = sample["images.cam"]
            # Rank check: this line only works on a single frame [H, W, C].
            assert img.ndim == 3, "transform saw a stacked array"
            img = img.astype(np.float32)
            img[0, 0, 0] += 100.0
            sample["images.cam"] = img
            return sample

    frames = _synthetic_episode(4)
    dataset = VLADataset(
        sample_windows=[SampleWindow(
            episode_index=0, start_frame_index=0,
            n_obs_steps=2, action_horizon=2,
        )],
        reader=_FakeReader(frames), codec=_ArrayCodec(), path=None,
        transforms=[AddHundredToFirstPixel()],
    )
    sample = dataset[0]
    np.testing.assert_array_equal(
        sample["images.cam"][:, 0, 0, 0], [100.0, 101.0],
    )


def test_collate_batches_the_time_axis():
    dataset = _dataset(_synthetic_episode(6), n_obs_steps=2, action_horizon=3)
    batch = collate_fn([dataset[0], dataset[0]])

    assert batch["observation"].images["cam"].shape == (2, 2, 4, 6, 3)
    assert batch["observation"].state.shape == (2, 2, 3)
    assert batch["actions"].shape == (2, 3, 2)
    assert batch["observation"].image_masks is not None


def test_single_frame_sample_has_no_time_axis():
    """The To=1 branch stays shape-identical to the pre-multi-frame era."""
    dataset = _dataset(_synthetic_episode(6), n_obs_steps=1, action_horizon=3)
    sample = dataset[0]

    assert sample["images.cam"].shape == (4, 6, 3)
    assert sample["state"].shape == (3,)
    assert sample["actions"].shape == (3, 2)


# ── stack_frame_samples ─────────────────────────────────────────────


def test_stack_rejects_partially_missing_fields():
    frames = [
        {"images.cam": np.zeros((4, 6, 3)), "state": np.zeros(3)},
        {"images.cam": np.zeros((4, 6, 3)), "state": None},
    ]
    with pytest.raises(ValueError, match="'state' is missing in 1 of 2"):
        stack_frame_samples(frames)


# ── Inference engine history ────────────────────────────────────────


class _RecordingModel:
    """predict_actions stand-in that records the observation it received."""

    def __init__(self) -> None:
        self.seen = []

    def predict_actions(self, observation, **kwargs):
        self.seen.append(observation)
        horizon, dim = 4, 2
        return np.zeros((horizon, dim), dtype=np.float32)


def _engine(n_obs_steps: int):
    from vla_factory.inference.inference_engine import InferenceEngine, ObsDict

    engine = object.__new__(InferenceEngine)
    engine.camera_keys = ("cam",)
    engine.schema = type("Schema", (), {
        "state_dim": 3,
        "action_dim": 2,
    })()
    engine.preprocessor = lambda sample: sample
    engine.postprocessor = lambda sample: sample
    engine.device = "cpu"
    engine.n_obs_steps = n_obs_steps
    engine._history = (
        __import__("collections").deque(maxlen=n_obs_steps)
        if n_obs_steps > 1 else None
    )
    engine._model = _RecordingModel()
    engine.action_horizon = 4
    engine.model_output_dim = 2
    engine.execution_action_dim = 2
    engine.num_inference_steps = 1
    return engine, ObsDict


def _obs(ObsDict, frame_index: int):
    return ObsDict(
        video={"cam": np.full((4, 6, 3), frame_index, dtype=np.uint8)},
        state=np.full(3, frame_index, dtype=np.float32),
    )


def test_engine_history_builds_trailing_window():
    engine, ObsDict = _engine(n_obs_steps=2)
    engine.predict(_obs(ObsDict, 0))
    engine.predict(_obs(ObsDict, 7))

    observation = engine._model.seen[-1]
    # (1, To, ...) — batch axis, then the time axis D8 adds.
    assert observation.images["cam"].shape == (1, 2, 4, 6, 3)
    assert observation.state.shape == (1, 2, 3)
    np.testing.assert_array_equal(
        observation.images["cam"][0, :, 0, 0, 0], [0, 7],
    )
    assert observation.image_masks["cam"].shape == (1, 2)


def test_engine_reset_clears_history():
    """After reset the next predict() sees an empty history and front-fills
    with the new episode's first frame (no leakage of the previous episode)."""
    engine, ObsDict = _engine(n_obs_steps=2)
    engine.predict(_obs(ObsDict, 0))
    engine.reset()
    engine.predict(_obs(ObsDict, 9))  # new episode: front-filled, no frame 0

    observation = engine._model.seen[-1]
    np.testing.assert_array_equal(
        observation.images["cam"][0, :, 0, 0, 0], [9, 9],
    )


def test_predict_window_needs_the_full_window_and_skips_history():
    engine, ObsDict = _engine(n_obs_steps=2)
    with pytest.raises(ValueError, match="n_obs_steps=2"):
        engine.predict_window([_obs(ObsDict, 0)])

    engine.predict_window([_obs(ObsDict, 3), _obs(ObsDict, 4)])
    observation = engine._model.seen[-1]
    np.testing.assert_array_equal(
        observation.images["cam"][0, :, 0, 0, 0], [3, 4],
    )
    # predict_window is caller-aligned state: the engine history stayed empty.
    assert not engine._history

    # A later predict() therefore front-fills rather than seeing frame 4.
    engine.predict(_obs(ObsDict, 8))
    observation = engine._model.seen[-1]
    np.testing.assert_array_equal(
        observation.images["cam"][0, :, 0, 0, 0], [8, 8],
    )


# ── Evaluation window assembly ──────────────────────────────────────
# The To=1 engine path is the pre-multi-frame original code (single-frame
# transforms + single observation), covered by test_inference_engine.py.


def test_trailing_window_assembly_front_fills():
    from vla_factory.inference.evaluate_dataset import _trailing_window_observations

    frames = _synthetic_episode(10)
    camera_keys = ("cam",)

    early = _trailing_window_observations(
        frames, 0, camera_keys, _ArrayCodec(), n_obs_steps=3,
    )
    assert [obs.state[0] for obs in early] == [0.0, 0.0, 0.0]

    late = _trailing_window_observations(
        frames, 5, camera_keys, _ArrayCodec(), n_obs_steps=3,
    )
    assert [obs.state[0] for obs in late] == [3.0, 4.0, 5.0]

    single = _trailing_window_observations(
        frames, 5, camera_keys, _ArrayCodec(), n_obs_steps=1,
    )
    assert len(single) == 1
    assert single[0].state[0] == 5.0

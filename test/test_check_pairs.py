"""Tests for the resolver's explicit compatibility checks.

Only facts sharing an explicit vocabulary are compared: vector dimensions,
data cameras to model slots, control modes, and normalization statistics.
Robot camera/joint names are deliberately not matched to DataSchema names.

Two kinds of coverage:

* **Golden tests** (``TestGoldenRealData``) — the real 3-episode LeKiwi test
  dataset against the real ``act``/``pi0`` registry entries and the real
  ``lekiwi`` robot profile. These are the cases this phase was actually
  designed against (see the plan doc's "摸底" section) — no synthetic stand-in
  can substitute for them.
* **Unit tests** (the rest) — synthetic ``ModelMetadata``/``DataSchema``/
  ``RobotProfile``/``NormStats`` for failure branches the real fixture cannot
  reach (e.g. a ``fixed`` dim policy — no shipped model declares one).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from helpers import make_schema

from vla_factory.assembly.resolve import ResolutionError, resolve_from_facts as resolve_assembly
from vla_factory.assembly.resolve.errors import (
    ACTION_DIM_INCOMPATIBLE,
    CAMERA_SLOT_AMBIGUOUS,
    CAMERA_SLOT_UNRESOLVED,
    CONTROL_MODE_INCOMPATIBLE,
    ERROR_PARAMS,
    NORM_STATS_INSUFFICIENT,
    STATE_DIM_INCOMPATIBLE,
)
from vla_factory.assembly.resolve.checks import (
    _check_action_dim,
    _check_control_mode,
    _check_norm_stats,
    _check_state_dim,
)
from vla_factory.assembly.resolve.mappings import resolve_camera_mapping
from vla_factory.data.data_schema import ActionDim, CameraEntry, DataSchema, FeatureStats, NormStats
from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from vla_factory.robot import GripperConvention, JointGroup, RobotProfile

DATASET_PATH = _project_root / "test/data" / "lerobot_train_data_3_episodes"


def _stats(mean=(0.0,), std=(1.0,), q01=(), q99=()) -> NormStats:
    fs = FeatureStats(mean=list(mean), std=list(std), q01=list(q01), q99=list(q99))
    return NormStats(state=fs, action=fs, method="zscore")


# ── Golden tests: real dataset × real registry entries × real robot ──


class TestGoldenRealData:

    @staticmethod
    @pytest.fixture(scope="class")
    def schema():
        if not DATASET_PATH.exists():
            pytest.skip("test dataset not found")
        from vla_factory.data.reader.lerobot_v3 import LeRobotV3Reader
        return LeRobotV3Reader().get_schema(DATASET_PATH)

    @staticmethod
    @pytest.fixture(scope="class")
    def norm_stats():
        if not DATASET_PATH.exists():
            pytest.skip("test dataset not found")
        from vla_factory.data.reader.lerobot_v3 import LeRobotV3Reader
        return LeRobotV3Reader().get_norm_stats(DATASET_PATH)

    @staticmethod
    @pytest.fixture(scope="class")
    def lekiwi():
        from vla_factory.robot import get_robot_profile
        return get_robot_profile("lekiwi")

    def _metadata(self, name: str) -> ModelMetadata:
        from vla_factory.model.registry import list_entries
        return list_entries()[name]

    def test_act_no_robot_succeeds(self, schema, norm_stats):
        """No example recipe declares a robot today (verified against
        examples/*.yaml) — this is the resolve shape every real recipe
        currently exercises."""
        assembly = resolve_assembly(schema, norm_stats, self._metadata("act"))
        assert assembly.model_io_spec.action_dim == 8
        assert assembly.model_io_spec.state_dim == 6

    def test_pi0_no_robot_succeeds_and_exercises_camera_slots(self, schema, norm_stats):
        """pi0 has real vision_slots (unlike ACT's empty tuple), so this is
        the one golden case that actually runs the camera-slot check: the
        dataset's ``front``/``wrist`` cameras resolve to ``third_person_front``/
        ``wrist`` and each of pi0's three slots gets exactly one candidate."""
        assembly = resolve_assembly(schema, norm_stats, self._metadata("pi0"))
        assert assembly.model_io_spec.cameras == ("front", "wrist")

    @pytest.mark.parametrize("model_name", ["act", "pi0"])
    def test_robot_joint_names_do_not_gate_data_model_resolution(
        self, schema, norm_stats, lekiwi, model_name,
    ):
        """Generic data names and physical robot names are separate namespaces."""
        assembly = resolve_assembly(
            schema, norm_stats, self._metadata(model_name), robot_profile=lekiwi,
        )
        assert assembly.robot_ref["name"] == "lekiwi"


# ── State / action dim ────────────────────────────────────────────


class TestDimChecks:

    def test_state_dim_flexible_always_passes(self):
        schema = make_schema(state_dim=6)
        _check_state_dim(schema, ModelMetadata(name="stub", dim_policy="flexible"))

    def test_state_dim_fixed_mismatch_raises(self):
        schema = make_schema(state_dim=6)
        meta = ModelMetadata(name="stub", dim_policy="fixed", dim_policy_max=8)
        with pytest.raises(ResolutionError) as exc:
            _check_state_dim(schema, meta)
        assert exc.value.code == STATE_DIM_INCOMPATIBLE
        assert exc.value.params == {
            "field": "state", "data_dim": 6, "limit": 8, "limit_source": "metadata.dim_policy_max",
        }
        assert set(exc.value.params) == ERROR_PARAMS[STATE_DIM_INCOMPATIBLE]

    def test_state_dim_padded_to_max_allows_smaller(self):
        schema = make_schema(state_dim=6)
        meta = ModelMetadata(name="stub", dim_policy="padded_to_max", dim_policy_max=32)
        _check_state_dim(schema, meta)  # must not raise — 6 <= 32

    def test_state_dim_padded_to_max_rejects_larger(self):
        schema = make_schema(state_dim=40)
        meta = ModelMetadata(name="stub", dim_policy="padded_to_max", dim_policy_max=32)
        with pytest.raises(ResolutionError) as exc:
            _check_state_dim(schema, meta)
        assert exc.value.code == STATE_DIM_INCOMPATIBLE

    def test_action_dim_model_cap_exceeded(self):
        schema = make_schema(action_dim=40)
        meta = ModelMetadata(name="stub", dim_policy="padded_to_max", dim_policy_max=32)
        with pytest.raises(ResolutionError) as exc:
            _check_action_dim(schema, meta)
        assert exc.value.code == ACTION_DIM_INCOMPATIBLE
        assert set(exc.value.params) == ERROR_PARAMS[ACTION_DIM_INCOMPATIBLE]
        assert exc.value.params["limit_source"] == "metadata"

    def test_robot_joint_count_is_not_an_action_width_contract(self):
        """A profile does not declare the complete command-vector layout."""
        schema = make_schema(action_dim=6)
        meta = ModelMetadata(name="stub", dim_policy="flexible")
        _check_action_dim(schema, meta)


# ── Camera slots ───────────────────────────────────────────────────


class TestCameraSlots:

    def _schema_with_cameras(self, *entries: tuple[str, str]) -> DataSchema:
        cams = tuple(CameraEntry(key=k, semantic=s, semantic_source="inferred") for k, s in entries)
        return DataSchema(cameras_entries=cams)

    def test_no_declared_slots_is_a_noop(self):
        schema = self._schema_with_cameras(("front", "third_person_front"))
        meta = ModelMetadata(name="stub")  # vision_slots=() default (ACT-style)
        resolve_camera_mapping(schema, meta, {})

    def test_unique_match_passes(self):
        schema = self._schema_with_cameras(("front", "third_person_front"), ("wrist", "wrist"))
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",)),
        ))
        resolve_camera_mapping(schema, meta, {})

    def test_ambiguous_candidates_raise(self):
        """Two cameras both satisfy the same generalized slot."""
        schema = self._schema_with_cameras(
            ("cam_a", "third_person_front"), ("cam_b", "third_person_top"),
        )
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",)),
        ))
        with pytest.raises(ResolutionError) as exc:
            resolve_camera_mapping(schema, meta, {})
        assert exc.value.code == CAMERA_SLOT_AMBIGUOUS
        assert set(exc.value.params["candidates"]) == {"cam_a", "cam_b"}

    def test_unresolved_required_slot_with_zero_pad_is_not_an_error(self):
        """The default (and every shipped entry's) missing_slot_policy."""
        schema = self._schema_with_cameras()
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",), required=True),
        ), missing_slot_policy="zero_pad")
        resolve_camera_mapping(schema, meta, {})

    def test_unresolved_required_slot_with_error_policy_raises(self):
        schema = self._schema_with_cameras()
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",), required=True),
        ), missing_slot_policy="error")
        with pytest.raises(ResolutionError) as exc:
            resolve_camera_mapping(schema, meta, {})
        assert exc.value.code == CAMERA_SLOT_UNRESOLVED

    def test_robot_camera_names_are_outside_this_check(self):
        """Only DataSchema cameras participate in data-to-model mapping."""
        schema = self._schema_with_cameras()  # no data cameras at all
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="head", semantic_accepts=("third_person",), required=True),
        ), missing_slot_policy="zero_pad")
        resolve_camera_mapping(schema, meta, {})


# ── Control mode ───────────────────────────────────────────────────


class TestControlMode:

    def test_no_declared_modes_is_a_noop(self):
        schema = DataSchema(action_dims=(ActionDim(name="a", mode=None),))
        _check_control_mode(schema, ModelMetadata(name="stub"), None)

    def test_model_accepts_data_mode_passes(self):
        schema = DataSchema(action_dims=(ActionDim(name="a", mode="joint_pos"),))
        meta = ModelMetadata(name="stub", control_mode_pref=("joint_pos",))
        _check_control_mode(schema, meta, None)

    def test_model_rejects_data_mode_raises(self):
        schema = DataSchema(action_dims=(ActionDim(name="a", mode="joint_delta"),))
        meta = ModelMetadata(name="stub", control_mode_pref=("joint_pos",))
        with pytest.raises(ResolutionError) as exc:
            _check_control_mode(schema, meta, None)
        assert exc.value.code == CONTROL_MODE_INCOMPATIBLE
        assert exc.value.params["data_modes"] == ["joint_delta"]

    def test_robot_rejects_data_mode_even_if_model_accepts(self):
        schema = DataSchema(action_dims=(ActionDim(name="a", mode="joint_vel"),))
        meta = ModelMetadata(name="stub", control_mode_pref=("joint_pos", "joint_vel"))
        robot = RobotProfile(name="stub", joints=JointGroup(names=("j1",)),
                              control_modes=("joint_pos",))
        with pytest.raises(ResolutionError) as exc:
            _check_control_mode(schema, meta, robot)
        assert exc.value.code == CONTROL_MODE_INCOMPATIBLE


# ── Norm stats ─────────────────────────────────────────────────────


class TestNormStats:

    def test_no_declared_method_is_a_noop(self):
        schema = make_schema(action_dim=4)
        _check_norm_stats(schema, NormStats(), ModelMetadata(name="stub"))

    def test_mean_std_satisfied_passes(self):
        schema = make_schema(action_dim=4)
        _check_norm_stats(schema, _stats(mean=[0.0] * 4, std=[1.0] * 4),
                          ModelMetadata(name="stub", vector_normalization="mean_std"))

    def test_mean_std_missing_raises(self):
        schema = make_schema(action_dim=4)
        empty = NormStats(state=None, action=None)
        with pytest.raises(ResolutionError) as exc:
            _check_norm_stats(schema, empty, ModelMetadata(name="stub", vector_normalization="mean_std"))
        assert exc.value.code == NORM_STATS_INSUFFICIENT
        assert set(exc.value.params["missing"]) == {"mean", "std"}

    def test_quantile_missing_raises(self):
        """The exact pi05 scenario: mean/std present, q01/q99 not."""
        schema = make_schema(action_dim=4)
        stats = _stats(mean=[0.0] * 4, std=[1.0] * 4)  # no q01/q99
        with pytest.raises(ResolutionError) as exc:
            _check_norm_stats(schema, stats, ModelMetadata(name="stub", vector_normalization="quantile"))
        assert exc.value.code == NORM_STATS_INSUFFICIENT
        assert set(exc.value.params["missing"]) == {"q01", "q99"}

"""Tests for the phase-2 Check Pairs stage (architecture §7.4 / §4.2.2).

Six matrix rows only — state dim, action dim, camera slots, control mode, norm
stats, joint order — matching the subset architecture §7.4 names explicitly
for phase 2. See ``docs/plans/phase2-resolution-diagnostics.cn.md`` for why
language / gripper / rotation / frequency / safety are out of scope.

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

from vla_factory.assembly.resolver import ResolutionError, resolve_assembly
from vla_factory.assembly.resolver.errors import (
    ACTION_DIM_INCOMPATIBLE,
    CAMERA_SLOT_AMBIGUOUS,
    CAMERA_SLOT_UNRESOLVED,
    CONTROL_MODE_INCOMPATIBLE,
    JOINT_ORDER_AMBIGUOUS,
    JOINT_ORDER_MISMATCH,
    NORM_STATS_INSUFFICIENT,
    STATE_DIM_INCOMPATIBLE,
)
from vla_factory.assembly.resolver.resolver import (
    _check_action_dim,
    _check_camera_slots,
    _check_control_mode,
    _check_joint_order,
    _check_norm_stats,
    _check_state_dim,
)
from vla_factory.data.manifest import ActionDim, CameraEntry, DataSchema, FeatureStats, NormStats, StateDim
from vla_factory.model.interfaces.model import ModelMetadata, VisionSlot
from vla_factory.robot.profile import GripperConvention, JointGroup, RobotProfile

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
        from vla_factory.data.formats.lerobot_v3 import LeRobotV3Reader
        return LeRobotV3Reader().get_schema(DATASET_PATH)

    @staticmethod
    @pytest.fixture(scope="class")
    def norm_stats():
        if not DATASET_PATH.exists():
            pytest.skip("test dataset not found")
        from vla_factory.data.formats.lerobot_v3 import LeRobotV3Reader
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
        assert assembly.canonical_interface.action_dim == 8
        assert assembly.canonical_interface.state_dim == 6

    def test_pi0_no_robot_succeeds_and_exercises_camera_slots(self, schema, norm_stats):
        """pi0 has real vision_slots (unlike ACT's empty tuple), so this is
        the one golden case that actually runs the camera-slot check: the
        dataset's ``front``/``wrist`` cameras resolve to ``third_person_front``/
        ``wrist`` and each of pi0's three slots gets exactly one candidate."""
        assembly = resolve_assembly(schema, norm_stats, self._metadata("pi0"))
        assert assembly.canonical_interface.cameras == ("front", "wrist")

    def test_act_with_lekiwi_robot_fails_on_action_joint_order(self, schema, norm_stats, lekiwi):
        """The state side (real names, ``shoulder_pan.pos`` etc.) uniquely
        embeds into LeKiwi's 9 joints; the action side (this fixture's own
        ``meta/info.json`` literally names its 8 action dims ``dim_0``..
        ``dim_7`` — verified by reading the file directly) matches none of
        them. This is exactly the case the plan doc's Context section flagged
        during scoping, not a contrived example."""
        with pytest.raises(ResolutionError) as exc:
            resolve_assembly(schema, norm_stats, self._metadata("act"), robot_profile=lekiwi)
        err = exc.value.to_dict()
        assert err["code"] == JOINT_ORDER_MISMATCH
        assert err["path"] == "schema.action.joint_order"
        assert set(err["params"]["unmatched_names"]) == {f"dim_{i}" for i in range(8)}

    def test_pi0_with_lekiwi_robot_fails_the_same_way(self, schema, norm_stats, lekiwi):
        """Same failure as ACT — the generic action names are a dataset
        property, independent of which model is being resolved against."""
        with pytest.raises(ResolutionError) as exc:
            resolve_assembly(schema, norm_stats, self._metadata("pi0"), robot_profile=lekiwi)
        assert exc.value.code == JOINT_ORDER_MISMATCH

    def test_state_side_alone_embeds_uniquely_into_lekiwi(self, schema, lekiwi):
        """Isolates the state-only half of the two tests above: real dim names
        with the ``.pos`` suffix (data-module §8.3 keeps it) must not be
        compared as raw strings against the robot's suffix-free joint names —
        decision D4's motivating bug. Calling the check function directly
        (rather than through resolve_assembly) proves state-side success is
        independent of the action-side failure, not incidentally masked by it."""
        _check_joint_order("state", schema.state_dims, lekiwi)  # must not raise


# ── State / action dim ────────────────────────────────────────────


class TestDimChecks:

    def test_state_dim_flexible_always_passes(self):
        schema = make_schema(state_dim=6)
        _check_state_dim(schema, ModelMetadata(name="stub", dim_policy="flexible"), {})

    def test_state_dim_fixed_mismatch_raises(self):
        schema = make_schema(state_dim=6)
        meta = ModelMetadata(name="stub", dim_policy="fixed", dim_policy_max=8)
        with pytest.raises(ResolutionError) as exc:
            _check_state_dim(schema, meta, {})
        assert exc.value.code == STATE_DIM_INCOMPATIBLE
        assert exc.value.params == {
            "field": "state", "data_dim": 6, "limit": 8, "limit_source": "metadata.dim_policy_max",
        }

    def test_state_dim_padded_to_max_allows_smaller(self):
        schema = make_schema(state_dim=6)
        meta = ModelMetadata(name="stub", dim_policy="padded_to_max", dim_policy_max=32)
        _check_state_dim(schema, meta, {})  # must not raise — 6 <= 32

    def test_state_dim_padded_to_max_rejects_larger(self):
        schema = make_schema(state_dim=40)
        meta = ModelMetadata(name="stub", dim_policy="padded_to_max", dim_policy_max=32)
        with pytest.raises(ResolutionError) as exc:
            _check_state_dim(schema, meta, {})
        assert exc.value.code == STATE_DIM_INCOMPATIBLE

    def test_state_dim_base_contract_refinement_is_authoritative(self):
        """A checkpoint's own reported state_dim overrides the family's
        dim_policy_max (Materialize already prefers it; Check Pairs must too,
        or the two stages could disagree)."""
        schema = make_schema(state_dim=10)
        meta = ModelMetadata(name="stub", dim_policy="padded_to_max", dim_policy_max=32)
        with pytest.raises(ResolutionError) as exc:
            _check_state_dim(schema, meta, {"state_dim_from_contract": 8})
        assert exc.value.params["limit_source"] == "base_contract.state_dim"

    def test_action_dim_model_cap_exceeded(self):
        schema = make_schema(action_dim=40)
        meta = ModelMetadata(name="stub", dim_policy="padded_to_max", dim_policy_max=32)
        with pytest.raises(ResolutionError) as exc:
            _check_action_dim(schema, meta, {"action_dim": 32}, None)
        assert exc.value.code == ACTION_DIM_INCOMPATIBLE
        assert exc.value.params["limit_source"] == "metadata"

    def test_action_dim_robot_joint_count_exceeded(self):
        """Model side is flexible (no cap) but the robot only has 3 joints —
        8 action dims physically cannot address them."""
        schema = make_schema(action_dim=8)
        meta = ModelMetadata(name="stub", dim_policy="flexible")
        robot = RobotProfile(name="tiny", joints=JointGroup(names=("j1", "j2", "j3")))
        with pytest.raises(ResolutionError) as exc:
            _check_action_dim(schema, meta, {}, robot)
        assert exc.value.code == ACTION_DIM_INCOMPATIBLE
        assert exc.value.params["limit_source"] == "robot.joints"

    def test_action_dim_robot_with_more_joints_than_data_is_fine(self):
        """The opposite direction (robot has MORE joints, e.g. LeKiwi's base
        absent from arm-only data) must not be flagged — see the docstring on
        ``_check_action_dim``."""
        schema = make_schema(action_dim=6)
        meta = ModelMetadata(name="stub", dim_policy="flexible")
        robot = RobotProfile(name="lekiwi_like", joints=JointGroup(names=tuple(f"j{i}" for i in range(9))))
        _check_action_dim(schema, meta, {}, robot)  # must not raise


# ── Camera slots ───────────────────────────────────────────────────


class TestCameraSlots:

    def _schema_with_cameras(self, *entries: tuple[str, str]) -> DataSchema:
        cams = tuple(CameraEntry(key=k, semantic=s, semantic_source="inferred") for k, s in entries)
        return DataSchema(cameras_entries=cams)

    def test_no_declared_slots_is_a_noop(self):
        schema = self._schema_with_cameras(("front", "third_person_front"))
        meta = ModelMetadata(name="stub")  # vision_slots=() default (ACT-style)
        _check_camera_slots(schema, meta, None)

    def test_unique_match_passes(self):
        schema = self._schema_with_cameras(("front", "third_person_front"), ("wrist", "wrist"))
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",)),
        ))
        _check_camera_slots(schema, meta, None)

    def test_ambiguous_candidates_raise(self):
        """Two cameras both satisfy the same generalized slot."""
        schema = self._schema_with_cameras(
            ("cam_a", "third_person_front"), ("cam_b", "third_person_top"),
        )
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",)),
        ))
        with pytest.raises(ResolutionError) as exc:
            _check_camera_slots(schema, meta, None)
        assert exc.value.code == CAMERA_SLOT_AMBIGUOUS
        assert set(exc.value.params["candidates"]) == {"cam_a", "cam_b"}

    def test_unresolved_required_slot_with_zero_pad_is_not_an_error(self):
        """The default (and every shipped entry's) missing_slot_policy."""
        schema = self._schema_with_cameras()
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",), required=True),
        ), missing_slot_policy="zero_pad")
        _check_camera_slots(schema, meta, None)

    def test_unresolved_required_slot_with_error_policy_raises(self):
        schema = self._schema_with_cameras()
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",), required=True),
        ), missing_slot_policy="error")
        with pytest.raises(ResolutionError) as exc:
            _check_camera_slots(schema, meta, None)
        assert exc.value.code == CAMERA_SLOT_UNRESOLVED

    def test_robot_camera_semantic_reused_not_redeclared(self):
        """D3: robot cameras reuse infer_camera_semantic() — no RobotProfile
        field changes needed. ``left_camera``/``right_camera`` (RoboTwin-style
        raw device names) resolve to nothing and simply don't participate."""
        schema = self._schema_with_cameras()  # no data cameras at all
        meta = ModelMetadata(name="stub", vision_slots=(
            VisionSlot(name="head", semantic_accepts=("third_person",), required=True),
        ), missing_slot_policy="zero_pad")
        robot = RobotProfile(name="robotwin_like", joints=JointGroup(names=("j1",)),
                              cameras=("head_camera", "left_camera", "right_camera"))
        # head_camera -> third_person_front (unique hit); left/right_camera
        # resolve to nothing and don't create ambiguity.
        _check_camera_slots(schema, meta, robot)


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


# ── Joint order ────────────────────────────────────────────────────


class TestJointOrder:

    def test_no_robot_is_a_noop(self):
        _check_joint_order("state", (StateDim(name="shoulder_pan.pos"),), None)

    def test_suffix_is_stripped_before_matching(self):
        """Decision D4's core fix — the bare bug this check exists to catch."""
        robot = RobotProfile(name="stub", joints=JointGroup(names=("shoulder_pan",)))
        _check_joint_order("state", (StateDim(name="shoulder_pan.pos"),), robot)  # must not raise

    def test_unmatched_name_raises_mismatch(self):
        robot = RobotProfile(name="stub", joints=JointGroup(names=("shoulder_pan",)))
        with pytest.raises(ResolutionError) as exc:
            _check_joint_order("action", (ActionDim(name="dim_0"),), robot)
        assert exc.value.code == JOINT_ORDER_MISMATCH

    def test_robot_superset_of_data_is_fine(self):
        """The robot may have joints the dataset never recorded (LeKiwi's
        mobile base) — that's a subset embedding, not a mismatch."""
        robot = RobotProfile(name="stub", joints=JointGroup(names=("base_x", "shoulder_pan")))
        _check_joint_order("state", (StateDim(name="shoulder_pan.pos"),), robot)

    def test_duplicate_stripped_names_raise_ambiguous(self):
        """Two data dims strip down to the same robot joint name."""
        robot = RobotProfile(name="stub", joints=JointGroup(names=("gripper",)))
        dims = (StateDim(name="gripper.pos"), StateDim(name="gripper.vel"))
        with pytest.raises(ResolutionError) as exc:
            _check_joint_order("state", dims, robot)
        assert exc.value.code == JOINT_ORDER_AMBIGUOUS
        assert exc.value.params["duplicate_names"] == ["gripper"]

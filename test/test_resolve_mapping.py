"""Tests for the phase-3 Resolve Mapping / Plan Pipeline stages (§7.4 phase 3).

Coverage mirrors phase 2's split:

* **Golden tests** (``TestGoldenRealData``) — the real 3-episode LeKiwi fixture
  against the real ``act`` / ``pi0`` / ``pi05`` registry entries. Expectations
  are written inline rather than in a side file so a reviewer reads the change
  and the expectation together.
* **Equivalence test** (``TestPlanMatchesBuiltPipeline``) — the planned
  ``data_to_model`` calls against the pipeline the *production* build path
  actually constructs for the same inputs. Nothing executes a plan yet, so
  without this the plans would be unverified data; with it, phase 4 has a
  judgement criterion for switching the downstream over.
* **Unit tests** — synthetic descriptions for branches the real fixture cannot
  reach (camera override failures, joint mapping against a robot).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from helpers import make_schema

from vla_factory.assembly import MappingSource
from vla_factory.assembly.resolve import (
    CAMERA_MAPPING_INVALID,
    ResolutionError,
    resolve_from_facts as resolve_assembly,
)
from vla_factory.data.data_schema import ActionDim, DataSchema, FeatureStats, NormStats
from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from vla_factory.user_interface import AssemblyOverrides

DATASET_PATH = _project_root / "test/data" / "lerobot_train_data_3_episodes"


def _metadata(name: str) -> ModelMetadata:
    from vla_factory.model.registry import list_entries
    return list_entries()[name]


def _calls(plan) -> list[tuple[str, dict]]:
    return [(c.type, c.args) for c in plan.calls]


# ── Golden: real fixture × real registry entries ──────────────────


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

    def test_act_mappings_and_plans(self, schema, norm_stats):
        """ACT declares no vision slots and no internal width: cameras map
        one-to-one onto the dataset's, nothing pads, and the declared
        no width reconciliation is needed because 8 → 8 is a no-op."""
        a = resolve_assembly(schema, norm_stats, _metadata("act"))

        assert a.camera_mapping.entries == (
            {
                "model_slot": "front", "data_source": "front",
                "source": MappingSource.INFERRED,
            },
            {
                "model_slot": "wrist", "data_source": "wrist",
                "source": MappingSource.INFERRED,
            },
        )
        assert len(a.state_mapping.entries) == 6
        assert a.state_mapping.entries[0] == {
            "model_index": 0, "data_dim_index": 0,
            "data_name": "shoulder_pan.pos",
            "source": MappingSource.INFERRED,
        }
        assert len(a.action_mapping.entries) == 8
        assert all("padded" not in e for e in a.action_mapping.entries)
        # requires_prompt=False → nothing to map.
        assert a.language_mapping.entries == ()

        assert _calls(a.data_to_model) == [
            ("image_to_float", {"range": [0.0, 1.0]}),
            ("image_layout", {"to": "CHW"}),
            ("image_normalize", {"mode": "imagenet"}),
            ("normalize_vector", {"fields": ["state", "actions"],
                                  "method": "zscore", "stats_ref": "norm_stats"}),
        ]
        assert _calls(a.model_to_robot) == [
            ("unnormalize_action", {"stats_ref": "norm_stats"}),
        ]

    def test_pi0_camera_override_is_the_complete_mapping(self, schema, norm_stats):
        """``examples/pi0_lora.yaml`` maps two of pi0's three slots and
        documents the third as intentionally unmapped. The override must
        therefore be complete: inferring ``wrist`` into ``right_wrist_0_rgb``
        would claim a camera ``adapters/openpi.py`` never passes (it hands unlisted
        roles a -1 placeholder + zero mask)."""
        overrides = AssemblyOverrides(camera_mapping={
            "base_0_rgb": "front", "left_wrist_0_rgb": "wrist",
        })
        a = resolve_assembly(schema, norm_stats, _metadata("pi0"), overrides=overrides)

        assert a.camera_mapping.entries == (
            {
                "model_slot": "base_0_rgb", "data_source": "front",
                "source": MappingSource.OVERRIDE,
            },
            {
                "model_slot": "left_wrist_0_rgb", "data_source": "wrist",
                "source": MappingSource.OVERRIDE,
            },
            {
                "model_slot": "right_wrist_0_rgb", "data_source": None,
                "source": MappingSource.INFERRED,
            },
        )

    def test_pi0_pads_both_vectors_to_32(self, schema, norm_stats):
        a = resolve_assembly(schema, norm_stats, _metadata("pi0"))

        # Mapping records the six/eight real relationships only. The 32-wide
        # model interface and its padding operation live in their own products.
        assert a.model_io_spec.state_dim == 32
        assert a.model_io_spec.action_dim == 32
        assert len(a.state_mapping.entries) == schema.state_dim == 6
        assert len(a.action_mapping.entries) == schema.action_dim == 8
        assert all("padded" not in e for e in a.state_mapping.entries)
        assert all("padded" not in e for e in a.action_mapping.entries)

        assert _calls(a.data_to_model) == [
            ("image_to_float", {"range": [-1.0, 1.0]}),
            ("image_layout", {"to": "CHW"}),
            ("resize_images", {"height": 224, "width": 224,
                               "mode": "pad", "interpolation": "bilinear"}),
            ("normalize_vector", {"fields": ["state", "actions"],
                                  "method": "zscore", "stats_ref": "norm_stats"}),
            ("pad_dimensions", {"target_dim": 32, "fields": ["state", "actions"]}),
            ("task_tokenize", {"max_length": 48, "discrete_state": False,
                               "tokenizer_repo": "google/paligemma-3b-pt-224"}),
        ]
        # Inverse of the two action-affecting calls, in reverse order — the
        # steps without an inverse (resize, tokenize, image ops) disappear.
        assert _calls(a.model_to_robot) == [
            ("unpad_action", {"target_dim": 8}),
            ("unnormalize_action", {"stats_ref": "norm_stats"}),
        ]

    def test_pi05_plans_quantile_inverse_and_state_tokenization_dependency(
        self, schema, norm_stats,
    ):
        """pi05 differs from pi0 in two ways the plan must carry: quantile
        normalization (so the inverse is the quantile one) and tokenizing
        *before* padding (the state must be digitized at its native width)."""
        a = resolve_assembly(schema, norm_stats, _metadata("pi05"))

        types = [t for t, _ in _calls(a.data_to_model)]
        assert types.index("task_tokenize") < types.index("pad_dimensions")
        normalize = dict(_calls(a.data_to_model))["normalize_vector"]
        assert normalize["method"] == "quantile"
        assert dict(_calls(a.data_to_model))["task_tokenize"] == {
            "max_length": 200, "discrete_state": True,
            "tokenizer_repo": "google/paligemma-3b-pt-224",
        }
        assert _calls(a.model_to_robot) == [
            ("unpad_action", {"target_dim": 8}),
            ("unnormalize_action_quantile", {"stats_ref": "norm_stats"}),
        ]

    def test_language_mapping_reads_the_dataset_task_field(self, schema, norm_stats):
        a = resolve_assembly(schema, norm_stats, _metadata("pi0"))
        assert a.language_mapping.entries == ({
            "model_field": "tokenized_prompt",
            "data_field": "task",
            "template": "{task}",
            "source": MappingSource.INFERRED,
        },)

    def test_resolution_is_deterministic(self, schema, norm_stats):
        first = resolve_assembly(schema, norm_stats, _metadata("pi0")).to_dict()
        second = resolve_assembly(schema, norm_stats, _metadata("pi0")).to_dict()
        assert first == second


# ── The planned calls are executable, and carry every argument ─────


class TestPlanIsExecutable:
    """A plan is only worth as much as the pipeline it instantiates.

    Phase 3 had to prove the plan matched a *second* implementation (the
    declaration-driven build path). That path is gone — there is one way to
    build a step now — so what is left to prove is that every planned call
    really constructs, with the arguments the plan states and nothing else
    filled in behind its back.
    """

    @staticmethod
    def _assembly_for(recipe_path: str):
        from vla_factory.assembly import resolve_assembly
        from vla_factory.user_interface import merge_model_config, parse_recipe

        recipe = merge_model_config(parse_recipe(str(_project_root / recipe_path)))
        recipe.data.path = str(DATASET_PATH)
        return recipe, resolve_assembly(recipe)

    @pytest.mark.parametrize("recipe_path", ["examples/act_lekiwi.yaml",
                                             "examples/pi0_lora.yaml"])
    def test_planned_calls_build_the_steps_they_describe(self, recipe_path):
        if not DATASET_PATH.exists():
            pytest.skip("test dataset not found")
        from vla_factory.assembly.transform import (
            TransformContext, TransformRegistry, build_pipeline,
        )

        _, assembly = self._assembly_for(recipe_path)
        plan = assembly.data_to_model

        pipeline = build_pipeline(
            plan, TransformContext(norm_stats=assembly.norm_stats),
        )
        built = list(pipeline.steps)
        assert [TransformRegistry.name_of(s) for s in built] == \
               [c.type for c in plan.calls]

        for step, call in zip(built, plan.calls):
            for key, planned in call.args.items():
                if key == "stats_ref":
                    continue    # a reference into the assembly, not a ctor arg
                actual = getattr(step, key)
                if isinstance(actual, tuple):
                    actual = list(actual)
                assert actual == planned, f"{call.type}.{key}"

    @pytest.mark.parametrize("recipe_path", ["examples/act_lekiwi.yaml",
                                             "examples/pi0_lora.yaml"])
    def test_reverse_plan_is_not_the_forward_list_reversed(self, recipe_path):
        """``model_to_robot`` keeps only the steps that declared an inverse."""
        if not DATASET_PATH.exists():
            pytest.skip("test dataset not found")
        from vla_factory.assembly.transform import TransformContext, build_pipeline

        _, assembly = self._assembly_for(recipe_path)
        forward = [c.type for c in assembly.data_to_model.calls]
        reverse = [c.type for c in assembly.model_to_robot.calls]

        assert reverse and reverse != list(reversed(forward))
        assert "unnormalize_action" in reverse
        # Image steps are input-only: no inverse, so they simply disappear.
        assert not any(c.startswith("image_") for c in reverse)
        # And the reverse plan instantiates too.
        build_pipeline(
            assembly.model_to_robot,
            TransformContext(norm_stats=assembly.norm_stats),
        )


# ── Unit: branches the real fixture cannot reach ──────────────────


def _pi0_like() -> ModelMetadata:
    return ModelMetadata(
        name="stub", action_dim=32, action_horizon=50,
        dim_policy="padded_to_max", dim_policy_max=32,
        vector_normalization="mean_std", image_input_range=(-1.0, 1.0),
        vision_slots=(
            VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",)),
        ),
    )


def _schema_with_cameras() -> DataSchema:
    return make_schema(state_dim=6, action_dim=6, cameras=("front", "wrist"))


def _usable_stats(dim: int = 6) -> NormStats:
    """Stats good enough to get past the phase-2 norm-stats check, so these
    tests fail on the thing they are about."""
    fs = FeatureStats(mean=[0.0] * dim, std=[1.0] * dim)
    return NormStats(state=fs, action=fs)


def test_camera_override_naming_an_unknown_slot_fails():
    with pytest.raises(ResolutionError) as exc:
        resolve_assembly(
            _schema_with_cameras(), _usable_stats(), _pi0_like(),
            overrides=AssemblyOverrides(camera_mapping={"nose_cam": "front"}),
        )
    err = exc.value.to_dict()
    assert err["code"] == CAMERA_MAPPING_INVALID
    assert err["path"] == "assembly.camera_mapping.nose_cam"
    assert err["params"] == {"field": "slot", "requested": "nose_cam",
                             "known": ["base_0_rgb"]}


def test_camera_override_naming_an_unknown_camera_fails():
    """The failure this code exists for: without it a typo silently degrades to
    slot padding and the model trains on a placeholder image."""
    with pytest.raises(ResolutionError) as exc:
        resolve_assembly(
            _schema_with_cameras(), _usable_stats(), _pi0_like(),
            overrides=AssemblyOverrides(camera_mapping={"base_0_rgb": "frotn"}),
        )
    err = exc.value.to_dict()
    assert err["code"] == CAMERA_MAPPING_INVALID
    assert err["params"] == {"field": "camera", "requested": "frotn",
                             "known": ["front", "wrist"]}


def test_model_io_width_generates_missing_padding_call():
    """The resolved interface directly drives width reconciliation."""
    metadata = ModelMetadata(
        name="stub", action_dim=32, action_horizon=50,
        dim_policy="padded_to_max", dim_policy_max=32,
        vector_normalization="mean_std", requires_prompt=False,
    )
    assembly = resolve_assembly(_schema_with_cameras(), _usable_stats(), metadata)
    assert assembly.model_io_spec.action_dim == 32
    assert _calls(assembly.data_to_model)[-1] == (
        "pad_dimensions", {"target_dim": 32, "fields": ["state", "actions"]},
    )


def test_transform_declaration_in_metadata_params_is_rejected():
    """A model entry cannot reopen the removed per-run transform surface."""
    metadata = ModelMetadata(
        name="stub", action_horizon=1, requires_prompt=False,
        vector_normalization="mean_std",
        params={"transforms": {"inputs": []}},
    )
    with pytest.raises(ValueError, match=r"params\['transforms'\]"):
        resolve_assembly(_schema_with_cameras(), _usable_stats(), metadata)


def test_a_planned_inverse_is_always_buildable():
    """Whatever ``inverse_call`` names must be a registered step that accepts
    the args it hands over — the plan is the only description of the reverse
    path now, so an inverse that cannot be built is a silently missing step in
    the deployed postprocessor."""
    from vla_factory.assembly.transform import (
        TransformContext, TransformRegistry, build_pipeline,
    )
    from vla_factory.assembly.transform.base import PlanContext, TransformStep
    from vla_factory.assembly.transform.plan import (
        TransformPipelinePlan, TransformStepCall,
    )

    @TransformRegistry.register("_probe_inverse")
    class Probe(TransformStep):
        def __init__(self, k: int = 1):
            self.k = k

        def __call__(self, sample):
            return sample

        @classmethod
        def compile_call(cls, cfg, ctx):
            return {"k": 7}

        @classmethod
        def inverse_call(cls, args, ctx):
            return ("unpad_action", {"target_dim": args["k"]}) if args else None

    try:
        args = Probe.compile_call({}, PlanContext())
        name, inverse_args = Probe.inverse_call(args, PlanContext())
        assert (name, inverse_args) == ("unpad_action", {"target_dim": 7})
        pipeline = build_pipeline(
            TransformPipelinePlan(
                calls=(TransformStepCall(type=name, args=inverse_args),),
            ),
            TransformContext(),
        )
        assert pipeline.steps[0].target_dim == 7
    finally:
        del TransformRegistry._steps["_probe_inverse"]


def test_unequal_vector_targets_produce_independent_pad_calls():
    """State and action targets are grouped only when their widths agree."""
    metadata = ModelMetadata(
        name="stub", action_dim=16, action_horizon=50,
        dim_policy="padded_to_max", dim_policy_max=32,
        requires_prompt=False,
    )
    a = resolve_assembly(_schema_with_cameras(), _usable_stats(), metadata)
    assert _calls(a.data_to_model) == [
        ("pad_dimensions", {"target_dim": 32, "fields": ["state"]}),
        ("pad_dimensions", {"target_dim": 16, "fields": ["actions"]}),
    ]
    # Only the actions half has an inverse on the model_to_robot path.
    assert _calls(a.model_to_robot) == [("unpad_action", {"target_dim": 6})]


def test_a_capping_dim_policy_without_a_cap_fails():
    """``padded_to_max`` with no ``dim_policy_max`` used to degrade silently:
    the IO spec reported the dataset's widths and no padding was planned, so a
    model expecting a fixed width was handed narrow tensors."""
    metadata = ModelMetadata(
        name="stub", action_horizon=50, requires_prompt=False,
        dim_policy="padded_to_max",          # ... but no dim_policy_max
        vector_normalization="mean_std",
    )
    with pytest.raises(ValueError, match="positive dim_policy_max"):
        resolve_assembly(_schema_with_cameras(), _usable_stats(), metadata)


@pytest.mark.parametrize("dim_policy", ["fixed", "padded_to_max"])
@pytest.mark.parametrize("limit", [0, -1])
def test_bounded_dim_policy_requires_a_positive_limit(dim_policy, limit):
    metadata = ModelMetadata(
        name="stub", dim_policy=dim_policy, dim_policy_max=limit,
        action_horizon=1, requires_prompt=False,
    )
    with pytest.raises(ValueError, match="positive dim_policy_max"):
        resolve_assembly(_schema_with_cameras(), _usable_stats(), metadata)


def test_per_slot_image_sizes_survive_when_the_dataset_already_matches():
    """Slots may declare different resolutions; a common resize step cannot
    serve them, so this only resolves when no resizing is needed at all."""
    metadata = ModelMetadata(
        name="stub", action_horizon=50, requires_prompt=False,
        vector_normalization="mean_std", image_input_range=(0.0, 1.0),
        vision_slots=(
            VisionSlot(name="head", semantic_accepts=("third_person",),
                       resolution=(224, 224)),
            VisionSlot(name="wrist", semantic_accepts=("wrist",),
                       resolution=(256, 256)),
        ),
    )
    schema = make_schema(
        state_dim=6, action_dim=6, cameras=("front", "wrist"),
        image_sizes={"front": (224, 224), "wrist": (256, 256)},
    )
    assembly = resolve_assembly(
        schema, _usable_stats(), metadata,
        overrides=AssemblyOverrides(
            camera_mapping={"head": "front", "wrist": "wrist"},
        ),
    )
    assert assembly.model_io_spec.camera_shapes == {
        "front": (224, 224), "wrist": (256, 256),
    }
    assert "resize_images" not in [c.type for c in assembly.data_to_model.calls]


def test_planner_derives_operations_from_interface_facts():
    metadata = ModelMetadata(
        name="stub", action_horizon=1, requires_prompt=False,
        image_input_range=(0.0, 1.0), image_layout="CHW",
        vector_normalization="mean_std",
    )
    assembly = resolve_assembly(_schema_with_cameras(), _usable_stats(), metadata)
    assert [call.type for call in assembly.data_to_model.calls] == [
        "image_to_float", "image_layout", "normalize_vector",
    ]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("image_layout", "NHWC", "unsupported image_layout"),
        ("image_resize_mode", "crop", "unsupported image_resize_mode"),
    ],
)
def test_invalid_pipeline_fact_fails_during_resolution(field, value, message):
    metadata = ModelMetadata(
        name="stub", action_horizon=1, requires_prompt=False,
        **{field: value},
    )
    with pytest.raises(ValueError, match=message):
        resolve_assembly(_schema_with_cameras(), _usable_stats(), metadata)


def test_model_input_size_rejects_non_positive_dimensions():
    """An invalid model interface fails before a resize call is planned."""
    metadata = ModelMetadata(
        name="stub", training_paradigm="from_scratch",
        requires_prompt=False, vector_normalization="mean_std",
        params={
            "action_horizon": 50,
            "input_image_size": [0, 224],
        },
    )
    with pytest.raises(ValueError, match="must be positive"):
        resolve_assembly(_schema_with_cameras(), _usable_stats(), metadata)


def test_model_image_size_without_resize_policy_fails():
    """A target shape does not silently select stretch vs letterbox."""
    metadata = ModelMetadata(
        name="stub", action_horizon=1, requires_prompt=False,
        image_input_range=(0.0, 1.0),
        vision_slots=(VisionSlot(name="front", resolution=(224, 224)),),
    )
    schema = make_schema(
        state_dim=6, action_dim=6, cameras=("front",),
        image_sizes={"front": (480, 640)},
    )
    with pytest.raises(ValueError, match="image_resize_mode"):
        resolve_assembly(schema, _usable_stats(), metadata)


def test_unimplementable_normalization_fails_with_a_message_not_a_keyerror():
    """``min_max`` is a legal ``vector_normalization`` value that no
    NormalizeVector method implements."""
    metadata = ModelMetadata(
        name="stub", action_horizon=1, requires_prompt=False,
        vector_normalization="min_max",
    )
    stats = FeatureStats(min=[0.0] * 6, max=[1.0] * 6)
    with pytest.raises(ValueError, match="no NormalizeVector method"):
        resolve_assembly(
            _schema_with_cameras(), NormStats(state=stats, action=stats), metadata,
        )

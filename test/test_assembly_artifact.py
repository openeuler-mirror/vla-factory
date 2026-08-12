"""The ``assembly.json`` artifact: version envelope + declaration drift.

These cover the two ways a checkpoint can be served under a contract it was not
trained with — an artifact this build cannot read, and a model declaration that
changed underneath it — plus the classification guard that keeps the drift check
honest as ``ModelMetadata`` grows.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import make_assembly, make_schema

from vla_factory.assembly.artifact import (
    ASSEMBLY_FORMAT_VERSION,
    AssemblyArtifactError,
    AssemblyDeclarationDrift,
    check_declaration_drift,
    load_assembly_artifact,
    save_assembly_artifact,
    unclassified_metadata_fields,
)
from vla_factory.model.registry import list_entries


def _act_assembly():
    schema = make_schema(
        state_dim=6, action_dim=8, cameras=("front",),
        image_sizes={"front": (480, 640)},
    )
    return make_assembly(schema, "act")


def test_round_trip_through_disk(tmp_path):
    assembly = _act_assembly()
    path = tmp_path / "assembly.json"
    save_assembly_artifact(path, assembly)

    payload = json.loads(path.read_text())
    assert payload["format_version"] == ASSEMBLY_FORMAT_VERSION
    # The envelope carries the version; the assembly's own shape is untouched.
    assert payload["assembly"] == assembly.to_dict()
    assert load_assembly_artifact(path).to_dict() == assembly.to_dict()


def test_unknown_format_version_fails_conservatively(tmp_path):
    """An artifact outlives the process that wrote it: a version this build
    does not know cannot be read on a best-effort basis (§1.7)."""
    path = tmp_path / "assembly.json"
    save_assembly_artifact(path, _act_assembly())
    payload = json.loads(path.read_text())
    payload["format_version"] = 999
    path.write_text(json.dumps(payload))

    with pytest.raises(AssemblyArtifactError, match="format_version"):
        load_assembly_artifact(path)


def test_a_file_that_is_not_an_artifact_fails(tmp_path):
    path = tmp_path / "assembly.json"
    path.write_text(json.dumps({"camera_mapping": {}}))
    with pytest.raises(AssemblyArtifactError):
        load_assembly_artifact(path)


def _saved(tmp_path, mutate) -> Path:
    path = tmp_path / "assembly.json"
    save_assembly_artifact(path, _act_assembly())
    payload = json.loads(path.read_text())
    mutate(payload["assembly"])
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize(
    "key", ["schema_ref", "norm_stats_ref", "metadata_ref", "model_io_spec"],
)
def test_a_missing_description_is_refused(tmp_path, key):
    """``from_dict`` fills absences with empty defaults — right in memory, wrong
    off disk, where absent means the artifact is incomplete."""
    path = _saved(tmp_path, lambda body: body.pop(key))
    with pytest.raises(AssemblyArtifactError, match=key):
        load_assembly_artifact(path)


@pytest.mark.parametrize("key", ["data_to_model", "model_to_robot"])
def test_an_unresolved_pipeline_is_refused(tmp_path, key):
    """The dangerous one is ``model_to_robot``: loaded as an empty plan it would
    build a postprocessor that does nothing, and a policy would send normalized
    actions to a robot with the shape check still passing."""
    path = _saved(tmp_path, lambda body: body.pop(key))
    with pytest.raises(AssemblyArtifactError, match=key):
        load_assembly_artifact(path)

    path = _saved(tmp_path, lambda body: body[key].__setitem__("resolved", False))
    with pytest.raises(AssemblyArtifactError, match=key):
        load_assembly_artifact(path)


def test_a_missing_interface_fact_is_refused(tmp_path):
    """Deleting a fact from the snapshot must not exempt it from drift detection."""
    path = _saved(tmp_path, lambda body: body["metadata_ref"].pop("image_input_range"))
    with pytest.raises(AssemblyArtifactError, match="image_input_range"):
        load_assembly_artifact(path)


def test_build_pipeline_refuses_an_unresolved_plan():
    from vla_factory.assembly.resolver.types import TransformPipelinePlan
    from vla_factory.assembly.transforms import TransformContext, build_pipeline

    with pytest.raises(ValueError, match="unresolved"):
        build_pipeline(TransformPipelinePlan(), TransformContext())


# ── Declaration drift ─────────────────────────────────────────────


def test_matching_declaration_passes():
    check_declaration_drift(_act_assembly(), list_entries()["act"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_input_range", (-1.0, 1.0)),      # SigLIP-style range, not ACT's
        ("vector_normalization", "quantile"),
        ("requires_prompt", True),
        ("dim_policy", "fixed"),
    ],
)
def test_interface_drift_is_refused(field, value):
    """None of these change a single tensor shape — the weights would load and
    the model would simply be fed the wrong inputs. That is why the check
    exists instead of relying on ``load_state_dict(strict=True)``."""
    assembly = _act_assembly()
    drifted = replace(list_entries()["act"], **{field: value})
    with pytest.raises(AssemblyDeclarationDrift, match=field):
        check_declaration_drift(assembly, drifted)


def test_non_interface_changes_do_not_block_serving():
    """Install hints and trainable-component patterns describe how a model is
    trained or installed, not what tensors it exchanges."""
    assembly = _act_assembly()
    metadata = replace(
        list_entries()["act"],
        install_hint="pip install something-else",
        support_lora=True,
    )
    check_declaration_drift(assembly, metadata)


def test_every_metadata_field_is_classified():
    """A new ``ModelMetadata`` field must be declared part of the interface
    contract or explicitly outside it — silence would let a future fact change
    what the model consumes while drift detection looks away."""
    assert unclassified_metadata_fields() == []

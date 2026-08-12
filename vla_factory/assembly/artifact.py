"""The ``assembly.json`` artifact: a versioned snapshot of a resolved composition.

Training writes one next to every checkpoint; inference executes it. It is the
*execution contract* of that checkpoint — the IO spec, mappings and pipeline
plans that were resolved from the model declaration in force **at training
time**, together with the dataset description and statistics they were resolved
against.

Why a snapshot and not a re-resolve at deploy time
--------------------------------------------------
Resolution is pure, so the deploy process could run it again — but it would run
it against *today's* declaration. A model entry that changed its image input
range, normalization method or camera slots since training would produce a
different, perfectly valid-looking pipeline, and the weights would still load
(none of those facts change a tensor shape). The failure would be silent and
would look like "accuracy mysteriously dropped". Pinning the contract makes that
impossible, and lets the engine detect the drift instead (see
``inference/infer.py``).

Format version
--------------
``format_version`` is written from the first release, and an unknown one is a
conservative failure rather than a best-effort read: the artifact outlives the
process that wrote it, and there is no way to guess the meaning of a shape that
did not exist yet. The version lives in the envelope, so ``ResolvedAssembly``'s
own ``to_dict()`` shape is untouched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from vla_factory.model.interfaces.model import ModelMetadata

from .resolver import ResolvedAssembly

# Bump only when the serialized shape of ResolvedAssembly changes in a way a
# previous reader would misinterpret; ship the migration in the same commit.
ASSEMBLY_FORMAT_VERSION = 1


class AssemblyArtifactError(Exception):
    """The saved assembly cannot be read as this version's contract."""


def assembly_artifact_dict(assembly: ResolvedAssembly) -> dict[str, Any]:
    """The on-disk envelope for a resolved assembly."""
    return {
        "format_version": ASSEMBLY_FORMAT_VERSION,
        "assembly": assembly.to_dict(),
    }


def save_assembly_artifact(path: str | Path, assembly: ResolvedAssembly) -> None:
    """Write ``assembly.json`` (envelope + resolved assembly)."""
    with open(Path(path), "w") as f:
        json.dump(assembly_artifact_dict(assembly), f, indent=2)


def load_assembly_artifact(path: str | Path) -> ResolvedAssembly:
    """Read ``assembly.json`` back, or fail with a readable reason.

    Raises
    ------
    FileNotFoundError
        No artifact at *path*.
    AssemblyArtifactError
        Unparsable, or written by a format version this build cannot read.
    """
    path = Path(path)
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise AssemblyArtifactError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or "assembly" not in payload:
        raise AssemblyArtifactError(
            f"{path} is not an assembly artifact (expected an object with "
            "'format_version' and 'assembly')."
        )

    version = payload.get("format_version")
    if version != ASSEMBLY_FORMAT_VERSION:
        raise AssemblyArtifactError(
            f"{path} has format_version={version!r}, but this build reads "
            f"version {ASSEMBLY_FORMAT_VERSION}. Use the framework version that "
            "wrote this checkpoint, or retrain with the current one."
        )

    body = payload["assembly"]
    if not isinstance(body, dict):
        raise AssemblyArtifactError(f"{path}: 'assembly' must be an object.")
    _validate_v1(path, body)
    return ResolvedAssembly.from_dict(body)


def _validate_v1(path: Path, body: dict[str, Any]) -> None:
    """Refuse an artifact that is missing part of the contract.

    ``ResolvedAssembly.from_dict`` fills absent fields with empty defaults,
    which is right for constructing one in memory and wrong for reading one off
    disk: an artifact whose ``model_to_robot`` is gone would load as an empty
    unresolved plan and produce a **silent no-op postprocessor** — a policy
    happily sending normalized actions to a robot, with the shape check still
    passing. Absent is not empty here; absent is invalid.
    """
    for key in ("schema_ref", "norm_stats_ref", "metadata_ref", "model_io_spec"):
        if not isinstance(body.get(key), dict) or not body[key]:
            raise AssemblyArtifactError(
                f"{path}: '{key}' is missing or empty; the artifact does not "
                "describe a complete composition."
            )

    for key in ("data_to_model", "model_to_robot"):
        plan = body.get(key)
        if not isinstance(plan, dict) or not plan.get("resolved"):
            raise AssemblyArtifactError(
                f"{path}: '{key}' is missing or not resolved. Both execution "
                "paths must be planned — an unresolved one would silently "
                "become a pipeline that does nothing."
            )

    missing = [f for f in INTERFACE_FACTS if f not in body["metadata_ref"]]
    if missing:
        raise AssemblyArtifactError(
            f"{path}: 'metadata_ref' is missing interface facts {missing}. "
            "Drift detection compares exactly these, so an absent one would "
            "silently exempt itself from the check."
        )


# ── Snapshot vs. current declaration ──────────────────────────────
#
# The facts below are the model's *interface contract* — every named field the
# composition resolver reads. If one of them changed since training, the
# snapshot describes a model the installed code no longer implements, and none
# of these changes alter a single tensor shape: an image range flipped from
# [0,1] to [-1,1], quantile normalization swapped for z-score, or a renamed
# vision slot all load their weights happily and simply behave wrongly. That is
# why this is checked explicitly rather than left to ``load_state_dict``.
INTERFACE_FACTS: tuple[str, ...] = (
    "name",
    "action_dim",
    "action_horizon",
    "dim_policy",
    "dim_policy_max",
    "vision_slots",
    "missing_slot_policy",
    "image_input_range",
    "image_normalize_mode",
    "vector_normalization",
    "requires_prompt",
    "language_template",
    "control_mode_pref",
    "expected_hz",
    "history_frames",
)

# Declaration fields deliberately outside the contract: they describe how a
# model is trained, wrapped or installed, not what tensors it exchanges, so
# changing one must not stop an existing checkpoint from serving.
NON_INTERFACE_FACTS: frozenset[str] = frozenset({
    "backend",
    "action_head_type",
    "training_paradigm",
    "components",
    "requires_augmentation",
    "support_lora",
    "support_full",
    "support_freeze",
    "install_hint",
    "params",
})


class AssemblyDeclarationDrift(AssemblyArtifactError):
    """The model declaration changed since this checkpoint was trained."""


def declaration_snapshot(metadata: ModelMetadata) -> dict[str, Any]:
    """The model declaration in the same JSON shape the assembly stores."""
    return json.loads(json.dumps(asdict(metadata)))


def check_declaration_drift(
    assembly: ResolvedAssembly, metadata: ModelMetadata,
) -> None:
    """Fail if the current declaration contradicts the assembly's snapshot.

    Only :data:`INTERFACE_FACTS` are compared — see the note above for why the
    rest are exempt.
    """
    current = declaration_snapshot(metadata)
    stored = assembly.metadata_ref or {}
    # A fact the snapshot does not carry counts as drift, not as "nothing to
    # compare": treating absence as agreement would mean deleting a key from
    # the artifact silently disables the check for it.
    drifted = {
        key: (stored.get(key, "<absent>"), current.get(key))
        for key in INTERFACE_FACTS
        if stored.get(key, "<absent>") != current.get(key)
    }
    if not drifted:
        return
    detail = "; ".join(
        f"{key}: trained with {was!r}, now declared {now!r}"
        for key, (was, now) in sorted(drifted.items())
    )
    raise AssemblyDeclarationDrift(
        f"Model {metadata.name!r} no longer declares the interface this "
        f"checkpoint was trained under — {detail}. The weights would still "
        "load; the inputs would be wrong. Serve this checkpoint with the "
        "framework version that trained it, or retrain against the current "
        "declaration."
    )


def unclassified_metadata_fields() -> list[str]:
    """Declaration fields that are in neither list (a test guards this is empty).

    A new ``ModelMetadata`` field must be classified deliberately: part of the
    interface contract, or explicitly outside it. Silence would mean a future
    fact changes what the model consumes while drift detection looks away.
    """
    known = set(INTERFACE_FACTS) | NON_INTERFACE_FACTS
    return sorted(f.name for f in fields(ModelMetadata) if f.name not in known)

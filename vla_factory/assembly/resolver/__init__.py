"""Composition-resolution sub-package (assembly resolver).

Public surface:
- ``resolve_assembly`` — deterministic ``data × model × robot`` → ResolvedAssembly
- ``ResolvedAssembly`` and the serializable spec / mapping types
- ``ResolutionError`` + stable error codes

Phase 0 (architecture §7.4): terms & data structures + Load / Materialize /
Validate + resolve dry-run. Mapping and TransformPipeline derivation lands later.
"""

from .errors import (
    INVALID_DESCRIPTION,
    MISSING_INPUT,
    METADATA_CONTRACT_CONFLICT,
    ResolutionError,
    UNKNOWN_MODEL,
    UNKNOWN_ROBOT,
    make_error,
)
from .resolver import resolve_assembly
from .types import (
    ActionMapping,
    CameraMapping,
    CanonicalInterface,
    JointMapping,
    LanguageMapping,
    ResolvedAssembly,
    StateMapping,
    TransformPipelineSpec,
    TransformStepSpec,
)

__all__ = [
    "resolve_assembly",
    "ResolvedAssembly",
    "CanonicalInterface",
    "TransformStepSpec",
    "TransformPipelineSpec",
    "CameraMapping",
    "StateMapping",
    "ActionMapping",
    "LanguageMapping",
    "JointMapping",
    "ResolutionError",
    "make_error",
    "MISSING_INPUT",
    "INVALID_DESCRIPTION",
    "METADATA_CONTRACT_CONFLICT",
    "UNKNOWN_MODEL",
    "UNKNOWN_ROBOT",
]

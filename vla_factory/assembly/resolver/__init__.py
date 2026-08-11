"""Composition-resolution sub-package (assembly resolver).

Public surface:
- ``resolve_assembly`` — deterministic ``data × model × robot`` → ResolvedAssembly
- ``ResolvedAssembly`` and the serializable plan / mapping types
- ``ResolutionError`` + stable error codes

Stages: Load → Validate → Check Pairs → Plan Pipeline → Build
Interface → Resolve Mapping → Emit. ``robot_to_model`` is the one product not
derivable yet — it needs the joint-reorder step, which has no implementation.
"""

from .errors import (
    CAMERA_MAPPING_INVALID,
    INVALID_DESCRIPTION,
    MISSING_INPUT,
    PIPELINE_WIDTH_MISMATCH,
    UNSUPPORTED_OVERRIDE,
    ResolutionError,
    UNKNOWN_MODEL,
    UNKNOWN_ROBOT,
    make_error,
)
from .resolver import resolve_assembly
from .types import (
    ActionMapping,
    CameraMapping,
    ModelIOSpec,
    JointMapping,
    LanguageMapping,
    ResolvedAssembly,
    StateMapping,
    TransformPipelinePlan,
    TransformStepCall,
)

__all__ = [
    "resolve_assembly",
    "ResolvedAssembly",
    "ModelIOSpec",
    "TransformStepCall",
    "TransformPipelinePlan",
    "CameraMapping",
    "StateMapping",
    "ActionMapping",
    "LanguageMapping",
    "JointMapping",
    "ResolutionError",
    "make_error",
    "MISSING_INPUT",
    "INVALID_DESCRIPTION",
    "UNKNOWN_MODEL",
    "UNKNOWN_ROBOT",
    "CAMERA_MAPPING_INVALID",
    "PIPELINE_WIDTH_MISMATCH",
    "UNSUPPORTED_OVERRIDE",
]

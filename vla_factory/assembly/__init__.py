"""Resolve a recipe or explicit facts into a train/deploy execution contract."""

from .resolve_assembly import (
    FieldMapping,
    InvalidAssemblyError,
    MappingSource,
    ModelIOSpec,
    ModelInterfaceMismatch,
    ResolvedAssembly,
    resolve_assembly,
)
from .resolve import ResolutionError, resolve_from_facts

__all__ = [
    "resolve_assembly", "resolve_from_facts", "ResolvedAssembly", "ModelIOSpec",
    "FieldMapping", "MappingSource", "ResolutionError", "InvalidAssemblyError",
    "ModelInterfaceMismatch",
]

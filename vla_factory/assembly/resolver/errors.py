"""Structured resolution errors (architecture §4.2.5).

The error contract keeps exactly three stable concepts:

* ``code``   — a stable machine error code (tests / CLI / tools key on this);
* ``path``   — the resolution target the error refers to (not necessarily a
               user recipe field path);
* ``params`` — the JSON-serializable facts needed to render a message.

User-readable text is **not** part of the stable contract. Each ``code`` is
produced only through a dedicated constructor in :data:`FACTORIES`, which fixes
the allowed ``params`` keys for that code — callers never assemble free-form
params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ── Stable error codes ────────────────────────────────────────────

# A required input description was missing or unreadable.
MISSING_INPUT = "missing_input"
# A description is internally invalid (unknown field, bad enum value, ...).
INVALID_DESCRIPTION = "invalid_description"
# ModelMetadata × BaseContract could not be merged (capability boundary breach).
METADATA_CONTRACT_CONFLICT = "metadata_contract_conflict"
# The named model is not in the registry.
UNKNOWN_MODEL = "unknown_model"
# The named robot profile is not registered.
UNKNOWN_ROBOT = "unknown_robot"


@dataclass
class ResolutionError(Exception):
    """Structured composition-resolution failure.

    Attributes
    ----------
    code : str
        One of the ``*_code`` constants above.
    path : str
        Dotted resolution target this error refers to (e.g. ``"schema"``,
        ``"model.action_dim"``).
    params : dict
        JSON-serializable facts. The allowed keys are fixed per ``code`` by the
        dedicated constructor in :data:`FACTORIES`.
    """

    code: str
    path: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Allow ``raise ResolutionError(code=..., path=..., params=...)`` to
        # behave both as a dataclass and as an exception.
        super().__init__(f"[{self.code}] {self.path}: {self.params}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "path": self.path, "params": dict(self.params)}


# ── Dedicated constructors (fix the allowed params per code) ──────
#
# Each entry maps a code to a function ``(path, **facts) -> ResolutionError``.
# The function's signature documents and enforces the allowed params keys for
# that code; tests and the CLI should build errors through these helpers so the
# params schema cannot drift.


def _missing_input(path: str, *, field_name: str, detail: str = "") -> ResolutionError:
    return ResolutionError(
        code=MISSING_INPUT,
        path=path,
        params={"field": field_name, "detail": detail},
    )


def _invalid_description(
    path: str, *, field_name: str, value: Any = None, detail: str = ""
) -> ResolutionError:
    return ResolutionError(
        code=INVALID_DESCRIPTION,
        path=path,
        params={"field": field_name, "value": value, "detail": detail},
    )


def _metadata_contract_conflict(
    path: str, *, field_name: str, metadata_value: Any, contract_value: Any
) -> ResolutionError:
    return ResolutionError(
        code=METADATA_CONTRACT_CONFLICT,
        path=path,
        params={
            "field": field_name,
            "metadata_value": metadata_value,
            "contract_value": contract_value,
        },
    )


def _unknown_model(path: str, *, model_name: str, known: list[str]) -> ResolutionError:
    return ResolutionError(
        code=UNKNOWN_MODEL,
        path=path,
        params={"model_name": model_name, "known": known},
    )


def _unknown_robot(path: str, *, robot_name: str, known: list[str]) -> ResolutionError:
    return ResolutionError(
        code=UNKNOWN_ROBOT,
        path=path,
        params={"robot_name": robot_name, "known": known},
    )


# code → constructor. Callers should go through this mapping so the allowed
# params shape stays coupled to the code.
FACTORIES: dict[str, Callable[..., ResolutionError]] = {
    MISSING_INPUT: _missing_input,
    INVALID_DESCRIPTION: _invalid_description,
    METADATA_CONTRACT_CONFLICT: _metadata_contract_conflict,
    UNKNOWN_MODEL: _unknown_model,
    UNKNOWN_ROBOT: _unknown_robot,
}


def make_error(code: str, path: str, **params: Any) -> ResolutionError:
    """Build a ``ResolutionError`` through the dedicated constructor for *code*.

    Centralising construction here guarantees each code only ever carries its
    allowed params keys.
    """
    factory = FACTORIES.get(code)
    if factory is None:  # pragma: no cover - defensive
        raise ValueError(f"Unknown resolution error code {code!r}")
    return factory(path, **params)

"""Shared test helpers.

``make_schema`` mirrors the pre-phase-1 flat ``DataSchema`` constructor so that
test fixtures stay readable while the real ``DataSchema`` stores the entry-table
form (data-module §8.3). Production code reads the derived compatibility
properties (decision D2); only these test fixtures build a schema directly.

``make_assembly`` runs the real resolver: a model factory now takes a
``ResolvedAssembly``, and hand-building one in each test would let a fixture
claim a composition the resolver would never produce.
"""

from __future__ import annotations

from vla_factory.assembly.resolver import ResolvedAssembly, resolve_assembly
from vla_factory.data.manifest import (
    ActionDim, CameraEntry, DataSchema, FeatureStats, NormStats, StateDim,
)


def make_schema(
    *,
    state_dim: int = 0,
    action_dim: int = 0,
    cameras: tuple[str, ...] = (),
    image_sizes: dict[str, tuple[int, int]] | None = None,
    fps: int = 30,
    has_language: bool = False,
    total_episodes: int = 0,
    total_frames: int = 0,
    robot_type: str = "unknown",
    state_keys: tuple[str, ...] = (),
    action_keys: tuple[str, ...] = (),
) -> DataSchema:
    image_sizes = image_sizes or {}
    # Every dimension gets a name unless the caller supplies its own: a real
    # reader always fills them (the composition resolver validates it, since the
    # deploy side reads the dimension→motor-key correspondence back), so a
    # nameless fixture would describe a dataset no reader produces. Tests that
    # are *about* missing names build a DataSchema directly.
    state_dims = tuple(
        StateDim(
            name=state_keys[i] if i < len(state_keys) else f"state_{i}",
            source_field="observation.state",
        )
        for i in range(state_dim)
    )
    action_dims = tuple(
        ActionDim(
            name=action_keys[i] if i < len(action_keys) else f"action_{i}",
            source_field="action",
        )
        for i in range(action_dim)
    )
    cameras_entries = tuple(
        CameraEntry(
            key=c,
            resolution=tuple(image_sizes[c]) if c in image_sizes else None,
        )
        for c in cameras
    )
    return DataSchema(
        episodes=total_episodes,
        total_frames=total_frames,
        robot_ref=None if robot_type in (None, "", "unknown") else str(robot_type),
        cameras_entries=cameras_entries,
        state_dims=state_dims,
        action_dims=action_dims,
        temporal_fps=fps,
        instruction_task_field="task" if has_language else None,
    )


def _unit_stats(dim: int) -> FeatureStats:
    # Quantiles are filled in too, so the same fixture satisfies both declared
    # normalization methods (pi05 resolves against quantile stats).
    return FeatureStats(
        mean=[0.0] * dim, std=[1.0] * dim,
        q01=[-1.0] * dim, q99=[1.0] * dim,
    )


def make_norm_stats(*, state_dim: int = 0, action_dim: int = 0) -> NormStats:
    """Unit statistics wide enough for the schema they accompany."""
    return NormStats(state=_unit_stats(state_dim), action=_unit_stats(action_dim))


def make_assembly(
    schema: DataSchema,
    model_name: str,
    *,
    recipe=None,
    norm_stats: NormStats | None = None,
    overrides: dict | None = None,
) -> ResolvedAssembly:
    """Resolve a real assembly for a registered model against *schema*.

    The model's declared tunables are taken from the recipe when one is given
    (so a per-run override of transforms / action_horizon is honoured), else
    from the declaration itself.
    """
    from vla_factory.model.registry import list_entries

    metadata = list_entries()[model_name]
    if norm_stats is None:
        norm_stats = make_norm_stats(
            state_dim=schema.state_dim, action_dim=schema.action_dim,
        )
    return resolve_assembly(
        schema=schema,
        norm_stats=norm_stats,
        metadata=metadata,
        overrides=overrides or None,
        model_config=getattr(recipe, "model_config", None),
        model_path=getattr(recipe, "model_path", None),
    )

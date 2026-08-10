"""Shared test helpers.

``make_schema`` mirrors the pre-phase-1 flat ``DataSchema`` constructor so that
test fixtures stay readable while the real ``DataSchema`` stores the entry-table
form (data-module §8.3). Production code reads the derived compatibility
properties (decision D2); only these test fixtures build a schema directly.
"""

from __future__ import annotations

from vla_factory.data.manifest import ActionDim, CameraEntry, DataSchema, StateDim


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
    state_dims = tuple(
        StateDim(
            name=state_keys[i] if i < len(state_keys) else None,
            source_field="observation.state",
        )
        for i in range(state_dim)
    )
    action_dims = tuple(
        ActionDim(
            name=action_keys[i] if i < len(action_keys) else None,
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

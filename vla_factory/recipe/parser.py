"""Strict YAML parsing for the public TrainRecipe structure."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from vla_factory.recipe.train_recipe import (
    AssemblyOverrides,
    DataConfig,
    FinetuningConfig,
    ModelConfig,
    OutputConfig,
    RobotConfig,
    TrainingConfig,
    TrainRecipe,
)


def parse_recipe(path: str | Path) -> TrainRecipe:
    """Parse one YAML recipe file without consulting live model declarations."""
    path = Path(path)
    return _build_recipe(_load_yaml(path.read_text(encoding="utf-8")))


def parse_recipe_from_string(content: str) -> TrainRecipe:
    """Parse recipe YAML text, primarily for tests and generated recipes."""
    return _build_recipe(_load_yaml(content))


def _load_yaml(content: str) -> dict[str, Any]:
    raw = yaml.safe_load(content) or {}
    if not isinstance(raw, dict):
        raise TypeError("recipe YAML root must be a mapping")
    return raw


def _build_recipe(raw: dict[str, Any]) -> TrainRecipe:
    _reject_unknown(
        raw,
        {
            "model",
            "data",
            "robot",
            "overrides",
            "finetuning",
            "training",
            "output",
        },
        "recipe",
    )
    return TrainRecipe(
        model=_parse_model(raw.get("model")),
        data=_parse_data(raw.get("data")),
        robot=_parse_robot(raw.get("robot")),
        overrides=_parse_overrides(raw.get("overrides")),
        finetuning=_parse_finetuning(raw.get("finetuning")),
        training=_parse_training(raw.get("training")),
        output=_parse_output(raw.get("output")),
    )


def _parse_model(value: Any) -> ModelConfig:
    if isinstance(value, str):
        if not value:
            raise ValueError("model cannot be empty")
        if "/" not in value:
            return ModelConfig(name=value)
        if value.endswith("/"):
            raise ValueError("model path must not end with '/'")
        name = value.rsplit("/", 1)[-1]
        if not name:
            raise ValueError("model path must end with a model name")
        return ModelConfig(name=name, path=value)

    block = _mapping(value, "model", required=True)
    _reject_unknown(block, {"name", "path", "config"}, "model")
    name = block.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("model.name is required and must be a non-empty string")
    path = block.get("path")
    if path is not None and not isinstance(path, str):
        raise TypeError("model.path must be a string or null")
    config = block.get("config", {})
    if not isinstance(config, dict):
        raise TypeError("model.config must be a mapping")
    return ModelConfig(name=name, path=path, config=dict(config))


def _parse_robot(value: Any) -> RobotConfig:
    if value is None:
        return RobotConfig()
    if isinstance(value, str):
        if not value:
            raise ValueError("robot cannot be an empty string")
        return RobotConfig(name=value)
    block = _mapping(value, "robot")
    _reject_unknown(block, {"name"}, "robot")
    name = block.get("name", "")
    if not isinstance(name, str):
        raise TypeError("robot.name must be a string")
    return RobotConfig(name=name)


def _parse_data(value: Any) -> DataConfig:
    block = _mapping(value, "data")
    _reject_unknown(block, {item.name for item in fields(DataConfig)}, "data")
    return DataConfig(
        path=_string(block.get("path", ""), "data.path"),
        format=_string(block.get("format", "auto"), "data.format"),
        video_codec=_string(
            block.get("video_codec", "auto"), "data.video_codec"
        ),
    )


def _parse_overrides(value: Any) -> AssemblyOverrides:
    block = _mapping(value, "overrides")
    _reject_unknown(
        block,
        {item.name for item in fields(AssemblyOverrides)},
        "overrides",
    )
    camera_mapping = block.get("camera_mapping")
    if camera_mapping is not None:
        if not isinstance(camera_mapping, dict) or not all(
            isinstance(key, str) and isinstance(camera, str)
            for key, camera in camera_mapping.items()
        ):
            raise TypeError(
                "overrides.camera_mapping must be a string-to-string mapping or null"
            )
        camera_mapping = dict(camera_mapping)
    default_task = block.get("default_task")
    if default_task is not None and not isinstance(default_task, str):
        raise TypeError("overrides.default_task must be a string or null")
    return AssemblyOverrides(
        camera_mapping=camera_mapping,
        default_task=default_task,
    )


def _parse_finetuning(value: Any) -> FinetuningConfig:
    block = _mapping(value, "finetuning")
    _reject_unknown(block, {"strategy", "config"}, "finetuning")
    strategy = block.get("strategy", "full")
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("finetuning.strategy must be a non-empty string")
    config = block.get("config", {})
    if not isinstance(config, dict):
        raise TypeError("finetuning.config must be a mapping")
    return FinetuningConfig(strategy=strategy, config=dict(config))


def _parse_training(value: Any) -> TrainingConfig:
    block = _mapping(value, "training")
    valid = {item.name for item in fields(TrainingConfig)}
    _reject_unknown(block, valid, "training")
    return TrainingConfig(
        backend=_string(block.get("backend", "pytorch"), "training.backend"),
        lr=_float(block.get("lr", 1e-4), "training.lr"),
        lr_backbone=_optional_float(
            block.get("lr_backbone"), "training.lr_backbone"
        ),
        batch_size=_integer(
            block.get("batch_size", 8), "training.batch_size"
        ),
        total_steps=_integer(
            block.get("total_steps", 10000), "training.total_steps"
        ),
        gradient_checkpointing=_boolean(
            block.get("gradient_checkpointing", False),
            "training.gradient_checkpointing",
        ),
        num_workers=_integer(
            block.get("num_workers", 4), "training.num_workers"
        ),
    )


def _parse_output(value: Any) -> OutputConfig:
    block = _mapping(value, "output")
    valid = {item.name for item in fields(OutputConfig)}
    _reject_unknown(block, valid, "output")
    return OutputConfig(
        output_dir=_string(
            block.get("output_dir", "outputs/default"), "output.output_dir"
        ),
        report_to=_string(block.get("report_to", "none"), "output.report_to"),
        logging_steps=_integer(
            block.get("logging_steps", 50), "output.logging_steps"
        ),
        save_steps=_integer(block.get("save_steps", 5000), "output.save_steps"),
        save_total_limit=_integer(
            block.get("save_total_limit", 3), "output.save_total_limit"
        ),
        overwrite_output_dir=_boolean(
            block.get("overwrite_output_dir", False),
            "output.overwrite_output_dir",
        ),
    )


def _mapping(
    value: Any,
    path: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    if value is None:
        if required:
            raise ValueError(f"{path} is required")
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    return value


def _reject_unknown(data: dict, valid: set[str], path: str) -> None:
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(
            f"Unknown {path} field(s): {unknown}; supported fields are "
            f"{sorted(valid)}"
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be an integer") from exc


def _float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be a number") from exc


def _optional_float(value: Any, path: str) -> float | None:
    return None if value is None else _float(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


__all__ = ["parse_recipe", "parse_recipe_from_string"]

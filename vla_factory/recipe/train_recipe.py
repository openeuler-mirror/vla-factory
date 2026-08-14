"""Strongly typed public structure of a VLA Factory training recipe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Select a registered model and optionally its pretrained checkpoint."""

    name: str = ""
    path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Select a dataset and the registered reader/codec used to read it."""

    path: str = ""
    format: str = "auto"
    video_codec: str = "auto"


@dataclass
class RobotConfig:
    """Select an optional registered robot profile."""

    name: str = ""


@dataclass
class AssemblyOverrides:
    """User choices applied while resolving data × model × robot relations."""

    camera_mapping: dict[str, str] | None = None
    default_task: str | None = None


@dataclass
class FinetuningConfig:
    """Select a registered fine-tuning strategy and its owned configuration."""

    strategy: str = "full"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Framework-level training-loop configuration."""

    backend: str = "pytorch"
    lr: float = 1e-4
    lr_backbone: float | None = None
    batch_size: int = 8
    total_steps: int = 10000
    gradient_checkpointing: bool = False
    num_workers: int = 4


@dataclass
class OutputConfig:
    """Output, logging, and checkpoint configuration."""

    output_dir: str = "outputs/default"
    report_to: str = "none"
    logging_steps: int = 50
    save_steps: int = 5000
    save_total_limit: int = 3
    overwrite_output_dir: bool = False


@dataclass
class TrainRecipe:
    """Complete user-authored training recipe.

    The object mirrors the public YAML blocks exactly. Relations derived from
    data, model, and robot descriptions belong to ``ResolvedAssembly`` rather
    than this user-choice structure.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    overrides: AssemblyOverrides = field(default_factory=AssemblyOverrides)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical public YAML representation."""
        result = asdict(self)
        if not self.robot.name:
            result["robot"] = None
        override_values = {
            key: value
            for key, value in result["overrides"].items()
            if value is not None
        }
        result["overrides"] = override_values or None
        return result


__all__ = [
    "AssemblyOverrides",
    "DataConfig",
    "FinetuningConfig",
    "ModelConfig",
    "OutputConfig",
    "RobotConfig",
    "TrainingConfig",
    "TrainRecipe",
]

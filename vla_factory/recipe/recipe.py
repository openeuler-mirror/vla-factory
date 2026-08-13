"""Configuration dataclasses for a TrainRecipe.

These mirror the YAML structure defined in the v2.0 architecture.  A YAML
file is parsed into a ``TrainRecipe`` object which is then consumed by the
training engine.  All fields have sensible defaults so a minimal YAML works.

For a fully-annotated example see ``configs/reference.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Data ───────────────────────────────────────────────────────────


@dataclass
class DataSourceConfig:
    """Where to find training data.

    Fields
    ------
    path : str
        Root directory or file path of the dataset.
    format : str
        Dataset storage format. One of:
          - ``auto``       : auto-detect from path contents
          - ``lerobot-v3`` : HuggingFace LeRobot v3 (Parquet + MP4)
          - ``hdf5``       : HDF5 (Robomimic, ALOHA raw)
          - ``rlds``       : TFRecord-based RLDS (Open X-Embodiment)
          - ``zarr``       : Chunked Zarr arrays (BridgeData V2)
    """

    path: str = ""
    format: str = "auto"
    video_codec: str = "auto"


@dataclass
class DataConfig:
    """Which dataset to train on.

    Only the source: how episodes are sliced into samples follows the model's
    temporal contract (observation frames and chunk length), and the train/val
    split is a fixed framework policy — neither is a choice the recipe restates.

    Fields
    ------
    source : DataSourceConfig
        Where to find data and how to read it.
    """

    source: DataSourceConfig = field(default_factory=DataSourceConfig)


# ── Fine-tuning ───────────────────────────────────────────────────


@dataclass
class LoraConfig:
    """LoRA (Low-Rank Adaptation) parameters — field names align with peft.

    Only used when ``finetuning.strategy == "lora"``. Common fields mirror
    ``peft.LoraConfig`` (``r``, ``lora_alpha``, ``lora_dropout``, ...) so users
    familiar with peft don't relearn names; they're forwarded as-is.
    ``target_components`` is vla-factory's own abstraction on top (component →
    subtree → peft target_modules), since different model ecosystems name their
    linear layers differently and we don't want users to memorize each.

    Fields
    ------
    r / rank : int  (alias; both accepted for backward compat)
        LoRA rank. Higher = more capacity, more parameters. Typical: 8, 16, 32.
    lora_alpha / alpha : int  (alias; both accepted)
        LoRA scaling factor. Effective scaling = ``lora_alpha / r``.
        Typically set equal to ``r``.
    lora_dropout : float
        Dropout on LoRA adapters. Forwarded to peft. Default 0.0.
    use_rslora : bool
        Rank-stabilized LoRA (scaling = alpha / sqrt(r)). Default False.
    init_lora_weights : bool | str
        Init scheme. Forwarded to peft. Default ``"gaussian"`` (matches RLinf's
        openpi LoRA). Other peft values: True/False/"eva"/"pissa"/...
    target_components : list[str]
        Which model components to apply LoRA to (vla abstraction).
        Names must match keys in the model's ``ModelMetadata.components``.
        Examples:
          - PI0:       ``["llm"]`` (PaliGemma VLM) or ``["llm", "action_expert"]``
          - OpenVLA:   ``["llm", "action_head"]``
    """

    # peft-aligned (aliased for backward compat: r/rank, lora_alpha/alpha both read).
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = False
    init_lora_weights: object = "gaussian"

    # vla-factory abstraction (component → subtree → peft target_modules).
    target_components: list[str] = field(default_factory=list)


# ── Augmentation ──────────────────────────────────────────────────


@dataclass
class OutputConfig:
    """Output, logging, and checkpoint configuration.

    Fields
    ------
    output_dir : str
        Directory for checkpoints, logs, and final model weights.
    report_to : str
        Training logger backend. One of: ``none``, ``tensorboard``, ``wandb``.
    logging_steps : int
        Log training metrics every N steps.
    save_steps : int
        Save a checkpoint every N steps.
    save_total_limit : int
        Maximum number of checkpoints to keep (oldest deleted first).
    overwrite_output_dir : bool
        If True, delete existing output_dir contents before training.
    """

    output_dir: str = "outputs/default"
    report_to: str = "none"  # none | tensorboard | wandb
    logging_steps: int = 50
    save_steps: int = 5000
    save_total_limit: int = 3
    overwrite_output_dir: bool = False


@dataclass
class AugmentationConfig:
    """Data augmentation applied during training only (disabled at eval).

    Fields
    ------
    random_crop : bool
        Apply random spatial crop to input images.
    crop_scale : tuple[float, float]
        Minimum and maximum crop area ratio when ``random_crop`` is True.
        Example: ``(0.9, 1.0)`` crops between 90%-100% of the image.
    color_jitter : float
        Strength of color jitter augmentation (0.0 = disabled).
        Typical: 0.0 - 0.3.
    """

    random_crop: bool = False
    crop_scale: tuple[float, float] = (0.9, 1.0)
    color_jitter: float = 0.0


# ── Robot / assembly (composition selection + controlled override) ──


@dataclass
class RobotConfig:
    """Robot selection (mirrors YAML ``robot``).

    Only the robot *name* is declared here — it resolves to a ``RobotProfile``.
    All body facts (joints, gripper, safety bounds, …) live in the profile, not
    in the recipe.
    """

    name: str = ""


@dataclass
class AssemblyConfig:
    """Controlled overrides for the data × model × robot composition.

    These are only consulted when the composition resolver cannot uniquely
    determine a relationship, or when the user wants a non-default strategy
    (architecture §3.1 "组合调整区"). Every field is optional and defaults to
    "unset". They never override objective facts (shapes, checkpoint slots,
    joint topology, fixed dim caps).

    Only overrides a resolver stage actually consumes live here. An adjustment
    nothing reads would be a field a user can set and watch do nothing, so the
    frequency and gripper knobs return when their checks do (see the deferred
    table in ``docs/plans/phase2-resolution-diagnostics.cn.md``).
    """

    camera_mapping: dict[str, str] | None = None
    default_task: str | None = None


# ── Top-level recipe ──────────────────────────────────────────────


@dataclass
class TrainRecipe:
    """Complete training recipe parsed from a single YAML file.

    Fields
    ------
    model_name : str
        Registered model name. Must match a ``@register_vla`` entry.
        Examples: ``act``, ``pi0``, ``openvla-7b``.
    model_path : str | None
        Path to pretrained weights.  ``None`` means train from scratch
        (used with ``training_paradigm == "from_scratch"`` models like ACT).
    finetuning_strategy : str
        Which parameters to train. One of:
          - ``full``      : all parameters trainable (ACT from scratch)
          - ``lora``      : LoRA adapters on selected components
          - ``freeze``    : freeze specified components, train the rest
          - ``selective`` : only train specified components
    lora_config : LoraConfig | None
        LoRA parameters.  Required when ``strategy == "lora"``.
    freeze_components : list[str] | None
        Component names to freeze.  Used when ``strategy == "freeze"``.
        Example: ``["vision_encoder"]`` to freeze the visual backbone.
    trainable_components : list[str] | None
        Component names to train.  Used when ``strategy == "selective"``.
        All other components are frozen.
    backend : str
        Training backend.  One of ``pytorch``, ``jax`` (reserved).
    lr : float
        Base learning rate.
    lr_backbone : float | None
        Separate (usually lower) learning rate for the visual backbone.
        Used by ACT (ResNet).  ``None`` means same as ``lr``.
    batch_size : int
        Training batch size (per GPU).
    total_steps : int
        Total number of training optimizer steps.
    gradient_checkpointing : bool
        Enable gradient checkpointing to reduce VRAM at the cost of ~30% speed.
    augmentation : AugmentationConfig
        Training-time data augmentation settings.
    output_dir : str
        Directory for checkpoints, logs, and final model weights.
    """

    # Model
    model_name: str = ""
    model_path: str | None = None
    model_config: dict = field(default_factory=dict)  # free-form, adapter-specific

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Robot / assembly (composition selection + controlled override)
    robot: RobotConfig = field(default_factory=RobotConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)

    # Fine-tuning
    finetuning_strategy: str = "full"
    lora_config: LoraConfig | None = None
    freeze_components: list[str] | None = None
    trainable_components: list[str] | None = None

    # Training
    backend: str = "pytorch"
    lr: float = 1e-4
    lr_backbone: float | None = None
    batch_size: int = 8
    total_steps: int = 10000
    gradient_checkpointing: bool = False
    num_workers: int = 4
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Output
    output: OutputConfig = field(default_factory=OutputConfig)

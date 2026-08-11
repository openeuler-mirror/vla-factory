"""Configuration dataclasses for a TrainRecipe.

These mirror the YAML structure defined in the v2.0 architecture.  A YAML
file is parsed into a ``TrainRecipe`` object which is then consumed by the
training engine.  All fields have sensible defaults so a minimal YAML works.

For a fully-annotated example see ``configs/reference.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Action spec ────────────────────────────────────────────────────


@dataclass
class ActionSpecConfig:
    """Robot action space definition (mirrors YAML ``action_spec``).

    Fields
    ------
    action_dim : int
        Number of action dimensions output by the policy.
        Examples: 6 (SO101 single arm), 14 (ALOHA bimanual), 7 (delta EEF + gripper).
    action_horizon : int
        Number of future timesteps the model predicts per inference call.
        Also called ``chunk_size`` in some frameworks.
        Examples: 50 (PI0), 100 (ACT), 7 (OpenVLA).
    action_type : str
        Semantic type of the action vector. One of:
          - ``joint_pos``   : absolute joint positions
          - ``delta_joint`` : joint position deltas
          - ``delta_eef``   : end-effector pose deltas (x,y,z,rx,ry,rz,gripper)
          - ``se3``         : SE(3) end-effector pose
          - ``tokenized``   : discretized action tokens (OpenVLA)
    bounds_low / bounds_high : list[float] | None
        Per-dimension action bounds for clipping / normalization.
        Length must equal ``action_dim``.  ``None`` means unbounded.
    """

    action_dim: int = 6
    action_horizon: int = 50
    action_type: str = "joint_pos"
    bounds_low: list[float] | None = None
    bounds_high: list[float] | None = None


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
class SamplerConfig:
    """How raw episodes are sliced into training samples.

    Fields
    ------
    type : str
        Sampling strategy. Currently only ``sliding_window``.
    n_obs_steps : int
        Number of observation frames in the input window.
        Most models use 1 (single frame); some use 2-3 for temporal stacking.
    action_horizon : int
        Number of future action steps per sample.
        Should match ``ActionSpecConfig.action_horizon`` in most cases.
    """

    type: str = "sliding_window"
    n_obs_steps: int = 1
    action_horizon: int = 50


@dataclass
class SplitConfig:
    """Train / validation data splitting strategy.

    Fields
    ------
    strategy : str
        How to partition the data. One of:
          - ``episode`` : split by whole episodes (recommended — no data leakage)
          - ``random``  : randomly assign individual samples (not yet implemented)
          - ``task``    : split by task name (not yet implemented)
    train_ratio : float
        Fraction of data used for training (0.0 - 1.0).
        The remaining ``1 - train_ratio`` fraction goes to validation.
    seed : int
        Random seed for reproducible splits.
    """

    strategy: str = "episode"
    train_ratio: float = 0.9
    seed: int = 42


@dataclass
class DataConfig:
    """Complete data pipeline configuration.

    Fields
    ------
    source : DataSourceConfig
        Where to find data and how to read it.
    sampler : SamplerConfig
        How to slice episodes into training samples.
    split : SplitConfig
        How to divide data into train / val sets.
    """

    source: DataSourceConfig = field(default_factory=DataSourceConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


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

    Consumed by the ``resolve`` dry-run today; ``train()`` behaviour is
    unchanged until the resolver takes over downstream execution.
    """

    camera_mapping: dict[str, str] | None = None
    accept_fps_mismatch: bool | None = None
    gripper_flip: bool | None = None
    default_task: str | None = None


def get_camera_mapping(recipe: "TrainRecipe", *, warn_legacy: bool = True):
    """Resolve the camera mapping with the architecture §3.1 zone priority.

    ``assembly.camera_mapping`` (组合调整区) is the preferred home; the legacy
    ``model.config.camera_mapping`` (组合选择区) is accepted for backwards
    compatibility with a deprecation warning. Returns a plain dict or None.
    """
    import logging

    if recipe.assembly.camera_mapping:
        return dict(recipe.assembly.camera_mapping)
    legacy = (recipe.model_config or {}).get("camera_mapping")
    if legacy:
        if warn_legacy:
            logging.getLogger(__name__).warning(
                "camera_mapping is declared under model.config (legacy). Move it "
                "to the `assembly:` block (architecture §3.1) — model.config "
                "support will be removed once the resolver owns this relationship."
            )
        return dict(legacy)
    return None


def get_default_task(recipe: "TrainRecipe", *, warn_legacy: bool = True) -> str | None:
    """Resolve the task-text fallback with the architecture §3.1 zone priority.

    Same split as :func:`get_camera_mapping`: the language fallback describes a
    data/model relationship, so it belongs to the ``assembly:`` block; the legacy
    ``model.config.default_task`` still works with a deprecation warning.
    """
    import logging

    if recipe.assembly.default_task:
        return recipe.assembly.default_task
    legacy = (recipe.model_config or {}).get("default_task")
    if legacy:
        if warn_legacy:
            logging.getLogger(__name__).warning(
                "default_task is declared under model.config (legacy). Move it "
                "to the `assembly:` block (architecture §3.1) — model.config "
                "support will be removed once the resolver owns this relationship."
            )
        return legacy
    return None


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
    inference_steps : int
        **Deprecated** — use ``model.config.num_inference_steps``. Denoising
        steps are a model knob, not a training one; the parser forwards a value
        set here (with a warning) and the field is removed when the recipe is
        slimmed down. Nothing reads this field directly.
    augmentation : AugmentationConfig
        Training-time data augmentation settings.
    output_dir : str
        Directory for checkpoints, logs, and final model weights.
    """

    # Model
    model_name: str = ""
    model_path: str | None = None
    model_config: dict = field(default_factory=dict)  # free-form, adapter-specific

    # Action space
    action_spec: ActionSpecConfig = field(default_factory=ActionSpecConfig)

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
    inference_steps: int = 1
    num_workers: int = 4
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Output
    output: OutputConfig = field(default_factory=OutputConfig)

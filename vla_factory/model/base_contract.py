"""Base checkpoint contract reader + model-config reporter.

A pretrained VLA checkpoint (lerobot / openpi pytorch format) ships a
``config.json`` describing the input contract it was trained with — which
camera roles it expects (``observation.images.<role>``), the state/action
dims, image resolution, etc. This module reads that contract so the framework
can:

  * validate a recipe's ``camera_mapping`` against what the base *actually*
    expects (roles + counts), instead of hard-coding a fixed role set;
  * power ``vlafactory-cli list --config <recipe>`` — showing the user, for
    any base checkpoint, which cameras it needs and how their dataset maps on.

The contract is read from the base ``config.json`` (source of truth) — never
from a hard-coded lookup table — so newly released checkpoints work without
framework changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid runtime import cycles
    from vla_factory.config.recipe import TrainRecipe
    from vla_factory.data.manifest import DataSchema


@dataclass(frozen=True)
class BaseContract:
    """Input contract extracted from a base checkpoint's ``config.json``."""

    # camera role name -> tensor shape, e.g. {"base_0_rgb": (3, 224, 224)}
    camera_roles: dict[str, tuple[int, ...]] = field(default_factory=dict)
    state_dim: int | None = None
    action_dim: int | None = None          # output action dim (= pad target)
    max_action_dim: int | None = None
    image_resolution: tuple[int, int] | None = None
    model_type: str | None = None
    repo_or_path: str = ""

    @property
    def camera_role_names(self) -> list[str]:
        return list(self.camera_roles.keys())


def _load_config_json(model_path: str) -> dict:
    """Load ``config.json`` from a local dir, a local file, or an HF repo id."""
    p = Path(model_path)
    if p.is_dir():
        cfg_file = p / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"No config.json in directory {model_path!r}")
        return json.loads(cfg_file.read_text())
    if p.is_file():
        return json.loads(p.read_text())
    # Otherwise treat as a HuggingFace repo id (e.g. "lerobot/pi0_base").
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            f"Cannot resolve {model_path!r}: not a local path, and "
            "`huggingface_hub` is not installed. Install the model's extra "
            "(run `bash scripts/install.sh pi0` to set up the openpi environment)."
        ) from e
    cfg_path = hf_hub_download(repo_id=model_path, filename="config.json")
    return json.loads(Path(cfg_path).read_text())


def load_base_contract(model_path: str | None) -> "BaseContract | None":
    """Read a base checkpoint's input contract.

    Returns ``None`` if *model_path* is falsy (no base configured). Raises if
    the path cannot be resolved or ``config.json`` is malformed.
    """
    if not model_path:
        return None
    cfg = _load_config_json(model_path)

    camera_roles: dict[str, tuple[int, ...]] = {}
    state_dim: int | None = None
    for key, feat in (cfg.get("input_features") or {}).items():
        shape = tuple(feat.get("shape", ())) if isinstance(feat, dict) else ()
        ftype = (feat.get("type") or "").upper() if isinstance(feat, dict) else ""
        if key.startswith("observation.images.") and ftype == "VISUAL":
            camera_roles[key[len("observation.images."):]] = shape
        elif ftype == "STATE" or key == "observation.state":
            state_dim = shape[0] if shape else None

    action_dim: int | None = None
    out = cfg.get("output_features") or {}
    act = out.get("action") if isinstance(out, dict) else None
    if isinstance(act, dict):
        shape = act.get("shape") or []
        action_dim = shape[0] if shape else None

    img_res = cfg.get("image_resolution")
    return BaseContract(
        camera_roles=camera_roles,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action_dim=cfg.get("max_action_dim"),
        image_resolution=tuple(img_res) if isinstance(img_res, (list, tuple)) else None,
        model_type=cfg.get("type"),
        repo_or_path=model_path,
    )


def check_camera_mapping(
    mapping: dict | None,
    contract: "BaseContract | None",
    dataset_cameras: list[str],
) -> dict:
    """Validate ``{base_role: dataset_camera}`` against the base contract.

    Returns a report dict:
      ``mapped``   — [(role, dataset_camera), ...] explicitly mapped & valid
      ``empty``    — [role, ...] base roles with no dataset camera (placeholder)
      ``unused``   — [dataset_camera, ...] not referenced by any mapping
      ``errors``   — [str, ...] hard problems (illegal role, missing camera)
      ``warnings`` — [str, ...] soft hints (view-semantics nudge, …)
    """
    report: dict = {"mapped": [], "empty": [], "unused": [], "errors": [], "warnings": []}

    if contract is None:
        report["warnings"].append(
            "model.path not set; base contract unreadable, camera_mapping not validated."
        )
        return report

    base_roles = contract.camera_role_names
    mapping = mapping or {}
    dataset_set = set(dataset_cameras)

    # 1. mapping keys must be a subset of the base's expected roles.
    for role in mapping:
        if role not in base_roles:
            report["errors"].append(
                f"camera_mapping key {role!r} is not in the base's expected roles {base_roles}."
            )

    # 2. mapping values must be real dataset cameras.
    used: set[str] = set()
    for role, cam in mapping.items():
        if role not in base_roles:
            continue
        if cam not in dataset_set:
            report["errors"].append(
                f"camera_mapping {role!r}<-{cam!r}: {cam!r} is not a dataset camera {dataset_cameras}."
            )
        else:
            report["mapped"].append((role, cam))
            used.add(cam)

    # 3. base roles not present in the mapping → placeholder slots.
    for role in base_roles:
        if role not in mapping:
            report["empty"].append(role)

    # 4. dataset cameras not referenced by any mapping → unused.
    for cam in dataset_cameras:
        if cam not in used:
            report["unused"].append(cam)

    # 5. view-semantics nudge (pi0-class models are sensitive to view roles).
    if report["mapped"]:
        pairs = ", ".join(f"{cam}->{role}" for role, cam in report["mapped"])
        report["warnings"].append(
            f"VLA models are sensitive to view semantics — verify {pairs} are correct "
            f"(don't map by count alone)."
        )
    return report


def describe_model_config(recipe: "TrainRecipe", schema: "DataSchema | None" = None) -> str:
    """Build a human-readable report: model metadata + base contract + camera_mapping check.

    Powers ``vlafactory-cli list --config <recipe>``. Safe to call with
    *schema*=None (skips the dataset-camera diff).
    """
    from vla_factory.model.registry import list_entries  # local import; avoids cycles

    lines: list[str] = []

    # ── Registry metadata ──
    try:
        entries = list_entries()
    except Exception as e:  # registry load error (e.g. a broken entry)
        entries = {}
        lines.append(f"(failed to load model registry: {e})")
    meta = entries.get(recipe.model_name)

    if meta:
        caps = [meta.backend, meta.action_head_type]
        if meta.training_paradigm == "pretrained_finetune":
            caps.append("finetune-only")
        strat = [s for s, on in (("full", meta.support_full), ("lora", meta.support_lora),
                                 ("freeze", meta.support_freeze)) if on]
        if strat:
            caps.append("supports " + "/".join(strat))
        lines.append(f"Model:   {recipe.model_name}  (" + " · ".join(caps) + ")")
    else:
        lines.append(f"Model:   {recipe.model_name}  (not registered; known: {sorted(entries)})")

    lines.append(f"Base:    {recipe.model_path or '(not set)'}")

    # ── Base contract ──
    contract = None
    if recipe.model_path:
        try:
            contract = load_base_contract(recipe.model_path)
        except Exception as e:
            lines.append(f"\n⚠ could not read base contract: {e}")

    if contract:
        pad_target = contract.max_action_dim or contract.action_dim
        lines.append(
            f"Action:  dim={contract.action_dim}  pad->{pad_target}  "
            f"(max_action_dim={contract.max_action_dim})"
        )
        if contract.image_resolution:
            lines.append(
                f"Image:   {contract.image_resolution[0]}x{contract.image_resolution[1]}  "
                f"(type={contract.model_type})"
            )
        lines.append("")
        lines.append(f"Base expects cameras ({len(contract.camera_role_names)}):")
        for role, shape in contract.camera_roles.items():
            sh = "x".join(str(s) for s in shape) if shape else "?"
            lines.append(f"  {role:20s} ({sh})")
    elif recipe.model_path:
        lines.append("(base contract parse failed)")
    else:
        lines.append("(no base — model.path not set)")

    # ── Dataset cameras ──
    dataset_cameras = list(schema.cameras) if schema else []
    if schema is not None:
        lines.append("")
        lines.append(f"Dataset provides cameras ({len(dataset_cameras)}):")
        lines.append("  " + (", ".join(dataset_cameras) if dataset_cameras else "(none)"))

    # ── camera_mapping check ──
    mapping = (recipe.model_config or {}).get("camera_mapping")
    report = check_camera_mapping(mapping, contract, dataset_cameras)

    lines.append("")
    if mapping is None:
        if contract:
            lines.append("camera_mapping: (not in recipe) declare {role: dataset_camera} per base role.")
    elif mapping:
        lines.append("camera_mapping (from recipe):")
        for role, cam in report["mapped"]:
            lines.append(f"  {role:20s} ← {cam}")
        for role in report["empty"]:
            lines.append(f"  {role:20s} <- <EMPTY>   (placeholder)")

    if report["errors"]:
        lines.append("")
        lines.extend(f"✗ {e}" for e in report["errors"])
    if report["mapped"] or report["empty"]:
        lines.append("")
        lines.append(
            f"✓ mapping valid: {len(report['mapped'])} real + {len(report['empty'])} placeholder."
        )
    if report["unused"]:
        lines.append(f"⚠ dataset cameras unused: {report['unused']}")
    if report["warnings"]:
        lines.append("")
        lines.extend(f"⚠ {w}" for w in report["warnings"])

    return "\n".join(lines)

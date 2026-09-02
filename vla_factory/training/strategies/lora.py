"""LoRA fine-tuning strategy via peft.

Injects PEFT LoRA adapters into the model's trainable components. Unlike
freeze/selective (which toggle ``requires_grad``), LoRA wraps linear layers
with low-rank adapters; the base weights stay frozen.

Default behavior (the contract a bare recipe gets)
--------------------------------------------------
A recipe that only sets ``type: lora`` + ``r`` + ``lora_alpha`` gets LoRA on
every declared component subtree: ``components`` defaults to ``"all"``
(every key of ``metadata.components``, each subtree wrapped individually),
``freeze_components`` defaults to ``[]`` (nothing extra frozen), and
``target_modules`` defaults to ``"all-linear"`` (peft's special string
matching every Linear/Conv1D inside each wrapped subtree). This is the
simplest "LoRA everything" and is parameter-equivalent to openpi's low-mem
configs (``gemma_2b_lora`` + ``gemma_300m_lora`` — both VLM and action
expert get LoRA) and to llamafactory's ``lora_target="all"`` default.

Design
------
``finetuning.config.components`` names components declared in
``ModelMetadata.components`` (e.g. ``"llm" ->
["paligemma_with_expert.paligemma."]``). The field is named ``components`` —
not ``target_components`` — to match freeze/selective: every strategy names
its subject the same way (the keys of ``metadata.components``), and the
``target_`` prefix carried no extra meaning (freeze's/components are equally
"targets"). Each prefix locates a subtree to wrap; only those subtrees get
adapters.

The default ``"all"`` (a string, not an empty list) expands to every key of
``metadata.components`` — each declared component goes through the same
per-subtree path once. A list (e.g. ``["llm"]``) restricts LoRA to those
subtrees only.

Freeze semantics: peft freezes the base weights INSIDE every wrapped
subtree; parameters OUTSIDE them (for pi0 with ``components: ["llm"]``:
action expert, state/action/time projections) keep ``requires_grad=True``
and are fully fine-tuned. The same boundary holds under ``"all"``, because
each subtree is wrapped on its own and the projections live outside them
all. So ``components: ["llm"]`` means "LoRA adapters on the VLM + full
FT of everything else", like openpi's ``paligemma_variant="gemma_2b_lora"``
paired with a non-lora action expert. The stats log below splits the two so
the numbers aren't misread as adapter size.

``finetuning.config.freeze_components`` (default ``[]``) names subtrees to
freeze (``requires_grad=False``) instead of full-FT. This closes the one gap
``_wrap_subtree`` could not cover on its own: it only knows "LoRA inside the
selected subtree, full-FT outside", so a recipe could not express "action_expert
frozen + llm LoRA". With ``freeze_components: ["action_expert"]`` that combo
is expressible; empty (the default) keeps the original full-FT-outside
behavior. A component must not appear in both ``components`` and
``freeze_components`` (validated at config parse for lists, and again in
``apply_lora`` after ``"all"`` expansion — the parse-time check cannot see
through the string). The freeze runs on the *unwrapped* model, before any
peft injection: the prefixes match clean parameter names, and peft cannot
rename what it never touched. The freeze itself reuses
``basic._freeze_components`` so the freeze strategy and the lora-with-freeze
path share one implementation.

``finetuning.config.target_modules`` (default ``"all-linear"``) is the
linear-layer-name set peft matches inside the wrapped subtree. It is forwarded
to peft verbatim — so any peft special string (``"all-linear"`` matches every
Linear/Conv1D), a regex string, or an explicit list (``["q_proj","v_proj"]``)
works. It is *not* a model-level fact: different backbones share the same
peft matching semantics, and a recipe's choice of which linear layers to
adapt is a per-run training decision, not a model-interface declaration.

What this design CANNOT express (known limitation): per-component different
LoRA configs (e.g. rank 16 on the VLM and rank 32 on the action expert). A
single peft_config is shared across every subtree's ``get_peft_model`` call,
so r/alpha/target_modules are uniform across all wrapped subtrees. openpi
achieves per-component
differences by giving each variant its own config; that is out of scope here
until a recipe needs it.

Adapter storage: the strategy finalizes by merging adapters into the base model
and returns a clean inference state dict. Checkpoint persistence itself remains
in ``training/checkpoint.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from vla_factory.model.model_interface import ModelMetadata
from vla_factory.training.strategies.base import FinetuningStrategy
from vla_factory.training.strategies.basic import _freeze_components
from vla_factory.training.strategies.registry import register_strategy
from vla_factory.utils.format import human_count

logger = logging.getLogger(__name__)

# peft's "all-linear" special string selects every Linear/Conv1D layer
# inside the wrapped subtree. This is the default for target_modules and
# aligns with llamafactory's lora_target="all" and openpi's LoRA-on-attention
# +ffn behavior. A recipe may override it per-run (e.g. ["q_proj","v_proj"]).
_DEFAULT_TARGET_MODULES = "all-linear"


@dataclass(frozen=True)
class LoraConfig:
    """Strict LoRA configuration owned by :class:`LoRAStrategy`."""

    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = False
    init_lora_weights: object = "gaussian"
    # Renamed from ``target_components`` to align with freeze/selective: every
    # strategy names its subject the same way (keys of metadata.components).
    # The "target_" prefix carried no extra meaning.
    #
    # Default "all" (the string) wraps every key of metadata.components, i.e.
    # LoRA on the whole model — matching llamafactory's lora_target="all"
    # behavior and openpi's low-mem configs (gemma_2b_lora + gemma_300m_lora,
    # i.e. both VLM and action expert get LoRA). Empty list/None is NOT a
    # valid value; the default is "all", not "nothing". The string form is
    # used so the default survives round-trip and reads like what it is.
    components: list[str] | str = "all"
    # Subtrees to freeze (requires_grad=False) instead of full-FT. Lets a
    # recipe express "action_expert frozen + llm LoRA" — the one gap
    # _wrap_subtree could not cover (it only knows "LoRA inside, full-FT
    # outside"). Empty (the default) keeps the original full-FT-outside
    # behavior. A component may appear in `components` OR `freeze_components`,
    # but never both (raising an overlap error is the validation's job).
    freeze_components: list[str] = field(default_factory=list)
    # peft target_modules: linear-layer names peft matches inside the wrapped
    # subtree. Default "all-linear" (peft special string selecting every
    # Linear/Conv1D), aligning with llamafactory's lora_target="all" and
    # openpi's attention+ffn LoRA. Override per-run, e.g. ["q_proj","v_proj"]
    # to match openpi's LoRA target set exactly. A list[str] or another peft
    # special string ("all-linear", a regex) is forwarded to peft verbatim.
    target_modules: list[str] | str = _DEFAULT_TARGET_MODULES

    def __post_init__(self) -> None:
        if isinstance(self.r, bool) or not isinstance(self.r, int) or self.r <= 0:
            raise ValueError("finetuning.config.r must be a positive integer")
        if (
            isinstance(self.lora_alpha, bool)
            or not isinstance(self.lora_alpha, int)
            or self.lora_alpha <= 0
        ):
            raise ValueError(
                "finetuning.config.lora_alpha must be a positive integer"
            )
        if not isinstance(self.lora_dropout, (int, float)) or not (
            0.0 <= self.lora_dropout < 1.0
        ):
            raise ValueError(
                "finetuning.config.lora_dropout must be in [0, 1)"
            )
        if not isinstance(self.use_rslora, bool):
            raise TypeError("finetuning.config.use_rslora must be a boolean")
        # components: "all" (default) or a non-empty list[str]. An empty list
        # is rejected so "forgot to set it" doesn't silently mean "no LoRA".
        # An empty list is a value error (the type list[str] is fine, it just
        # names nothing); a non-str/empty-str element is a type error.
        if isinstance(self.components, str):
            if self.components != "all":
                raise ValueError(
                    "finetuning.config.components as a string must be 'all'; "
                    f"got {self.components!r}"
                )
        elif not isinstance(self.components, list):
            raise TypeError(
                "finetuning.config.components must be 'all' or a list of names"
            )
        elif not self.components:
            raise ValueError(
                "finetuning.config.components must not be empty "
                f"(use 'all' for whole-model LoRA; known: {[]})"
            )
        elif any(
            not isinstance(name, str) or not name for name in self.components
        ):
            raise TypeError(
                "finetuning.config.components must be a list of non-empty "
                "strings"
            )
        if not isinstance(self.freeze_components, list) or any(
            not isinstance(name, str) or not name
            for name in self.freeze_components
        ):
            raise TypeError(
                "finetuning.config.freeze_components must be a list of names"
            )
        # target_modules: "all-linear" (default), another non-empty string
        # (treated by peft as a regex/special), or a non-empty list[str].
        # An empty list is rejected so "forgot to set it" ≠ "match nothing".
        if isinstance(self.target_modules, str):
            if not self.target_modules:
                raise ValueError(
                    "finetuning.config.target_modules must be a non-empty "
                    "string (e.g. 'all-linear') or a non-empty list"
                )
        elif (
            not isinstance(self.target_modules, list)
            or not self.target_modules
            or any(
                not isinstance(name, str) or not name
                for name in self.target_modules
            )
        ):
            raise TypeError(
                "finetuning.config.target_modules must be 'all-linear', a "
                "non-empty string, or a non-empty list of names"
            )
        # A component in BOTH components and freeze_components is a
        # contradiction (LoRA a subtree AND freeze it) — catch it here so the
        # failure is a clear config error, not "peft wrapped it then freeze
        # froze the adapters too".
        if (
            isinstance(self.components, list)
            and self.freeze_components
        ):
            overlap = set(self.components) & set(self.freeze_components)
            if overlap:
                raise ValueError(
                    "finetuning.config: components and freeze_components must "
                    f"not overlap; got {sorted(overlap)} in both"
                )


@register_strategy("lora")
class LoRAStrategy(FinetuningStrategy[LoraConfig]):
    config_type = LoraConfig

    def prepare_model(self, model, config, metadata):
        return apply_lora(model, config, metadata)

    def finalize_model(self, model):
        return merge_lora_adapters(model)

    def state_dict(self, model):
        return _strip_peft_prefix(model.state_dict())


def apply_lora(
    model: nn.Module,
    config: LoraConfig,
    metadata: ModelMetadata,
) -> nn.Module:
    """Inject LoRA adapters into the model's components.

    Each selected component subtree is wrapped in place with its own
    ``get_peft_model`` call (one shared peft config). Linears outside every
    wrapped subtree — e.g. pi0's state/action/time projections under the
    default ``"all"`` — are never seen by peft and keep full-parameter
    training.

    Returns the model with the wrapped subtrees re-mounted in place; the
    top-level object never changes.

    Raises ``ValueError`` if the model doesn't support LoRA, a component
    name is unknown, ``components`` overlaps ``freeze_components`` after
    ``"all"`` expansion, or a component resolves to no real subtree.
    """
    from peft import LoraConfig as PeftLoraConfig

    if not metadata.support_lora:
        raise ValueError(
            f"Model {metadata.name!r} does not support LoRA "
            f"(support_lora=False). Use a different finetuning strategy."
        )

    # 1. Expand components: "all" (the default) expands to every key of
    # metadata.components so a bare `finetuning: {type: lora, config: {r, lora_alpha}}`
    # recipe LoRAs every declared subtree. A list is taken verbatim. An empty
    # list is rejected by LoraConfig.__post_init__, so we never reach here
    # with nothing-to-wrap silently. Dedupe order-preserving: a repeated
    # name would wrap the same subtree twice (a PeftModel nested inside a
    # PeftModel).
    if config.components == "all":
        targets = list(metadata.components.keys())
        if not targets:
            raise ValueError(
                f"LoRA: components='all' but {metadata.name!r} declares no "
                f"components (metadata.components is empty)."
            )
    else:
        targets = list(config.components)
        if not targets:
            raise ValueError(
                "LoRA: finetuning.config.components is empty. "
                f"Known components for {metadata.name!r}: {list(metadata.components)}."
            )
    targets = list(dict.fromkeys(targets))

    # 2. Resolve components → subtree paths via metadata.components. Dedupe
    # across components: two names may declare the same prefix, and wrapping
    # a path twice would nest PeftModels.
    subtree_paths: list[str] = []
    for comp in targets:
        if comp not in metadata.components:
            # Skipping would silently shrink the adapter surface — the run
            # succeeds, trains fewer layers than the recipe asked for, and only
            # a log line records it. Fail instead (same rule as full.py).
            raise ValueError(
                f"LoRA: components entry {comp!r} is not declared in "
                f"ModelMetadata.components for model {metadata.name!r}. "
                f"Available components: {list(metadata.components)}."
            )
        subtree_paths.extend(metadata.components[comp])
    subtree_paths = list(dict.fromkeys(subtree_paths))

    if not subtree_paths:
        raise ValueError(
            f"LoRA: components {targets} resolved to no subtree paths. "
            f"metadata.components={metadata.components}"
        )

    # 3. Overlap check AFTER "all" expansion. LoraConfig.__post_init__ only
    # sees the list form, so a freeze_components entry naming a declared
    # component used to slip through when components='all' — and the freeze
    # then silently no-oped because peft had renamed the wrapped parameters
    # out from under the prefixes.
    overlap = set(targets) & set(config.freeze_components)
    if overlap:
        raise ValueError(
            "finetuning.config: components and freeze_components must "
            f"not overlap; got {sorted(overlap)} in both"
        )

    # 4. Freeze BEFORE wrapping. _freeze_components matches metadata
    # prefixes against parameter names; once peft wraps a subtree, those
    # names gain "base_model.model." segments and the prefixes miss (a
    # zero match raises in basic._freeze_components rather than silently
    # training). The overlap check above guarantees frozen components are
    # outside every wrapped subtree, so the subsequent wrapping cannot
    # disturb them. Reuses basic._freeze_components so the freeze strategy
    # and this path share one implementation.
    if config.freeze_components:
        _freeze_components(model, config.freeze_components, metadata)

    # 5. Forward peft-aligned fields to peft.LoraConfig as-is; vla only adds
    # the components → subtree mapping around it. target_modules is
    # forwarded verbatim (the "all-linear" default or a recipe override).
    # One config object is shared by every get_peft_model call — peft does
    # not mutate it.
    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_rslora=config.use_rslora,
        init_lora_weights=config.init_lora_weights,
        target_modules=config.target_modules,
    )

    # 6. Wrap each subtree in place, individually — never the whole model.
    # A whole-model wrap with target_modules="all-linear" would also adapt
    # and freeze linears OUTSIDE the declared components (pi0: the
    # state/action/time projections), inverting the "outside every wrapped
    # subtree stays full-FT" contract. openpi semantics preserved: only the
    # wrapped subtrees' base weights freeze; everything outside trains fully.
    for path in subtree_paths:
        _wrap_subtree(model, path, peft_config)

    _log_lora_stats(
        model, label=f"lora(r={config.r}, α={config.lora_alpha}, {targets})",
    )
    return model


def _wrap_subtree(model: nn.Module, subtree_path: str, peft_config) -> None:
    """Wrap one component subtree in place.

    Walk to the submodule named by ``subtree_path`` (e.g.
    ``paligemma_with_expert.paligemma.``), wrap it with ``get_peft_model``,
    and re-attach so the parent sees the peft-wrapped version. peft never
    sees anything outside this subtree.
    """
    from peft import get_peft_model

    path = subtree_path.rstrip(".")
    parent, leaf = _resolve_parent(model, path)
    if parent is None or leaf is None:
        # Falling back to a whole-model wrap changes where adapters land
        # (every matching linear layer, not the declared subtree) while the
        # run still succeeds — the LoRA equivalent of freezing the wrong
        # component. Fail so the wrong mount surface cannot ship.
        raise ValueError(
            f"LoRA: subtree {path!r} (declared in ModelMetadata.components) "
            f"does not exist on {type(model).__name__}. Adapters would land "
            "on the whole model instead of the intended subtree."
        )
    subtree = getattr(parent, leaf)
    setattr(parent, leaf, get_peft_model(subtree, peft_config))


def _resolve_parent(model: nn.Module, dotted_path: str):
    """Walk a.b.c -> return (obj_at_a_b, "c") or (None, None) if not found.

    Tries the path on *model* first, then on ``model.model`` if present —
    vla-factory wrappers (PI0ModelWrapper) hold the upstream model at
    ``self.model``, while metadata.components paths are written against the
    upstream's own structure (e.g. ``paligemma_with_expert.paligemma.``).
    """
    parent, leaf = _walk(model, dotted_path)
    if parent is not None:
        return parent, leaf
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return _walk(inner, dotted_path)
    return None, None


def _walk(root: nn.Module, dotted_path: str):
    parts = dotted_path.split(".")
    obj = root
    for p in parts[:-1]:
        if not hasattr(obj, p):
            return None, None
        obj = getattr(obj, p)
    leaf = parts[-1]
    return obj, leaf if hasattr(obj, leaf) else None


def _log_lora_stats(model: nn.Module, label: str) -> None:
    """Log trainable vs total params, splitting adapters from full-FT params.

    The split matters: with subtree LoRA, most trainable params are usually
    the fully fine-tuned modules OUTSIDE the wrapped subtree (pi0: the action
    expert), not the adapters — a single "trainable" number reads like
    adapter size and misleads memory/lr decisions.
    """
    total = sum(p.numel() for p in model.parameters())
    adapter = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    )
    full_ft = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" not in n
    )
    trainable = adapter + full_ft
    logger.info(
        "Strategy [%s]: %s (%s) trainable / %s (%s) total (%.2f%%) — "
        "%s (%s) LoRA adapters + %s (%s) full-FT (modules outside the wrapped "
        "subtree, trained like openpi's freeze filter)",
        label,
        f"{trainable:,}", human_count(trainable),
        f"{total:,}", human_count(total),
        100.0 * trainable / max(total, 1),
        f"{adapter:,}", human_count(adapter),
        f"{full_ft:,}", human_count(full_ft),
    )


def merge_lora_adapters(model: nn.Module) -> nn.Module:
    """Merge whole-model or subtree PEFT wrappers into their base weights."""
    from peft import PeftModel

    if isinstance(model, PeftModel):
        _merge_lora_layers_inplace(model)
        merged = model.merge_and_unload()
        _merge_peft_subtrees(merged)
        return merged
    _merge_peft_subtrees(model)
    return model


def _merge_peft_subtrees(module: nn.Module) -> None:
    """Recursively replace PEFT children with their merged base modules."""
    from peft import PeftModel

    for name, child in list(module.named_children()):
        if isinstance(child, PeftModel):
            _merge_lora_layers_inplace(child)
            merged = child.merge_and_unload()
            setattr(module, name, merged)
            _merge_peft_subtrees(merged)
        else:
            _merge_peft_subtrees(child)


def _merge_lora_layers_inplace(peft_model: nn.Module) -> None:
    """Fold LoRA deltas into base weights in bounded row chunks."""
    from peft.tuners.lora import LoraLayer

    for _, layer in peft_model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        for adapter in list(layer.lora_A.keys()):
            if adapter in layer.merged_adapters:
                continue
            if not _merge_lora_layer_chunked(layer, adapter):
                layer.merge(safe_merge=False, adapter_names=[adapter])


def _merge_lora_layer_chunked(layer, adapter: str, chunk_rows: int = 256) -> bool:
    """Merge one vanilla LoRA adapter without materializing its full delta."""
    if adapter in getattr(layer, "lora_variant", {}):
        return False
    if getattr(layer, "fan_in_fan_out", False):
        return False

    base = layer.get_base_layer()
    weight_a = layer.lora_A[adapter].weight
    weight_b = layer.lora_B[adapter].weight
    scaling = layer.scaling[adapter]

    with torch.no_grad():
        out_dim = weight_b.shape[0]
        for start in range(0, out_dim, chunk_rows):
            end = min(start + chunk_rows, out_dim)
            delta = torch.mm(weight_b[start:end], weight_a) * scaling
            base.weight.data[start:end] += delta.to(base.weight.dtype)

        if layer.lora_bias[adapter]:
            if getattr(base, "bias", None) is None:
                raise RuntimeError(
                    "lora_bias=True but base layer has no bias; cannot merge."
                )
            base.bias.data += (
                layer.lora_B[adapter].bias * scaling
            ).to(base.bias.dtype)

    layer.merged_adapters.append(adapter)
    return True


def _strip_peft_prefix(state_dict: dict) -> dict:
    """Remove PEFT's residual wrapper segment from finalized state keys."""
    return {
        (key.replace(".base_model.model.", ".")
         if ".base_model.model." in key else key): value
        for key, value in state_dict.items()
    }

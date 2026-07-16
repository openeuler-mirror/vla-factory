"""Task instruction → tokenized_prompt transform.

Language-conditioned VLA models (pi0, pi05) need the task string tokenized
into input ids + an attention mask. The tokenizer is loaded (lazily) from an
explicit ``tokenizer_repo`` or, failing that, the base checkpoint
(``recipe.model_path``).

The task text is resolved with a never-failing fallback chain (mirroring
openpi, whose ``InjectDefaultPrompt`` supplies a fixed prompt and whose models
require the prompt tensor to exist)::

    sample["task"]  >  default_task (recipe)  >  ""  (empty prompt)

``sample["task"]`` is the per-frame dataset text during training and the
client-supplied task at deployment (lerobot host adapter / simulator adapter →
``ObsDict.language``). ``default_task`` lets datasets that recorded only a
``task_index`` without text (common in custom recordings) still supply a
meaningful prompt. The final ``""`` keeps the model runnable either way —
``PI0Pytorch.embed_prefix`` embeds the prompt unconditionally, so producing an
empty-but-valid prompt beats skipping the fields and crashing in the model.

pi05 (``discrete_state: true``): the state vector is part of the discrete
language input — it is digitized into 256 bins over [-1, 1] and embedded into
the prompt as ``Task: <task>, State: <bins>;\\nAction: `` (openpi
``PaligemmaTokenizer.tokenize`` with a state argument). The state must already
be normalized (quantile → roughly [-1, 1]) and NOT yet padded, so this step
runs after ``normalize_vector`` and before ``pad_dimensions`` in the pi05
profile.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import TransformStep
from .registry import TransformRegistry

logger = logging.getLogger(__name__)

# openpi PaligemmaTokenizer: state is digitized into 256 bins over [-1, 1].
_STATE_BINS = np.linspace(-1, 1, 256 + 1)[:-1]


def build_prompt(task: str, state: np.ndarray | None = None) -> str:
    """Build the model prompt string, mirroring openpi's PaligemmaTokenizer.

    With ``state`` (pi05 discrete-state format)::

        Task: <cleaned task>, State: <256-bin digitized state>;\\nAction:{space}

    Without ``state`` (pi0), the cleaned task text is returned as-is.
    """
    cleaned = str(task).strip().replace("_", " ").replace("\n", " ")
    if state is None:
        return cleaned
    discretized = np.digitize(state, bins=_STATE_BINS) - 1
    state_str = " ".join(map(str, discretized))
    return f"Task: {cleaned}, State: {state_str};\nAction: "


@TransformRegistry.register("task_tokenize")
class TaskTokenize(TransformStep):
    """Tokenize ``sample["task"]`` → ``tokenized_prompt`` + ``tokenized_prompt_mask``.

    Padded/truncated to ``max_length``. Task resolution never fails:
    ``sample["task"]`` > ``default_task`` > ``""`` (see module docstring), so
    every sample leaves this step with prompt fields. With
    ``discrete_state=True`` (pi05) the normalized state vector is digitized
    into the prompt (see :func:`build_prompt`).
    """

    def __init__(
        self,
        tokenizer_repo: str | None = None,
        max_length: int = 48,
        default_task: str | None = None,
        discrete_state: bool = False,
    ) -> None:
        self.tokenizer_repo = tokenizer_repo
        self.max_length = int(max_length)
        self.default_task = default_task
        self.discrete_state = bool(discrete_state)
        self._tokenizer = None
        self._warned_empty_task = False

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            repo = self.tokenizer_repo
            if not repo:
                raise ValueError(
                    "task_tokenize needs a tokenizer source: set `tokenizer_repo` "
                    "in the transform config, or point the recipe's model.path at "
                    "a base checkpoint that ships a tokenizer."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(repo)
        return self._tokenizer

    def __call__(self, sample: dict) -> dict:
        task = sample.pop("task", None)
        if task is None:
            task = self.default_task
        if task is None:
            # Final fallback: empty prompt (openpi-style). The model requires
            # the prompt tensor to exist; an empty one is valid input. Warn
            # once so a dataset that silently lost its task text is noticed.
            task = ""
            if not self._warned_empty_task:
                self._warned_empty_task = True
                logger.warning(
                    "task_tokenize: sample has no task text and no default_task "
                    "is set — using an empty prompt. Fine for single-task "
                    "finetuning; set model.config.default_task in the recipe "
                    "for a meaningful prompt."
                )

        state = None
        if self.discrete_state:
            state = sample.get("state")
            if state is None:
                raise ValueError(
                    "task_tokenize discrete_state=True requires a normalized "
                    "`state` in the sample (pi05 embeds the state into the "
                    "prompt); run normalize_vector before this step."
                )

        tok = self._ensure_tokenizer()
        enc = tok(
            build_prompt(task, state),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        sample["tokenized_prompt"] = enc["input_ids"][0].astype(np.int64)
        sample["tokenized_prompt_mask"] = enc["attention_mask"][0].astype(bool)
        return sample

    @classmethod
    def from_config(cls, cfg: dict, ctx=None) -> "TaskTokenize":
        repo = cfg.get("tokenizer_repo")
        if repo is None and ctx is not None:
            recipe = getattr(ctx, "recipe", None)
            repo = getattr(recipe, "model_path", None) if recipe else None
        default_task = cfg.get("default_task")
        if default_task is None and ctx is not None:
            recipe = getattr(ctx, "recipe", None)
            mc = getattr(recipe, "model_config", None) or {}
            default_task = mc.get("default_task")
        return cls(
            tokenizer_repo=repo,
            max_length=cfg.get("max_length", 48),
            default_task=default_task,
            discrete_state=cfg.get("discrete_state", False),
        )

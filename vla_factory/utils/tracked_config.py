"""``TrackedConfig`` — the model config surface's "unread key is an error" gate.

A model declares its tunable defaults in ``ModelMetadata.params`` and the recipe
overrides them through ``model.config``. Nothing in that path checks whether a
declared key is ever *read*, so a key nobody consumes is a silent no-op: the
user edits it, the run behaves identically, and no error is raised. Two such
keys shipped in the baseline profiles — ``num_inference_steps`` (the engine read
the model metadata instead) and ``tokenizer_max_length`` (the ``task_tokenize``
step carries its own ``max_length``).

Wrapping the resolved config in a ``TrackedConfig`` and asserting at the end of
the factory turns that class of mistake into a startup error. It also makes the
two entries behave alike: ACT surfaced unknown keys only because it forwards
``**cfg`` to lerobot's ``ACTConfig``, while pi0 reads keys one by one with
``cfg.get()`` and drops anything it does not recognise.

This is a ``MutableMapping`` rather than a ``dict`` subclass on purpose: CPython
merges a ``dict`` subclass through the concrete fast path, so ``**cfg`` would
never call an overridden ``__getitem__`` and every forwarded key would look
unread. A ``Mapping`` forces ``**`` through ``keys()`` + ``__getitem__``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping
from typing import Any


# Keys consumed outside the model factory, which therefore can never be observed
# as "read" on the factory's config object:
#   transforms           — training/loader.py and inference/infer.py build the
#                          pipeline from it
#   num_inference_steps  — inference/infer.py drives predict_actions with it
#   camera_mapping       — recipe.get_camera_mapping() (legacy model.config
#                          location, kept during the assembly-block migration)
#   default_task         — assembly/transforms/task_tokenize.py reads it off
#                          recipe.model_config directly
FRAMEWORK_CONSUMED_KEYS: frozenset[str] = frozenset({
    "transforms",
    "num_inference_steps",
    "camera_mapping",
    "default_task",
})


class TrackedConfig(MutableMapping):
    """A mapping that records which keys were read.

    Behaves like the plain ``dict`` the factories used before; ``get`` / ``pop``
    / ``**`` expansion / item access all count as a read.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        framework_keys: Iterable[str] = FRAMEWORK_CONSUMED_KEYS,
    ) -> None:
        self._data: dict[str, Any] = dict(data or {})
        # Pre-mark framework-consumed keys so their absence from the factory's
        # own reads is not reported as an unused declaration.
        self._read: set[str] = {k for k in framework_keys if k in self._data}

    # ── Mapping protocol (every read path funnels through __getitem__) ──

    def __getitem__(self, key: str) -> Any:
        self._read.add(key)
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TrackedConfig({self._data!r})"

    # ── Gate ──

    def unread(self) -> list[str]:
        """Declared keys that nothing has read yet, sorted for stable messages."""
        return sorted(set(self._data) - self._read)

    def assert_all_consumed(self, model_name: str) -> None:
        """Raise if any declared key went unread.

        Called at the end of a model factory, once every key the model cares
        about has been taken out of the config.
        """
        leftover = self.unread()
        if not leftover:
            return
        raise ValueError(
            f"{model_name}: config key(s) {leftover} were declared but never "
            "read. A key nothing consumes is a silent no-op — overriding it in "
            "a recipe would change nothing. Either wire it into the model "
            f"factory, or drop it from ModelMetadata.params for {model_name!r}."
        )

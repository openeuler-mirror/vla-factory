"""Step-level profiling for the training loop.

Records the time spent in each stage of a training step — data loading,
host-to-device transfer, forward, backward, and optimizer step — and emits
a human-readable breakdown at every ``logging_steps`` so the bottleneck is
visible at a glance.

Integration point
-----------------
:class:`~vla_factory.training.trainer.VLATrainer` owns one
:class:`StepProfiler` and hooks it into the HuggingFace ``Trainer`` hot
points:

    dataloader            →  wrap_dataloader          (data_load)
    compute_loss          →  stage("h2d") / stage("forward")
    training_step return  →  record_backward_from()   (backward, by delta)
    optimizer.step        →  wrap_optimizer            (optimizer)
    lr_scheduler.step     →  wrap_lr_scheduler         (optimizer)
    log                   →  drain_and_report()        (emit)

The profiler is opt-in: ``VLATrainer(profile=...)`` controls whether a live
:class:`StepProfiler` is attached. When disabled, a single no-op instance is
used so every hook point stays on one code path without per-call guards.

Why a delta for backward
------------------------
``loss.backward()`` lives inside ``Trainer.training_step`` (via
``self.accelerator.backward``) and cannot be wrapped without forking the
method. Instead the forward end is timestamped inside ``compute_loss`` and
the backward span is the gap to ``training_step``'s return — the only work
in that gap is ``backward`` plus a cheap ``loss.detach()`` / loss-scaling,
which is folded in. The error is sub-millisecond and stable.

Window model
------------
A *micro-batch step* is one ``training_step`` call (one forward+backward).
One *optimizer step* (``global_step``) spans
``gradient_accumulation_steps`` micro-batches. The report window is keyed by
``logging_steps`` optimizer steps, so it holds
``logging_steps * gradient_accumulation_steps`` micro-batch samples. Every
stage measured here is on the main-process serial path, so
``sum(stages) + gap == wall`` exactly; the gap is grad clipping, callbacks,
and loop overhead that no stage covers. Dataloader-worker prefetch runs in
parallel to GPU compute but never re-enters the main-process timing — a
near-zero ``data_load`` therefore means "prefetch kept up" rather than "data
loading was free".
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

# Ordered stage list — the report is sorted by total time, but this is the
# canonical order used for accounting (sum + gap = wall) and labels.
#
# data_load, h2d, forward, backward, optimizer all sit on the main-process
# serial path, so their sum plus the framework "gap" (grad clip, callbacks,
# loop overhead) equals the end-to-end wall time exactly. Parallelism happens
# behind the scenes in dataloader workers (prefetch) and does NOT re-enter the
# main-process timing — see the module docstring.
STAGES: tuple[str, ...] = (
    "data_load",
    "h2d",
    "forward",
    "backward",
    "optimizer",
)

# Human-readable labels for the report (kept short to fit the column).
STAGE_LABEL: dict[str, str] = {
    "data_load": "data_load",
    "h2d": "h2d",
    "forward": "forward",
    "backward": "backward",
    "optimizer": "optimizer",
}

# Stages that occupy the GPU compute stream — used for the "GPU busy" line so
# the user can tell a GPU-bound run (high busy %) from a data/overhead-bound
# one (low busy %). h2d is excluded: it is a PCIe transfer that overlaps GPU
# compute rather than occupying the compute units.
GPU_STAGES: tuple[str, ...] = ("forward", "backward", "optimizer")


class StepProfiler:
    """Per-step stage timer aggregated over a logging window.

    When ``enabled`` is False the profiler is a complete no-op: every stage
    context, wrap, and report call short-circuits, so the trainer's hook
    points cost nothing but a flag check. This keeps a single code path for
    both profiled and unprofiled runs rather than scattering ``if`` guards
    across every hook.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        # Times for the in-flight micro-batch step.
        self._current: dict[str, float] = {}
        # Aggregated micro-batch steps since the last report.
        self._window: list[dict[str, float]] = []
        # Forward-end timestamp, set by mark_forward_end and consumed by
        # record_backward_from. None outside a step.
        self._t_fwd_end: float | None = None
        # Wall-clock span of the whole window (first stage start -> report).
        self._wall_start: float = perf_counter()
        # Whether a profile report has already been emitted (gates the very
        # first window, which is polluted by compile/warmup).
        self._reported_once: bool = False

    # -- per-stage timing -------------------------------------------

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a contiguous block and accumulate it into the current step.

        A no-op (no timing) when disabled, so the context manager protocol is
        preserved without the cost of a perf_counter call pair.
        """
        if not self.enabled:
            yield
            return
        t0 = perf_counter()
        try:
            yield
        finally:
            self._current[name] = self._current.get(name, 0.0) + (perf_counter() - t0)

    def mark_forward_end(self) -> None:
        """Record the forward->backward boundary (called inside compute_loss)."""
        if not self.enabled:
            return
        self._t_fwd_end = perf_counter()

    def record_backward_from(self) -> None:
        """Close the backward span opened by :meth:`mark_forward_end`.

        Called right after ``training_step`` returns. If no forward-end was
        marked (e.g. compute_loss was bypassed) the call is a no-op, so the
        profiler degrades gracefully on unusual code paths.
        """
        if not self.enabled:
            return
        t = self._t_fwd_end
        if t is not None:
            self._current["backward"] = self._current.get("backward", 0.0) + (perf_counter() - t)
            self._t_fwd_end = None

    def commit_step(self) -> None:
        """Seal the current micro-batch step into the window."""
        if not self.enabled:
            return
        if self._current:
            self._window.append(self._current)
        self._current = {}

    # -- wrapping external objects ----------------------------------

    def wrap_dataloader(self, dataloader: Any) -> Any:
        """Return a loader whose ``__next__`` is timed under ``data_load``."""
        if not self.enabled:
            return dataloader
        profiler = self

        class _ProfiledIterator:
            def __init__(self, it: Iterator[Any]) -> None:
                self._it = it

            def __iter__(self) -> "_ProfiledIterator":
                return self

            def __next__(self) -> Any:
                with profiler.stage("data_load"):
                    return next(self._it)

            def __len__(self) -> int:
                return len(self._it)  # type: ignore[arg-type]

        class _ProfiledLoader:
            def __init__(self, loader: Any) -> None:
                self._loader = loader

            def __iter__(self) -> _ProfiledIterator:
                return _ProfiledIterator(iter(self._loader))

            def __getattr__(self, name: str) -> Any:
                # Delegate dataset / collate_fn / batch_size etc.
                return getattr(self._loader, name)

        return _ProfiledLoader(dataloader)

    def wrap_optimizer(self, optimizer: Any) -> Any:
        """Time ``optimizer.step`` under the ``optimizer`` stage."""
        if not self.enabled or getattr(optimizer, "_vf_profiled", False):
            return optimizer
        original_step = optimizer.step

        def _profiled_step(*args: Any, **kwargs: Any) -> Any:
            with self.stage("optimizer"):
                return original_step(*args, **kwargs)

        optimizer.step = _profiled_step  # type: ignore[method-assign]
        optimizer._vf_profiled = True  # type: ignore[attr-defined]
        return optimizer

    def wrap_lr_scheduler(self, scheduler: Any) -> Any:
        """Time ``scheduler.step`` under the ``optimizer`` stage.

        LR scheduling runs in the same optimizer-sync block, so it shares the
        stage rather than getting its own row.
        """
        if (
            not self.enabled
            or scheduler is None
            or getattr(scheduler, "_vf_profiled", False)
        ):
            return scheduler
        original_step = scheduler.step

        def _profiled_step(*args: Any, **kwargs: Any) -> Any:
            with self.stage("optimizer"):
                return original_step(*args, **kwargs)

        scheduler.step = _profiled_step  # type: ignore[method-assign]
        scheduler._vf_profiled = True  # type: ignore[attr-defined]
        return scheduler

    # -- reporting ------------------------------------------------

    def drain_and_report(
        self,
        global_step: int,
        *,
        gradient_accumulation_steps: int = 1,
    ) -> str:
        """Aggregate the window, reset it, and return a formatted report.

        Returns an empty string when disabled, or when the window is empty
        (e.g. the very first ``log`` fires before any committed step). The
        first real window is still emitted but flagged, because it usually
        includes JIT/compile warmup that inflates forward/backward.
        """
        if not self.enabled:
            return ""
        window = self._window
        self._window = []
        if not window:
            return ""

        wall = perf_counter() - self._wall_start
        self._wall_start = perf_counter()

        first = not self._reported_once
        self._reported_once = True

        n_micro = len(window)
        n_optim = max(1, n_micro // max(1, gradient_accumulation_steps))
        return _format_report(
            window=window,
            wall=wall,
            global_step=global_step,
            n_micro=n_micro,
            n_optim=n_optim,
            first=first,
        )


def _format_report(
    *,
    window: Iterable[dict[str, float]],
    wall: float,
    global_step: int,
    n_micro: int,
    n_optim: int,
    first: bool,
) -> str:
    """Render the per-stage timing table.

    Stages are sorted by total time descending so the bottleneck is always
    at the top, marked with ``◀``. ``sum + gap`` must equal the wall time,
    which makes any silent gap (grad clipping, callbacks, loop overhead)
    visible rather than hidden inside a stage.
    """
    totals: dict[str, float] = defaultdict(float)
    maxs: dict[str, float] = defaultdict(float)
    mins: dict[str, float] = defaultdict(lambda: float("inf"))
    counts: dict[str, int] = defaultdict(int)

    for step in window:
        for stage in STAGES:
            v = step.get(stage)
            if v is None or v <= 0:
                continue
            totals[stage] += v
            counts[stage] += 1
            if v > maxs[stage]:
                maxs[stage] = v
            if v < mins[stage]:
                mins[stage] = v

    tracked = sum(totals.values())
    gap = max(0.0, wall - tracked)
    gpu_busy = sum(totals[s] for s in GPU_STAGES)

    present = [s for s in STAGES if counts[s] > 0]
    if not present:
        return ""
    present.sort(key=lambda s: totals[s], reverse=True)
    bottleneck = present[0]

    def _pct(x: float) -> str:
        return f"{(x / wall * 100):5.1f}%" if wall else "   —  "

    def _ms(x: float) -> str:
        return f"{x * 1000:7.1f}ms"

    width = 71
    bar = "─" * width
    lines: list[str] = []
    lines.append("┌─ Step Profile " + bar[: width - 15] + "┐")
    lines.append(
        f"│  端到端: {n_micro} micro-steps · global_step={global_step}"
        + ("  [warmup]" if first else "")
    )
    if wall > 0:
        lines.append(
            f"│  total {wall:6.2f}s · {n_micro / wall:5.2f} steps/s · "
            f"{n_optim / wall:5.2f} optim-steps/s"
        )
    lines.append("├" + bar + "┤")
    lines.append(
        f"│  {'stage':<12} {'total':>7}  {'share':>6}  {'avg':>9}  "
        f"{'max':>9}  {'min':>9}"
    )
    lines.append("│  " + "─" * (width - 4))

    for stage in present:
        c = counts[stage]
        t = totals[stage]
        avg = t / c if c else 0.0
        mn = mins[stage] if mins[stage] != float("inf") else 0.0
        mx = maxs[stage]
        marker = " ◀" if stage == bottleneck else "  "
        lines.append(
            f"│  {STAGE_LABEL[stage]:<12} {t:6.2f}s  {_pct(t)}  "
            f"{_ms(avg)}  {_ms(mx)}  {_ms(mn)}{marker}"
        )

    lines.append("│  " + "─" * (width - 4))
    lines.append(
        f"│  {'sum':<12} {tracked:6.2f}s  {_pct(tracked)}  "
        f"{'(各阶段之和)':>29}"
    )
    lines.append(
        f"│  {'gap':<12} {gap:6.2f}s  {_pct(gap)}  "
        f"{'(grad clip/回调/循环)':>29}"
    )
    lines.append("└" + bar + "┘")
    if wall > 0 and bottleneck:
        b_share = totals[bottleneck] / wall * 100
        gpu_pct = gpu_busy / wall * 100
        lines.append(
            f"  瓶颈: {STAGE_LABEL[bottleneck]} ({b_share:.1f}%) · "
            f"GPU占用 {gpu_pct:.1f}% (fwd+bwd+opt)"
        )
    return "\n".join(lines)

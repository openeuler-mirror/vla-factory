"""Transform base class: forward-only sample-level step.

Every transform — whether or not it has a logical inverse — implements the
``TransformStep`` interface.  The pipeline layer is **entirely forward-only**:
there is no ``inverse()`` method.  The normalize↔unnormalize symmetry is
realised through two independent forward-only classes whose formulas live in
the same lookup table, not through a generic inverse mechanism.

ColorJitter, RandomCrop, ResizeImages, PadDimensions are first-class citizens
that never need to pretend they are invertible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TransformStep(ABC):
    """A single forward-only transform step.

    A step takes and returns a ``dict`` with the same flat-key structure used
    by ``VLADataset.__getitem__`` — typically ``"state"``, ``"actions"`` and
    ``"images.<cam>"`` keys.

    Subclasses must implement ``__call__``, ``state_dict`` and
    ``from_state_dict``.  Serialisation is required so that a pipeline can be
    persisted to a checkpoint and reconstructed at inference time (the actual
    checkpoint round-trip lands in a later phase; the interface is in place
    now).
    """

    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        """Apply this step.  Take a sample dict, return the transformed sample."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for checkpoint storage.

        Large shared objects (e.g. dataset ``NormStats``) are NOT stored here —
        they are injected as ``**shared`` at reconstruction time.
        """
        ...

    @classmethod
    @abstractmethod
    def from_state_dict(cls, d: dict, **shared) -> "TransformStep":
        """Reconstruct an instance from a ``state_dict``.

        ``shared`` carries objects not stored in the dict (e.g.
        ``stats=norm_stats``); steps that don't need it simply ignore it.
        """
        ...

"""Floor version check shared by the lerobot-riding adapters.

ACT and the pi family all wrap lerobot 0.5 policies. A stale environment
still on lerobot 0.4 imports several of those modules fine but runs
pre-0.5 semantics (no ``action_is_pad`` requirement in forward, no
processor layer), which would fail obscurely mid-training or — worse —
quietly diverge from what the L1 parity tests verified against 0.5.1. The
``_try_import_*`` helpers refuse anything below ``LEROBOT_MIN`` up front,
so the factory raises its standard install-hint ``ImportError`` instead.

Not a registered adapter module (no ``@register_vla``); like
``lerobot_pi.py`` it is shared code living beside the entries, imported
lazily-safe by the registry's pkgutil scan.
"""

from __future__ import annotations

import importlib.metadata
import re

# Floor, not a pin: any lerobot 0.5.x+ the extras resolve is accepted.
LEROBOT_MIN = "0.5.0"

_MIN_PARTS = tuple(int(p) for p in re.findall(r"\d+", LEROBOT_MIN)[:2])


def lerobot_version_supported() -> bool:
    """True when the installed lerobot meets ``LEROBOT_MIN``.

    Also False when lerobot is not installed at all — callers treat both the
    same way (refuse, then let the factory's install-hint ImportError speak).
    """
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        return False
    parts = tuple(int(p) for p in re.findall(r"\d+", version)[:2])
    return parts >= _MIN_PARTS

"""Human-readable formatting helpers for logs."""

from __future__ import annotations


def human_count(n: int) -> str:
    """Format a count with a compact unit suffix: 850 → "850",
    606_343_712 → "606.3M", 3_529_679_120 → "3.5B".
    """
    for div, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n >= div:
            return f"{n / div:.1f}{suffix}"
    return str(n)

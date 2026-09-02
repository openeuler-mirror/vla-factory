"""L1 parity: vector normalisation epsilons against their upstream.

The epsilon in ``(x - mean) / (std + eps)`` is invisible on ordinary data and
decisive on near-zero-std dimensions — a constant gripper, a locked joint. At
std = 0, lerobot's 1e-8 amplifies by 1e8 while openpi's 1e-6 amplifies by 1e6:
a 100x difference in what the model is fed, with nothing at runtime to flag it.
Issue #7 opened on exactly this class of failure ("是 1e-6 还是 1e-8").

**This file is the reference shape for L1 constant blocks.** Every embedded
constant carries:

  1. the upstream repo + pinned commit (or version) it was read from,
  2. the file:line and the verbatim expression,
  3. the date it was verified.

Plus two mechanical guards, because a hand-transcribed constant can encode the
same misreading it is meant to catch:

  * ``test_openpi_pin_has_not_moved`` — the recorded commit must still equal the
    one ``scripts/install.sh`` installs. Bumping the pin without re-reading the
    constants turns this red.
  * ``test_*_eps_matches_the_installed_*`` — where the upstream is actually
    installed, the constant is re-derived from live upstream code rather than
    trusted. Skipped elsewhere; that is a skip, not a separate tier.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from vla_factory.assembly.transform.normalize import (
    DEFAULT_QUANTILE_EPS,
    DEFAULT_ZSCORE_EPS,
    NormalizeVector,
)
from vla_factory.data.data_schema import FeatureStats, NormStats
from vla_factory.model.registry import list_entries

pytestmark = pytest.mark.l1

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────
# UPSTREAM: openpi @ 15a9616a00943ada6c20a0f158e3adb39df2ccac
#   src/openpi/transforms.py:139  Normalize._normalize
#       return (x - mean) / (std + 1e-6)
#   src/openpi/transforms.py:145  Normalize._normalize_quantile
#       return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
# Verified: 2026-07-28
# ─────────────────────────────────────────────────────────────────────
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_ZSCORE_EPS = 1e-6
OPENPI_QUANTILE_EPS = 1e-6

# ─────────────────────────────────────────────────────────────────────
# UPSTREAM: lerobot 0.4.4 (pyproject `[act]` extra; exact PyPI release)
#   lerobot/processor/normalize_processor.py:94   eps: float = 1e-8
#   lerobot/processor/normalize_processor.py:335  denom = std + self.eps
# Verified: 2026-07-28
# ─────────────────────────────────────────────────────────────────────
LEROBOT_ZSCORE_EPS = 1e-8
LEROBOT_VERSION = "0.4.4"


def _profile_eps(model_name: str) -> float:
    """The eps a model's shipped metadata declares."""
    metadata = list_entries()[model_name]
    assert metadata.vector_normalization_eps is not None, (
        f"{model_name} metadata must declare `vector_normalization_eps` explicitly "
        "— the epsilon is a per-upstream contract and should be visible in the "
        "model declaration, not inherited silently from a code default."
    )
    return float(metadata.vector_normalization_eps)


# ── The profiles carry their own upstream's epsilon ───────────────────


def test_pi0_profile_uses_openpi_zscore_eps():
    """pi0 normalises with z-score, so it must use openpi's 1e-6, not lerobot's."""
    assert _profile_eps("pi0") == OPENPI_ZSCORE_EPS


def test_pi05_profile_uses_openpi_quantile_eps():
    assert _profile_eps("pi05") == OPENPI_QUANTILE_EPS


def test_act_profile_uses_lerobot_eps():
    assert _profile_eps("act") == LEROBOT_ZSCORE_EPS


def test_pi0_and_act_epsilons_are_not_the_same_value():
    """Guard the guard: if these ever converge, the tests above stop proving
    anything and a copy-paste between profiles would go unnoticed."""
    assert _profile_eps("pi0") != _profile_eps("act")


# ── The declared value is the one actually applied ───────────────────


def test_declared_eps_reaches_the_arithmetic():
    """A constant-std dimension makes the epsilon the entire denominator.

    With mean=0 and std=0, ``(x - 0) / (0 + eps)`` is exactly ``x / eps`` — so
    the output pins down which epsilon ran, with no tolerance games.
    """
    stats = NormStats(state=FeatureStats(mean=[0.0], std=[0.0]))
    for eps in (OPENPI_ZSCORE_EPS, LEROBOT_ZSCORE_EPS):
        step = NormalizeVector(stats, fields=("state",), method="zscore", eps=eps)
        out = step({"state": np.array([1.0], dtype=np.float32)})
        np.testing.assert_allclose(out["state"], [1.0 / eps], rtol=1e-6)


def test_the_two_epsilons_differ_by_two_orders_of_magnitude():
    """Quantify the stake: this is why the value cannot be 'close enough'."""
    stats = NormStats(state=FeatureStats(mean=[0.0], std=[0.0]))
    x = np.array([1.0], dtype=np.float32)

    openpi_out = NormalizeVector(
        stats, fields=("state",), method="zscore", eps=OPENPI_ZSCORE_EPS
    )({"state": x.copy()})["state"]
    lerobot_out = NormalizeVector(
        stats, fields=("state",), method="zscore", eps=LEROBOT_ZSCORE_EPS
    )({"state": x.copy()})["state"]

    assert lerobot_out[0] / openpi_out[0] == pytest.approx(100.0)


# ── Anti-transcription guards ────────────────────────────────────────


def test_openpi_pin_has_not_moved():
    """The recorded commit must still be the one install.sh installs.

    Bumping the pin without re-reading the constants above is the way an
    embedded golden goes stale silently; this turns that into a red test.
    """
    install_sh = (_PROJECT_ROOT / "scripts" / "install.sh").read_text()
    match = re.search(r'OPENPI_REF="([0-9a-f]{40})"', install_sh)
    assert match, "OPENPI_REF not found in scripts/install.sh — update this test"
    assert match.group(1) == OPENPI_COMMIT, (
        f"scripts/install.sh pins openpi at {match.group(1)}, but the constants "
        f"in this file were read from {OPENPI_COMMIT}. Re-read the upstream "
        "expressions cited above and update both the values and the header."
    )


def test_lerobot_pin_has_not_moved():
    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text()
    match = re.search(r'act = \["lerobot==([^\"]+)"\]', pyproject)
    assert match, "[act] must pin lerobot to an exact compatible release"
    assert match.group(1) == LEROBOT_VERSION


@pytest.mark.skipif(
    importlib.util.find_spec("openpi") is None,
    reason="openpi not installed (bash scripts/install.sh pi0)",
)
def test_openpi_eps_matches_the_installed_upstream():
    """Re-derive both epsilons from live openpi instead of trusting the header.

    openpi hardcodes the value inside the method, so it cannot be imported —
    but with std = 0 the normaliser returns ``x / eps``, which inverts cleanly.
    """
    from openpi.shared.normalize import NormStats as OpenpiNormStats
    from openpi.transforms import Normalize as OpenpiNormalize

    one = np.array([1.0], dtype=np.float32)
    zero = np.array([0.0], dtype=np.float32)

    zscore_stats = OpenpiNormStats(mean=zero, std=zero)
    derived_zscore = 1.0 / OpenpiNormalize(None)._normalize(one, zscore_stats)[0]
    assert derived_zscore == pytest.approx(OPENPI_ZSCORE_EPS, rel=1e-3)

    # Quantile: q01 = q99 = 0 → (x - 0) / (0 + eps) * 2 - 1 → x*2/eps - 1.
    quantile_stats = OpenpiNormStats(mean=zero, std=zero, q01=zero, q99=zero)
    normalized = OpenpiNormalize(None, use_quantiles=True)._normalize_quantile(
        one, quantile_stats
    )[0]
    derived_quantile = 2.0 / (normalized + 1.0)
    assert derived_quantile == pytest.approx(OPENPI_QUANTILE_EPS, rel=1e-3)


def test_lerobot_eps_matches_the_installed_upstream():
    """lerobot exposes eps as a dataclass field, so read it directly.

    Guarded on the *submodule*, not the package: ``lerobot`` alone being
    importable proves nothing, since the module path moved across versions
    (the openpi environment ships one without ``lerobot.processor`` at all).
    A guard that is coarser than what the test imports turns a version skew
    into a red test instead of a skip.
    """
    normalize_processor = pytest.importorskip(
        "lerobot.processor.normalize_processor",
        reason="installed lerobot has no processor.normalize_processor "
               "(module path differs across versions)",
    )

    step = getattr(normalize_processor, "NormalizerProcessorStep", None)
    if step is None or not hasattr(step, "eps"):
        pytest.skip("installed lerobot exposes no class-level normalizer eps")
    assert step.eps == LEROBOT_ZSCORE_EPS


# ── Code defaults stay put for pre-`eps` checkpoints ─────────────────


def test_code_defaults_are_the_historical_values():
    """A checkpoint saved before the `eps` key exists resolves through these.

    ``inference_metadata/recipe.yaml`` stores the *resolved* transform list and
    is replayed verbatim at inference — an old one carries no `eps`, so these
    defaults are what keep it reproducing its training-time normalisation.
    Changing them silently re-normalises every existing checkpoint.
    """
    assert DEFAULT_ZSCORE_EPS == LEROBOT_ZSCORE_EPS
    assert DEFAULT_QUANTILE_EPS == OPENPI_QUANTILE_EPS

    stats = NormStats(state=FeatureStats(mean=[0.0], std=[0.0]))
    step = NormalizeVector(stats, fields=("state",), method="zscore")  # no eps
    out = step({"state": np.array([1.0], dtype=np.float32)})
    np.testing.assert_allclose(out["state"], [1.0 / DEFAULT_ZSCORE_EPS], rtol=1e-6)

"""VLA model adaptation workflow package.

Functional boundary: everything the adaptation loop executes lives here.
``run.py`` at the ``agent/`` root is the thin interactive entry;
``python -m adapt_vla_model`` (run from ``agent/``) is the Judge CLI.
Layout, dependencies only pointing downward:

- ``runtime``   — infrastructure: JobManifest, JSON io, OpenCode invocation,
                  bwrap sandbox builder, provider-failure retry
- ``prompts``   — role-local Agent boundary (Reference / Adapter / Diagnosis)
- ``contract``  — Judge-owned Policy protocol copied into every bundle
- ``benchmarks/`` — Judge-owned benchmark discovery plus the benchmark data
                  itself (``<platform>/<task>/``: spec.json, contracts, run.py)
- ``judge``     — the Judge: workspace prep, compliance, preflight, parity,
                  repair state machine, verdicts, CLI
- ``parity/``   — loss-parity capture scripts, executed by file path inside
                  candidate venvs (kept self-contained on purpose; they rely
                  on the script directory being on ``sys.path``)
- policy servers — ``robotwin_policy_server.py`` is copied into Reference
  bundles as the benchmark-owned serving contract; ``factory_policy_server.py``
  wraps the Factory side. Both are path-executed standalones.
- ``test_vla_adapt_harness.py`` — the harness test suite (run with
  ``python -m unittest`` from ``agent/``)
"""

from __future__ import annotations

__all__ = ["judge", "prompts", "runtime", "contract", "benchmarks"]

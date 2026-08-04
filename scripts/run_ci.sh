#!/usr/bin/env bash
# Entry point for the vla-factory CI daemon.
# Delegates to scripts/ci/run_ci.py (interactive config + daemon launch).
exec python3 "$(dirname "$0")/ci/run_ci.py" "$@"

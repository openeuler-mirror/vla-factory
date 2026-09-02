#!/usr/bin/env bash
# vla-factory CI daemon entry point.
#
# On every new PR head, the daemon runs L0 in the configured base environment,
# L1/L2 in the act environment, and L1 in the pi environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/run_ci.sh" >&2
  exit 2
fi

exec python3 "$SCRIPT_DIR/ci/run_ci.py"

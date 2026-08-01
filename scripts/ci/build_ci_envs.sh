#!/usr/bin/env bash
# Build the three CI test environments in one shot.
#
# Usage:
#   bash ci/build_ci_envs.sh [base|act|pi]...      # default: all three
#
# Idempotent: an environment that already has its marker package installed is
# left alone. Delete the environment directory to force a rebuild.
#
# After installation, the python paths are printed — paste them into run_ci.py
# (or just press Enter there, the defaults already point here).
#
# Why three environments rather than one: openpi pins lerobot to an old commit
# through a uv git source, so no single environment can hold both openpi and a
# current lerobot. Each environment covers the part of L1 it can; cases whose
# upstream is absent skip themselves via pytest.importorskip.
#
#   base  core + [dev] only, CPU torch — deliberately CI-shaped. Running L0
#         here (rather than in a fully-loaded environment) is what exposes a
#         mis-scoped skip guard instead of hiding it.
#   act   + lerobot        → ACT parity, L2 overfit smoke. CPU torch is enough;
#         the smoke test is a CPU run by design.
#   pi    + openpi via scripts/install.sh, which picks the torch CUDA index
#         from the local compute capability (Blackwell → cu128).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_PREFIX="${VLAF_ENV_PREFIX:-$HOME/envs}"
PY_VERSION="${VLAF_PY_VERSION:-3.12}"
CPU_TORCH_INDEX="https://download.pytorch.org/whl/cpu"

# uv rather than conda or plain venv, for three reasons: CLAUDE.md makes uv the
# repository's dependency workflow; conda would demand accepting the
# anaconda.com channel terms, which an automated setup script has no business
# doing on its operator's behalf; and `python -m venv` needs ensurepip, which
# Debian/Ubuntu ship separately (a bare machine has neither it nor pip). uv
# bootstraps from a static binary and brings its own interpreter if the system
# one is unsuitable, so it is the only option that works on an untouched host.
# It also matches pi, which scripts/install.sh builds as a uv venv.
ensure_uv() {
  local candidate
  for candidate in uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      UV="$candidate"; return
    fi
  done
  echo "== bootstrapping uv (no pip on this host) =="
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh >/dev/null
  else
    echo "provision: need curl or wget to bootstrap uv" >&2
    exit 1
  fi
  UV="$HOME/.local/bin/uv"
  [[ -x "$UV" ]] || { echo "provision: uv bootstrap failed" >&2; exit 1; }
}
ensure_uv
echo "   uv: $("$UV" --version)"

# The default 30 s per-request budget is not enough for multi-hundred-MB torch
# wheels on a slow link; a mid-download timeout aborts the whole provision.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
mkdir -p "$ENV_PREFIX"

# Report the exact interpreter paths at the end — these are what go into
# ci/remote_gate.sh, and guessing them wrong is the most likely setup mistake.
declare -a SUMMARY=()

have_module() {  # <python> <module>
  "$1" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$2') else 1)" 2>/dev/null
}

provision_venv() {  # <label> <marker module> <extra pip args...>
  local label="$1" marker="$2"; shift 2
  local env_dir="$ENV_PREFIX/$label"
  local python_bin="$env_dir/bin/python"

  if [[ -x "$python_bin" ]] && have_module "$python_bin" "$marker"; then
    echo "== $label: already provisioned, skipping =="
  else
    echo "== $label: creating $env_dir =="
    [[ -x "$python_bin" ]] || "$UV" venv --python "$PY_VERSION" "$env_dir"
    # --no-sources is load-bearing: pyproject routes torch/torchvision to the
    # cu126 index via [tool.uv.sources] so that a GPU install resolves without
    # extra flags. base and act want the CPU build, and that routing overrides
    # a plain --index-url — without --no-sources these environments silently
    # pull ~3 GB of CUDA libraries that neither tier ever executes.
    "$UV" pip install --python "$python_bin" --quiet --no-sources \
        --index-url "$CPU_TORCH_INDEX" torch torchvision
    "$UV" pip install --python "$python_bin" --quiet --no-sources "$@"
  fi
  SUMMARY+=("$label $python_bin")
}

provision_via_install() {  # <label> <model> <marker module>
  local label="$1" model="$2" marker="$3"
  local env_dir="$ENV_PREFIX/$label"
  local python_bin="$env_dir/bin/python"

  if [[ -x "$python_bin" ]] && have_module "$python_bin" "$marker"; then
    echo "== $label: already provisioned, skipping =="
  else
    echo "== $label: installing via scripts/install.sh --model $model =="
    bash "$REPO_ROOT/scripts/install.sh" --model "$model" --venv "$env_dir"
  fi
  SUMMARY+=("$label $python_bin")
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(base act pi)

for target in "${targets[@]}"; do
  case "$target" in
    base) provision_venv base pytest -e ".[dev]" ;;
    act)  provision_via_install act act lerobot ;;
    pi)   provision_via_install pi  pi0 openpi ;;
    *)    echo "provision: unknown environment '$target' (expected base|act|pi)" >&2; exit 1 ;;
  esac
done

echo
echo "=========================================="
echo "  CI environments ready"
echo "=========================================="
echo
echo "  Run scripts/run_ci.sh and use these paths"
echo "  (or just press Enter — defaults match):"
echo
for entry in "${SUMMARY[@]}"; do
   read -r label python_bin <<<"$entry"
   printf "    VLAF_ENV_%-5s %s\n" "${label^^}" "$python_bin"
done
echo

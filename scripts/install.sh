#!/usr/bin/env bash
# install.sh — set up vla-factory model environment(s) with uv.
#
# Usage:
#   bash scripts/install.sh [--model {act|pi0|pi05}] [--venv <dir>] [-y|--yes]
#
#   --model   act | pi0 | pi05
#             Omit it to install ALL model environments (act, pi0, pi05) —
#             the script prints what will be installed and asks for
#             confirmation (yes/y) before proceeding. Use -y/--yes to skip
#             the prompt in non-interactive contexts.
#   --venv    venv directory    (default: ./.{model}, e.g. ./.act;
#                                only valid together with --model)
#   -y|--yes  assume yes at the all-models confirmation prompt
#
# Each model gets a minimal install — only its own deps, no cross-contamination:
#
#   act   lerobot from PyPI + CPU torch. No openpi, no CUDA, no patches.
#   pi0   openpi (pinned git source) + CUDA torch + transformers_replace patch.
#   pi05  same as pi0 (shares the openpi/pi0 codebase, pi05=True flag).
#
# Override torch backend for pi0: VLA_TORCH_BACKEND=cu126|cu128
# Override PyPI mirror:           VLA_PYPI_INDEX=https://...

set -euo pipefail

# ── Parse args ───────────────────────────────────────────────────────

MODEL=""
VENV_DIR=""
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --venv)    VENV_DIR="$2"; shift 2 ;;
    --venv=*)  VENV_DIR="${1#*=}"; shift ;;
    -y|--yes)  ASSUME_YES=1; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)         echo "install.sh: unexpected argument '$1'"; exit 1 ;;
  esac
done

if [[ -n "$VENV_DIR" && -z "$MODEL" ]]; then
  echo "install.sh: --venv is only valid together with --model (all-models mode uses ./.act ./.pi0 ./.pi05)"
  exit 1
fi

case "$MODEL" in
  act|pi0|pi05|"") ;;
  *) echo "install.sh: unknown model '$MODEL' (expected act|pi0|pi05)"; exit 1 ;;
esac

# ── All-models mode: notice + confirmation ───────────────────────────

if [[ -z "$MODEL" ]]; then
  MODELS=(act pi0 pi05)
  echo ""
  echo "== no --model given: installing ALL model environments =="
  echo "   act   -> ./.act   (lerobot + CPU torch)"
  echo "   pi0   -> ./.pi0   (openpi + CUDA torch)"
  echo "   pi05  -> ./.pi05  (openpi + CUDA torch)"
  echo ""
  echo "This creates 3 separate venvs and downloads multi-GB dependencies"
  echo "(torch CUDA wheels, openpi source); it can take a long while and"
  echo "use significant disk space."
  echo ""
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    echo "--yes given: skipping confirmation."
  elif [ ! -t 0 ]; then
    echo "install.sh: stdin is not a terminal — refusing to install everything"
    echo "without confirmation. Re-run with -y/--yes, or pick one model:"
    echo "  bash scripts/install.sh --model {act|pi0|pi05}"
    exit 1
  else
    read -r -p "Proceed with installing ALL environments? [yes/y] " answer || true
    ans="${answer:-}"
    ans="${ans,,}"
    case "$ans" in
      yes|y) ;;
      *) echo "Cancelled. Install a single environment with: bash scripts/install.sh --model {act|pi0|pi05}"; exit 1 ;;
    esac
  fi
else
  MODELS=("$MODEL")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Upgrade mode (set by build_ci_envs.sh --upgrade): without --upgrade, uv pip
# install treats already-satisfied requirements as done, so a rerun after a
# dependency bump would change nothing.
#   VLA_UPGRADE=1  resync repo-declared deps (install unsatisfied only — fast)
#   VLA_UPGRADE=2  additionally bump every dep to newest allowed (-U) — slow,
#                  re-downloads torch-sized wheels on upstream point releases
UV_UPGRADE=()
[[ "${VLA_UPGRADE:-0}" = "2" ]] && UV_UPGRADE=(--upgrade)

PYTHON_VERSION="3.12"

# openpi is pinned to a known-good commit for reproducibility (no release tags
# upstream). Bump deliberately after verifying compatibility.
OPENPI_REF="15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_DIR=".local-deps/openpi"
LEROBOT_COMMIT="${LEROBOT_COMMIT:-0cf864870cf29f4738d3ade893e6fd13fbd7cdb5}"
LEROBOT_DIR=".local-deps/lerobot"

# ── Shared helpers ───────────────────────────────────────────────────

# Ensure uv is available, with --torch-backend support.
if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv
fi
if ! uv pip install --help 2>/dev/null | grep -q -- "--torch-backend"; then
  pip install -q -U uv
fi

# Pick a PyPI mirror. Tsinghua is fastest from CN networks; overseas → PyPI.
detect_pypi_index() {
  if [ -n "${VLA_PYPI_INDEX:-}" ]; then
    echo "$VLA_PYPI_INDEX"; return
  fi
  if curl -sfS --max-time 3 -o /dev/null \
       "https://pypi.tuna.tsinghua.edu.cn/simple/setuptools/" 2>/dev/null; then
    echo "https://pypi.tuna.tsinghua.edu.cn/simple"; return
  fi
  echo "https://pypi.org/simple"
}
export UV_DEFAULT_INDEX="$(detect_pypi_index)"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

retry_uv() {
  local attempt max="${VLA_UV_ATTEMPTS:-5}"
  for attempt in $(seq 1 "$max"); do
    uv "$@" && return 0
    [[ "$attempt" -eq "$max" ]] && return 1
    echo "uv failed (attempt $attempt/$max); retrying in $((attempt * 5))s..." >&2
    sleep $((attempt * 5))
  done
}

# Pick a PyTorch CUDA wheel index from the GPU's compute capability (sm_xx).
#   compute_cap >= 10.0 (Blackwell) → cu128
#   compute_cap <  10.0 (Hopper and earlier) → cu126
# cu126/cu128 wheels bundle their own CUDA runtime; driver 550+ runs both.
detect_cuda_index() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cu126"; return
  fi
  local cc
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [ -z "$cc" ]; then echo "cu126"; return; fi
  local major=${cc%%.*}
  if [ "$major" -ge 10 ] 2>/dev/null; then echo "cu128"
  else echo "cu126"; fi
}

# ── Per-model install ────────────────────────────────────────────────
# Runs in a subshell (see main loop) so each venv activation is contained.

install_one_model() {
  local model="$1"
  local venv_dir
  if [[ -n "$VENV_DIR" ]]; then venv_dir="$VENV_DIR"; else venv_dir="./.$model"; fi

  # ── Create venv ────────────────────────────────────────────────────

  if [ ! -f "$venv_dir/bin/activate" ]; then
    retry_uv venv "$venv_dir" --python "$PYTHON_VERSION"
  fi
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"

  # ═══════════════════════════════════════════════════════════════════
  #  ACT — minimal: lerobot from PyPI + CPU torch
  # ═══════════════════════════════════════════════════════════════════
  if [[ "$model" == "act" ]]; then
    echo ""
    echo "== installing act (lerobot + CPU torch) =="

    retry_uv pip install \
      --default-index "$UV_DEFAULT_INDEX" \
      --torch-backend cpu \
      --index-strategy first-index \
      --no-sources-package torch \
      --no-sources-package torchvision \
      ${UV_UPGRADE[@]+"${UV_UPGRADE[@]}"} \
      -e ".[act,dev]"

    echo ""
    echo "Done."
    echo "  venv:   $(cd "$venv_dir" && pwd)"
    echo "  python: $(cd "$venv_dir" && pwd)/bin/python"
    echo ""
    echo "Activate and run:"
    echo "  source $(cd "$venv_dir" && pwd)/bin/activate"
    echo "  vlafactory-cli train --config examples/act.yaml"
    return 0
  fi

  # ═══════════════════════════════════════════════════════════════════
  #  PI0 / PI05 — full: openpi source + CUDA torch + transformers patch
  # ═══════════════════════════════════════════════════════════════════

  local cuda_index="${VLA_TORCH_BACKEND:-$(detect_cuda_index)}"
  case "$cuda_index" in
    cu126|cu128) ;;
    *) echo "install.sh: unsupported VLA_TORCH_BACKEND '$cuda_index' (expected cu126|cu128)"; return 1 ;;
  esac

  local uv_pip_flags=(
    --default-index "$UV_DEFAULT_INDEX"
    --torch-backend "$cuda_index"
    --index-strategy first-index
    --no-sources-package torch
    --no-sources-package torchvision
  )

  ensure_lerobot_source() {
    local tarball=".local-deps/lerobot.tar.gz"
    if [[ -f "$LEROBOT_DIR/pyproject.toml" && "$(cat "$LEROBOT_DIR/.vla-pin" 2>/dev/null)" == "$LEROBOT_COMMIT" ]]; then
      echo "lerobot source present at $LEROBOT_DIR (pin ${LEROBOT_COMMIT:0:8})."
      return
    fi
    rm -rf "$LEROBOT_DIR"
    echo "downloading lerobot tarball @ ${LEROBOT_COMMIT:0:8}..."
    mkdir -p .local-deps
    curl -fSL \
      --retry 8 --retry-delay 3 --retry-all-errors --retry-connrefused \
      --continue-at - \
      -o "$tarball" \
      "https://github.com/huggingface/lerobot/archive/${LEROBOT_COMMIT}.tar.gz"
    tar xzf "$tarball" -C .local-deps
    rm -f "$tarball"
    mv ".local-deps/lerobot-${LEROBOT_COMMIT}" "$LEROBOT_DIR"
    printf '%s\n' "$LEROBOT_COMMIT" > "$LEROBOT_DIR/.vla-pin"
  }

  ensure_openpi_source() {
    local tarball=".local-deps/openpi.tar.gz"
    # Freshness is decided by the pinned REF, not mere directory presence:
    # a PR bumping OPENPI_REF must re-download on --upgrade, not keep
    # serving the stale tree.
    if [[ -f "$OPENPI_DIR/pyproject.toml" && "$(cat "$OPENPI_DIR/.vla-pin" 2>/dev/null)" == "$OPENPI_REF" ]]; then
      echo "openpi source present at $OPENPI_DIR (pin ${OPENPI_REF:0:8})."
    else
      rm -rf "$OPENPI_DIR"
      echo "downloading openpi tarball @ ${OPENPI_REF:0:8}..."
      mkdir -p .local-deps
      curl -fSL \
        --retry 8 --retry-delay 3 --retry-all-errors --retry-connrefused \
        --continue-at - \
        -o "$tarball" \
        "https://github.com/Physical-Intelligence/openpi/archive/${OPENPI_REF}.tar.gz"
      tar xzf "$tarball" -C .local-deps
      rm -f "$tarball"
      mv ".local-deps/openpi-${OPENPI_REF}" "$OPENPI_DIR"
      printf '%s\n' "$OPENPI_REF" > "$OPENPI_DIR/.vla-pin"
    fi

    # uv pip install of a git dependency can fail on openpi's workspace source.
    # Make the workspace member an explicit local path for this extracted copy.
    sed -i \
      's/openpi-client = { workspace = true }/openpi-client = { path = "packages\/openpi-client" }/' \
      "$OPENPI_DIR/pyproject.toml"

    if [ "${VLA_LOCAL_LEROBOT:-0}" = "1" ]; then
      ensure_lerobot_source
      sed -i \
        's|lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" }|lerobot = { path = "../lerobot" }|' \
        "$OPENPI_DIR/pyproject.toml"
    fi
  }

  echo ""
  echo "== installing $model (openpi + $cuda_index torch) =="

  if [ "${VLA_LOCAL_LEROBOT:-0}" = "1" ]; then
    ensure_lerobot_source
  fi
  ensure_openpi_source
  retry_uv pip install "${uv_pip_flags[@]}" ${UV_UPGRADE[@]+"${UV_UPGRADE[@]}"} "$OPENPI_DIR"

  # Apply openpi's transformers_replace patch (SigLIP/PaliGemma/Gemma dtype fixes
  # required by PI0Pytorch). Safe overwrite — the patch is version-matched to the
  # pinned openpi commit.
  SP=$(python -c "import site; print(site.getsitepackages()[0])")
  cp -rf "$SP/openpi/models_pytorch/transformers_replace"/* "$SP/transformers/"

  # Install vla-factory itself (editable).
  retry_uv pip install "${uv_pip_flags[@]}" ${UV_UPGRADE[@]+"${UV_UPGRADE[@]}"} -e ".[dev]"

  # Optional: faster HF downloads for the multi-GB base weights.
  retry_uv pip install --default-index "$UV_DEFAULT_INDEX" -q hf_transfer || true

  echo ""
  echo "Done."
  echo "  venv:   $(cd "$venv_dir" && pwd)"
  echo "  python: $(cd "$venv_dir" && pwd)/bin/python"
  echo ""
  echo "Activate and run:"
  echo "  source $(cd "$venv_dir" && pwd)/bin/activate"
  echo "  vlafactory-cli train --config examples/$model.yaml"
}

# All-models mode reports per-model outcome and keeps going past failures
# (set -e would otherwise abort the whole run on the first bad model).
failed_models=()
for m in "${MODELS[@]}"; do
  # Subshell: contains the venv activation and function-local helpers.
  if ( install_one_model "$m" ); then
    echo "== $m: OK =="
  else
    failed_models+=("$m")
    echo "== $m: FAILED — continuing with remaining models ==" >&2
  fi
done
if [[ ${#failed_models[@]} -gt 0 ]]; then
  echo
  echo "install.sh: finished WITH FAILURES: ${failed_models[*]}" >&2
  exit 1
fi

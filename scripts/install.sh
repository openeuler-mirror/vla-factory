#!/usr/bin/env bash
# install.sh — set up a vla-factory model environment with uv.
#
# Usage:
#   bash scripts/install.sh --model {act|pi0|pi05} [--venv <dir>]
#
#   --model   act | pi0 | pi05  (required)
#   --venv    venv directory    (default: ./.{model}, e.g. ./.act)
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --venv)    VENV_DIR="$2"; shift 2 ;;
    --venv=*)  VENV_DIR="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)         echo "install.sh: unexpected argument '$1'"; exit 1 ;;
  esac
done

[[ -z "$MODEL" ]] && { echo "install.sh: --model is required (act|pi0|pi05)"; exit 1; }
# Default venv dir = model name (./act, ./pi0, ...).
[[ -z "$VENV_DIR" ]] && VENV_DIR="./.$MODEL"

case "$MODEL" in
  act|pi0|pi05) ;;
  *) echo "install.sh: unknown model '$MODEL' (expected act|pi0|pi05)"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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

# ── Create venv ──────────────────────────────────────────────────────

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  retry_uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ═══════════════════════════════════════════════════════════════════════
#  ACT — minimal: lerobot from PyPI + CPU torch
# ═══════════════════════════════════════════════════════════════════════
if [[ "$MODEL" == "act" ]]; then
  echo ""
  echo "== installing act (lerobot + CPU torch) =="

  retry_uv pip install \
    --default-index "$UV_DEFAULT_INDEX" \
    --torch-backend cpu \
    --index-strategy first-index \
    --no-sources-package torch \
    --no-sources-package torchvision \
    -e ".[act,dev]"

  echo ""
  echo "Done."
  echo "  venv:   $(cd "$VENV_DIR" && pwd)"
  echo "  python: $(cd "$VENV_DIR" && pwd)/bin/python"
  echo ""
  echo "Activate and run:"
  echo "  source $(cd "$VENV_DIR" && pwd)/bin/activate"
  echo "  vlafactory-cli train --config examples/act.yaml"
  exit 0
fi

# ═══════════════════════════════════════════════════════════════════════
#  PI0 / PI05 — full: openpi source + CUDA torch + transformers patch
# ═══════════════════════════════════════════════════════════════════════

CUDA_INDEX="${VLA_TORCH_BACKEND:-$(detect_cuda_index)}"
case "$CUDA_INDEX" in
  cu126|cu128) ;;
  *) echo "install.sh: unsupported VLA_TORCH_BACKEND '$CUDA_INDEX' (expected cu126|cu128)"; exit 1 ;;
esac

UV_PIP_FLAGS=(
  --default-index "$UV_DEFAULT_INDEX"
  --torch-backend "$CUDA_INDEX"
  --index-strategy first-index
  --no-sources-package torch
  --no-sources-package torchvision
)

ensure_lerobot_source() {
  local tarball=".local-deps/lerobot.tar.gz"
  if [ -f "$LEROBOT_DIR/pyproject.toml" ]; then
    echo "lerobot source present at $LEROBOT_DIR."
    return
  fi
  rm -rf "$LEROBOT_DIR"
  echo "downloading lerobot tarball @ ${LEROBOT_COMMIT:0.8}..."
  mkdir -p .local-deps
  curl -fSL \
    --retry 8 --retry-delay 3 --retry-all-errors --retry-connrefused \
    --continue-at - \
    -o "$tarball" \
    "https://github.com/huggingface/lerobot/archive/${LEROBOT_COMMIT}.tar.gz"
  tar xzf "$tarball" -C .local-deps
  rm -f "$tarball"
  mv ".local-deps/lerobot-${LEROBOT_COMMIT}" "$LEROBOT_DIR"
}

ensure_openpi_source() {
  local tarball=".local-deps/openpi.tar.gz"
  if [ -f "$OPENPI_DIR/pyproject.toml" ]; then
    echo "openpi source present at $OPENPI_DIR."
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
echo "== installing $MODEL (openpi + $CUDA_INDEX torch) =="

if [ "${VLA_LOCAL_LEROBOT:-0}" = "1" ]; then
  ensure_lerobot_source
fi
ensure_openpi_source
retry_uv pip install "${UV_PIP_FLAGS[@]}" "$OPENPI_DIR"

# Apply openpi's transformers_replace patch (SigLIP/PaliGemma/Gemma dtype fixes
# required by PI0Pytorch). Safe overwrite — the patch is version-matched to the
# pinned openpi commit.
SP=$(python -c "import site; print(site.getsitepackages()[0])")
cp -rf "$SP/openpi/models_pytorch/transformers_replace"/* "$SP/transformers/"

# Install vla-factory itself (editable).
retry_uv pip install "${UV_PIP_FLAGS[@]}" -e ".[dev]"

# Optional: faster HF downloads for the multi-GB base weights.
retry_uv pip install --default-index "$UV_DEFAULT_INDEX" -q hf_transfer || true

echo ""
echo "Done."
echo "  venv:   $(cd "$VENV_DIR" && pwd)"
echo "  python: $(cd "$VENV_DIR" && pwd)/bin/python"
echo ""
echo "Activate and run:"
echo "  source $(cd "$VENV_DIR" && pwd)/bin/activate"
echo "  vlafactory-cli train --config examples/$MODEL.yaml"

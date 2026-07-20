#!/usr/bin/env bash
# install.sh — set up a vla-factory model environment with uv.
#
# Usage:
#   bash scripts/install.sh [venv_dir] [model]
#     venv_dir  default .venv
#     model     default pi0  (pi0 | pi05; `act` to be added)
#
# This wraps openpi, whose strict == pins and in-place transformers patch make
# a plain `pip install -e ".[pi0]"` unresolvable. uv (PubGrub) handles the
# pins; the transformers_replace patch is applied after install.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${1:-.venv}"
MODEL="${2:-pi0}"
PYTHON_VERSION="3.12"

# openpi is pinned to a known-good commit for reproducibility (no release tags
# upstream). Bump deliberately after verifying compatibility.
OPENPI_REF="15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_DIR=".local-deps/openpi"
LEROBOT_COMMIT="${LEROBOT_COMMIT:-0cf864870cf29f4738d3ade893e6fd13fbd7cdb5}"
LEROBOT_DIR=".local-deps/lerobot"

case "$MODEL" in
  pi0|pi05)
    ;;
  act)
    echo "install.sh: 'act' not yet supported here; use 'pip install -e \".[act]\"' for now." >&2
    exit 1
    ;;
  *)
    echo "install.sh: unknown model '$MODEL' (expected pi0|pi05)" >&2
    exit 1
    ;;
esac

# Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv
fi
if ! uv pip install --help 2>/dev/null | grep -q -- "--torch-backend"; then
  pip install -q -U uv
fi

# Pick a PyTorch CUDA wheel index from the GPU's compute capability (sm_xx),
# NOT the driver's CUDA version — a Blackwell GPU (sm_100+, e.g. RTX 5090 sm_120)
# reports driver CUDA 12.4 but needs cu128 wheels (torch 2.8+) which ship the
# sm_100/sm_120 kernels; cu126's torch 2.7.1 tops out at sm_90 and would fail
# with "no kernel image is available for execution on the device".
#   compute_cap >= 10.0 (Blackwell) → cu128
#   compute_cap < 10.0 (Hopper sm_90 and earlier, e.g. RTX 4080 sm_89) → cu126
# cu126/cu128 wheels bundle their own CUDA runtime; driver 550+ runs both.
# Override with VLA_TORCH_BACKEND=cu126|cu128.
detect_cuda_index() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cu126"; return
  fi
  local cc
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [ -z "$cc" ]; then echo "cu126"; return; fi
  local major=${cc%%.*}
  if [ "$major" -ge 10 ] 2>/dev/null; then
    echo "cu128"
  else
    echo "cu126"
  fi
}
CUDA_INDEX="${VLA_TORCH_BACKEND:-$(detect_cuda_index)}"
case "$CUDA_INDEX" in
  cu126|cu128)
    ;;
  *)
    echo "install.sh: unsupported VLA_TORCH_BACKEND '$CUDA_INDEX' (expected cu126|cu128)" >&2
    exit 1
    ;;
esac

# Pick a default PyPI index for non-torch packages. The nvidia-*-cu12 CUDA libs
# that torch depends on exist on PyPI / Tsinghua (→ files.pythonhosted.org) with
# the same sha256 as on pypi.nvidia.com; some proxies (autodl's turbo tunnel)
# reset the pypi.nvidia.com connection mid-fetch, so routing those wheels
# through a mirror avoids that host entirely. Auto-detected to stay portable:
# Tsinghua is fastest from CN networks (autodl); overseas falls back to PyPI.
# Override with VLA_PYPI_INDEX.
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

retry_uv() {
  local attempt
  local max_attempts="${VLA_UV_ATTEMPTS:-5}"
  for attempt in $(seq 1 "$max_attempts"); do
    if uv "$@"; then
      return 0
    fi
    if [ "$attempt" -eq "$max_attempts" ]; then
      return 1
    fi
    echo "uv failed (attempt $attempt/$max_attempts); retrying in $((attempt * 5))s..." >&2
    sleep $((attempt * 5))
  done
}

ensure_lerobot_source() {
  local tarball=".local-deps/lerobot.tar.gz"
  if [ -f "$LEROBOT_DIR/pyproject.toml" ]; then
    echo "lerobot source present at $LEROBOT_DIR."
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
    # Keep openpi off fragile GitHub git transport in weak-network environments.
    sed -i \
      's|lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" }|lerobot = { path = "../lerobot" }|' \
      "$OPENPI_DIR/pyproject.toml"
  fi
}

UV_PIP_FLAGS=(
  --default-index "$UV_DEFAULT_INDEX"
  --torch-backend "$CUDA_INDEX"
  --index-strategy first-index
  --no-sources-package torch
  --no-sources-package torchvision
)

# Create the venv.
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  retry_uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install openpi from a local tarball extraction. Installing it as a direct git
# dependency can fail because openpi's pyproject references openpi-client as a
# workspace member, and uv builds git dependencies outside that workspace.
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

# Install vla-factory itself (editable). Keep any local path sources such as the
# autodl lerobot workaround, but let --torch-backend pick the torch wheels.
retry_uv pip install "${UV_PIP_FLAGS[@]}" -e .

# Optional: faster HF downloads for the multi-GB base weights.
retry_uv pip install --default-index "$UV_DEFAULT_INDEX" -q hf_transfer || true

echo ""
echo "Done. Activate and run:"
echo "  source $VENV_DIR/bin/activate"
echo "  vlafactory-cli list"
echo "  vlafactory-cli train --config examples/pi0.yaml"

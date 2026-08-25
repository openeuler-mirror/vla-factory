#!/usr/bin/env bash
# Build the three CI test environments in one shot.
#
# Usage:
#   bash scripts/ci/build_ci_envs.sh [-u|--upgrade] [base|act|pi]...
#                                       # default: all three
#
# Idempotent: an environment that already has its marker package installed is
# left alone (with a hint on how to resync it). Delete the environment
# directory to force a rebuild from scratch.
#
#   --upgrade  resync existing environments to the CURRENT REPO-DECLARED deps
#              instead of skipping them — run this after PRs changed
#              dependencies. Only unsatisfied requirements are installed, so
#              it downloads just the delta (fast, cache-friendly).
#   --latest   with --upgrade: additionally bump every dependency to the
#              newest allowed upstream release (-U). Big downloads: torch-
#              sized wheels re-download whenever upstream shipped a newer
#              point release, regardless of what any PR changed.
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
SKIPPED=0

have_module() {  # <python> <module>
  "$1" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$2') else 1)" 2>/dev/null
}

install_torchcodec_for_python() {  # <python>
  local python_bin="$1" torch_version tc_version
  if have_module "$python_bin" torchcodec; then
    return
  fi
  torch_version="$("$python_bin" -c 'import torch; print(torch.__version__)')"
  tc_version=""
  case "$torch_version" in
    2.7.*)  tc_version="0.4.0" ;;
    2.8.*)  tc_version="0.7.0" ;;
    2.9.*)  tc_version="0.9.0" ;;
    2.10.*) tc_version="0.10.0" ;;
    2.11.*) tc_version="0.11.0" ;;
    2.12.*) tc_version="0.15.0" ;;
    *)      tc_version="" ;;
  esac
  if [ -n "$tc_version" ]; then
    echo "== base: installing torchcodec==$tc_version for torch ${torch_version%%+*} =="
    "$UV" pip install --python "$python_bin" --quiet --no-sources "torchcodec==$tc_version"
  else
    echo "== base: installing latest torchcodec =="
    "$UV" pip install --python "$python_bin" --quiet --no-sources "torchcodec>=0.4.0"
  fi
}

provision_venv() {  # <label> <marker module> <extra pip args...>
  local label="$1" marker="$2"; shift 2
  local env_dir="$ENV_PREFIX/$label"
  local python_bin="$env_dir/bin/python"

  if [[ -x "$python_bin" ]] && have_module "$python_bin" "$marker" && [[ "$UPGRADE" -eq 0 ]]; then
    echo "== $label: already provisioned, skipping =="
    SKIPPED=$((SKIPPED + 1))
  else
    if [[ "$UPGRADE" -eq 1 ]]; then
      echo "== $label: upgrading in place =="
    [[ "$FULL" -eq 1 ]] && echo "   (--latest: bumping all packages to newest allowed — expect big downloads)"
    else
      echo "== $label: creating $env_dir =="
    fi
    [[ -x "$python_bin" ]] || "$UV" venv --python "$PY_VERSION" "$env_dir"
    # --no-sources is load-bearing: pyproject routes torch/torchvision to the
    # cu126 index via [tool.uv.sources] so that a GPU install resolves without
    # extra flags. base and act want the CPU build, and that routing overrides
    # a plain --index-url — without --no-sources these environments silently
    # pull ~3 GB of CUDA libraries that neither tier ever executes.
    #
    # Fresh builds stay quiet; upgrades show uv's download/install progress —
    # -U re-downloads torch-sized wheels and a silent multi-minute download
    # is indistinguishable from a hang.
    local quiet=(--quiet)
    [[ "$UPGRADE" -eq 1 ]] && quiet=()
    echo "  -> torch/torchvision (CPU) ..."
    "$UV" pip install --python "$python_bin" --no-sources \
        ${quiet[@]+"${quiet[@]}"} \
        ${_upgrade_flags[@]+"${_upgrade_flags[@]}"} \
        --index-url "$CPU_TORCH_INDEX" torch torchvision
    echo "  -> project dependencies ..."
    "$UV" pip install --python "$python_bin" --no-sources \
        ${quiet[@]+"${quiet[@]}"} \
        ${_upgrade_flags[@]+"${_upgrade_flags[@]}"} "$@"
  fi
  SUMMARY+=("$label $python_bin")
}

provision_via_install() {  # <label> <model> <marker module>
  local label="$1" model="$2" marker="$3"
  local env_dir="$ENV_PREFIX/$label"
  local python_bin="$env_dir/bin/python"

  if [[ -x "$python_bin" ]] && have_module "$python_bin" "$marker" && [[ "$UPGRADE" -eq 0 ]]; then
    echo "== $label: already provisioned, skipping =="
    SKIPPED=$((SKIPPED + 1))
  else
    echo "== $label: installing via scripts/install.sh --model $model =="
    # VLA_UPGRADE: 1 = resync repo deps (no -U), 2 = also bump to newest (-U)
    VLA_UPGRADE="$((UPGRADE + FULL))" bash "$REPO_ROOT/scripts/install.sh" --model "$model" --venv "$env_dir"
  fi
  SUMMARY+=("$label $python_bin")
}

UPGRADE=0
FULL=0
targets=()
for arg in "$@"; do
  case "$arg" in
    -u|--upgrade) UPGRADE=1 ;;
    --latest)     UPGRADE=1; FULL=1 ;;
    base|act|pi)  targets+=("$arg") ;;
    *) echo "provision: unknown environment '$arg' (expected base|act|pi, --upgrade, --latest)" >&2; exit 1 ;;
  esac
done
[[ ${#targets[@]} -eq 0 ]] && targets=(base act pi)

# Safe empty-array expansion for set -u (bash < 4.4 would treat "" as unbound).
# Light sync (default): no -U — uv installs only unsatisfied requirements.
# --latest adds -U: everything moves to newest allowed, cache misses abound.
_upgrade_flags=()
[[ "$UPGRADE" -eq 1 && "$FULL" -eq 1 ]] && _upgrade_flags=(-U)

for target in "${targets[@]}"; do
  case "$target" in
    base)
      provision_venv base pytest -e ".[dev]"
      install_torchcodec_for_python "$ENV_PREFIX/base/bin/python"
      ;;
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

if [[ "$SKIPPED" -gt 0 ]]; then
  echo
  echo "  已跳过 $SKIPPED 个已存在的环境；如需同步依赖（新合入 PR 后环境可能有变化），执行:"
  echo "    bash scripts/ci/build_ci_envs.sh --upgrade          # 同步项目声明的依赖（快，只装增量）"
  echo "    bash scripts/ci/build_ci_envs.sh --upgrade --latest # 连同允许范围内的新版本全部刷新（慢，大下载）"
  echo "    （以上命令均可追加 base / act / pi 只处理指定环境）"
fi
echo

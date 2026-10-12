"""Shared, non-agent pieces for the VLA adaptation harness."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Judge tasks have their own bounded contract.  --pure prevents a repository
# plugin/skill from replacing that task with a broad, unrelated workflow.
# OpenCode also treats paths such as /tmp as external even when bwrap mounts an
# isolated tmpfs there.  --auto accepts that tool-level request; bwrap remains
# the actual filesystem boundary.
#: Built-in provider models for the default deployment. Another deployment
#: overrides the whole set via the ``OPENCODE_MODELS`` environment variable
#: (comma-separated, first entry becomes the default) instead of editing code.
_BUILTIN_OPENCODE_MODELS = (
    "csi-provider/GLM-5.2",
    "csi-provider/DeepSeek-V4.1-Flash",
    "csi-provider/DeepSeek-V4-Flash-0731",
    "csi-provider/DeepSeek-V4-Pro",
    "csi-provider/GLM-5.3",
    "csi-provider/GLM-5.3-Flash",
    "csi-provider/Qwen3.7-Plus",
    "csi-provider/Qwen3.8-Flash",
    "csi-provider/Qwen3.8-Max",
    "csi-provider/kimi-k2.7-code",
    "csi-provider/kimi-k3",
)
OPENCODE_MODELS = tuple(
    model.strip()
    for model in os.environ.get("OPENCODE_MODELS", "").split(",")
    if model.strip()
) or _BUILTIN_OPENCODE_MODELS
DEFAULT_OPENCODE_MODEL = OPENCODE_MODELS[0]
OPENCODE_HEADLESS = ("opencode", "run", "--pure", "--auto", "--format", "json")
OPENCODE_INFRASTRUCTURE_BACKOFF_SECONDS = (5.0, 15.0, 30.0)
RUNTIME_FILE = "runtime.json"
SANDBOX_HOME = Path("/home/opencode")


class RetryableInfrastructureError(RuntimeError):
    """OpenCode could not reach its model provider after bounded retries."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


@dataclass(frozen=True)
class JobManifest:
    job_id: str
    upstream_repo: str
    vla: str
    checkpoint: str
    factory_base: str
    benchmark_name: str
    benchmark_revision: str
    comparison: dict[str, float]
    checkpoint_lock: dict[str, Any] | None = None
    opencode_model: str = DEFAULT_OPENCODE_MODEL


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(job_dir: Path) -> JobManifest:
    raw = load_json(job_dir / "manifest.json")
    raw.setdefault("benchmark_name", "robotwin-beat-block-hammer")
    raw.setdefault("opencode_model", DEFAULT_OPENCODE_MODEL)
    return JobManifest(**raw)


def write_manifest(path: Path, manifest: JobManifest) -> None:
    write_json(path, asdict(manifest))


def agent_state_directory(workspace: Path) -> Path:
    """Return job-owned OpenCode state that is deliberately outside its worktree.

    OpenCode snapshots its active workspace. Keeping HOME in that workspace makes
    its database and snapshot repository snapshot themselves recursively.
    """
    return workspace.resolve().parent / ".agent-state" / workspace.name


def normalized_network_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Return the common network environment used by Agents and Judge jobs.

    A local SOCKS ``ALL_PROXY`` often breaks Python installers that do not have
    SOCKS support even when an HTTP(S) proxy is available. Drop only that
    redundant fallback, and do so for both execution contexts. If SOCKS is the
    only configured proxy, preserve it rather than silently disabling network
    access.
    """
    env = dict(os.environ if source is None else source)
    has_http_proxy = any(env.get(name) for name in (
        "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
    ))
    if has_http_proxy:
        for name in ("ALL_PROXY", "all_proxy"):
            if env.get(name, "").lower().startswith("socks://"):
                env.pop(name, None)
    return env


def dependency_environment(workspace: Path, *, sandbox: bool = False) -> dict[str, str]:
    """Return the role-local dependency contract shared by Agent and Judge.

    The Agent HOME is a bind mount of :func:`agent_state_directory`, so the
    sandbox paths below and the host paths used by Judge name the same cache.
    The cache is persistent across repair attempts, while every Judge runtime
    remains a newly-created venv.
    """
    state = agent_state_directory(workspace)
    uv_cache = state / ".cache" / "uv"
    pip_cache = state / ".cache" / "pip"
    uv_cache.mkdir(parents=True, exist_ok=True)
    pip_cache.mkdir(parents=True, exist_ok=True)
    if sandbox:
        uv_cache = SANDBOX_HOME / ".cache" / "uv"
        pip_cache = SANDBOX_HOME / ".cache" / "pip"

    env = {
        "UV_CACHE_DIR": str(uv_cache),
        "PIP_CACHE_DIR": str(pip_cache),
        "UV_TORCH_BACKEND": "cpu",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }

    # A Harness-level source override wins. Otherwise, if only one installer
    # has an index configured, make the other installer use that same source.
    source_env = os.environ
    index = (
        source_env.get("VLA_PACKAGE_INDEX_URL")
        or source_env.get("UV_INDEX_URL")
        or source_env.get("PIP_INDEX_URL")
    )
    if index:
        env.update({"UV_INDEX_URL": index, "PIP_INDEX_URL": index})
    return env


def _sandboxed_opencode_command(command: list[str], workspace: Path,
                                readonly_inputs: tuple[tuple[Path, Path], ...] = ()) -> list[str]:
    """Start OpenCode with a sealed workspace and its read-only credentials.

    OpenCode's provider selection and credentials are user configuration, not
    Agent inputs.  Mount only those files into a workspace-local HOME; never
    expose the user's complete home directory or its Python environments.
    """
    bwrap = shutil.which("bwrap")
    opencode = shutil.which("opencode")
    uv = shutil.which("uv")
    node = shutil.which("node")
    if not bwrap or not opencode or not uv or not node:
        raise RuntimeError("Agent sandbox requires bwrap, opencode, uv, and node on the Harness PATH")

    workspace = workspace.resolve()
    agent_state = agent_state_directory(workspace)
    agent_state.mkdir(parents=True, exist_ok=True)
    sandbox_home = SANDBOX_HOME
    host_home = Path.home().resolve()
    opencode_path = Path(opencode).resolve()
    uv_path = Path(uv).resolve()
    node_path = Path(node).resolve()
    nvm_root = next((parent for parent in opencode_path.parents if parent.name == ".nvm"), None)

    readonly_mounts = [Path(path) for path in ("/usr", "/bin", "/lib", "/lib64", "/etc") if Path(path).exists()]
    if nvm_root is not None:
        readonly_mounts.append(nvm_root)
    else:
        readonly_mounts.extend((opencode_path.parent, node_path.parent))
    if not str(uv_path).startswith("/usr/"):
        readonly_mounts.append(uv_path.parent)

    # Preserve only the provider configuration required to select and
    # authenticate the same model as the unsandboxed OpenCode invocation.
    credential_mounts = (
        (host_home / ".config" / "opencode", sandbox_home / ".config" / "opencode"),
        (host_home / ".local" / "share" / "opencode" / "auth.json", sandbox_home / ".local" / "share" / "opencode" / "auth.json"),
        (host_home / ".local" / "state" / "opencode" / "model.json", sandbox_home / ".local" / "state" / "opencode" / "model.json"),
        (host_home / ".local" / "state" / "opencode" / "kv.json", sandbox_home / ".local" / "state" / "opencode" / "kv.json"),
    )
    for source, destination in credential_mounts:
        if source.exists():
            state_destination = agent_state / destination.relative_to(sandbox_home)
            state_destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                state_destination.touch(exist_ok=True)

    sandbox = [
        bwrap, "--die-with-parent", "--unshare-all", "--share-net", "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--dir", "/home", "--bind", str(agent_state), str(sandbox_home),
    ]
    for path in dict.fromkeys(readonly_mounts):
        sandbox.extend(("--ro-bind", str(path), str(path)))
    for source, destination in readonly_inputs:
        source, destination = source.resolve(), destination.resolve()
        if not source.is_dir() or not destination.is_absolute() or ".." in destination.parts:
            raise ValueError(f"invalid read-only Agent input: {source} -> {destination}")
        sandbox.extend(("--dir", str(destination.parent), "--ro-bind", str(source), str(destination)))
    sandbox.extend(("--bind", str(workspace), str(workspace)))
    for source, destination in credential_mounts:
        if source.exists():
            sandbox.extend(("--ro-bind", str(source), str(destination)))
    dependency_env = dependency_environment(workspace, sandbox=True)
    sandbox.extend((
        "--setenv", "HOME", str(sandbox_home),
        "--setenv", "XDG_CONFIG_HOME", str(sandbox_home / ".config"),
        "--setenv", "XDG_DATA_HOME", str(sandbox_home / ".local" / "share"),
        "--setenv", "XDG_STATE_HOME", str(sandbox_home / ".local" / "state"),
        "--setenv", "XDG_CACHE_HOME", str(sandbox_home / ".cache"),
        "--setenv", "PATH", os.pathsep.join((str(node_path.parent), str(uv_path.parent), "/usr/bin", "/bin")),
    ))
    for name, value in dependency_env.items():
        sandbox.extend(("--setenv", name, value))
    sandbox.extend(("--chdir", str(workspace)))
    return [*sandbox, *command]


def _event_message(event: dict[str, Any], fallback: str) -> str:
    """Produce a compact, useful console line from an OpenCode JSON event."""
    part = event.get("part", {})
    if event.get("type") == "error":
        error = event.get("error", {})
        data = error.get("data", {}) if isinstance(error, dict) else {}
        if not isinstance(data, dict):
            data = {}
        return data.get("message") or (error.get("message") if isinstance(error, dict) else None) or str(error)
    if event.get("type") == "tool_use":
        state = part.get("state", {})
        request = state.get("input", {})
        summary = request.get("description") or request.get("filePath") or request.get("command") or part.get("tool")
        return f"tool {state.get('status', 'unknown')}: {summary}"
    return part.get("text") or event.get("type") or fallback


def _retryable_provider_failure(event: dict[str, Any]) -> tuple[str, int | None] | None:
    """Return retry details for transient top-level provider failures."""
    if event.get("type") != "error":
        return None
    error = event.get("error", {})
    data = error.get("data", {}) if isinstance(error, dict) else {}
    if not isinstance(data, dict):
        data = {}
    message = data.get("message") or (error.get("message") if isinstance(error, dict) else None) or str(error)
    raw_status = data.get("statusCode", error.get("statusCode") if isinstance(error, dict) else None)
    try:
        status_code = int(raw_status) if raw_status is not None else None
    except (TypeError, ValueError):
        status_code = None
    lowered = str(message).lower()
    transient_message = any(token in lowered for token in (
        "bad gateway", "service unavailable", "gateway timeout", "too many requests",
        "rate limit", "connection timed out", "connection timeout", "connect timeout",
        "connection reset", "connection refused", "network error", "socket hang up",
    ))
    retryable = (
        data.get("isRetryable") is True
        or status_code in {408, 429}
        or (status_code is not None and 500 <= status_code <= 599)
        or transient_message
    )
    return (str(message), status_code) if retryable else None


def run_opencode(role: str, prompt: str, artifact_dir: Path, workspace: Path,
                 readonly_inputs: tuple[tuple[Path, Path], ...] = (), *,
                 model: str = DEFAULT_OPENCODE_MODEL) -> None:
    if not workspace.is_dir():
        raise ValueError(f"workspace does not exist: {workspace}")
    command = _sandboxed_opencode_command(
        [*OPENCODE_HEADLESS, "--model", model, "--dir", str(workspace), prompt],
        workspace,
        readonly_inputs,
    )
    output = artifact_dir / "opencode-events.jsonl"
    total_attempts = len(OPENCODE_INFRASTRUCTURE_BACKOFF_SECONDS) + 1
    for attempt in range(1, total_attempts + 1):
        print(f"[{role}] OpenCode started; workspace={workspace}; infrastructure-attempt={attempt}/{total_attempts}")
        provider_error: str | None = None
        retryable_error: tuple[str, int | None] | None = None
        mode = "w" if attempt == 1 else "a"
        with output.open(mode, encoding="utf-8") as file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=normalized_network_environment(),
            )
            assert process.stdout is not None
            for line in process.stdout:
                file.write(line)
                file.flush()
                try:
                    event = json.loads(line)
                    message = _event_message(event, line.strip())
                    if event.get("type") == "error":
                        error = event.get("error", {})
                        data = error.get("data", {}) if isinstance(error, dict) else {}
                        if not isinstance(data, dict):
                            data = {}
                        provider_error = data.get("message") or (
                            error.get("message") if isinstance(error, dict) else None
                        ) or str(error)
                        retryable_error = _retryable_provider_failure(event)
                except json.JSONDecodeError:
                    message = line.strip()
                if message:
                    print(f"[{role}] {message}")
            result = process.wait()
        print(f"[{role}] OpenCode finished with exit code {result}; events={output}")
        if not result:
            return
        if retryable_error is None:
            detail = f": {provider_error}" if provider_error else ""
            raise RuntimeError(f"OpenCode {role} agent failed{detail}; see {output}")
        message, status_code = retryable_error
        if attempt == total_attempts:
            status = f" HTTP {status_code}" if status_code is not None else ""
            raise RetryableInfrastructureError(
                f"OpenCode {role} provider failed after {attempt} infrastructure attempts{status}: "
                f"{message}; see {output}",
                attempts=attempt,
                status_code=status_code,
            )
        delay = OPENCODE_INFRASTRUCTURE_BACKOFF_SECONDS[attempt - 1]
        print(f"[{role}] retryable provider failure; retrying in {delay:g}s without consuming a repair")
        time.sleep(delay)


def require_outputs(role: str, artifact_dir: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (artifact_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"OpenCode {role} agent finished without required outputs: {', '.join(missing)}")

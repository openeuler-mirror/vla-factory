"""Role-local Agent prompts and OpenCode task entry points.

One module per concern boundary, three sections: the Reference Agent
(reconstruct the upstream VLA as a benchmark Policy), the Adapter Agent
(adapt it into a VLA-Factory bundle), and the read-only Parity Diagnosis
Agent (attribute a loss mismatch for repair routing). Each section owns its
prompt text, its OpenCode invocation, and its deliverable checks.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from .runtime import JobManifest, load_manifest, require_outputs, run_opencode



# -----------------------------------------------------------------------
# Reference Agent
# -----------------------------------------------------------------------


def reference_prompt(manifest: JobManifest, artifact_dir: Path, feedback: str | None = None) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", manifest.vla)
    lines = [
            f"You are the Reference Agent for VLA {manifest.vla}.",
            f"Upstream repo: {manifest.upstream_repo}",
            f"Checkpoint: {manifest.checkpoint}",
            "Your assigned artifact directory is `.judge/` inside the current workspace.",
            "The Judge has already acquired and locked the requested checkpoint at `/judge-inputs/checkpoint`. "
            "Use that read-only directory as the only source checkpoint; do not download, modify, or replace it. "
            "When your code runs under the Judge (outside the Agent Sandbox), obtain the same path from the "
            "`JUDGE_CHECKPOINT_DIR` environment variable; fall back to `/judge-inputs/checkpoint` only for Sandbox use. "
            "Read the upstream README and docs quick-start before implementation. Read `.judge/REFERENCE_BUNDLE_CONTRACT.md` "
            "first. Reconnaissance is bounded: use rg then targeted line ranges; do not read whole source files over 250 lines. "
            "After at most 12 tool calls, create the deliverable skeleton in `.judge/` before doing further investigation. "
            "Do not inspect unrelated rollout, training, or deployment code unless a specific missing model API requires it. "
            "and implement that contract. "
            f"Deliver install_{name}.sh, smoke_test_{name}.py, policy_impl.py, policy_spec.json, and runtime.json. "
            "policy_impl.py's compute_loss(Policy Mapping) applies official preprocessing then native_model.forward without training. "
            "Policy Mapping camera names are benchmark names, not necessarily upstream feature names: declare and apply "
            "the model-specific mapping inside policy_impl.py before official preprocessing. "
            "Do not create or persist Judge parity samples, compare values, choose tolerances, run a parity verdict, or run "
            "RoboTwin yourself. You may construct an in-memory synthetic Policy Mapping solely for the required smoke checks. "
            "The install script must create/install the upstream runtime dependencies. The smoke test must import the new "
            "model, load the requested checkpoint enough to prove initialization, instantiate build_policy(checkpoint), and "
            "call reset plus act with a minimal valid observation and require a finite action chunk matching the "
            "declared action schema. Also smoke-check compute_loss with an in-memory "
            "Policy Mapping using the benchmark camera names from REFERENCE_BUNDLE_CONTRACT.md and valid action targets/mask; "
            "require a finite scalar loss. This is a runtime smoke check, not a parity comparison or Judge sample. "
            "Run install then the smoke test yourself. Before stopping, perform these Judge-equivalent simple checks: "
            f"verify install_{name}.sh, smoke_test_{name}.py, policy_impl.py, policy_spec.json, and runtime.json are "
            "regular files in `.judge/` (the same requirement enforced by "
            "require_outputs), validate policy_spec.json's required keys/version/safe relative paths/action schema, "
            "and parse runtime.json to verify its declared command. Verify build_policy returns reset, act, and compute_loss. "
            "Verify its compute_loss is finite using only an in-memory valid Mapping. After all checks pass, stop. Do not start the TCP server, write an E2E client, or run "
            "RoboTwin; Judge owns those checks. "
            "policy_spec.json must declare contract_version, implementation, "
            "install_script, smoke_test, input schema, and the action schema required by the benchmark contract; it must not contain a python field. "
            "Judge creates a fresh role-isolated runtime and supplies JUDGE_BOOTSTRAP_PYTHON, JUDGE_RUNTIME_DIR, and JUDGE_PYTHON to your install script. "
            "Use the bootstrap Python only to create the venv at JUDGE_RUNTIME_DIR, then install and run only through JUDGE_PYTHON. "
            "Honor the Harness-provided UV_CACHE_DIR, PIP_CACHE_DIR, UV_INDEX_URL/PIP_INDEX_URL, proxy variables, and UV_TORCH_BACKEND; do not replace them with hard-coded mirrors or proxy settings. "
            "Use uv/pip's resolver normally. Never inspect archive-v0 or directly symlink/copy unpacked cache entries into site-packages, and do not search HOME or the host filesystem for another cache or environment. "
            "Fail clearly when required JUDGE_* variables are absent; do not put a fallback runtime path in the deliverable installer. "
            "For Sandbox-only local testing, create a separate temporary venv, set all three JUDGE_* variables explicitly, and run the same installer and smoke test; it is never the Judge runtime. "
            "Paths in policy_spec.json are filenames relative to `.judge/` (for example, `install_" + name
            + ".sh`), never `.judge/...`. Create runtime.json exactly as required by "
            "REFERENCE_BUNDLE_CONTRACT.md. Its protocol and command are benchmark-owned; do not invent a live server for an offline benchmark. "
    ]
    if feedback:
        lines.extend((
            "This is a repair attempt. You are the Reference Agent: repair only the upstream Reference policy/runtime artifacts, not the Factory bundle.",
            "Your previous Agent session has ended. The following object identifies the failed stage and candidate, includes bounded Judge log tails, and inventories the retained files for your role. Read the failure report and inspect those retained files before editing; preserve valid work and correct only the reported failure.",
            "Repair context follows:",
            feedback,
            "If parity_repair_handoff is present, it is Judge-owned semantic evidence, not a scalar target. "
            "Read /judge-inputs/parity-diagnosis/diagnosis.json and verify cited files in the read-only /judge-inputs/reference "
            "and /judge-inputs/factory mounts. Implement only the Reference-side correction supported by that evidence.",
            "Read the referenced log file before editing. Do not modify Judge-owned contracts, runners, benchmarks, "
            "tolerances, or the failure report. Re-run the smallest relevant validation after your fix.",
        ))
    return "\n".join(lines)


def run_reference_agent(job_dir: Path, workspace: Path, artifact_dir: Path, feedback: str | None = None,
        readonly_inputs: tuple[tuple[Path, Path], ...] = ()) -> None:
    manifest = load_manifest(job_dir)
    checkpoint = job_dir / "judge-inputs" / "checkpoint"
    run_opencode("reference", prompt(manifest, artifact_dir, feedback), artifact_dir, workspace,
                 ((checkpoint, Path("/judge-inputs/checkpoint")), *readonly_inputs),
                 model=manifest.opencode_model)
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", manifest.vla)
    require_outputs("reference", artifact_dir, (
        f"install_{name}.sh", f"smoke_test_{name}.py", "policy_impl.py", "policy_spec.json", "runtime.json",
    ))


# -----------------------------------------------------------------------
# Adapter Agent
# -----------------------------------------------------------------------


def adapter_prompt(manifest: JobManifest, artifact_dir: Path, feedback: str | None = None) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", manifest.vla)
    lines = [
            f"You are the Adapter Agent for VLA {manifest.vla}.",
            f"Upstream repo: {manifest.upstream_repo}",
            "Checkpoint source: Judge-owned (do not record its identifier in Factory artifacts).",
            f"Factory base: {manifest.factory_base}",
            "Your assigned artifact directory is `.judge/` inside the current workspace.",
            "The Judge has already acquired and locked the requested source checkpoint at `/judge-inputs/checkpoint`. "
            "Use that read-only directory as the only source checkpoint; do not download, modify, or replace it. "
            "Read VLA-Factory README, docs quick-start, and `.judge/FACTORY_BUNDLE_CONTRACT.md` before implementation. "
            "Keep reconnaissance bounded: use rg then targeted line ranges; do not read whole source files over 250 lines. "
            "After at most 12 tool calls, create the declared deliverable skeleton in `.judge/` before doing further investigation. "
            "A complete, isolated copy of the upstream checkout is at `.judge/upstream_source`; "
            "inspect any part of it you need for the model API, dependencies, configuration, or processors. "
            "It is an independent git checkout already pinned to the manifest's upstream revision; use its source, git metadata, pyproject.toml, "
            "and lockfile for bounded recon. "
            "Adapt the model in the Factory workspace and convert the source checkpoint into `.judge/factory_checkpoint/` "
            "according to FACTORY_BUNDLE_CONTRACT.md. The objective is behavioral fidelity, not merely a bootable bundle: "
            "after handoff, Judge will give the upstream Reference policy and the Factory model the same benchmark observation, "
            "training targets/mask, and RNG state, then compare their native forward loss. Preserve the upstream checkpoint's "
            "forward behavior across the Factory boundary; the smoke test below is only a restoration gate. "
            "After that bounded recon, implement the Factory adaptation and create the declared deliverables; then install and run the smoke test. "
            "A design, investigation, or route selection is not a completion condition. Stop only after all required deliverables exist and the "
            "required smoke test has passed. "
            "Deliver install_"
            f"{name}.sh, smoke_test_{name}.py, factory_bundle.json, and factory_checkpoint/. The install script must install "
            "the Factory/model dependencies. The smoke test must instantiate VLA-Factory's InferenceEngine from "
            "factory_checkpoint and run one prediction through the benchmark format declared in FACTORY_BUNDLE_CONTRACT.md. It must restore from the Factory checkpoint in the bundle contract's "
            "fresh-offline-cache lifecycle, without a network or pre-existing user cache; this is the required smoke test. Run that install then smoke yourself; after it "
            "passes, stop. Before stopping, perform these Judge-equivalent simple checks: verify install_"
            f"{name}.sh, smoke_test_{name}.py, and factory_bundle.json are regular files in `.judge/` (the same requirement enforced "
            "by require_outputs); verify factory_checkpoint/model.pt, weights.json, inference_metadata/assembly.json, and "
            "inference_metadata/recipe.yaml exist; and validate factory_bundle.json's required fields, safe relative paths, and "
            "supported weight format. Judge creates a fresh role-isolated runtime and supplies JUDGE_BOOTSTRAP_PYTHON, JUDGE_RUNTIME_DIR, and JUDGE_PYTHON to your install script. "
            "Use the bootstrap Python only to create the venv at JUDGE_RUNTIME_DIR, then install and run only through JUDGE_PYTHON. "
            "Honor the Harness-provided UV_CACHE_DIR, PIP_CACHE_DIR, UV_INDEX_URL/PIP_INDEX_URL, proxy variables, and UV_TORCH_BACKEND; do not replace them with hard-coded mirrors or proxy settings. "
            "Use uv/pip's resolver normally. Never inspect archive-v0 or directly symlink/copy unpacked cache entries into site-packages, and do not search HOME or the host filesystem for another cache or environment. "
            "Fail clearly when required JUDGE_* variables are absent; do not put a fallback runtime path in the deliverable installer. "
            "For Sandbox-only local testing, create a separate temporary venv, set all three JUDGE_* variables explicitly, and run the same installer and smoke test; it is never the Judge runtime. If model restoration needs non-weight assets such as a config, processor, or tokenizer, "
            "place them under factory_checkpoint/model_artifacts/, record the safe relative directory in model.config.model_artifacts, and resolve it relative to that checkpoint; do not put them in the Factory source tree. "
            "For conversion, the source checkpoint configuration and processors are authoritative: preserve its model input/output "
            "widths, preprocessing, and parameter dtypes at the model boundary. Map the benchmark contract outside that model boundary. A full `factory_checkpoint/model.pt` bundle must "
            "declare `inference_needs_base_checkpoint=False`: bundled model assets are part of that checkpoint, not an external base checkpoint. A model that needs "
            "an external upstream/base checkpoint is not a valid native bundle for this handoff. Saved recipes and assemblies use only bundle-relative asset paths. This is a hard handoff gate: after you finish, "
            "the harness will reject the artifact unless `metadata_ref.inference_needs_base_checkpoint` is exactly `false` and the saved "
            "`recipe.model.path` is `null`. This is a required static bundle check, not a parity test. "
    ]
    if feedback:
        lines.extend((
            "This is a repair attempt. You are the Adapter Agent: repair only the Factory adapter/runtime/bundle artifacts, not the upstream Reference policy.",
            "Your previous Agent session has ended. The following object identifies the failed stage and candidate, includes bounded Judge log tails, and inventories the retained files for your role. Read the failure report and inspect those retained files before editing; preserve valid work and correct only the reported failure.",
            "Repair context follows:",
            feedback,
            "If parity_repair_handoff is present, it is Judge-owned semantic evidence, not a scalar target. "
            "Read /judge-inputs/parity-diagnosis/diagnosis.json and verify cited files in the read-only /judge-inputs/reference "
            "and /judge-inputs/factory mounts. Implement equivalent Reference behavior only in your Factory artifacts.",
            "Read the referenced log file before editing. Do not modify Judge-owned contracts, runners, benchmarks, "
            "tolerances, or the failure report. Re-run the smallest relevant validation after your fix.",
        ))
    return "\n".join(lines)


def run_adapter_agent(job_dir: Path, workspace: Path, artifact_dir: Path, feedback: str | None = None,
        readonly_inputs: tuple[tuple[Path, Path], ...] = ()) -> None:
    manifest = load_manifest(job_dir)
    checkpoint = job_dir / "judge-inputs" / "checkpoint"
    run_opencode("adapter", adapter_prompt(manifest, artifact_dir, feedback), artifact_dir, workspace,
                 ((checkpoint, Path("/judge-inputs/checkpoint")), *readonly_inputs),
                 model=manifest.opencode_model)
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", manifest.vla)
    require_outputs("adapter", artifact_dir, (
        f"install_{name}.sh", f"smoke_test_{name}.py", "factory_bundle.json",
    ))
    # Deferred to break the prompts <-> judge import cycle (judge imports
    # this module at top level for its entry points).
    from .judge import factory_bundle_spec
    factory_bundle_spec(artifact_dir, manifest)


# -----------------------------------------------------------------------
# Parity Diagnosis Agent
# -----------------------------------------------------------------------


def run_diagnosis(job_dir: Path, failure_report: dict) -> dict:
    manifest = load_manifest(job_dir)
    workspace = job_dir / "artifacts" / "parity-diagnosis"
    workspace.mkdir(parents=True, exist_ok=True)
    parity = job_dir / "artifacts" / "parity"
    reference = job_dir / "workspaces" / "upstream" / ".judge"
    factory = job_dir / "workspaces" / "factory" / ".judge"
    prompt = """You are the Parity Diagnosis Agent. You are read-only: do not modify
Reference, Factory, Judge runners, samples, tolerances, or contracts. The prior
Agent sessions ended. Inspect the read-only Reference artifacts at
/judge-inputs/reference, Factory artifacts at /judge-inputs/factory, and Judge
parity evidence at /judge-inputs/parity. Determine which role should receive
the next repair, based on concrete evidence rather than a numerical target.

Write diagnosis.json in your current writable workspace. It must follow this
schema (the fact keys are deliberately model-agnostic and may describe any
VLA-specific processor, modality, state/action, tokenizer, architecture, or
checkpoint behavior):
{
  "version": "parity-diagnosis/v1",
  "owner": "reference" | "adapter" | "both" | "inconclusive",
  "confidence": "low" | "medium" | "high",
  "summary": "one non-numeric statement of the suspected semantic divergence",
  "evidence": ["specific, falsifiable observations"],
  "next_checks": ["ordered checks for the selected repair role(s)"],
  "repair_handoffs": {
    "reference": {
      "objective": "non-numeric repair objective",
      "semantic_facts": [{
        "key": "open namespaced key, e.g. embodiment.tag",
        "value": "observed value or JSON value",
        "source": {"role": "reference" | "adapter" | "judge", "path": "path relative to that read-only input", "line_hint": "optional"}
      }],
      "differences": [{
        "dimension": "open comparison dimension, e.g. action.mapping",
        "reference": "observed reference behavior/value",
        "factory": "observed factory behavior/value",
        "impact": "why it can affect native forward behavior",
        "evidence": [{"role": "reference" | "adapter" | "judge", "path": "relative path", "line_hint": "optional"}]
      }],
      "investigation_order": ["concrete, ordered, non-numeric checks"]
    },
    "adapter": {"...same handoff schema..."}
  }
}

Include a handoff only for a selected role. A handoff is not a fixed list of
VLA fields: use semantic_facts and differences to express only the facts this
model needs. Every fact or difference must cite a read-only source path so the
repair Agent can verify it. For example, an embodiment difference can cite the
Reference policy's processor registration and the Factory adapter's selected
tag; another VLA might instead cite token layout, image preprocessing, or
action masking. Do not include factory/reference losses, absolute error,
tolerances, or any scalar target in diagnosis.json.

Do not report or optimize a scalar loss. If evidence cannot distinguish the
roles, choose inconclusive rather than guessing.

Judge failure context follows:
""" + json.dumps(failure_report, indent=2, sort_keys=True)
    run_opencode(
        "parity-diagnosis", prompt, workspace, workspace,
        ((reference, Path("/judge-inputs/reference")),
         (factory, Path("/judge-inputs/factory")),
         (parity, Path("/judge-inputs/parity"))),
        model=manifest.opencode_model,
    )
    require_outputs("parity-diagnosis", workspace, ("diagnosis.json",))
    return json.loads((workspace / "diagnosis.json").read_text(encoding="utf-8"))

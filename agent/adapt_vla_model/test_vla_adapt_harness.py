import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from adapt_vla_model import judge as harness
from adapt_vla_model import prompts, runtime
import run as workflow_entry
from adapt_vla_model.runtime import DEFAULT_OPENCODE_MODEL, OPENCODE_HEADLESS, OPENCODE_MODELS, require_outputs
from adapt_vla_model.benchmarks import parity_dataset, resolve as resolve_benchmark
from adapt_vla_model.benchmarks.robotwin.beat_block_hammer.run import _cuda_link_dir, _result_rate
from adapt_vla_model.parity.parity_io import read_sample, write_sample
from adapt_vla_model.parity.factory_parity_runner import _codec_format
from adapt_vla_model.contract import Policy
from adapt_vla_model.benchmarks.robotwin.beat_block_hammer.run import _robotwin_python
from adapt_vla_model.robotwin_policy_server import _load


class HarnessTest(unittest.TestCase):
    def test_resume_cli_loads_job_and_saved_repair_budget_without_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            job.mkdir()
            harness.write_manifest(job / "manifest.json", harness.JobManifest(
                "job", "repo", "model", "owner/model", "base", "bench", "rev", {},
            ))
            harness.write_json(job / "workflow-state.json", {
                "version": "workflow-state/v1", "stage": "init",
                "repairs_used": 0, "max_repairs": 6,
            })
            with patch.object(
                workflow_entry, "run_workflow",
                return_value={"verdict": "PASS", "reason": "Judge passed", "attempts": 0},
            ) as run_workflow:
                result = workflow_entry.main(["--resume", str(job)])

            values = run_workflow.call_args.args[0]
            self.assertEqual(0, result)
            self.assertEqual(str(job.resolve()), values.job_dir)
            self.assertTrue(values.resume)
            self.assertEqual(6, values.max_repairs)
            self.assertEqual(DEFAULT_OPENCODE_MODEL, values.opencode_model)
            self.assertEqual(6, run_workflow.call_args.kwargs["max_repairs"])

    def test_interactive_model_selection_accepts_menu_number(self):
        with patch("builtins.input", return_value="1"):
            self.assertEqual("csi-provider/GLM-5.2", workflow_entry._choose_opencode_model())

    def test_prepare_workspaces_finishes_existing_valid_partial_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            upstream = job / "workspaces" / "upstream"
            factory = job / "workspaces" / "factory"
            (job / "artifacts").mkdir(parents=True)
            (upstream / ".git").mkdir(parents=True)
            factory.mkdir(parents=True)
            (factory / ".git").write_text("gitdir: worktree", encoding="utf-8")
            harness.write_manifest(job / "manifest.json", harness.JobManifest(
                "job", "repo", "model", "owner/model", "factory-rev",
                "bench", "benchmark-rev", {},
            ))

            def revision(repository, ref="HEAD"):
                del ref
                return "factory-rev" if Path(repository) == factory else "upstream-rev"

            with (
                patch.object(harness, "_revision", side_effect=revision),
                patch.object(harness, "factory_root", return_value=Path(directory) / "source"),
                patch.object(harness, "_run") as run_command,
                patch.object(harness, "_seed_bundle"),
                patch.object(harness, "_stage_upstream_source"),
            ):
                harness.prepare_workspaces(job)

            run_command.assert_not_called()
            lock = harness.load_json(job / "workspace-lock.json")
            self.assertEqual("upstream-rev", lock["upstream"]["revision"])
            self.assertEqual("factory-rev", lock["factory"]["revision"])

    def test_checkpoint_source_parser_supports_hubs_urls_and_local_dirs(self):
        self.assertEqual(
            {"provider": "modelscope", "model_id": "amap_cvlab/ABot-M0-Pretrain", "revision": "master"},
            harness.parse_checkpoint_source("https://www.modelscope.cn/models/amap_cvlab/ABot-M0-Pretrain/files"),
        )
        self.assertEqual(
            {"provider": "huggingface", "model_id": "lerobot/smolvla_base", "revision": "v2"},
            harness.parse_checkpoint_source("https://huggingface.co/lerobot/smolvla_base/tree/v2"),
        )
        with tempfile.TemporaryDirectory() as directory:
            parsed = harness.parse_checkpoint_source(directory)
        self.assertEqual("local", parsed["provider"])

    def test_interrupted_init_resumes_checkpoint_before_workspaces(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            (job / "artifacts").mkdir(parents=True)
            partial = job / "judge-inputs" / "checkpoint" / "blob.incomplete"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial")
            harness.write_manifest(job / "manifest.json", harness.JobManifest(
                "job", "repo", "model", "owner/model", "base", "bench", "rev", {},
            ))
            lock = {
                "provider": "huggingface", "model_id": "owner/model",
                "revision": "commit", "path": "judge-inputs/checkpoint",
                "files": {"model.bin": 1},
            }
            with (
                patch.object(harness, "acquire_checkpoint", return_value=lock) as acquire,
                patch.object(harness, "resolve_factory_base", return_value="base"),
                patch.object(harness, "prepare_workspaces") as prepare,
            ):
                harness.init_job(argparse.Namespace(job_dir=str(job)))

            acquire.assert_called_once_with(job.resolve(), "owner/model")
            prepare.assert_called_once_with(job.resolve())
            self.assertEqual(lock, harness.load_manifest(job).checkpoint_lock)

    def test_checkpoint_acquisition_retains_partial_files_on_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            partial = job / "judge-inputs" / "checkpoint" / "blob.incomplete"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"keep me")
            source = {
                "provider": "huggingface", "model_id": "owner/model", "revision": "commit",
            }
            failure = harness.RetryableInfrastructureError("stalled", attempts=4)
            with (
                patch.object(harness, "_locked_huggingface_source", return_value=source),
                patch.object(harness, "_download_huggingface_snapshot", side_effect=failure),
            ):
                with self.assertRaises(harness.RetryableInfrastructureError):
                    harness.acquire_checkpoint(job, "owner/model")

            self.assertEqual(b"keep me", partial.read_bytes())

    def test_huggingface_downloader_disables_xet_and_reuses_local_dir(self):
        class CompleteProcess:
            returncode = 0

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint"
            source = {
                "provider": "huggingface", "model_id": "owner/model", "revision": "commit",
            }
            with patch.object(harness.subprocess, "Popen", return_value=CompleteProcess()) as popen:
                harness._download_huggingface_snapshot(source, target)

            command = popen.call_args.args[0]
            environment = popen.call_args.kwargs["env"]
            self.assertIn(str(target), command)
            self.assertEqual("1", environment["HF_HUB_DISABLE_XET"])
            self.assertEqual("0", environment["HF_HUB_ENABLE_HF_TRANSFER"])

    def test_checkpoint_revision_lock_is_reused_across_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            state = job / "judge-inputs" / harness.CHECKPOINT_DOWNLOAD_STATE
            harness.write_json(state, {
                "version": "checkpoint-download/v1", "status": "downloading",
                "source": "owner/model", "provider": "huggingface",
                "model_id": "owner/model", "revision": "locked-commit",
            })
            locked = harness._locked_huggingface_source(
                job, "owner/model",
                {"provider": "huggingface", "model_id": "owner/model", "revision": "main"},
            )
            self.assertEqual("locked-commit", locked["revision"])

    def test_legacy_partial_download_recovers_revision_from_hub_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            metadata = (
                job / "judge-inputs" / "checkpoint" / ".cache" /
                "huggingface" / "download" / "config.json.metadata"
            )
            metadata.parent.mkdir(parents=True)
            revision = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
            metadata.write_text(f"{revision}\netag\n0\n", encoding="utf-8")

            with patch("huggingface_hub.HfApi.model_info") as model_info:
                locked = harness._locked_huggingface_source(
                    job, "owner/model",
                    {"provider": "huggingface", "model_id": "owner/model", "revision": "main"},
                )

            model_info.assert_not_called()
            self.assertEqual(revision, locked["revision"])
            state = harness.load_json(
                job / "judge-inputs" / harness.CHECKPOINT_DOWNLOAD_STATE
            )
            self.assertEqual(revision, state["revision"])

    def test_checkpoint_infrastructure_failure_is_incomplete_not_candidate_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            failure = harness.RetryableInfrastructureError("stalled", attempts=4)
            with patch.object(harness, "init_job", side_effect=failure):
                result = harness.run_workflow(argparse.Namespace(job_dir=str(job)), max_repairs=6)

            self.assertEqual("INCOMPLETE", result["verdict"])
            self.assertEqual(0, result["attempts"])
            self.assertEqual("checkpoint infrastructure retries exhausted", result["reason"])
            state = json.loads((job / "workflow-state.json").read_text(encoding="utf-8"))
            self.assertEqual(0, state["repairs_used"])

    def test_failure_router_uses_stage_message_not_workspace_log_path(self):
        self.assertEqual(
            "adapter",
            harness._failure_candidate(
                "factory smoke failed; see /tmp/job/workspaces/factory/.judge/judge-smoke.log"
            ),
        )
        self.assertIsNone(harness._failure_candidate("benchmark parity dataset is missing: /data/example"))

    def test_parity_repair_report_exposes_directions_not_loss_values(self):
        with tempfile.TemporaryDirectory() as directory:
            report = harness._failure_report(
                Path(directory), attempt=0, stage="judge",
                error=harness.ParityMismatchError(),
            )
        self.assertIsNone(report["candidate"])
        context = report["repair_context"]
        self.assertEqual("loss_parity", context["gate"])
        self.assertEqual("mismatch", context["outcome"])
        rendered = json.dumps(report)
        self.assertNotIn("factory_loss", rendered)
        self.assertNotIn("reference_loss", rendered)
        self.assertIn("preprocessing and action/state normalization", rendered)

    def test_parity_handoff_accepts_model_specific_facts_with_cited_evidence(self):
        handoff = {
            "objective": "align the Factory processor setup with the cited Reference behavior",
            "semantic_facts": [{
                "key": "embodiment.tag",
                "value": "NEW_EMBODIMENT",
                "source": {"role": "reference", "path": "policy_impl.py", "line_hint": "around processor setup"},
            }],
            "differences": [{
                "dimension": "action.mapping",
                "reference": "six joint-position action values",
                "factory": "synthetic DROID action representation",
                "impact": "the processor receives different modality semantics",
                "evidence": [
                    {"role": "reference", "path": "policy_impl.py", "line_hint": "compute_loss"},
                    {"role": "adapter", "path": "factory_checkpoint/inference_metadata/assembly.json"},
                ],
            }],
            "investigation_order": ["verify processor registration", "remove synthetic mapping if unsupported"],
        }
        harness._validate_parity_handoff("adapter", handoff)

    def test_parity_handoff_rejects_unsafe_evidence_path(self):
        handoff = {
            "objective": "test", "semantic_facts": [{
                "key": "processor.tag", "value": "x",
                "source": {"role": "reference", "path": "../policy_impl.py"},
            }], "differences": [], "investigation_order": [],
        }
        with self.assertRaisesRegex(RuntimeError, "safe relative path"):
            harness._validate_parity_handoff("adapter", handoff)

    def test_agent_retry_clears_only_job_owned_residue(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            workspace = job / "workspaces" / "factory"
            artifact = workspace / ".judge"
            artifact.mkdir(parents=True)
            (job / "manifest.json").write_text(json.dumps({
                "job_id": "job", "upstream_repo": "repo", "vla": "smolvla", "checkpoint": "ckpt",
                "factory_base": "base", "benchmark_name": "robotwin-beat-block-hammer", "benchmark_revision": "rev",
                "comparison": {},
            }), encoding="utf-8")
            (job / "workspace-lock.json").write_text("{}", encoding="utf-8")
            subprocess.run(["git", "-C", str(workspace), "init"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(workspace), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(workspace), "config", "user.name", "Harness test"], check=True)
            tracked = workspace / "tracked.txt"
            tracked.write_text("base", encoding="utf-8")
            subprocess.run(["git", "-C", str(workspace), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(workspace), "commit", "-m", "base"], check=True, capture_output=True, text=True)
            tracked.write_text("agent edit", encoding="utf-8")
            (workspace / "generated.py").write_text("agent residue", encoding="utf-8")
            (workspace / ".smolvla").mkdir()
            (workspace / ".local-deps").mkdir()
            (workspace / "outputs").mkdir()
            (artifact / "FACTORY_BUNDLE_CONTRACT.md").write_text("judge input", encoding="utf-8")
            (artifact / "upstream_source").mkdir()
            (artifact / "factory_bundle.json").write_text("agent output", encoding="utf-8")
            (artifact / "agent-home").mkdir()

            harness._clear_agent_residue(job, "adapter", workspace, artifact)

            self.assertEqual("base", tracked.read_text(encoding="utf-8"))
            self.assertFalse((workspace / "generated.py").exists())
            self.assertFalse((workspace / ".smolvla").exists())
            self.assertFalse((workspace / ".local-deps").exists())
            self.assertFalse((workspace / "outputs").exists())
            self.assertTrue((artifact / "FACTORY_BUNDLE_CONTRACT.md").is_file())
            self.assertTrue((artifact / "upstream_source").is_dir())
            self.assertFalse((artifact / "factory_bundle.json").exists())
            self.assertFalse((artifact / "agent-home").exists())

    def test_opencode_provider_error_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            artifact = workspace / ".judge"
            artifact.mkdir(parents=True)
            error_event = json.dumps({
                "type": "error",
                "error": {"data": {"message": "Payment Required"}},
            })
            process = type("Process", (), {
                "stdout": iter([error_event + "\n"]),
                "wait": lambda self: 1,
            })()
            with patch.object(runtime, "_sandboxed_opencode_command", return_value=["opencode"]), \
                 patch.object(runtime.subprocess, "Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "Payment Required"):
                    runtime.run_opencode("adapter", "prompt", artifact, workspace)

    def test_opencode_runner_passes_the_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            artifact = workspace / ".judge"
            artifact.mkdir(parents=True)
            process = type("Process", (), {
                "stdout": iter(()),
                "wait": lambda self: 0,
            })()
            with (
                patch.object(runtime, "_sandboxed_opencode_command", return_value=["opencode"]) as sandbox,
                patch.object(runtime.subprocess, "Popen", return_value=process),
            ):
                runtime.run_opencode(
                    "adapter", "prompt", artifact, workspace,
                    model="csi-provider/GLM-5.3",
                )

        command = sandbox.call_args.args[0]
        self.assertEqual("csi-provider/GLM-5.3", command[command.index("--model") + 1])

    def test_opencode_classifies_transient_provider_errors(self):
        for data in (
            {"message": "Too Many Requests", "statusCode": 429},
            {"message": "Bad Gateway", "statusCode": 502},
            {"message": "Service Unavailable", "statusCode": 503},
            {"message": "connection timed out"},
            {"message": "provider requested retry", "isRetryable": True},
        ):
            with self.subTest(data=data):
                self.assertIsNotNone(runtime._retryable_provider_failure({
                    "type": "error", "error": {"data": data},
                }))
        self.assertIsNone(runtime._retryable_provider_failure({
            "type": "error", "error": {"data": {"message": "Payment Required", "statusCode": 402}},
        }))

    def test_opencode_retries_transient_provider_failure_in_place(self):
        class Process:
            def __init__(self, lines, returncode):
                self.stdout = iter(lines)
                self.returncode = returncode

            def wait(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            artifact = workspace / ".judge"
            artifact.mkdir(parents=True)
            error_event = json.dumps({
                "type": "error",
                "error": {"data": {"message": "Bad Gateway", "statusCode": 502, "isRetryable": True}},
            }) + "\n"
            processes = [Process([error_event], 1), Process([], 0)]
            with (
                patch.object(runtime, "_sandboxed_opencode_command", return_value=["opencode"]),
                patch.object(runtime, "OPENCODE_INFRASTRUCTURE_BACKOFF_SECONDS", (0.0,)),
                patch.object(runtime.subprocess, "Popen", side_effect=processes) as popen,
                patch.object(runtime.time, "sleep") as sleep,
            ):
                runtime.run_opencode("adapter", "prompt", artifact, workspace)

        self.assertEqual(2, popen.call_count)
        sleep.assert_called_once_with(0.0)

    def test_opencode_raises_infrastructure_error_after_retry_exhaustion(self):
        class Process:
            def __init__(self, line):
                self.stdout = iter([line])

            def wait(self):
                return 1

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            artifact = workspace / ".judge"
            artifact.mkdir(parents=True)
            error_event = json.dumps({
                "type": "error",
                "error": {"data": {"message": "Service Unavailable", "statusCode": 503}},
            }) + "\n"
            with (
                patch.object(runtime, "_sandboxed_opencode_command", return_value=["opencode"]),
                patch.object(runtime, "OPENCODE_INFRASTRUCTURE_BACKOFF_SECONDS", (0.0,)),
                patch.object(runtime.subprocess, "Popen", side_effect=[
                    Process(error_event), Process(error_event),
                ]),
                patch.object(runtime.time, "sleep"),
            ):
                with self.assertRaises(runtime.RetryableInfrastructureError) as raised:
                    runtime.run_opencode("adapter", "prompt", artifact, workspace)

        self.assertEqual(2, raised.exception.attempts)
        self.assertEqual(503, raised.exception.status_code)

    def test_exhausted_infrastructure_retries_are_not_candidate_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            error = runtime.RetryableInfrastructureError(
                "Bad Gateway", attempts=4, status_code=502,
            )
            with (
                patch.object(harness, "init_job"),
                patch.object(harness, "run_agent", side_effect=error),
            ):
                result = harness.run_workflow(argparse.Namespace(job_dir=str(job)), max_repairs=2)

            state = json.loads((job / "workflow-state.json").read_text(encoding="utf-8"))
            failure = result["last_failure"]
        self.assertEqual("INCOMPLETE", result["verdict"])
        self.assertEqual(0, result["attempts"])
        self.assertEqual(0, state["repairs_used"])
        self.assertEqual("infrastructure:agent:reference", failure["stage"])
        self.assertEqual("RetryableInfrastructureError", failure["error_type"])

    def test_initial_reference_and_adapter_agents_run_in_parallel(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            rendezvous = threading.Barrier(2)
            roles = set()
            lock = threading.Lock()

            def run_initial_agent(args):
                with lock:
                    roles.add(args.role)
                rendezvous.wait(timeout=2)

            with (
                patch.object(harness, "init_job"),
                patch.object(harness, "run_agent", side_effect=run_initial_agent),
                patch.object(harness, "judge", return_value={"verdict": "PASS"}),
            ):
                result = harness.run_workflow(argparse.Namespace(job_dir=str(job)), max_repairs=2)

        self.assertEqual("PASS", result["verdict"])
        self.assertEqual({"reference", "adapter"}, roles)

    def test_selected_reference_and_adapter_repairs_run_in_parallel(self):
        rendezvous = threading.Barrier(2)
        roles = set()
        lock = threading.Lock()

        def repair(_job_dir, role, _report):
            with lock:
                roles.add(role)
            rendezvous.wait(timeout=2)

        with patch.object(harness, "_repair_agent", side_effect=repair):
            errors = harness._repair_roles(Path("/job"), ("reference", "adapter"), {})

        self.assertEqual({}, errors)
        self.assertEqual({"reference", "adapter"}, roles)

    def test_infrastructure_failure_during_repair_does_not_consume_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            (job / "workspaces" / "factory").mkdir(parents=True)
            (job / "workspace-lock.json").write_text("{}", encoding="utf-8")
            (job / "manifest.json").write_text(json.dumps({
                "job_id": "job", "upstream_repo": "repo", "vla": "model", "checkpoint": "ckpt",
                "factory_base": "base", "benchmark_name": "robotwin-beat-block-hammer",
                "benchmark_revision": "rev", "comparison": {},
            }), encoding="utf-8")
            infrastructure_error = runtime.RetryableInfrastructureError(
                "Service Unavailable", attempts=4, status_code=503,
            )

            def run_initial_agent(args):
                if args.role == "adapter":
                    raise RuntimeError("factory bundle invalid")

            with (
                patch.object(harness, "init_job"),
                patch.object(harness, "run_agent", side_effect=run_initial_agent),
                patch.object(harness, "_repair_agent", side_effect=infrastructure_error),
            ):
                result = harness.run_workflow(argparse.Namespace(job_dir=str(job)), max_repairs=2)

            state = json.loads((job / "workflow-state.json").read_text(encoding="utf-8"))
            reports = sorted((job / "artifacts" / "attempts").glob("*.json"))
        self.assertEqual("INCOMPLETE", result["verdict"])
        self.assertEqual(0, result["attempts"])
        self.assertEqual(0, state["repairs_used"])
        self.assertEqual(2, len(reports))

    def test_agent_sandbox_hides_host_environments_but_mounts_opencode_auth(self):
        if not shutil.which("bwrap"):
            self.skipTest("bwrap is required by the Harness Agent sandbox")
        host_home = Path.home().resolve()
        # The sandbox may ro-bind tool roots that live under the host home
        # (``.nvm`` for node/opencode, ``.local`` for uv, opencode state).
        # Probe that any OTHER host-home entry is invisible instead.
        mounted_roots = {".nvm", ".local", ".config"}
        hidden_entry = next(
            (child for child in sorted(host_home.iterdir()) if child.name not in mounted_roots),
            None,
        )
        if hidden_entry is None:
            self.skipTest("no unmounted host-home entry to probe sandbox isolation")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "factory"
            (workspace / ".judge").mkdir(parents=True)
            check_auth = "test -f \"$HOME/.local/share/opencode/auth.json\"" if (
                Path.home() / ".local/share/opencode/auth.json"
            ).is_file() else "true"
            command = runtime._sandboxed_opencode_command(
                ["/bin/sh", "-c", "test -d \"$PWD\" && "
                 "test \"$UV_CACHE_DIR\" = \"$HOME/.cache/uv\" && "
                 "test \"$PIP_CACHE_DIR\" = \"$HOME/.cache/pip\" && "
                 "test \"$UV_TORCH_BACKEND\" = cpu && "
                 f"test ! -e {hidden_entry} && " + check_auth],
                workspace,
            )
            self.assertIn(str(runtime.agent_state_directory(workspace)), command)
            self.assertNotIn(str(workspace / ".judge" / "agent-home"), command)
            subprocess.run(command, check=True)

    def test_policy_contract_covers_serving_and_parity(self):
        self.assertTrue(all(hasattr(Policy, name) for name in ("reset", "act", "compute_loss")))

    def test_policy_sample_round_trip(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is required to serialize a policy sample")

        observation = {
            "robotwin_observation": {
                "observation": {"head_camera": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}},
                "joint_action": {"vector": np.arange(14, dtype=np.float32)},
            },
            "instruction": "task",
            "step": 3,
            "training": {"actions": np.zeros((2, 14), dtype=np.float32), "action_is_pad": np.zeros(2, dtype=bool)},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            write_sample(observation, {"seed": 42}, path)
            loaded, metadata = read_sample(path)
        self.assertEqual(42, metadata["seed"])
        self.assertEqual("task", loaded["instruction"])
        self.assertEqual(3, loaded["step"])
        np.testing.assert_array_equal(observation["training"]["actions"], loaded["training"]["actions"])
        np.testing.assert_array_equal(np.arange(6, dtype=np.float32), loaded["robotwin_observation"]["joint_action"]["left_arm"])

    def test_lerobot_policy_sample_round_trip_preserves_host_keys(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is required to serialize a policy sample")

        observation = {
            "lerobot_observation": {
                "front": np.zeros((2, 2, 3), dtype=np.uint8),
                "wrist": np.ones((2, 2, 3), dtype=np.uint8),
                "shoulder_pan.pos": 1.25,
                "gripper.pos": 0.5,
            },
            "instruction": "pick up cube", "step": 3,
            "training": {"actions": np.zeros((2, 6), dtype=np.float32), "action_is_pad": np.zeros(2, dtype=bool)},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            write_sample(observation, {"seed": 42}, path)
            loaded, metadata = read_sample(path)
        self.assertEqual("lerobot", metadata["platform"])
        self.assertEqual(1.25, loaded["lerobot_observation"]["shoulder_pan.pos"])
        self.assertEqual(0.5, loaded["lerobot_observation"]["gripper.pos"])
        np.testing.assert_array_equal(observation["lerobot_observation"]["front"], loaded["lerobot_observation"]["front"])

    def test_robotwin_runner_requires_its_own_python(self):
        with patch.dict(os.environ, {"ROBOTWIN_PY": ""}):
            with self.assertRaisesRegex(ValueError, "ROBOTWIN_PY"):
                _robotwin_python()

    def test_reference_environment_uses_cpu_and_removes_socks_proxy(self):
        with patch.dict(os.environ, {"ALL_PROXY": "socks://127.0.0.1:7897", "HTTPS_PROXY": "http://proxy"}):
            env = harness._reference_environment()
        self.assertNotIn("ALL_PROXY", env)
        self.assertEqual("http://proxy", env["HTTPS_PROXY"])
        self.assertEqual("", env["CUDA_VISIBLE_DEVICES"])

    def test_network_environment_preserves_the_only_available_proxy(self):
        env = runtime.normalized_network_environment({"ALL_PROXY": "socks://proxy"})
        self.assertEqual("socks://proxy", env["ALL_PROXY"])

    def test_dependency_environment_aligns_uv_and_pip_indexes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"VLA_PACKAGE_INDEX_URL": "https://packages.example/simple"}, clear=True,
        ):
            env = runtime.dependency_environment(Path(directory) / "factory")
        self.assertEqual("https://packages.example/simple", env["UV_INDEX_URL"])
        self.assertEqual(env["UV_INDEX_URL"], env["PIP_INDEX_URL"])

    def test_factory_environment_uses_cpu_for_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            env = harness._hermetic_environment(Path(directory) / ".judge")
        self.assertEqual("", env["CUDA_VISIBLE_DEVICES"])

    def test_reference_environment_uses_a_persistent_judge_home(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / ".judge"
            env = harness._reference_environment(artifact)
        self.assertEqual(str(artifact / "judge-reference-home"), env["HOME"])
        self.assertEqual(str(artifact / "judge-reference-home" / "hf"), env["HF_HOME"])

    def test_reference_judge_reuses_the_agent_dependency_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspaces" / "upstream"
            artifact = workspace / ".judge"
            artifact.mkdir(parents=True)
            agent_env = runtime.dependency_environment(workspace, sandbox=True)
            judge_env = harness._reference_environment(artifact)
            cache_root = runtime.agent_state_directory(workspace) / ".cache"
        self.assertEqual("/home/opencode/.cache/uv", agent_env["UV_CACHE_DIR"])
        self.assertEqual(str(cache_root / "uv"), judge_env["UV_CACHE_DIR"])
        self.assertEqual(str(cache_root / "pip"), judge_env["PIP_CACHE_DIR"])
        self.assertEqual("cpu", judge_env["UV_TORCH_BACKEND"])

    def test_factory_judge_reuses_the_factory_agent_dependency_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspaces" / "factory"
            artifact = workspace / ".judge"
            artifact.mkdir(parents=True)
            env = harness._hermetic_environment(artifact)
            cache_root = runtime.agent_state_directory(workspace) / ".cache"
        self.assertEqual(str(cache_root / "uv"), env["UV_CACHE_DIR"])
        self.assertEqual(str(cache_root / "pip"), env["PIP_CACHE_DIR"])

    def test_reference_environment_exposes_the_judge_locked_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            artifact = job / "workspaces" / "upstream" / ".judge"
            checkpoint = job / "judge-inputs" / "checkpoint"
            checkpoint.mkdir(parents=True)
            env = harness._reference_environment(artifact)
        self.assertEqual(str(checkpoint), env["JUDGE_CHECKPOINT_DIR"])
        self.assertEqual(str(checkpoint), env["CHECKPOINT"])

    def test_parity_dataset_is_platform_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data" / "beat_block_hammer" / "demo_randomized"
            dataset.mkdir(parents=True)
            with patch.dict(os.environ, {"ROBOTWIN_ROOT": str(root)}):
                path, revision = parity_dataset("robotwin-beat-block-hammer")
        self.assertEqual(dataset, path)
        self.assertEqual("robotwin-beat-block-hammer-data/v1", revision)

    def test_lerobot_parity_dataset_uses_the_configured_cache_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "leisaac" / "pickup_cude_30_v2"
            dataset.mkdir(parents=True)
            with patch.dict(os.environ, {"LEROBOT_DATASET_ROOT": str(root)}):
                path, revision = parity_dataset("lerobot-leisaac-pickup-cude-30-v2")
        self.assertEqual(dataset, path)
        self.assertEqual("leisaac/pickup_cude_30_v2@v2.1", revision)

    def test_lerobot_benchmark_contract_is_six_dimensional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "leisaac" / "pickup_cude_30_v2" / "meta"
            dataset.mkdir(parents=True)
            names = [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
            ]
            (dataset / "info.json").write_text(json.dumps({
                "codebase_version": "v2.1", "robot_type": "so101_follower",
                "features": {
                    "action": {"shape": [6], "names": names},
                    "observation.state": {"shape": [6], "names": names},
                    "observation.images.front": {}, "observation.images.wrist": {},
                },
            }), encoding="utf-8")
            with patch.dict(os.environ, {"LEROBOT_DATASET_ROOT": str(root)}):
                benchmark = resolve_benchmark("lerobot-leisaac-pickup-cude-30-v2")
        self.assertEqual("lerobot", benchmark.platform)
        self.assertEqual({"type": "joint_position", "dim": 6}, benchmark.action_schema)
        self.assertEqual(6, len(benchmark.state_keys))

    def test_init_parser_accepts_a_judge_selected_benchmark(self):
        args = harness.parser().parse_args([
            "init", "/tmp/job", "--job-id", "job", "--upstream-repo", "repo", "--vla", "smolvla",
            "--checkpoint", "checkpoint", "--factory-base", "main",
            "--benchmark", "lerobot-leisaac-pickup-cude-30-v2",
        ])
        self.assertEqual("lerobot-leisaac-pickup-cude-30-v2", args.benchmark)

    def test_parity_codec_uses_the_detected_robotwin_format(self):
        class Reader:
            def get_schema(self, path):
                return type("Schema", (), {"source_format": "robotwin_hdf5"})()

        self.assertEqual("robotwin", _codec_format(Reader(), Path("dataset"), "auto"))

    def test_cuda_link_uses_the_real_runtime_not_a_dangling_env_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "env" / "lib" / "libcudart.so.12.4.127"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("", encoding="utf-8")
            broken = runtime.parent / "libcudart.so"
            broken.symlink_to("libcudart.so.12.1.105")
            link = _cuda_link_dir(root / "env", root / "artifacts") / "libcudart.so"
            self.assertEqual(runtime, link.resolve())

    def test_opencode_runner_is_isolated_from_workspace_plugins(self):
        self.assertIn("--pure", OPENCODE_HEADLESS)
        self.assertIn("--auto", OPENCODE_HEADLESS)
        self.assertEqual(DEFAULT_OPENCODE_MODEL, "csi-provider/GLM-5.2")
        self.assertEqual(OPENCODE_MODELS[0], DEFAULT_OPENCODE_MODEL)
        self.assertIn("csi-provider/GLM-5.3", OPENCODE_MODELS)

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            payload = {
                "job_id": "demo", "upstream_repo": "repo", "vla": "model", "checkpoint": "ckpt",
                "factory_base": "base", "benchmark_name": "robotwin-beat-block-hammer", "benchmark_revision": "bench",
                "comparison": {"success_rate": 0}, "opencode_model": "csi-provider/GLM-5.3",
            }
            (job / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            manifest = harness.load_manifest(job)
            self.assertEqual("demo", manifest.job_id)
            self.assertEqual("csi-provider/GLM-5.3", manifest.opencode_model)

    def test_old_manifest_gets_the_fixed_benchmark_name(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            payload = {
                "job_id": "old", "upstream_repo": "repo", "vla": "model", "checkpoint": "ckpt",
                "factory_base": "base", "benchmark_revision": "bench", "comparison": {"success_rate": 0},
            }
            (job / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            manifest = harness.load_manifest(job)
            self.assertEqual("robotwin-beat-block-hammer", manifest.benchmark_name)
            self.assertEqual(DEFAULT_OPENCODE_MODEL, manifest.opencode_model)

    def test_policy_spec_uses_judge_owned_checkpoint_not_agent_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for name in ("policy_impl.py", "install_model.sh", "smoke_test_model.py"):
                (bundle / name).write_text("", encoding="utf-8")
            spec = {
                "contract_version": "robotwin-policy/v1", "implementation": "policy_impl.py",
                "install_script": "install_model.sh", "smoke_test": "smoke_test_model.py",
                "input_schema": {"platform": "robotwin"},
                "action_schema": {"type": "qpos", "action_horizon": 50, "action_dim": 14},
            }
            (bundle / "policy_spec.json").write_text(json.dumps(spec), encoding="utf-8")
            manifest = harness.JobManifest(
                "job", "repo", "model", "ckpt", "base", "robotwin-beat-block-hammer", "rev", {"success_rate": 0},
            )
            self.assertEqual(spec, harness.policy_spec(bundle, manifest))
            spec["python"] = "candidate-owned"
            (bundle / "policy_spec.json").write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime is Judge-owned"):
                harness.policy_spec(bundle, manifest)
            del spec["python"]
            spec["checkpoint"] = "candidate-owned"
            (bundle / "policy_spec.json").write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Judge-owned"):
                harness.policy_spec(bundle, manifest)
            del spec["checkpoint"]
            spec["action_schema"]["action_horizon"] = 0
            (bundle / "policy_spec.json").write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "action_horizon"):
                harness.policy_spec(bundle, manifest)

    def test_policy_spec_uses_lerobot_action_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for name in ("policy_impl.py", "install_model.sh", "smoke_test_model.py"):
                (bundle / name).write_text("", encoding="utf-8")
            spec = {
                "contract_version": "robotwin-policy/v1", "implementation": "policy_impl.py",
                "install_script": "install_model.sh", "smoke_test": "smoke_test_model.py",
                "input_schema": {"platform": "lerobot"},
                "action_schema": {"type": "joint_position", "action_horizon": 50, "action_dim": 6},
            }
            (bundle / "policy_spec.json").write_text(json.dumps(spec), encoding="utf-8")
            manifest = harness.JobManifest(
                "job", "repo", "model", "ckpt", "base", "lerobot-leisaac-pickup-cude-30-v2", "rev", {},
            )
            self.assertEqual(spec, harness.policy_spec(bundle, manifest))

    def test_factory_bundle_is_a_native_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            checkpoint = bundle / "factory_checkpoint"
            metadata = checkpoint / "inference_metadata"
            metadata.mkdir(parents=True)
            for name in ("model.pt", "weights.json"):
                (checkpoint / name).write_text('{"format": "bare_full"}' if name == "weights.json" else "", encoding="utf-8")
            for name in ("assembly.json", "recipe.yaml"):
                (metadata / name).write_text("{}", encoding="utf-8")
            for name in ("install_model.sh", "smoke_test_model.py"):
                (bundle / name).write_text("", encoding="utf-8")
            spec = {"format": "vla-factory-inference/v1", "factory_checkpoint": "factory_checkpoint",
                    "install_script": "install_model.sh", "smoke_test": "smoke_test_model.py"}
            (bundle / "factory_bundle.json").write_text(json.dumps(spec), encoding="utf-8")
            manifest = harness.JobManifest("job", "repo", "model", "ckpt", "base", "bench", "rev", {"success_rate": 0})
            self.assertEqual(spec, harness.factory_bundle_spec(bundle, manifest))
            spec["python"] = "candidate-owned"
            (bundle / "factory_bundle.json").write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Judge-owned"):
                harness.factory_bundle_spec(bundle, manifest)
            del spec["python"]
            spec["checkpoint"] = "candidate-owned"
            (bundle / "factory_bundle.json").write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Judge-owned"):
                harness.factory_bundle_spec(bundle, manifest)

    def test_judge_runtime_path_is_fixed_and_never_read_from_agent_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / ".judge"
            runtime = bundle / "judge-runtime" / "bin" / "python"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("#!/bin/sh\n", encoding="utf-8")
            runtime.chmod(0o755)
            self.assertEqual(str(runtime), harness._judge_python(bundle, "factory"))
            runtime.unlink()
            with self.assertRaisesRegex(RuntimeError, "did not create Judge runtime"):
                harness._judge_python(bundle, "factory")

    def test_factory_preflight_runs_only_candidate_install_and_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / ".judge"
            bundle.mkdir()
            spec = {
                "install_script": "install_model.sh",
                "smoke_test": "smoke_test_model.py",
            }
            manifest = harness.JobManifest(
                "job", "repo", "model", "ckpt", "base",
                "robotwin-beat-block-hammer", "rev", {},
            )
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch.object(harness, "factory_bundle_spec", return_value=spec),
                patch.object(harness, "_hermetic_environment", return_value={}),
                patch.object(harness, "_judge_bootstrap_python", return_value="/bootstrap/python"),
                patch.object(harness, "_judge_python", return_value="/runtime/python"),
                patch.object(harness.subprocess, "run", return_value=completed) as run,
            ):
                result = harness.run_preflight(bundle, manifest, "factory")

        self.assertEqual("/runtime/python", result["judge_python"])
        self.assertEqual([
            ["bash", str(bundle / "install_model.sh")],
            ["/runtime/python", str(bundle / "smoke_test_model.py")],
        ], [call.args[0] for call in run.call_args_list])

    def test_reference_and_factory_preflights_run_in_parallel(self):
        rendezvous = threading.Barrier(2)
        candidates = set()
        lock = threading.Lock()
        manifest = harness.JobManifest(
            "job", "repo", "model", "ckpt", "base",
            "robotwin-beat-block-hammer", "rev", {},
        )

        def preflight(_path, _manifest, candidate):
            with lock:
                candidates.add(candidate)
            rendezvous.wait(timeout=2)
            return {"candidate": candidate}

        with patch.object(harness, "run_preflight", side_effect=preflight):
            reference, factory = harness._run_preflights(Path("/reference"), Path("/factory"), manifest)

        self.assertEqual("reference", reference["candidate"])
        self.assertEqual("factory", factory["candidate"])
        self.assertEqual({"reference", "factory"}, candidates)

    def test_reference_and_factory_loss_capture_run_in_parallel(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            reference_dir = job / "workspaces" / "upstream" / ".judge"
            factory_dir = job / "workspaces" / "factory" / ".judge"
            reference_dir.mkdir(parents=True)
            factory_dir.mkdir(parents=True)
            rendezvous = threading.Barrier(2)
            capture_logs = set()
            lock = threading.Lock()

            def run_logged(command, _cwd, log, _env=None):
                if log.name in {"reference.log", "factory.log"}:
                    with lock:
                        capture_logs.add(log.name)
                    rendezvous.wait(timeout=2)
                elif log.name == "compare.log":
                    report = Path(command[command.index("--report") + 1])
                    report.write_text('{"status": "PASS"}', encoding="utf-8")

            manifest = harness.JobManifest(
                "job", "repo", "model", "ckpt", "base",
                "robotwin-beat-block-hammer", "rev", {},
            )
            benchmark = type("Benchmark", (), {"platform": "robotwin", "state_keys": ()})()
            with (
                patch.object(harness, "resolve_benchmark", return_value=benchmark),
                patch.object(harness, "_run_logged", side_effect=run_logged),
            ):
                result = harness._run_parity(
                    job, manifest, reference_dir, factory_dir,
                    {"judge_python": "/reference/python", "implementation": "policy_impl.py"},
                    {"judge_python": "/factory/python", "factory_checkpoint": "factory_checkpoint"},
                    job / "dataset", "revision",
                )

        self.assertEqual("PASS", result["status"])
        self.assertEqual({"reference.log", "factory.log"}, capture_logs)

    def test_judge_bootstrap_python_uses_explicit_judge_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            bootstrap = Path(directory) / "python3.12"
            bootstrap.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            bootstrap.chmod(0o755)
            with patch.dict(os.environ, {"VLA_JUDGE_BOOTSTRAP_PYTHON": str(bootstrap)}, clear=False):
                self.assertEqual(str(bootstrap.resolve()), harness._judge_bootstrap_python())

    def test_repair_inventory_lists_role_files_but_not_runtime_or_upstream_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            artifact = workspace / ".judge"
            runtime = artifact / "judge-runtime"
            upstream = artifact / "upstream_source"
            runtime.mkdir(parents=True)
            upstream.mkdir()
            (artifact / "policy_impl.py").write_text("policy", encoding="utf-8")
            (runtime / "python").write_text("runtime", encoding="utf-8")
            (upstream / "README.md").write_text("source", encoding="utf-8")
            inventory = harness._repair_inventory(workspace, artifact)
        paths = {entry["path"] for entry in inventory["artifact_files"]}
        self.assertIn(".judge/policy_impl.py", paths)
        self.assertNotIn(".judge/judge-runtime/python", paths)
        self.assertNotIn(".judge/upstream_source/README.md", paths)

    def test_contract_implementation_is_loaded_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            implementation = Path(directory) / "policy_impl.py"
            implementation.write_text(
                "class Demo:\n"
                "    def __init__(self, checkpoint): self.checkpoint = checkpoint\n"
                "    def reset(self, task_info): pass\n"
                "    def act(self, observation): return [[0.0]]\n"
                "    def compute_loss(self, observation): return 0.0\n"
                "def build_policy(checkpoint): return Demo(checkpoint)\n",
                encoding="utf-8",
            )
            policy = _load(implementation, "checkpoint")
            self.assertEqual("checkpoint", policy.checkpoint)
            self.assertEqual([[0.0]], policy.act({}))

    def test_contract_implementation_requires_compute_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            implementation = Path(directory) / "policy_impl.py"
            implementation.write_text(
                "class Demo:\n"
                "    def reset(self, task_info): pass\n"
                "    def act(self, observation): return [[0.0]]\n"
                "def build_policy(checkpoint): return Demo()\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TypeError, "compute_loss"):
                _load(implementation, "checkpoint")

    def test_parity_runner_uses_policy_implementation(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is required to serialize a policy sample")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            implementation = root / "policy_impl.py"
            implementation.write_text(
                "class Policy:\n"
                "    def compute_loss(self, observation): return 1.25\n"
                "def build_policy(checkpoint): return Policy()\n",
                encoding="utf-8",
            )
            sample = root / "sample.npz"
            write_sample({
                "robotwin_observation": {
                    "observation": {"head_camera": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}},
                    "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
                },
                "instruction": "task", "step": 0,
                "training": {"actions": np.zeros((1, 14), dtype=np.float32), "action_is_pad": np.zeros(1, dtype=bool)},
            }, {"seed": 42, "precision": "fp32", "mode": "eval"}, sample)
            output = root / "loss.json"
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("parity") / "native_parity_runner.py"),
                "--implementation", str(implementation), "--checkpoint", "ckpt",
                "--sample", str(sample), "--output", str(output),
            ], check=True)
            self.assertEqual(1.25, json.loads(output.read_text(encoding="utf-8"))["loss"])

    def test_robotwin_result_is_one_bounded_success_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "timestamp" / "_result.txt"
            result.parent.mkdir()
            result.write_text("0.45\n", encoding="utf-8")
            self.assertEqual(0.45, _result_rate(root))

    def test_agent_workspace_is_job_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            with self.assertRaisesRegex(ValueError, "not prepared"):
                harness.workspace_for_role(job, "reference")
            (job / "workspace-lock.json").write_text("{}", encoding="utf-8")
            (job / "manifest.json").write_text(json.dumps({
                "job_id": "job", "upstream_repo": "repo", "vla": "model", "checkpoint": "ckpt",
                "factory_base": "base", "benchmark_name": "robotwin-beat-block-hammer", "benchmark_revision": "rev",
                "comparison": {"success_rate": 0},
            }), encoding="utf-8")
            (job / "workspaces" / "upstream").mkdir(parents=True)
            (job / "workspaces" / "factory").mkdir()
            upstream = job / "workspaces" / "upstream"
            (upstream / "pyproject.toml").write_text("# upstream package metadata", encoding="utf-8")
            source = upstream / "src" / "lerobot" / "policies" / "model"
            source.mkdir(parents=True)
            (source / "model.py").write_text("# upstream", encoding="utf-8")
            common = source.parent / "common"
            common.mkdir()
            (common / "vla_utils.py").write_text("# shared upstream API", encoding="utf-8")
            (upstream / ".git").mkdir()
            (upstream / ".judge").mkdir()
            (upstream / ".judge" / "policy_impl.py").write_text("reference artifact", encoding="utf-8")
            subprocess.run(["git", "-C", str(upstream), "init"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(upstream), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(upstream), "config", "user.name", "Harness test"], check=True)
            subprocess.run(["git", "-C", str(upstream), "add", "pyproject.toml", "src"], check=True)
            subprocess.run(["git", "-C", str(upstream), "commit", "-m", "upstream source"], check=True, capture_output=True, text=True)
            self.assertEqual(job / "workspaces" / "upstream", harness.workspace_for_role(job, "reference"))
            self.assertEqual(job / "workspaces" / "factory", harness.workspace_for_role(job, "adapter"))
            self.assertEqual(job / "workspaces" / "factory" / ".judge", harness.bundle_for_role(job, "adapter"))
            self.assertFalse((job / "workspaces" / "factory" / ".judge" / "upstream_source").exists())
            harness._stage_upstream_source(job, job / "workspaces" / "factory" / ".judge")
            staged = job / "workspaces" / "factory" / ".judge" / "upstream_source"
            self.assertTrue((staged / "pyproject.toml").is_file())
            self.assertTrue((staged / "src" / "lerobot" / "policies" / "model" / "model.py").is_file())
            self.assertTrue((staged / "src" / "lerobot" / "policies" / "common" / "vla_utils.py").is_file())
            self.assertTrue((staged / ".git").is_dir())
            self.assertEqual("repo", subprocess.run(
                ["git", "-C", str(staged), "remote", "get-url", "origin"], check=True,
                capture_output=True, text=True,
            ).stdout.strip())
            self.assertFalse((staged / ".judge").exists())

    def test_agent_bundles_contain_only_role_owned_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            # Pin the robotwin benchmark: this test asserts the robotwin
            # contract's content, which must not depend on the mutable default.
            bundle = harness._seed_bundle(
                Path(directory), benchmark="robotwin-beat-block-hammer", role="reference",
            )
            self.assertTrue((bundle / "REFERENCE_BUNDLE_CONTRACT.md").is_file())
            self.assertIn("robotwin_observation", (bundle / "REFERENCE_BUNDLE_CONTRACT.md").read_text(encoding="utf-8"))
            self.assertTrue((bundle / "policy_contract.py").is_file())
            self.assertTrue((bundle / "robotwin_policy_server.py").is_file())
            self.assertFalse((bundle / "native_parity_runner.py").exists())
            self.assertFalse((bundle / "FACTORY_BUNDLE_CONTRACT.md").exists())
            lerobot_bundle = harness._seed_bundle(
                Path(directory) / "lerobot", benchmark="lerobot-leisaac-pickup-cude-30-v2", role="reference",
            )
            self.assertIn("action_dim\": 6", (lerobot_bundle / "REFERENCE_BUNDLE_CONTRACT.md").read_text(encoding="utf-8"))
            self.assertFalse((lerobot_bundle / "robotwin_policy_server.py").exists())
            lerobot_factory = harness._seed_bundle(
                Path(directory) / "lerobot-factory", benchmark="lerobot-leisaac-pickup-cude-30-v2", role="adapter",
            )
            lerobot_contract = (lerobot_factory / "FACTORY_BUNDLE_CONTRACT.md").read_text(encoding="utf-8")
            self.assertIn("LerobotHostObsAdapter", lerobot_contract)
            self.assertIn('"format": "vla-factory-inference/v1"', lerobot_contract)
            self.assertIn('"factory_checkpoint": "factory_checkpoint"', lerobot_contract)
            self.assertIn("JUDGE_RUNTIME_DIR", lerobot_contract)
            self.assertIn("JUDGE_BOOTSTRAP_PYTHON", lerobot_contract)
            factory_bundle = harness._seed_bundle(
                Path(directory) / "factory", benchmark="robotwin-beat-block-hammer", role="adapter",
            )
            self.assertTrue((factory_bundle / "FACTORY_BUNDLE_CONTRACT.md").is_file())
            self.assertIn("HF_HUB_OFFLINE=1", (factory_bundle / "FACTORY_BUNDLE_CONTRACT.md").read_text(encoding="utf-8"))
            self.assertIn(
                "complete set of runtime dependencies",
                (factory_bundle / "FACTORY_BUNDLE_CONTRACT.md").read_text(encoding="utf-8"),
            )
            for name in ("policy_contract.py", "factory_policy_server.py", "parity_io.py",
                         "native_parity_runner.py", "factory_parity_runner.py"):
                self.assertFalse((factory_bundle / name).exists(), name)

    def test_agent_output_check_lists_missing_contract_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "policy_impl.py"):
                require_outputs("reference", Path(directory), ("policy_impl.py",))

    def test_agents_have_separate_parity_responsibilities(self):
        manifest = harness.JobManifest("job", "repo", "smolvla", "ckpt", "base", "bench", "rev", {})
        adapter_prompt = prompts.adapter_prompt(manifest, Path("/bundle"))
        reference_prompt = prompts.reference_prompt(manifest, Path("/bundle"))
        self.assertIn("behavioral fidelity", adapter_prompt)
        self.assertIn("compare their native forward loss", adapter_prompt)
        self.assertIn("only a restoration gate", adapter_prompt)
        self.assertIn("same benchmark observation", adapter_prompt)
        self.assertIn("A design, investigation, or route selection", adapter_prompt)
        self.assertIn("independent git checkout", adapter_prompt)
        self.assertIn("fresh-offline-cache", adapter_prompt)
        self.assertIn("source checkpoint configuration", adapter_prompt)
        self.assertIn("inference_needs_base_checkpoint=False", adapter_prompt)
        self.assertIn("factory_checkpoint/model_artifacts/", adapter_prompt)
        self.assertIn("model.config.model_artifacts", adapter_prompt)
        self.assertIn("`/judge-inputs/checkpoint`", adapter_prompt)
        self.assertIn("do not download, modify, or replace it", adapter_prompt)
        self.assertNotIn("adapt-new-vla-model/SKILL.md", adapter_prompt)
        robotwin_contract = Path(__file__).with_name("benchmarks") / "robotwin" / "beat_block_hammer" / "FACTORY_BUNDLE_CONTRACT.md"
        self.assertIn("Judge handoff manifest", robotwin_contract.read_text(encoding="utf-8"))
        self.assertIn("model_artifacts/", robotwin_contract.read_text(encoding="utf-8"))
        self.assertNotIn("native_bridge.py", reference_prompt)
        self.assertIn("require_outputs", reference_prompt)
        self.assertIn("in-memory synthetic Policy Mapping", reference_prompt)
        self.assertIn("declared action schema", reference_prompt)
        self.assertIn("filenames relative to `.judge/`", reference_prompt)
        self.assertNotIn("Do not use /tmp", reference_prompt)
        self.assertIn("factory_checkpoint/model.pt", adapter_prompt)
        self.assertIn("require_outputs", adapter_prompt)
        self.assertIn("REFERENCE_BUNDLE_CONTRACT.md", reference_prompt)
        self.assertIn("`/judge-inputs/checkpoint`", reference_prompt)
        self.assertNotIn("verify_factory_bundle.py", adapter_prompt)


    def test_agent_prompts_keep_artifacts_inside_the_workspace(self):
        manifest = harness.JobManifest("job", "repo", "smolvla", "ckpt", "base", "bench", "rev", {})
        artifact_dir = Path("/outside/workspace/.judge")
        for prompt in (prompts.reference_prompt(manifest, artifact_dir), prompts.adapter_prompt(manifest, artifact_dir)):
            self.assertIn("`.judge/`", prompt)
            self.assertNotIn(str(artifact_dir), prompt)


if __name__ == "__main__":
    unittest.main()

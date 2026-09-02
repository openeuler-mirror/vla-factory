"""L0 tests for the CLI's argparse surface and command dispatch.

``vla_factory/user_interface/cli.py`` is the framework's user-facing contract: which
subcommands exist, which arguments they require, and — for ``train`` — that the
CLI overrides actually reach the trainer instead of being parsed and dropped.
Before Issue #7 only ``deploy`` had any coverage here.

Every test stubs the work the branch would do (training, inference,
preprocessing), so this file stays a pure argparse/dispatch test: no dataset,
no checkpoint, no model.
"""

from __future__ import annotations

import sys

import pytest

from vla_factory.user_interface import cli
from vla_factory.utils.constants import INFERENCE_META_DIR, RECIPE_FILE


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["vlafactory-cli", *args])


# ── Top-level dispatch ───────────────────────────────────────────────


def test_no_command_prints_help_and_exits_nonzero(monkeypatch, capsys):
    _argv(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "usage: vlafactory-cli" in capsys.readouterr().out


def test_unknown_command_is_rejected(monkeypatch, capsys):
    _argv(monkeypatch, "finetune")
    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "invalid choice: 'finetune'" in capsys.readouterr().err


@pytest.mark.parametrize("command,missing", [
    ("train", "--config"),
    ("evaluate", "--checkpoint"),
    ("infer", "--checkpoint"),
    ("deploy", "--checkpoint"),
])
def test_required_argument_is_enforced(monkeypatch, capsys, command, missing):
    """Each subcommand must refuse to start without its mandatory input."""
    _argv(monkeypatch, command)
    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert missing in capsys.readouterr().err


def test_evaluate_requires_dataset_as_well(monkeypatch, capsys):
    _argv(monkeypatch, "evaluate", "--checkpoint", "ckpt")
    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "--dataset" in capsys.readouterr().err


# ── train: overrides must reach the trainer ──────────────────────────


@pytest.fixture
def captured_train(monkeypatch):
    """Replace the training implementation with a recorder."""
    captured: dict = {}

    def _fake_train(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return {"loss": 0.0}

    monkeypatch.setattr("vla_factory.training.train.train", _fake_train)
    return captured


def test_train_forwards_config_and_no_overrides(monkeypatch, captured_train, capsys):
    """Without override flags the trainer must receive None, not a stand-in value."""
    _argv(monkeypatch, "train", "--config", "recipe.yaml")
    cli.main()

    assert captured_train["config"] == "recipe.yaml"
    assert captured_train["override_steps"] is None
    assert captured_train["override_batch_size"] is None
    assert captured_train["override_output_dir"] is None
    assert "Training complete" in capsys.readouterr().out


def test_train_forwards_all_overrides(monkeypatch, captured_train):
    """--steps/--batch-size/--output-dir are the documented recipe overrides."""
    _argv(
        monkeypatch, "train",
        "--config", "recipe.yaml",
        "--steps", "123",
        "--batch-size", "7",
        "--output-dir", "/tmp/out",
    )
    cli.main()

    assert captured_train["override_steps"] == 123
    assert captured_train["override_batch_size"] == 7
    assert captured_train["override_output_dir"] == "/tmp/out"


def test_train_rejects_non_integer_steps(monkeypatch, capsys):
    _argv(monkeypatch, "train", "--config", "recipe.yaml", "--steps", "many")
    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "--steps" in capsys.readouterr().err


# ── list ─────────────────────────────────────────────────────────────


def test_list_prints_registered_models(monkeypatch, capsys):
    """`list` with no --config enumerates the registry, install hint included."""
    _argv(monkeypatch, "list")
    cli.main()

    out = capsys.readouterr().out
    assert "act" in out
    assert "backend=" in out and "install=" in out


def test_list_reports_empty_registry(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_entries", dict)
    _argv(monkeypatch, "list")
    cli.main()

    assert "No models registered." in capsys.readouterr().out


# ── infer: recipe resolution from the checkpoint ─────────────────────


def test_infer_without_config_and_without_saved_recipe_exits(monkeypatch, capsys, tmp_path):
    """No --config and nothing in inference_metadata/ is a hard stop, not a guess."""
    _argv(monkeypatch, "infer", "--checkpoint", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "no saved recipe found" in capsys.readouterr().out


def test_infer_falls_back_to_the_checkpoint_recipe(monkeypatch, capsys, tmp_path):
    """With no --config the saved inference_metadata/recipe.yaml must be used."""
    meta = tmp_path / INFERENCE_META_DIR
    meta.mkdir()
    saved_recipe = meta / RECIPE_FILE
    saved_recipe.write_text("model: {name: act}\n")

    seen: dict = {}
    monkeypatch.setattr(
        "vla_factory.inference.evaluate_dataset.infer_dataset_sample",
        lambda **kw: seen.update(kw) or {"action": "ok"},
    )

    _argv(monkeypatch, "infer", "--checkpoint", str(tmp_path))
    cli.main()

    assert seen["config"] == str(saved_recipe)
    assert seen["dataset_index"] == 0, "default sample index"
    assert "Inference result" in capsys.readouterr().out


def test_infer_finds_the_recipe_one_level_up(monkeypatch, tmp_path):
    """checkpoint-NNN/ subdirs resolve the recipe from the run root."""
    run_root = tmp_path
    (run_root / INFERENCE_META_DIR).mkdir()
    saved_recipe = run_root / INFERENCE_META_DIR / RECIPE_FILE
    saved_recipe.write_text("model: {name: act}\n")
    checkpoint = run_root / "checkpoint-100"
    checkpoint.mkdir()

    seen: dict = {}
    monkeypatch.setattr(
        "vla_factory.inference.evaluate_dataset.infer_dataset_sample",
        lambda **kw: seen.update(kw) or {},
    )

    _argv(monkeypatch, "infer", "--checkpoint", str(checkpoint))
    cli.main()

    assert seen["config"] == str(saved_recipe)

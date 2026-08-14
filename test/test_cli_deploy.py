"""CLI tests for the deploy command."""

from __future__ import annotations

import sys

import pytest

from vla_factory import cli


def _run_invalid_deploy(monkeypatch, command: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vlafactory-cli",
            command,
            "--checkpoint",
            "unused-for-argument-validation",
            "--max-loop-freq-hz",
            "0",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_deploy_command(monkeypatch, capsys):
    _run_invalid_deploy(monkeypatch, "deploy")

    captured = capsys.readouterr()
    assert "--max-loop-freq-hz must be a positive number" in captured.err


def test_serve_command_is_not_registered(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["vlafactory-cli", "serve"])

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "invalid choice: 'serve'" in capsys.readouterr().err

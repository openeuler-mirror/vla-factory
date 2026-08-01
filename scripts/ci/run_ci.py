#!/usr/bin/env python3
"""Interactive launcher for the vla-factory CI daemon.

Prompts for configuration with auto-detected defaults, validates each
value, saves to ``~/.vlaf_ci.conf`` for next-run defaults, then starts
the daemon.

Usage::

    python3 ci/runner/run_ci.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONF_FILE = Path.home() / ".vlaf_ci.conf"
REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Auto-detect defaults ─────────────────────────────────────────────

def detect_token() -> str:
    """Token from config.json (repo root) or saved conf."""
    config = REPO_ROOT / "config.json"
    if config.exists():
        try:
            data = json.loads(config.read_text())
            return data.get("gitcode", {}).get("token", "")
        except Exception:
            pass
    return ""


def detect_env_default(label: str) -> str:
    """Default python path for an env label — matches build_ci_envs.sh output."""
    p = Path.home() / "envs" / label / "bin" / "python"
    return str(p) if p.exists() else ""


# ── Helpers ──────────────────────────────────────────────────────────

def load_conf() -> dict[str, str]:
    if CONF_FILE.exists():
        conf = {}
        for line in CONF_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                conf[k.strip()] = v.strip()
        return conf
    return {}


def save_conf(conf: dict[str, str]) -> None:
    lines = ["# vla-factory CI daemon configuration (auto-generated)",
             "# Edit or delete this file to reconfigure.\n"]
    for k, v in conf.items():
        lines.append(f"{k}={v}")
    CONF_FILE.write_text("\n".join(lines) + "\n")
    print(f"  saved → {CONF_FILE}")


def ask(prompt: str, default: str = "", required: bool = False,
        validate=None) -> str:
    """Prompt for a value with default and validation."""
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"  {prompt}{suffix}: ").strip() or default
        if required and not val:
            print("    必填")
            continue
        if validate:
            err = validate(val)
            if err:
                print(f"    {err}")
                continue
        return val


def warn_path(val: str) -> str | None:
    """Soft validation: warn but don't block."""
    if val and not Path(val).exists():
        return f"注意: 路径不存在 ({val}),确认是否正确"
    return None


def validate_python(val: str) -> str | None:
    if not val:
        return None
    if not Path(val).exists():
        return f"解释器不存在: {val}"
    return None


# ── Interactive setup ────────────────────────────────────────────────

def interactive_setup() -> dict[str, str]:
    old = load_conf()

    if old:
        print("发现已有配置 (~/.vlaf_ci.conf):\n")
        for k, v in old.items():
            display = v if k != "VLAF_GITCODE_TOKEN" else v[:8] + "..."
            print(f"  {k} = {display}")
        choice = input("\n  回车 = 使用已有配置, r = 重新配置: ").strip().lower()
        if choice != "r":
            return old

    # Merge saved conf + auto-detected defaults.
    d = {
        "VLAF_GITCODE_TOKEN": old.get("VLAF_GITCODE_TOKEN") or detect_token(),
        "VLAF_BASE_DIR": old.get("VLAF_BASE_DIR") or str(Path.home() / "vla-factory-ci"),
        "VLAF_POLL_INTERVAL": old.get("VLAF_POLL_INTERVAL", "30"),
        "VLAF_ENV_BASE": old.get("VLAF_ENV_BASE") or detect_env_default("base") or sys.executable,
        "VLAF_ENV_ACT": old.get("VLAF_ENV_ACT") or detect_env_default("act"),
        "VLAF_ENV_PI": old.get("VLAF_ENV_PI") or detect_env_default("pi"),
    }

    print("\n" + "=" * 60)
    print("  vla-factory CI daemon 配置")
    print("  (方括号内为默认值, 直接回车采用)")
    print("=" * 60 + "\n")

    conf = {}
    conf["VLAF_GITCODE_TOKEN"] = ask(
        "GitCode token", d["VLAF_GITCODE_TOKEN"], required=True)

    conf["VLAF_BASE_DIR"] = ask(
        "CI 目录 (不存在会自动 clone)", d["VLAF_BASE_DIR"])

    conf["VLAF_POLL_INTERVAL"] = ask(
        "轮询间隔 (秒)", d["VLAF_POLL_INTERVAL"])

    print("\n  测试环境 (act/pi 留空则跳过该 tier):")
    print("  未跑过 build_ci_envs.sh? 留空即可, 之后手动指定\n")

    conf["VLAF_ENV_BASE"] = ask(
        "base python (L0)", d["VLAF_ENV_BASE"],
        required=True, validate=validate_python)

    conf["VLAF_ENV_ACT"] = ask(
        "act python (L1+L2, 留空跳过)", d["VLAF_ENV_ACT"],
        validate=validate_python)

    conf["VLAF_ENV_PI"] = ask(
        "pi python (L1, 留空跳过)", d["VLAF_ENV_PI"],
        validate=validate_python)

    print()
    save_conf(conf)
    return conf


# ── Summary ──────────────────────────────────────────────────────────

def print_summary(conf: dict[str, str]) -> None:
    envs = []
    for label in ("base", "act", "pi"):
        py = conf.get(f"VLAF_ENV_{label.upper()}")
        if py:
            envs.append(label)

    tier_map = {"base": "L0", "act": "L1+L2", "pi": "L1"}
    print("=" * 60)
    print("  配置摘要")
    print("=" * 60)
    print(f"  repo:  {conf['VLAF_BASE_DIR']}")
    print(f"  poll:  {conf.get('VLAF_POLL_INTERVAL', '30')}s")
    print(f"  envs:")
    for e in envs:
        print(f"    {e:6s} → {tier_map.get(e, '?'):8s}  ({conf[f'VLAF_ENV_{e.upper()}']})")
    print("=" * 60 + "\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    conf = interactive_setup()
    print_summary(conf)

    for k, v in conf.items():
        os.environ[k] = v

    sys.path.insert(0, str(Path(__file__).parent))
    import daemon as daemon_mod
    daemon_mod.main()


if __name__ == "__main__":
    main()

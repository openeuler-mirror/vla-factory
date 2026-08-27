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
import shutil
import sys
from pathlib import Path

CONF_FILE = Path.home() / ".vlaf_ci.conf"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Known headless agent CLIs and the command template each one needs.
# The template is executed via sh -c by the daemon; {prompt} is the
# generated review prompt file, {output} the JSON the agent must produce.
# zcode is deliberately absent: it is an Electron desktop app with no
# headless mode — invoking it just opens a window and never produces output.
# The only supported agent for now — others are being verified one by
# one and can be wired manually via VLAF_AGENT_CMD.
AGENT_CHOICES: dict[str, str] = {
    # stdin, not argv: prompts embed the full PR diff and blow past the
    # ~128KB single-argument kernel limit ("argument list too long").
    "claude": ('claude -p --no-session-persistence '
               '--disable-slash-commands --tools "" < {prompt} > {output}'),
}


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


def ask_agent_cmd() -> str:
    """Pick the headless agent command for /vla-factory review.

    Offers the known CLIs (marking any found on PATH), an OpenAI-compatible
    API adapter, manual entry, or skip. Returns the filled-in template for
    conf['VLAF_AGENT_CMD'].
    """
    detected = next((n for n in AGENT_CHOICES if shutil.which(n.split()[0])), "")
    names = list(AGENT_CHOICES)
    print("\n  无头 agent 命令 — /vla-factory review 用它执行检视提示词")
    if not detected and shutil.which("zcode"):
        print("  注意: 检测到 zcode, 但它是桌面应用、没有无头模式, 不能用于 review")
    print("  （模板经 shell 执行；{prompt}=提示词文件路径, {output}=结果 JSON 路径）")
    for i, n in enumerate(names, 1):
        mark = "  ← 已检测到" if n == detected else ""
        print(f"    {i}) {n}{mark}")
    print(f"    {len(names) + 1}) 其他（手动输入完整命令模板）")
    if detected:
        print(f"    回车 = 采用 {detected}")
    else:
        print("    回车 = 跳过（/vla-factory review 将不可用, help/retest 不受影响）")
    while True:
        c = input(f"  选择 [1-{len(names) + 1}]: ").strip().lower()
        if not c:
            return AGENT_CHOICES[detected] if detected else ""
        if c.isdigit() and 1 <= int(c) <= len(names):
            return AGENT_CHOICES[names[int(c) - 1]]
        if c == str(len(names) + 1):
            return ask("完整命令模板", "")
        if c in AGENT_CHOICES:
            return AGENT_CHOICES[c]
        print("    无效选择，请输入编号或名称")


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
    """Write ~/.vlaf_ci.conf as a fully annotated template.

    The file doubles as the documentation of every supported knob: keys
    the operator configured keep their values, the rest are written with
    their defaults (or commented out when optional).
    """
    def val(k: str, default: str = "") -> str:
        return conf.get(k, default)

    lines = [
        "# vla-factory CI daemon 配置（由 run_ci.sh 生成；可直接编辑，",
        "# 删除本文件后重新运行 scripts/run_ci.sh 可重新配置）。格式: k=v。",
        "",
        "# ── 必填 ──",
        f"VLAF_GITCODE_TOKEN={val('VLAF_GITCODE_TOKEN')}",
        f"VLAF_ENV_BASE={val('VLAF_ENV_BASE')}",
        "",
        "# ── CI 行为（默认值如下，按需修改）──",
        f"VLAF_BASE_DIR={val('VLAF_BASE_DIR', str(Path.home() / 'vla-factory-ci'))}",
        f"VLAF_POLL_INTERVAL={val('VLAF_POLL_INTERVAL', '30')}",
        f"VLAF_CI_WORKERS={val('VLAF_CI_WORKERS', '5')}",
        f"VLAF_CMD_WORKERS={val('VLAF_CMD_WORKERS', '5')}",
        f"VLAF_TIER_TIMEOUT={val('VLAF_TIER_TIMEOUT', '1200')}",
        f"VLAF_MAX_RETRIES={val('VLAF_MAX_RETRIES', '3')}",
        f"VLAF_CHECKOUT_TTL_DAYS={val('VLAF_CHECKOUT_TTL_DAYS', '7')}",
        "",
        "# ── /vla-factory review ──",
        "# 无头 agent 命令模板：占位符 {prompt}=检视提示词文件、{output}=结果 JSON",
        "# 路径，经 shell 执行（可用重定向/$(...)）。检视提示词由 skills 的",
        "# vlafactory-code-review 技能根据 PR diff 自动生成，无需手写。",
        f"VLAF_AGENT_CMD={val('VLAF_AGENT_CMD', AGENT_CHOICES['claude'])}",
        f"VLAF_REVIEW_LANG={val('VLAF_REVIEW_LANG', 'zh')}",
        f"VLAF_AGENT_TIMEOUT={val('VLAF_AGENT_TIMEOUT', '600')}",
        f"VLAF_REVIEW_WORKERS={val('VLAF_REVIEW_WORKERS', '1')}",
        f"VLAF_REVIEW_GLOBAL_WORKERS={val('VLAF_REVIEW_GLOBAL_WORKERS', '4')}",
        f"VLAF_REVIEW_STEP_TIMEOUT={val('VLAF_REVIEW_STEP_TIMEOUT', '180')}",
        "# VLAF_REVIEW_GUIDE=              # 缺省使用新技能内置的 VLA Factory policy",
        "# VLAF_SKILL_DIR=                # 缺省使用仓库内 skills 子模块并自动同步最新 main",
        "# VLAF_NODE_BIN=                 # node 不在 daemon PATH 时显式指定",
        "",
        "# ── 可选测试环境（留空则跳过该环境）──",
        f"VLAF_ENV_ACT={val('VLAF_ENV_ACT')}",
        f"VLAF_ENV_PI={val('VLAF_ENV_PI')}",
        "",
    ]
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
        "VLAF_AGENT_CMD": old.get("VLAF_AGENT_CMD", ""),
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
    conf["VLAF_AGENT_CMD"] = ask_agent_cmd()
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
    agent = conf.get("VLAF_AGENT_CMD", "")
    print(f"  review agent: {'configured' if agent else 'NOT set — /vla-factory review will refuse'}")
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

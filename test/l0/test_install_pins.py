"""Vendored-source pin guard（#27 的 pin_has_not_moved 模式）。

OpenVLA 的装配语义（prompt 模板、256-bin 动作 token、label mask 公式）是对照
``OPENVLA_REF`` 的上游源码逐行转录的，分布在
``assemble_token_action_sequence`` 步骤与 adapter 注释里。换 pin（升级 vendored
上游）时这些转录必须重读核对——本测试把 pin 钉在已核对的参考值上：改 pin 而
不改这里，CI 直接失败，而不是在训练里静默漂移。
"""

from __future__ import annotations

import re
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"

# 已核对的上游参考：c8f03f4 之上的转录（prompt 模板 / 256-bin / label mask）
# 由 test/l1/test_openvla_parity.py 用真 prismatic 组件锁定。
OPENVLA_REF_EXPECTED = "c8f03f48af692657d3060c19588038c7220e9af9"


def test_openvla_pin_has_not_moved():
    text = INSTALL_SH.read_text()
    match = re.search(r'OPENVLA_REF="\$\{OPENVLA_REF:-([0-9a-f]{40})\}"', text)
    assert match, "OPENVLA_REF pin not found in scripts/install.sh"
    assert match.group(1) == OPENVLA_REF_EXPECTED, (
        "OPENVLA_REF moved — re-verify the transcribed assembly semantics "
        "(prompt template, 256-bin mapping, label mask) against the new "
        "upstream revision, then update OPENVLA_REF_EXPECTED here and the "
        "behavioral lock in test/l1/test_openvla_parity.py."
    )

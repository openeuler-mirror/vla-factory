"""L1 parity: LoRA 挂载 —— 对 canonical peft 与 openpi 契约两个参照。

**覆盖强度与 pipeline parity 不同，这点必须先说清楚。** 我们不调用 openpi 的 LoRA：
openpi 那边是 jax + 自研的 ``lora.Einsum``（``models/lora.py``），我们是在 pytorch
移植上用 peft 复现同一套语义。跨框架做逐张量对比需要 jax↔torch 权重搬运，代价远超
收益。所以这里有两个参照，粒度不同：

1. **canonical peft（同一套库、两条路径）** —— 张量级。
   ``apply_lora``（我们的子树遍历 + 重挂）对照手写的
   ``get_peft_model(subtree, cfg)``。验的是我们的**编排**，不是 peft 本身。
   这一路参考了 LLaMA-Factory ``train/test_utils.py::compare_model`` 的思路：
   拿"用同样的库、按最直白的方式做同一件事"当基准。

2. **openpi 契约（跨框架）** —— 契约级。冻结面语义、rank/alpha、scaling 公式。
   只能比"规则是否一致"，比不了权重。

外加两组不依赖参照物的不变量：

3. **dtype** —— adapter 必须 float32、base 保持低精度。目前靠 peft 的默认行为成立，
   没有任何东西守着；``entries/pi0.py`` 先把模型转 bfloat16、之后才注入 LoRA，谁在
   ``apply_lora`` 之后补一句 ``.to(dtype)`` 就会让 adapter 掉进 bf16（8 位尾数训
   LoRA 增量）。同样取自 LLaMA-Factory ``check_lora_model``。
4. **merge 数学** —— B=0 时恒等；填入非零 B 后 merge 必须真的改动 base 权重。

与 ``test/test_lora_strategy.py`` 的分工：那边用 **fake peft**，验的是我们的挂载
*逻辑*（属 L0）；这里用**真 peft**，验的是挂载*结果*与上游契约。
"""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vla_factory.training.strategies.lora import (
    LoraConfig as VlaLoraConfig,
    _DEFAULT_TARGET_MODULES,
    apply_lora,
)

# pytest 的 prepend 导入模式会把本文件所在目录插入 sys.path，故直接导入同目录 utils。
from utils import assert_state_dict_parity, summarize_lora_parameters

pytestmark = pytest.mark.l1

if importlib.util.find_spec("peft") is None:
    pytest.skip(
        "PEFT parity needs peft installed (pip install -e '.[pi0]')",
        allow_module_level=True,
    )

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_RANK = 16
_ALPHA = 16


# ── pi0 形状的替身：paligemma + action expert 两个子树 ──────────────


class _PaliGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class _ActionExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)

    def forward(self, x):
        return self.q_proj(x)


class _FakePI0(nn.Module):
    """结构对齐 openpi PI0Pytorch 的顶层块命名。"""

    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = _PaliGemma()
        self.paligemma_with_expert.gemma_expert = _ActionExpert()
        self.state_proj = nn.Linear(8, 8)  # 子树之外的投影层


class _Metadata:
    name = "pi0"
    support_lora = True
    components = {
        "llm": ["paligemma_with_expert.paligemma."],
        "action_expert": ["paligemma_with_expert.gemma_expert."],
    }


def _recipe(targets=("llm",)) -> VlaLoraConfig:
    return VlaLoraConfig(
        r=_RANK,
        lora_alpha=_ALPHA,
        target_components=list(targets),
        init_lora_weights="gaussian",
    )


def _peft_config():
    """手写参照用的 LoraConfig，字段与 apply_lora 构造的一致。

    ``_DEFAULT_TARGET_MODULES`` 是**输入**（声明要挂哪些层）而非被测对象，直接复用；
    被测的是 apply_lora 如何找到子树并把包裹结果挂回去。
    """
    from peft import LoraConfig

    return LoraConfig(
        r=_RANK,
        lora_alpha=_ALPHA,
        lora_dropout=0.0,
        use_rslora=False,
        init_lora_weights="gaussian",
        target_modules=_DEFAULT_TARGET_MODULES,
    )


def _canonical_reference(model: _FakePI0) -> _FakePI0:
    """教科书写法：直接取子树、包裹、挂回去。不走我们的路径解析代码。"""
    from peft import get_peft_model

    subtree = model.paligemma_with_expert.paligemma
    model.paligemma_with_expert.paligemma = get_peft_model(subtree, _peft_config())
    return model


# ══════════════════════════════════════════════════════════════════════
#  1. 对照 canonical peft —— 张量级
# ══════════════════════════════════════════════════════════════════════


def test_apply_lora_matches_a_hand_written_peft_wrap():
    """我们的子树遍历 + 重挂必须与手写 get_peft_model 产出同一个模型。

    ``_resolve_parent`` / ``_walk`` / ``setattr(parent, leaf, wrapped)`` 那段是
    lora.py 里最容易出错的地方——挂错一层、或者包裹了却没挂回去，模型照样能前向，
    只是 adapter 落在别处或根本没接进计算图。
    """
    base = _FakePI0()

    torch.manual_seed(0)
    ours = apply_lora(copy.deepcopy(base), _recipe(), _Metadata())

    torch.manual_seed(0)
    reference = _canonical_reference(copy.deepcopy(base))

    assert_state_dict_parity(
        ours.state_dict(), reference.state_dict(), label="apply_lora vs 手写 peft"
    )


def test_apply_lora_returns_the_same_top_level_object():
    """单子树目标必须原地重挂并返回原顶层对象。

    返回一个新的包裹对象会让调用方手里的引用变成没有 adapter 的旧模型——训练照跑，
    梯度落在没人用的副本上。
    """
    model = _FakePI0()
    assert apply_lora(model, _recipe(), _Metadata()) is model


def test_adapters_land_only_on_the_targeted_subtree():
    """target_components=["llm"] 时，action expert 与子树外的投影层不得有 adapter。"""
    model = apply_lora(_FakePI0(), _recipe(), _Metadata())

    adapter_names = [n for n, _ in model.named_parameters() if "lora_" in n]
    assert adapter_names, "没有注入任何 adapter"
    assert all("paligemma_with_expert.paligemma." in n for n in adapter_names), \
        f"adapter 落到了目标子树之外: {adapter_names[:4]}"
    assert summarize_lora_parameters(model).adapter_modules == {"q_proj", "o_proj"}


# ══════════════════════════════════════════════════════════════════════
#  2. 对照 openpi 契约 —— 契约级
# ══════════════════════════════════════════════════════════════════════


def test_freeze_surface_matches_openpi_semantics():
    """冻结面必须等价于 openpi ``Pi0Config.get_freeze_filter()``。

    上游语义（``models/pi0_config.py``，paligemma 走 lora、action expert 不走）::

        freeze(".*llm.*") & Not(".*llm.*_1.*") & Not(".*lora.*")

    翻成参数集合就是：paligemma 基座冻结、action expert 全量可训练、lora 可训练、
    子树外的投影层可训练。``entries/pi0.py`` 的注释声称"same as openpi's freeze
    filter"——在此之前没有任何测试守着这句话。
    """
    model = apply_lora(_FakePI0(), _recipe(), _Metadata())

    frozen, trainable = set(), set()
    for name, param in model.named_parameters():
        (trainable if param.requires_grad else frozen).add(name)

    assert all("lora_" in n for n in trainable if "paligemma_with_expert.paligemma." in n), \
        "paligemma 子树里除 lora 外不应有可训练参数"
    assert any("lora_" in n for n in trainable), "lora 参数必须可训练"
    assert any("gemma_expert" in n for n in trainable), \
        "action expert 必须全量可训练（openpi 的 Not(.*llm.*_1.*)）"
    assert any("state_proj" in n for n in trainable), \
        "子树之外的投影层必须可训练"
    assert frozen and all("paligemma_with_expert.paligemma." in n for n in frozen), \
        f"只有 paligemma 基座该被冻结，实际冻结了: {sorted(frozen)[:4]}"


@pytest.mark.skipif(
    importlib.util.find_spec("openpi") is None,
    reason="openpi 未安装（bash scripts/install.sh pi0）",
)
def test_scaling_formula_matches_openpi():
    """peft 的 scaling 必须与 openpi ``LoRAConfig.scaling_value`` 算出同一个数。

    openpi: ``alpha / rank``（``rslora=True`` 时 ``alpha / sqrt(rank)``）。
    公式错了不会报错，只会让 adapter 的有效学习率整体偏掉一个常数因子。
    """
    from openpi.models import gemma

    upstream = gemma.get_config("gemma_2b_lora").lora_configs["attn"]

    model = apply_lora(
        _FakePI0(),
        _recipe(),
        _Metadata(),
    )
    wrapped = model.paligemma_with_expert.paligemma
    peft_scaling = wrapped.base_model.model.q_proj.scaling["default"]

    # 用上游的 rank/alpha 复算，确认两边公式一致（而非恰好数值相同）。
    expected = upstream.alpha / upstream.rank
    assert upstream.scaling_value == pytest.approx(expected)
    assert peft_scaling == pytest.approx(_ALPHA / _RANK)

    rs_expected = upstream.alpha / math.sqrt(upstream.rank)
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    torch.manual_seed(0)
    rs_model = get_peft_model(
        _PaliGemma(),
        PeftLoraConfig(
            r=upstream.rank, lora_alpha=upstream.alpha, use_rslora=True,
            target_modules=["q_proj"],
        ),
    )
    assert rs_model.base_model.model.q_proj.scaling["default"] == pytest.approx(rs_expected), \
        "use_rslora 的缩放公式与 openpi 的 rslora 分支不一致"


@pytest.mark.skipif(
    importlib.util.find_spec("openpi") is None,
    reason="openpi 未安装（bash scripts/install.sh pi0）",
)
def test_example_recipe_rank_alpha_match_openpi_defaults():
    """示例 recipe 的 r/alpha 应与 openpi 的 paligemma LoRA 默认值一致。"""
    import yaml

    from openpi.models import gemma

    upstream = gemma.get_config("gemma_2b_lora").lora_configs["attn"]
    recipe = yaml.safe_load((_PROJECT_ROOT / "examples/pi0_lora.yaml").read_text())
    lora_block = recipe["finetuning"]["config"]

    assert lora_block["r"] == upstream.rank
    assert lora_block["lora_alpha"] == upstream.alpha


def test_openpi_initialises_both_lora_matrices_unlike_peft():
    """记录一处**已知且不打算修**的差异：初始化。

    openpi ``lora.Einsum.setup`` 对 ``lora_a`` 和 ``lora_b`` 都用
    ``init_fn=normal(stddev=0.01)``，因此 adapter 在 step 0 就带约 1e-4 量级的扰动；
    peft 无论 ``init_lora_weights`` 取什么值，**B 恒为 0**，是精确的恒等初始化。

    peft 不暴露 B 的初始化方式，要对齐就得绕过 peft，代价不成比例。这条断言把差异
    钉在测试里：哪天 peft 改了 B 的初始化，我们会立刻知道，而不是等训练变差。
    """
    model = apply_lora(_FakePI0(), _recipe(), _Metadata())

    b_matrices = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert b_matrices, "没有找到 lora_B"
    assert all(torch.all(p == 0) for p in b_matrices), (
        "peft 的 lora_B 不再是零初始化了——merge 等价断言与本文件的差异说明都需要复核"
    )


# ══════════════════════════════════════════════════════════════════════
#  3. dtype 不变量
# ══════════════════════════════════════════════════════════════════════


def test_adapters_stay_float32_on_a_bfloat16_base():
    """base 低精度、adapter float32。

    ``entries/pi0.py`` 先 ``to_bfloat16_for_selected_params`` 再注入 LoRA，所以 peft
    是在 bf16 基座上建 adapter 的。目前 peft 默认就把 adapter 建成 fp32，但这没有任何
    东西守着——adapter 掉进 bf16 意味着用 8 位尾数承载 LoRA 增量，静默劣化。
    """
    model = _FakePI0().to(dtype=torch.bfloat16)
    model = apply_lora(model, _recipe(), _Metadata())

    summary = summarize_lora_parameters(model)
    assert summary.adapter_dtypes == {"torch.float32"}, \
        f"adapter 不是 float32: {summary.adapter_dtypes}"
    assert all(grad for _, grad in summary.adapter_params.values()), \
        "adapter 必须可训练"

    frozen_dtypes = {
        dtype for name, (dtype, grad) in summary.base_params.items()
        if not grad and "paligemma_with_expert.paligemma." in name
    }
    assert frozen_dtypes == {"torch.bfloat16"}, \
        f"被冻结的基座权重应保持 bfloat16: {frozen_dtypes}"


# ══════════════════════════════════════════════════════════════════════
#  4. merge 数学
# ══════════════════════════════════════════════════════════════════════


def test_merge_is_identity_while_lora_b_is_zero():
    """B=0 时 adapter 是恒等，merge 前后前向输出必须逐位相等。

    这是最干净的二值信号：若 adapter 根本没接进计算图，或 merge 把缩放系数用错，
    这条立刻红。
    """
    from vla_factory.training.strategies.lora import merge_lora_adapters

    torch.manual_seed(0)
    model = apply_lora(_FakePI0(), _recipe(), _Metadata())
    model.eval()

    x = torch.randn(2, 8)
    with torch.no_grad():
        before = model.paligemma_with_expert.paligemma(x)   # peft 包裹态
        merged = merge_lora_adapters(model)
        after = merged.paligemma_with_expert.paligemma(x)   # 解包并合并之后

    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_merge_writes_the_adapter_delta_into_the_base_weight():
    """填入非零 B 之后，merge 必须真的改动 base 权重，且改动量 = B@A*scaling。

    只断言"merge 后前向不变"是不够的——一个什么都没做的 merge 同样满足它。这里用
    ``must_differ`` 反向断言 base 权重必须变，再逐元素核对增量。
    """
    from vla_factory.training.strategies.lora import merge_lora_adapters

    torch.manual_seed(0)
    model = apply_lora(_FakePI0(), _recipe(), _Metadata())
    q_proj = model.paligemma_with_expert.paligemma.base_model.model.q_proj

    with torch.no_grad():
        q_proj.lora_B["default"].weight.normal_(std=0.05)
        base_weight = q_proj.base_layer.weight.detach().clone()
        delta = (
            q_proj.lora_B["default"].weight
            @ q_proj.lora_A["default"].weight
        ) * q_proj.scaling["default"]
        expected = base_weight + delta

    before_state = {"q_proj.weight": base_weight}
    merged = merge_lora_adapters(model)
    after_state = {"q_proj.weight": merged.paligemma_with_expert.paligemma.q_proj.weight}

    assert_state_dict_parity(
        after_state, before_state,
        must_differ=("q_proj.weight",),
        label="merge 后的 base 权重",
    )
    torch.testing.assert_close(
        merged.paligemma_with_expert.paligemma.q_proj.weight, expected,
        rtol=1e-5, atol=1e-6,
    )

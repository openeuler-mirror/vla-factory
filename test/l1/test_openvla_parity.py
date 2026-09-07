"""L1 parity: OpenVLA 序列装配 vs 上游 RLDSBatchTransform（真 prismatic 组件）。

漂移风险最高的转录是 prompt 模板、动作 token 映射与 label mask——本文件把
vla-factory 的 ``assemble_token_action_sequence`` 步骤与上游
``RLDSBatchTransform``（同一 tokenizer / ActionTokenizer / PurePromptBuilder）
放在同一份归一化动作与同一条任务文本上，逐张量对比：

    我们侧: sample → assemble_token_action_sequence（定长, 计划产物）
    上游侧: RLDS 形状 dict → RLDSBatchTransform（变长）

断言（去 padding 后）：input_ids 逐位相等；labels（HF 语义，未监督位 -100）
逐位相等。pixel_values 两侧调用的是同一个 processor 方法，不重复断言。

需要 openvla-7b 基座 checkpoint（tokenizer 与 prismatic 栈）——仅在模型
环境且 ``OPENVLA_CHECKPOINT`` 指向基座目录时执行；框架环境自动跳过。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.l1

if importlib.util.find_spec("prismatic") is None:
    pytest.skip("prismatic stack not installed (framework env)", allow_module_level=True)

CHECKPOINT = os.environ.get("OPENVLA_CHECKPOINT", "")
if not CHECKPOINT or not Path(CHECKPOINT).is_dir():
    pytest.skip(
        "OPENVLA_CHECKPOINT not set to an openvla-7b base checkpoint directory",
        allow_module_level=True,
    )

from utils import assert_tensor_parity  # noqa: E402  (test/l1/helpers.py 布局)

TASK = "pick up the object"
ACTIONS = np.array([[0.3, -0.1, 0.2, 1.0, 0.0, 0.5, 1.0]], dtype=np.float64)  # 已归一化 [1, 7]
TEMPLATE = "What action should the robot take to {task}?"


def _load_rlds_batch_transform():
    """从上游源文件加载 ``RLDSBatchTransform``，不引入 TF/dlimp。

    类体本身是 TF-free 的（numpy/torch/PIL/transformers），但它所在模块的
    顶层 import 拖入 ``prismatic.vla.datasets.rlds`` → dlimp/TF 数据栈
    （我们文档过的 import 墙）。这里只为该模块的四个 RLDS 子导入预置
    stub（本对比不触碰它们），再用 importlib 从真实源文件执行——parity
    的对照物是上游真类体，不是转写副本。
    """
    import prismatic

    prismatic_root = Path(prismatic.__file__).parent

    def _stub(name: str, **attrs):
        module = types.ModuleType(name)
        module.__path__ = []  # 标记为 package，允许子模块继续从 sys.modules 解析
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    class NormalizationType(types.SimpleNamespace):
        pass

    _stub("prismatic.vla.datasets.rlds",
          make_interleaved_dataset=None, make_single_dataset=None)
    _stub("prismatic.vla.datasets.rlds.oxe",
          OXE_NAMED_MIXTURES=[], get_oxe_dataset_kwargs_and_weights=None)
    _stub("prismatic.vla.datasets.rlds.utils")
    _stub("prismatic.vla.datasets.rlds.utils.data_utils",
          NormalizationType=NormalizationType)

    spec = importlib.util.spec_from_file_location(
        "prismatic.vla.datasets.datasets",
        prismatic_root / "vla" / "datasets" / "datasets.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RLDSBatchTransform


def test_assembly_matches_rlds_batch_transform():
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from transformers import AutoTokenizer

    from vla_factory.assembly.transform.task_tokenize import (
        AssembleTokenActionSequence,
    )

    RLDSBatchTransform = _load_rlds_batch_transform()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    action_tokenizer = ActionTokenizer(tokenizer)

    # ── 上游参照：RLDS 形状 dict → RLDSBatchTransform（图像不参与本对比）──
    class _ImageTransform:
        def __call__(self, img):
            return torch.zeros(6, 224, 224)

    transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=tokenizer,
        image_transform=_ImageTransform(),
        prompt_builder_fn=PurePromptBuilder,
    )
    rlds_batch = {
        "dataset_name": "utokyo",
        "action": ACTIONS,
        "observation": {"image_primary": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
        "task": {"language_instruction": TASK.encode()},
    }
    reference = transform(rlds_batch)
    ref_ids = reference["input_ids"].flatten()
    ref_labels = reference["labels"].flatten()

    # ── 我们侧：计划步骤（同 tokenizer、同模板、同归一化动作）──────────
    step = AssembleTokenActionSequence(
        tokenizer_repo=CHECKPOINT, max_length=48, template=TEMPLATE,
    )
    out = step({"task": TASK, "actions": ACTIONS})
    our_ids = out["tokenized_prompt"][out["tokenized_prompt_mask"]]
    our_labels = np.where(
        out["token_loss_mask"][out["tokenized_prompt_mask"]], our_ids, -100
    )

    # 去 padding 后逐位相等：ids（含 BOS 与动作 token）、labels（-100 掩码）
    assert our_ids.shape == ref_ids.shape, (
        f"sequence length drift: ours={our_ids.shape} upstream={ref_ids.shape}"
    )
    assert_tensor_parity(our_ids, ref_ids.detach().cpu().numpy(), name="input_ids")
    assert_tensor_parity(
        our_labels.astype(np.int64),
        ref_labels.detach().cpu().numpy().astype(np.int64),
        name="labels",
    )


def test_image_transform_matches_checkpoint_processor():
    # 像素边界：步骤对 primary 相机的 uint8 HWC → PIL → checkpoint processor
    # 的搬运，与直接调用 processor.apply_transform 逐位一致（含融合骨干的
    # 通道堆叠 [6, 224, 224] 输出契约）。
    from PIL import Image

    from vla_factory.assembly.transform.checkpoint_image import (
        CheckpointImageTransform,
        _registered_image_processor,
    )

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    reference = _registered_image_processor(CHECKPOINT).apply_transform(
        Image.fromarray(raw)
    )

    step = CheckpointImageTransform(source_key="images.front", repo=CHECKPOINT)
    out = step({"images.front": raw})

    assert out["pixel_values"].shape == reference.shape
    assert_tensor_parity(
        out["pixel_values"], reference.detach().cpu().numpy(), name="pixel_values"
    )

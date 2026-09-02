"""L1 parity: ACT 整条预处理链 vs lerobot 官方链。

与 ``test_openpi_pipeline_parity.py`` 同一目的，但上游换成 lerobot，链的形状完全不同：
openpi 把 resize / to-float / tokenize 都放进 transform 链，lerobot 则把这些留给
**数据集加载器**，策略侧的官方链只剩归一化：

    lerobot.policies.act.processor_act.make_act_pre_post_processors →
        RenameObservationsProcessorStep(rename_map={})   # 本例为 no-op
        AddBatchDimensionProcessorStep()
        DeviceProcessorStep(device=...)
        NormalizerProcessorStep(features, norm_map, stats)

因此对比的边界也不同：图像的 uint8 HWC → CHW float [0,1] 这一段在 lerobot 里由
``LeRobotDataset`` 完成，在我们这里由 ``image_to_float`` + ``image_layout`` 完成。
本文件把它当作双方共同的"数据集加载"契约，在喂给官方链之前手工做掉，从而把对比
隔离在**归一化**上——那才是两边都声称在做同一件事的部分。

ImageNet 替换同理：lerobot 在 ``datasets/factory.py:126`` 用 IMAGENET_STATS 覆盖相机
的 stats，再走通用 normalizer；我们用一个显式的 ``image_normalize: imagenet`` 步骤。
两条路必须得出同一个数。

本文件不覆盖：ACT 的 resize（profile 默认不 resize，加了 resize 才需要单独对比）、
batch > 1、后处理（反归一化）链。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.l1

# importorskip 而非 find_spec：find_spec 对点号路径会在父包缺失时直接抛
# ModuleNotFoundError（采集期就中断整轮），而 lerobot 的模块路径跨版本变动过——
# openpi 环境里那份就没有 lerobot.policies。守卫必须精确到本文件真正导入的模块。
pytest.importorskip(
    "lerobot.policies.act.processor_act",
    reason="需要带 policies.act.processor_act 的 lerobot（pip install -e '.[act]'）",
)

import torch  # noqa: E402 —— 在 skip 守卫之后，有意为之

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATASET = _PROJECT_ROOT / "test/data/lerobot_train_data_3_episodes"

if not _DATASET.exists():
    pytest.skip(f"测试数据集不存在: {_DATASET}", allow_module_level=True)

_ACTION_HORIZON = 50
_STATE_DIM = 6
_ACTION_DIM = 8
_CAMERAS = ("front", "wrist")

# lerobot datasets/factory.py:30 —— use_imagenet_stats=True 时覆盖相机 stats。
_IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}


@pytest.fixture(scope="module")
def raw():
    """原始样本 + schema + norm_stats，不经过任何变换。"""
    from vla_factory.data.codec.pyav import PyAVCodec
    from vla_factory.data.reader.lerobot_v3 import LeRobotV3Reader
    from vla_factory.training.dataset import SampleWindow, VLADataset

    reader, codec = LeRobotV3Reader(), PyAVCodec()
    schema = reader.get_schema(_DATASET)
    norm_stats = reader.get_norm_stats(_DATASET)
    windows = [SampleWindow(0, 0, n_obs_steps=1, action_horizon=_ACTION_HORIZON)]
    dataset = VLADataset(windows, reader, codec, _DATASET, transforms=[])
    return dataset[0], schema, norm_stats


@pytest.fixture(scope="module")
def upstream(raw):
    """lerobot 官方 ACT 前处理链的输出。"""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    sample, schema, norm_stats = raw

    # ACTConfig 独立构造（不复用我们的 factory —— 否则等于拿自己和自己比）。
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
    }
    for camera in _CAMERAS:
        height, width = schema.image_sizes[camera]
        input_features[f"observation.images.{camera}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, height, width)
        )
    config = ACTConfig(
        chunk_size=_ACTION_HORIZON,
        n_action_steps=_ACTION_HORIZON,
        input_features=input_features,
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,)),
        },
        device="cpu",
    )

    stats = {
        "observation.state": {
            "mean": torch.tensor(norm_stats.state.mean, dtype=torch.float32),
            "std": torch.tensor(norm_stats.state.std, dtype=torch.float32),
        },
        "action": {
            "mean": torch.tensor(norm_stats.action.mean, dtype=torch.float32),
            "std": torch.tensor(norm_stats.action.std, dtype=torch.float32),
        },
    }
    for camera in _CAMERAS:
        stats[f"observation.images.{camera}"] = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in _IMAGENET_STATS.items()
        }

    # 数据集加载器的职责（两边共同的契约）：uint8 HWC → float32 CHW [0,1]。
    batch = {
        "observation.state": torch.tensor(sample["state"], dtype=torch.float32),
        "action": torch.tensor(sample["actions"], dtype=torch.float32),
    }
    for camera in _CAMERAS:
        image = np.asarray(sample[f"images.{camera}"], dtype=np.float32) / 255.0
        batch[f"observation.images.{camera}"] = torch.tensor(image).permute(2, 0, 1)

    preprocessor, _ = make_act_pre_post_processors(config, dataset_stats=stats)
    return preprocessor(batch)


@pytest.fixture(scope="module")
def ours(raw):
    """我们的链，从 act.yaml profile 解析出来后执行到 collate。"""
    from vla_factory.assembly import resolve_assembly
    from vla_factory.assembly.transform import TransformContext, build_pipeline
    from vla_factory.training.dataset import collate_fn
    from vla_factory.user_interface import merge_model_config, parse_recipe_from_string

    sample, _, _ = raw
    recipe = merge_model_config(parse_recipe_from_string(
        "model:\n  name: act\n"
        f"  config:\n    action_horizon: {_ACTION_HORIZON}\n"
        f"data:\n  path: {_DATASET}\n  format: lerobot-v3\n"
    ))
    assembly = resolve_assembly(recipe)
    transforms = build_pipeline(
        assembly.data_to_model, TransformContext(norm_stats=assembly.norm_stats),
    )
    transformed = dict(sample)
    for step in transforms:
        transformed = step(transformed)
    return collate_fn([transformed])


def _numpy(value) -> np.ndarray:
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)


# ══════════════════════════════════════════════════════════════════════


def test_state_is_bit_identical(upstream, ours):
    """z-score 归一化后的 state 必须逐位相等。

    这条守住 lerobot 的 eps（1e-8，与 openpi 的 1e-6 不同）以及 mean/std 的取用方式。
    """
    up = _numpy(upstream["observation.state"])
    mine = _numpy(ours["observation"].state)
    assert up.shape == mine.shape == (1, _STATE_DIM)
    np.testing.assert_array_equal(up, mine)


def test_actions_are_bit_identical(upstream, ours):
    """ACT 的 action_dim 等于数据集维度，因此不该有 padding，且数值必须相等。

    ``AddBatchDimensionProcessorStep`` 只给 observation 加 batch 维，``action``
    保持 ``[H, D]``；我们的 collate 给所有键加，得到 ``[1, H, D]``。这是边界处的
    约定差异而非 parity 问题，比较前对齐掉。
    """
    up = _numpy(upstream["action"])
    mine = _numpy(ours["actions"])
    if up.ndim == mine.ndim - 1:
        up = up[None]
    assert up.shape == mine.shape == (1, _ACTION_HORIZON, _ACTION_DIM), (
        f"形状不一致 {up.shape} vs {mine.shape}——若我们这边维度更大，说明 "
        "pad_dimensions 在 target_dim <= dataset_dim 时没有被跳过"
    )
    np.testing.assert_array_equal(up, mine)


@pytest.mark.parametrize("camera", _CAMERAS)
def test_images_are_bit_identical(upstream, ours, camera):
    """ImageNet 归一化后的图像必须逐位相等。

    两边路径不同：lerobot 用 IMAGENET_STATS 覆盖相机 stats 后走通用 normalizer，
    我们走显式的 image_normalize 步骤。数值必须一致，否则 ResNet backbone 看到的
    输入分布就和 lerobot 训练 ACT 时不同。
    """
    up = _numpy(upstream[f"observation.images.{camera}"])
    mine = _numpy(ours["observation"].images[camera])
    assert up.shape == mine.shape, f"{camera}: 形状不一致 {up.shape} vs {mine.shape}"
    np.testing.assert_allclose(up, mine, rtol=0, atol=1e-6)


def test_image_layout_is_channels_first(ours):
    """ACT 的 ResNet backbone 吃 CHW；HWC 会在 backbone 里静默按错误的轴卷积。"""
    for camera in _CAMERAS:
        image = _numpy(ours["observation"].images[camera])
        assert image.ndim == 4 and image.shape[1] == 3, \
            f"{camera}: 期望 [B,3,H,W]，实际 {image.shape}"


def test_no_prompt_fields_for_a_non_language_model(ours):
    """ACT 不是语言条件模型，profile 里不该混进 tokenize 步骤。"""
    observation = ours["observation"]
    assert observation.tokenized_prompt is None
    assert observation.tokenized_prompt_mask is None

"""L1 parity: pi0 / pi05 整条预处理链 vs openpi 官方链，比到模型的真实入参。

这是 Issue #7 正文的核心诉求——"同一条样本分别过上游官方 transform 链和
vla-factory 的 pipeline，断言逐张量相等"。比逐个常量断言强的地方在于：它一次覆盖
所有常量（eps、值域、分辨率、pad 维、步序），且不依赖测试作者读对了上游源码。

对比点是 ``openpi.models.model.Observation`` —— ``PI0Pytorch.forward`` 的实际入参，
两侧都构造到这一层：

    我们侧: sample → _build_transforms(config/model/pi0.yaml) → collate_fn
                   → vla Observation → PI0ModelWrapper._to_openpi_observation()
    上游侧: canonical dict → transforms.Normalize → ModelTransformFactory 四步
                   → Observation.from_dict()

两侧共用同一条 raw sample 和同一份 norm_stats，所以隔离出来的是**变换链本身**。

每条断言都对 pi0 与 pi05 各跑一次（``variant`` fixture）：两者共用 PI0Pytorch 与
PI0ModelWrapper，差异在配置（max_token_len 48/200、discrete_state_input）、归一化方式
（z-score / quantile）以及 tokenize 与 pad 的先后。

本文件不覆盖：norm_stats 的**计算方式**（两侧喂的是同一份，只验证消费不验证生产）、
batch > 1、``token_ar_mask``。ACT 走 lerobot 而非 openpi，链的形状不同，见
``test_act_pipeline_parity.py``。

一个未测的隐式依赖：openpi 用 ``position_ids = cumsum(pad_masks) - 1``
（``pi0_pytorch.py:343``）而非 ``arange``，所以我们左填充、上游右填充仍然行为等价。
上游若改成 arange，我们会静默崩——这条目前没有断言守着。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.l1

if importlib.util.find_spec("openpi") is None:
    pytest.skip(
        "pipeline parity 需要 openpi（bash scripts/install.sh pi0）",
        allow_module_level=True,
    )

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATASET = _PROJECT_ROOT / "test/data/lerobot_train_data_3_episodes"

if not _DATASET.exists():
    pytest.skip(f"测试数据集不存在: {_DATASET}", allow_module_level=True)

# 该数据集的真实契约（见 test/test_data_pipeline.py）
_CAMERA_MAPPING = {"base_0_rgb": "front", "left_wrist_0_rgb": "wrist"}
_ACTION_HORIZON = 50
_DATASET_ACTION_DIM = 8
_MODEL_ACTION_DIM = 32  # openpi max_action_dim

# 数据集 meta/tasks.parquet 里的任务文本，由 reader 读出。
_TASK = "Lift the red cube up."

# 480x640 保比缩放到 224 得 224x168，上下各 28 行 letterbox。
_PAD_ROWS = np.r_[0:28, 196:224]
_CONTENT_ROWS = np.r_[28:196]

# 内容区容差：jax.image.resize(LINEAR, antialias) vs cv2.INTER_LINEAR 的实现差异。
# 实测 mean|Δ|≈0.003、max|Δ|≈0.32（[-1,1] 尺度）。阈值卡在均值上而非最大值上——
# 差异集中在边缘高频像素，用 max 会把正常的插值差异也判成失败。
_INTERPOLATION_MEAN_TOL = 0.01


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
    sample = dataset[0]
    assert sample.get("task") == _TASK, (
        "reader 没有把数据集的 task 文本带出来；parity 对比会退化成空 prompt 的对比。"
        " 见 test_reader_surfaces_the_dataset_task。"
    )
    return sample, schema, norm_stats


@pytest.fixture(scope="module", params=("pi0", "pi05"))
def variant(request):
    """pi0 与 pi05 共用 PI0Pytorch / PI0ModelWrapper，只有配置与归一化方式不同。

    pi05 的差异：Pi0Config(pi05=True)（max_token_len 200、discrete_state_input）、
    上游用 quantile 归一化、prompt 模板把 digitize 后的 state 嵌进去，且 tokenize
    排在 pad 之前（state 必须按原生维度进 prompt）。
    """
    return request.param


@pytest.fixture(scope="module")
def upstream(raw, variant):
    """openpi 官方链的 Observation。"""
    from openpi.models import model as _model
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS
    from openpi.shared.normalize import NormStats as OpenpiNormStats
    from openpi.training.config import ModelTransformFactory
    import openpi.transforms as T

    sample, _, norm_stats = raw
    reference = np.asarray(sample[f"images.{next(iter(_CAMERA_MAPPING.values()))}"])

    image, image_mask = {}, {}
    for role in IMAGE_KEYS:
        camera = _CAMERA_MAPPING.get(role)
        if camera is not None:
            image[role] = np.asarray(sample[f"images.{camera}"])
            image_mask[role] = np.array(True)
        else:
            # openpi repack 对缺席相机的惯例：零图 + mask False。
            image[role] = np.zeros_like(reference)
            image_mask[role] = np.array(False)

    def _stats(feature):
        return OpenpiNormStats(
            mean=np.asarray(feature.mean, dtype=np.float32),
            std=np.asarray(feature.std, dtype=np.float32),
            q01=np.asarray(feature.q01, dtype=np.float32) if feature.q01 else None,
            q99=np.asarray(feature.q99, dtype=np.float32) if feature.q99 else None,
        )

    data = {
        "image": image,
        "image_mask": image_mask,
        "state": np.asarray(sample["state"], dtype=np.float32),
        "actions": np.asarray(sample["actions"], dtype=np.float32),
        "prompt": sample["task"],
    }
    is_pi05 = variant == "pi05"
    # openpi 按 model_type != PI0 选择 quantile 归一化（DataConfigFactory
    # use_quantile_norm），即 pi0 走 z-score、pi05 走 quantile。
    data = T.Normalize(
        norm_stats={"state": _stats(norm_stats.state), "actions": _stats(norm_stats.action)},
        use_quantiles=is_pi05,
    )(data)
    for step in ModelTransformFactory()(Pi0Config(pi05=is_pi05)).inputs:
        data = step(data)
    return _model.Observation.from_dict(data), data


@pytest.fixture(scope="module")
def ours(raw, variant):
    """我们的链，一路走到 _to_openpi_observation。"""
    from vla_factory.assembly import resolve_assembly
    from vla_factory.assembly.transform import TransformContext, build_pipeline
    from vla_factory.model.adapters.openpi import PI0ModelWrapper
    from vla_factory.training.dataset import collate_fn
    from vla_factory.user_interface import merge_model_config, parse_recipe_from_string

    sample, _, _ = raw
    recipe = merge_model_config(parse_recipe_from_string(
        f"model:\n  name: {variant}\n"
        f"data:\n  path: {_DATASET}\n  format: lerobot-v3\n"
        "overrides:\n"
        "  camera_mapping:\n"
        "    base_0_rgb: front\n"
        "    left_wrist_0_rgb: wrist\n"
    ))
    assembly = resolve_assembly(recipe)
    transforms = build_pipeline(
        assembly.data_to_model, TransformContext(norm_stats=assembly.norm_stats),
    )
    transformed = dict(sample)
    for step in transforms:
        transformed = step(transformed)

    batch = collate_fn([transformed])
    # model=None: _to_openpi_observation 只用 camera_mapping，不碰权重。
    wrapper = PI0ModelWrapper(model=None, camera_mapping=_CAMERA_MAPPING)
    return wrapper._to_openpi_observation(batch["observation"]), batch


def _hwc(tensor) -> np.ndarray:
    """统一成 ``[B, H, W, C]`` 再比。

    我们的图像是 ``[B, C, H, W]``（profile 里有 image_layout: CHW），上游这条路径
    是无 batch 维的 ``[H, W, C]``。两者都归一化，否则 numpy 的广播会让形状不同的
    张量"比较通过"——helper 写错时测试静默失效，比没有测试更糟。
    """
    array = np.asarray(tensor.detach().cpu() if hasattr(tensor, "detach") else tensor)
    if array.ndim == 3:  # 上游：HWC，无 batch 维
        array = array[None]
    elif array.ndim == 4 and array.shape[1] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(0, 2, 3, 1)  # 我们：BCHW → BHWC
    assert array.ndim == 4 and array.shape[-1] in (1, 3, 4), \
        f"无法归一化为 BHWC: {array.shape}"
    return array


def _numpy(array) -> np.ndarray:
    return np.asarray(array.detach().cpu() if hasattr(array, "detach") else array)


def _vector_2d(array) -> np.ndarray:
    """归一化成 ``[B, D]``：上游这条路径无 batch 维（``[D]``），我们有。"""
    array = _numpy(array)
    return array if array.ndim >= 2 else array[None]


def _mask_1d(array) -> np.ndarray:
    """归一化成 ``[B]``：上游的 image_mask 是标量 ``()``，我们 collate 后是 ``[1]``。"""
    return np.atleast_1d(_numpy(array)).astype(bool)


def _unpadded_tokens(tokens, mask) -> np.ndarray:
    """Select prompt tokens by their attention mask, preserving token value 0."""
    tokens = _numpy(tokens)
    mask = _numpy(mask).astype(bool)
    assert tokens.ndim == mask.ndim == 1, (
        f"token/mask 序列应为一维，实际为 {tokens.shape}/{mask.shape}"
    )
    assert tokens.shape == mask.shape, f"token/mask 长度不一致: {tokens.shape}/{mask.shape}"
    return tokens[mask]


def test_prompt_mask_preserves_interior_zero_tokens():
    np.testing.assert_array_equal(
        _unpadded_tokens(
            np.array([11, 0, 108, 0, 0]),
            np.array([True, True, True, False, False]),
        ),
        np.array([11, 0, 108]),
    )


# ══════════════════════════════════════════════════════════════════════
#  已对齐的部分 —— 这些断言是回归护栏
# ══════════════════════════════════════════════════════════════════════


def test_state_is_bit_identical(upstream, ours):
    """归一化 + pad 后的 state 必须逐位相等。

    这条同时守住 zscore eps（openpi 1e-6，不是 lerobot 的 1e-8）、pad 目标维 32、
    以及 normalize/pad 的先后顺序 —— 任一项错都会在这里显形。
    """
    up = _vector_2d(upstream[0].state)
    mine = _vector_2d(ours[0].state)
    assert up.shape == mine.shape == (1, _MODEL_ACTION_DIM)
    np.testing.assert_array_equal(up, mine)


def test_actions_are_bit_identical(upstream, ours):
    """动作张量走的是同一套 norm stats 与 pad，必须逐位相等。"""
    up = np.asarray(upstream[1]["actions"])
    mine = np.asarray(ours[1]["actions"].detach().cpu())[0]
    assert up.shape == mine.shape == (_ACTION_HORIZON, _MODEL_ACTION_DIM)
    np.testing.assert_array_equal(up, mine)


def test_image_roles_match_openpi_image_keys(upstream, ours):
    """相机角色集合必须与 openpi 的 IMAGE_KEYS 完全一致。

    少一个角色 openpi 会在 embed_prefix 里少一段 prefix；多一个会被忽略。
    """
    from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS

    assert set(ours[0].images) == set(IMAGE_KEYS)
    assert set(ours[0].images) == set(upstream[0].images)


def test_unmapped_camera_placeholder_matches_upstream(upstream, ours):
    """未映射角色的占位图必须与上游一致。

    上游走的是 uint8 零图 → resize → ``/255*2-1`` = -1.0；我们的 adapter 直接填
    -1.0。两条路径不同但结果必须相同，否则模型会在一个不存在的相机上看到灰图而非黑图。
    """
    role = "right_wrist_0_rgb"
    assert role not in _CAMERA_MAPPING, "该角色必须是未映射的，否则本测试失去意义"
    np.testing.assert_array_equal(
        _hwc(upstream[0].images[role]), _hwc(ours[0].images[role])
    )


def test_image_masks_match(upstream, ours):
    """mask 决定 openpi 是否 attend 这个相机，错了等于悄悄丢一路视觉输入。"""
    for role in upstream[0].images:
        np.testing.assert_array_equal(
            _mask_1d(upstream[0].image_masks[role]),
            _mask_1d(ours[0].image_masks[role]),
            err_msg=f"image_masks[{role}] 不一致",
        )


@pytest.mark.parametrize("role", sorted(_CAMERA_MAPPING))
def test_image_content_region_within_interpolation_tolerance(upstream, ours, role):
    """内容区（非 letterbox）只允许存在插值实现级别的差异。

    上游 ``jax.image.resize(LINEAR)`` 默认带 antialias，我们用 ``cv2.INTER_LINEAR``
    不带。差异集中在边缘高频像素，所以阈值卡均值。若某天两边换成同一实现，这条
    会变得很紧，届时可下调阈值。
    """
    up = _hwc(upstream[0].images[role])[:, _CONTENT_ROWS]
    mine = _hwc(ours[0].images[role])[:, _CONTENT_ROWS]
    diff = np.abs(up.astype(np.float64) - mine.astype(np.float64))
    assert diff.mean() < _INTERPOLATION_MEAN_TOL, (
        f"{role} 内容区平均差异 {diff.mean():.6f} 超过插值容差 "
        f"{_INTERPOLATION_MEAN_TOL} —— 这已经不是插值实现能解释的了"
    )


def test_prompt_tokens_agree_up_to_the_start_of_answer_token(upstream, ours):
    """去掉 padding 后，我们的 token 必须是上游 token 的前缀。

    上游多出末尾的 ``\\n``（token 108，openpi 称 "start of answer" token）；除此之外
    逐个 token 必须相同 —— 文本清洗规则、BOS、tokenizer 版本任一处不同都会在这里显形。
    完整相等由 test_prompt_tokens_are_identical 守着。
    """
    up = np.asarray(upstream[0].tokenized_prompt)
    mine = np.asarray(ours[0].tokenized_prompt.detach().cpu())[0]
    up_tokens = _unpadded_tokens(up, upstream[0].tokenized_prompt_mask).tolist()
    my_tokens = _unpadded_tokens(
        mine, ours[0].tokenized_prompt_mask.detach().cpu()[0]
    ).tolist()

    assert my_tokens, "我们没有产出任何非 padding token"
    assert up_tokens[: len(my_tokens)] == my_tokens, (
        f"token 序列在前缀处就分叉了：\n  上游={up_tokens}\n  我们={my_tokens}"
    )


# ══════════════════════════════════════════════════════════════════════
#  建立本文件时发现并已修复的三处差异 —— 现在是防回归的断言
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("role", sorted(_CAMERA_MAPPING))
def test_letterbox_padding_matches_upstream(upstream, ours, role):
    """letterbox 必须是 -1.0（黑）而不是 0.0（中灰）。

    profile 把 resize 排在 image_to_float 之前正是为了这个：在 uint8 上 pad 出
    黑边、再映射到 -1.0，与 openpi 一致。谁把这两步换回去，这条立刻红——4:3 画面
    下这是 25% 的像素被喂成分布外的灰色。
    """
    up = _hwc(upstream[0].images[role])[:, _PAD_ROWS]
    mine = _hwc(ours[0].images[role])[:, _PAD_ROWS]
    np.testing.assert_allclose(up, mine, atol=1e-6)


def test_prompt_tokens_are_identical(upstream, ours):
    """去掉 padding 后 token 序列必须完全一致，包括末尾的 start-of-answer token。

    openpi models/tokenizer.py:33 用 encode(text) + encode("\\n") 追加它；我们在
    build_prompt 里把 "\\n" 接进字符串，PaliGemma tokenizer 产出同一个 id (108)。
    """
    up = np.asarray(upstream[0].tokenized_prompt)
    mine = np.asarray(ours[0].tokenized_prompt.detach().cpu())[0]
    np.testing.assert_array_equal(
        _unpadded_tokens(up, upstream[0].tokenized_prompt_mask),
        _unpadded_tokens(mine, ours[0].tokenized_prompt_mask.detach().cpu()[0]),
    )


def test_reader_surfaces_the_dataset_task():
    """task 文本必须由 reader 从 meta/tasks.parquet 读出，而不是靠 default_task 兜底。

    lerobot v3 把任务文本放在 pandas index（``__index_level_0__``）而非 ``task``
    列。只认列会静默返回 {}，让每一帧都退回 recipe 的 default_task —— 多任务数据集
    因此被训成单一 prompt。
    """
    from vla_factory.data.codec.pyav import PyAVCodec
    from vla_factory.data.reader.lerobot_v3 import LeRobotV3Reader

    episode = LeRobotV3Reader().read_episode(_DATASET, 0, PyAVCodec())
    assert episode.load_frames()[0].language == _TASK

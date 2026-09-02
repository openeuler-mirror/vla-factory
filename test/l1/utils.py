"""Parity 断言工具（Issue #7 计划 v4 的 T-L1.1）。

失败信息是这类测试的全部价值所在：一条 "arrays are not equal" 只能说明"错了"，
定位还得靠人重跑一遍。这里的断言把"错在哪一步、哪个元素、差多少"直接打出来。

三个层次：

* :func:`assert_tensor_parity` —— 单个张量，报首个不匹配元素的位置与双方取值。
* :func:`assert_state_dict_parity` —— 整个 state_dict，key 集合差异与逐 key 数值分开报，
  并支持 ``must_differ``：断言某些 key **必须不同**（用来识破 no-op —— 一个没生效的
  操作和一个成功的操作，产物看起来一模一样）。
* :func:`summarize_lora_parameters` —— 把 peft 包裹后的模型拆成 adapter / base 两组，
  连同 dtype 与 requires_grad 一起返回，供 dtype 不变量断言使用。

``must_differ`` 与 LoRA 摘要的形态参考了 LLaMA-Factory 的
``train/test_utils.py``（``compare_model`` / ``check_lora_model``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    """torch 张量 / numpy 数组 → numpy（bfloat16 先提到 float32，numpy 不支持它）。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def _dtype_name(value: Any) -> str:
    return str(getattr(value, "dtype", type(value).__name__))


def assert_tensor_parity(
    actual: Any,
    reference: Any,
    *,
    name: str,
    rtol: float = 0.0,
    atol: float = 0.0,
    check_dtype: bool = False,
) -> None:
    """断言两个张量一致，失败时给出可直接定位的报告。

    默认 ``rtol=atol=0``（逐位相等）——parity 测试的默认期望就是完全一致，需要容差
    的地方应当显式写出来并注明理由，而不是让容差悄悄兜住真实差异。

    ``check_dtype`` 默认关闭：跨框架比较时 dtype 往往合法地不同（jax float32 vs
    torch bfloat16），需要时再显式打开。
    """
    left, right = _to_numpy(actual), _to_numpy(reference)

    if check_dtype and _dtype_name(actual) != _dtype_name(reference):
        raise AssertionError(
            f"{name}: dtype 不一致 —— actual={_dtype_name(actual)} "
            f"reference={_dtype_name(reference)}"
        )

    if left.shape != right.shape:
        raise AssertionError(
            f"{name}: 形状不一致 —— actual={left.shape} reference={right.shape}"
        )

    if left.dtype == bool or right.dtype == bool:
        if not np.array_equal(left.astype(bool), right.astype(bool)):
            mismatched = int((left.astype(bool) != right.astype(bool)).sum())
            raise AssertionError(
                f"{name}: 布尔张量不一致 —— {mismatched}/{left.size} 个元素不同"
            )
        return

    left_f = left.astype(np.float64)
    right_f = right.astype(np.float64)
    close = np.isclose(left_f, right_f, rtol=rtol, atol=atol, equal_nan=True)
    if close.all():
        return

    bad = np.argwhere(~close)
    first = tuple(int(i) for i in bad[0])
    diff = np.abs(left_f - right_f)
    raise AssertionError(
        f"{name}: {len(bad)}/{left.size} 个元素超出容差 "
        f"(rtol={rtol}, atol={atol})\n"
        f"  首个不匹配位置 {first}: actual={left_f[first]!r} reference={right_f[first]!r}\n"
        f"  max|Δ|={diff.max():.6g}  mean|Δ|={diff.mean():.6g}\n"
        f"  actual  shape={left.shape} dtype={left.dtype} "
        f"range=[{left_f.min():.6g}, {left_f.max():.6g}]\n"
        f"  reference shape={right.shape} dtype={right.dtype} "
        f"range=[{right_f.min():.6g}, {right_f.max():.6g}]"
    )


def assert_state_dict_parity(
    actual: dict[str, Any],
    reference: dict[str, Any],
    *,
    must_differ: Iterable[str] = (),
    rtol: float = 0.0,
    atol: float = 0.0,
    label: str = "state_dict",
) -> None:
    """断言两个 state_dict 的 key 集合与数值一致。

    ``must_differ`` 里的**子串**匹配到的 key 反过来断言"必须不同"。这条是识破 no-op
    的手段：一个没真正执行的操作（比如 merge 掉了个寂寞）产出的 state_dict，和成功
    执行的那个长得一模一样，只有反向断言能区分。
    """
    actual_keys, reference_keys = set(actual), set(reference)
    if actual_keys != reference_keys:
        only_actual = sorted(actual_keys - reference_keys)
        only_reference = sorted(reference_keys - actual_keys)
        raise AssertionError(
            f"{label}: key 集合不一致\n"
            f"  仅 actual 有 ({len(only_actual)}): {only_actual[:8]}\n"
            f"  仅 reference 有 ({len(only_reference)}): {only_reference[:8]}"
        )

    must_differ = tuple(must_differ)
    for key in sorted(actual_keys):
        expected_to_differ = any(pattern in key for pattern in must_differ)
        if expected_to_differ:
            left, right = _to_numpy(actual[key]), _to_numpy(reference[key])
            if left.shape == right.shape and np.allclose(
                left.astype(np.float64), right.astype(np.float64),
                rtol=max(rtol, 1e-6), atol=max(atol, 1e-8),
            ):
                raise AssertionError(
                    f"{label}[{key}]: 期望不同但完全相同 —— 该操作没有生效"
                )
            continue
        assert_tensor_parity(
            actual[key], reference[key], name=f"{label}[{key}]", rtol=rtol, atol=atol
        )


@dataclass
class LoraParameterSummary:
    """peft 包裹后模型的参数分组视图。"""

    #: 挂上 adapter 的叶子 module 名（如 ``{"q_proj", "o_proj"}``）
    adapter_modules: set[str] = field(default_factory=set)
    #: adapter 参数名 → (dtype 字符串, requires_grad)
    adapter_params: dict[str, tuple[str, bool]] = field(default_factory=dict)
    #: 其余参数名 → (dtype 字符串, requires_grad)
    base_params: dict[str, tuple[str, bool]] = field(default_factory=dict)

    @property
    def adapter_dtypes(self) -> set[str]:
        return {dtype for dtype, _ in self.adapter_params.values()}

    @property
    def base_dtypes(self) -> set[str]:
        return {dtype for dtype, _ in self.base_params.values()}

    @property
    def trainable(self) -> set[str]:
        return {
            name
            for name, (_, grad) in {**self.adapter_params, **self.base_params}.items()
            if grad
        }


def summarize_lora_parameters(model) -> LoraParameterSummary:
    """按 adapter / base 拆分参数，并记录 dtype 与 requires_grad。

    ``adapter_modules`` 取 ``lora_A`` / ``lora_B`` 前面那一段的最后一节，也就是被挂
    adapter 的那个叶子 module 名 —— 这正是"adapter 落在哪些层上"这个问题的答案。
    """
    summary = LoraParameterSummary()
    for name, param in model.named_parameters():
        info = (str(param.dtype), bool(param.requires_grad))
        if "lora_A" in name or "lora_B" in name:
            leaf = name.split(".lora_", maxsplit=1)[0].split(".")[-1]
            summary.adapter_modules.add(leaf)
            summary.adapter_params[name] = info
        else:
            summary.base_params[name] = info
    return summary

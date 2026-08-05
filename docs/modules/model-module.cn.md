# 模型抽象（model）模块设计

> 文档状态：**TODO** —— 本文档待补充。完成后对齐**当前已实现**的行为（架构文档描述目标架构，可能超前于实现），供读者参照学习。
> 对应架构：见 [总体架构 § 4.1.2 VLA 模型](../architecture/vla-factory-architecture.cn.md) 与 [§ 2.2 目录结构 `model/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

模型抽象模块隔离上游模型实现差异，让训练和推理只依赖最小协议。它持有 `VLAModel` 接口、`Observation` 规范类型、`ModelMetadata` / `BaseContract`，以及模型注册表与上游模型的薄 adapter。模型维度的描述（接口能力、默认预处理、相机槽位、推理步数等）随模型自身声明 YAML 发布，不在 recipe 里修改。

## 1. 核心对象

- `VLAModel` 协议（`model/interfaces/`）：`compute_loss(observation, actions, ...)` 与 `predict_actions(observation, ...)`；PyTorch 模型另实现 `parameters()` / `named_parameters()` / `train()` / `to()`。
- `Observation`：跨维度共享的规范样本类型（归属本模块，被 composition / training / inference 共同消费）。
- `ModelMetadata`：模型族静态能力描述（视觉槽位、维度策略、动作 horizon、normalization、可训练组件、微调方式、install_hint 等）。
- `BaseContract`：具体 checkpoint / 模型实例自述的输入槽位、维度、时序事实。
- 注册表：`@register_vla` 装饰器，`get_entry(name)` lazy import `model/registry/entries/*`。
- Thin Adapter：把 `Observation` 与上游模型 batch 互转，不复制上游模型代码。

## 2. 详细设计

TODO，后续补充：

- `VLAModel` 协议的完整方法签名与 backend 扩展（PyTorch / Jax）。
- `ModelMetadata` 字段全集与校验；`BaseContract` 合并规则（实例事实优先、不超 ModelMetadata 边界）。
- registry 的 lazy import、导入失败显式报错、optional dependency 延迟导入边界。
- checkpoint 加载策略（wrapper 与上游 key prefix 差异）。
- 新增模型 entry 的脚手架与 contract test。

## 3. 扩展方式

TODO：新增上游模型 adapter 的标准步骤；不得根据数据集名/机器人名做隐藏分支。

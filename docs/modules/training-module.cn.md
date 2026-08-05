# 微调层（training）模块设计

> 文档状态：**TODO** —— 本文档待补充。完成后对齐**当前已实现**的行为（架构文档描述目标架构，可能超前于实现），供读者参照学习。
> 对应架构：见 [总体架构 § 4.3 微调层](../architecture/vla-factory-architecture.cn.md#4-核心模块设计) 与 [§ 2.2 目录结构 `training/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

微调层（由 `training/` 模块实现）负责训练编排：把数据层产出的 Canonical IR（`Episode` / `Frame`）按具身组合中解析得到的 `data_to_model` TransformPipeline 组装成 `Observation` 样本，再做窗口采样与批处理，交给 `VLATrainer`。它消费具身组合，不重新推导三者关系。

## 1. 核心对象

- 入口 `train()`（`vla_factory/training/train.py`）。
- `VLATrainer`：HuggingFace `Trainer` 的薄子类，把 batch 桥接到 `model.compute_loss`，补充 VLA batch 适配、辅助 loss logging、`lr_backbone` 参数组。
- 微调策略（`training/strategies/`）：`full` / `lora` / `freeze` / `selective`，基于 `ModelMetadata.components` 与 `named_parameters()` 选择参数。
- Checkpoint 与 Final Model：`inference_metadata/`（recipe / schema / norm_stats）+ `<output_dir>/final/model.pt`。

## 2. 详细设计

TODO，后续补充：

- `train()` 的完整流程（parse recipe → prepare output_dir → resolve_composition → build dataloaders → VLATrainer.train → save final）。
- `Observation` 样本构建与 `collate_fn` 的细节。
- 各微调策略的参数选择规则与 LoRA（`target_components` → peft `target_modules`）映射。
- TrainingArguments 构建与混合精度 / 梯度累积 / 梯度检查点。
- checkpoint 写出布局与推理加载优先级（final → 根目录 → safetensors → checkpoint-*）。

## 3. 扩展方式

TODO：新增微调策略、自定义 collate 的约定；不得根据模型名硬编码参数选择。

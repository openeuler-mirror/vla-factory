# 用户表达层（recipe）模块设计

> 文档状态：**TODO** —— 本文档待补充。完成后对齐**当前已实现**的行为（架构文档描述目标架构，可能超前于实现），供读者参照学习。
> 对应架构：见 [总体架构 § 3 用户表达层](../architecture/vla-factory-architecture.cn.md#3-用户表达层) 与 [§ 2.2 目录结构 `recipe/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

用户表达层是 VLA Factory 的入口。它把用户可读的 YAML recipe 解析成训练和推理都能消费的结构化对象（`TrainRecipe` 及子 dataclass），并提供 CLI / API 入口。recipe 只承载用户的组合选择（数据/模型/机器人）、组合调整（`assembly`）与训练参数，不承载数据/模型/机器人三者之间的关系。

## 1. 核心对象

- `TrainRecipe` 及子 dataclass：`DataConfig` / `SamplerConfig` / `SplitConfig` / `LoraConfig` / `OutputConfig` / `AugmentationConfig` 等（见 `vla_factory/recipe/recipe.py`）。
- `parser.py`：YAML → `TrainRecipe`。
- `vlafactory-cli`：统一命令行入口（train / list / resolve / inspect ...）。

## 2. 详细设计

TODO，后续补充：

- recipe 解析与校验规则（含三个区的优先级、严格解析）。
- CLI 子命令与输出契约。
- 配置来源与合并策略（CLI > recipe > 框架默认值）。
- 训练产物中 `recipe.yaml` 的写出与部署侧复用。

## 3. 扩展方式

TODO：新增 recipe 字段、新增 CLI 子命令的约定。

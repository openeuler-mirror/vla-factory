# 组合解析层（assembly）模块设计

> 文档状态：**TODO** —— 本文档待补充。完成后对齐**当前已实现**的行为（架构文档描述目标架构，可能超前于实现），供读者参照学习。
> 对应架构：见 [总体架构 § 4.2 组合解析层](../architecture/vla-factory-architecture.cn.md#4-核心模块设计) 与 [§ 2.2 目录结构 `assembly/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

组合解析层把「数据集 × VLA 模型 × 机器人」三者解析为统一的**具身组合**（`ResolvedAssembly`），是微调层和推理层共同的上游。它是确定性的纯逻辑层：不创建模型、不构造 DataLoader、不启动训练、不加载部署平台、不依赖 GPU。本层是目标架构的演进方向，不代表所有能力均已实现。

本层还持有 `transforms/`（`TransformStep` / `TransformPipeline` / `TransformRegistry` 及各 step 实现），因为组合解析器规划 pipeline、具身组合产出三条 TransformPipelineSpec。

## 1. 核心对象

- `Resolver`（公共入口 `resolve_assembly()`）与解析阶段：Load → Materialize → Validate → Check Pairs → Build Interface → Resolve Mapping → Plan Pipeline → Emit。
- `ResolvedAssembly`：唯一成功产物，包含三者描述引用、canonical interface、五类 Mapping（Camera / State / Action / Language / Joint）、三条 TransformPipelineSpec（`data_to_model` / `robot_to_model` / `model_to_robot`）。
- `ResolutionError`：结构化失败（稳定 `code` / `path` / `params`）。
- 兼容性检查与转换等级 T1（确定性）/ T2（依赖条件）/ T3（直接失败）。
- `TransformStepSpec` / `TransformPipelineSpec` / `TransformRegistry` / `TransformStep` / `TransformPipeline`。

## 2. 详细设计

TODO，后续补充：

- 解析阶段的具体算法与不变量。
- 兼容性检查矩阵（状态/动作维度、相机槽位、语言、控制模式、夹爪、旋转、归一化、频率、关节顺序、安全范围）。
- Mapping 规则（唯一性、确定性、不依赖字典顺序）。
- TransformPipelineSpec 的声明式 schema 与 TransformRegistry 能力 metadata（风险、可逆性、inverse）。
- 三条 pipeline 的语义边界与正向/逆向不能靠列表反转。
- ResolutionError 的稳定错误码目录与 CLI 渲染。

## 3. 扩展方式

TODO：新增解析规则 / TransformStep 的约束（稳定 ID、显式适用/成功/失败条件、禁止名称硬编码、不含训练或部署运行时配置）。

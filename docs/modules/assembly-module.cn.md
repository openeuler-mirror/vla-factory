# 组合解析层（assembly）模块设计

> 文档状态：**已对齐当前实现**（2026-08-13）。
> 对应架构：见 [总体架构 § 4.2 组合解析层](../architecture/vla-factory-architecture.cn.md#4-核心模块设计) 与 [§ 2.2 目录结构 `assembly/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

组合解析层把 DataSchema、ModelMetadata 和可选 RobotProfile 解析为统一的 `ResolvedAssembly`，是训练和推理的共同上游。它是确定性的纯逻辑层：不创建模型、不构造 DataLoader、不启动训练、不加载部署平台、不依赖 GPU。

本层还持有 `transform/`（`TransformStep` / `TransformPipeline` / `TransformRegistry` 及各 step 实现），因为组合解析器规划 pipeline、具身组合产出三条 TransformPipelinePlan。

## 1. 核心对象

- `assembly/resolve_assembly.py`：阅读入口，集中放置带 I/O 的 `resolve_assembly(recipe)` 编排、解析结果类型及 `save()` / `load()` 持久化；`assembly/resolve/` 只放纯 `resolve_from_facts(...)` 与各项规则。
- `ResolvedAssembly`：唯一成功产物，包含描述快照、`ModelIOSpec`、统一的 `FieldMapping`（按用途使用 Camera / State / Action / Language 语义别名）和三个语义 Pipeline 入口。它自身提供 `save()` / `load()` / `check_model_compatibility()`；不再另设 artifact service，也没有版本信封。
- `ResolutionError`：结构化失败（稳定 `code` / `path` / `params`）。
- 兼容性检查与转换等级 T1（确定性）/ T2（依赖条件）/ T3（直接失败）。
- `TransformStepCall` / `TransformPipelinePlan` / `TransformRegistry` / `TransformStep` / `TransformPipeline`。
- `MappingSource`：Mapping 关系只有 `INFERRED` 与 `OVERRIDE` 两种来源。相机占位由 `data_source=None` 表达，不伪装成第三种来源；语言兜底由 `task_tokenize` call 表达，不重复写入 LanguageMapping。

## 2. 详细设计

解析顺序是 Require Inputs → Validate Descriptions → Check Pairs → Resolve Mappings → Build IO Spec → Plan Pipeline → Emit。`_require_inputs()` 只检查必需对象及类型；`_validate_descriptions()` 一次完成 DataSchema、ModelMetadata 和 RobotProfile 的内部自洽性检查，后续 IO 构造不再重复验证。`fixed` / `padded_to_max` 必须声明正数 `dim_policy_max`。Check Pairs 只比较有共享显式词表的事实，不用相机名或关节名猜测 DataSchema 与 RobotProfile 的对应。相机候选的校验与构造都由 Camera Mapping 负责，不存在第二套 matching 判断。

Platform Adapter 必须产出 checkpoint DataSchema 要求的相机 key、state 顺序和 action 语义。因此 `robot_to_model = data_to_model`。`model_to_robot` 恢复到 DataSchema action，Adapter 再把该 action 发送给平台。不支持跨机器人 checkpoint 复用或隐式字段重排。`ResolvedAssembly.load()` 严格要求两条输入计划值相等；不兼容旧 checkpoint，也不维护格式版本迁移。

## 3. 扩展方式

新增解析规则或 TransformStep 时，保持稳定 ID、显式适用/失败条件，禁止模型名、数据名或机器人名硬编码。模型输入尺寸和向量宽度必须来自 ModelMetadata / ModelIOSpec，不从 transform args 反向推导。无法编译必须抛错；`TransformPipelinePlan` 不携带 `resolved` 状态，空 calls 只表示已成功规划出的 identity。

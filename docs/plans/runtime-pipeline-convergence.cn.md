# 开发计划：运行时 Pipeline 收敛（DataSchema 作为统一边界）

> 状态：**已实施**（2026-08-13）。
>
> 本计划承接阶段 4/5，但不再沿用「为 `robot_to_model` 新增关节重排、相机改名、
> 夹爪翻转等通用 step」的方向。当前产品假设是：**训练数据由目标机器人录制，
> Platform Adapter 在运行时输出与该数据相同的 `DataSchema` 接口**。这是现有用法的
> 主路径，也是本轮重构应优先服务的范围。
>
> 代码、`ResolvedAssembly` 直接 JSON、架构/模块文档、CLAUDE.md 与模型适配指南已同步。

---

## 1. 目标与核心假设

### 1.1 目标

把三条公开语义路径收敛成两份实际执行计划：

```text
训练数据 ───────────────┐
                       ├─ DataSchema ── data_to_model ── Model
机器人 ─ Platform Adapter┘

Model raw action ── model_to_robot ── DataSchema action
                                      │
                                      └─ Platform Adapter ── Robot
```

公开结构仍保留：

```python
data_to_model
robot_to_model
model_to_robot
```

但只规划两次：

```python
data_to_model = plan_data_to_model(...)
robot_to_model = data_to_model
model_to_robot = plan_model_to_robot(...)
```

这里的相等是**计划内容相等**。解析失败直接抛错，不编码为计划状态；JSON round-trip 后不要求 Python 对象身份相同。

### 1.2 当前必须成立的假设

1. 训练数据来自目标机器人或与目标机器人使用相同的观测/动作接口；
2. Platform Adapter 负责把机器人 SDK 的原生输入组装为 `DataSchema`；
3. Adapter 输出的相机 key、state 顺序和 action 语义与录制数据一致；
4. `model_to_robot` 当前恢复到 `DataSchema action`，Adapter 负责把它发送给机器人；
5. 当前不支持跨机器人复用 checkpoint，也不支持运行时接口与录制接口不同的情况。

这些是本版本的显式能力边界，不在 resolver 中用名字启发式“补齐”。

---

## 2. 设计决策

### 2.1 DataSchema 是 Assembly 的运行时输入边界

Assembly 不接触机器人 SDK 的原生字段：

```text
robot-native observation
        │
        │ Platform Adapter
        ▼
DataSchema-compatible observation
        │
        ▼
data_to_model / robot_to_model
```

因此 resolver 不需要生成：

- `robot_to_canonical`；
- `data_to_canonical`；
- `canonical_to_model`；
- `rename_cameras`；
- `reorder_state`；
- `reorder_action`；
- `RobotIOSpec`。

“canonical”只保留为概念说明，不新增代码层级或子 Pipeline。

### 2.2 robot_to_model 直接复用 data_to_model

两者入口名称表达不同场景，但执行内容相同：

- `data_to_model`：训练、离线评估等数据侧入口；
- `robot_to_model`：机器人部署的语义入口；
- 两者输入都已经符合 `DataSchema`。

resolver 发出的两者必须满足：

```python
assembly.robot_to_model == assembly.data_to_model
```

不再用 `resolved=False` 表达未实现，也不再等待关节重排或夹爪翻转 step。

### 2.3 model_to_robot 的当前目标是 DataSchema action

`model_to_robot` 撤销模型侧 action 变换，例如：

```text
PI0 output (50, 32)
  → unpad_action(target_dim=8)
  → unnormalize_action
  → DataSchema action (50, 8)
```

当前命名保留，是因为在核心假设下：

```text
DataSchema action == 机器人录制/执行使用的 action 接口
```

模型原始输出与执行侧动作继续使用两个明确的宽度：

```python
model_output_dim = assembly.model_io_spec.action_dim
execution_action_dim = assembly.schema.action_dim
```

原始模型输出按前者校验，`model_to_robot` 结果按后者校验。

### 2.4 RobotProfile 不负责证明 DataSchema 的字段语义

`RobotProfile` 只描述机器人自身静态事实，并校验自身结构：

- joint names 与 types/limits 的长度关系；
- control mode 枚举合法；
- safety bounds 内部维度一致；
- 推荐控制频率为正；
- profile 内部不存在重复或非法字段。

本阶段不再根据名称硬校验：

- `RobotProfile.cameras` 与 `DataSchema.cameras`；
- `DataSchema.state_dims[].name` 与 robot joint names；
- `DataSchema.action_dims[].name` 与 robot joint names；
- 数据 action 维度与 robot joint 数量（除非未来 profile 能明确声明完整 command layout）。

原因是这些字段处在两个不同命名空间。`front` 与 `head_camera`、`joint_0` 与
`shoulder_pan` 可能指向同一物理对象；无法匹配只能表示“没有证据”，不能证明不兼容。

两侧都显式声明、使用同一受控词表且不依赖名称猜测的事实仍可硬校验。例如
`DataSchema.action_dims[].mode` 与 `RobotProfile.control_modes` 明确冲突时应失败；这不是
从字段名推断出来的关系。数据侧尚未声明单位时，则不能拿 RobotProfile 的单位单方面判错。

现有基于去后缀和字符串匹配的 JointMapping 不再作为 `resolved=True` 的执行关系，也不用于
发送命令或套用 safety bounds。最简单的实施选择是从 `ResolvedAssembly` 删除
`JointMapping` 及其无消费者的匹配/错误码；若改动评估发现仍有诊断消费者，则暂时保留为空且
`resolved=False`，但不能继续生成推测出的映射。

### 2.5 显式失败，不做隐式补偿

当前若 Platform Adapter 无法产生 DataSchema 所要求的接口，应在 Adapter 初始化或组装输入时
明确失败，而不是由 resolver 猜测相机或关节对应关系。

未来只有出现真实跨接口需求时，才设计显式 binding，例如：

```yaml
camera_binding:
  front: head_camera
action_binding:
  shoulder_pan.pos: arm_joint_0
```

该能力不属于本计划。

---

## 3. Resolver 最终流程

阶段顺序保持不变：

```text
Load
  → Validate
  → Check Pairs
  → Resolve Mappings
  → Build ModelIOSpec
  → Plan Pipelines
  → Emit ResolvedAssembly
```

各阶段职责收敛如下。

### 3.1 Validate

- 校验 `DataSchema` 自身结构；
- 校验可选 `RobotProfile` 自身结构；
- 校验受控 override；
- `fixed` / `padded_to_max` 必须声明正数 `dim_policy_max`；
- 不做 RobotProfile 与 DataSchema 的字段名推断。

### 3.2 Check Pairs

只保留能从可靠事实直接判定的 data × model 检查：

- state/action 维度策略；
- data camera 到 model vision slot；
- normalization stats；
- data/model/robot 都使用受控词表显式声明的 control mode；
- 其他已有、确实由参与方明确声明且无需名称猜测的契约。

删除 robot camera 名称推断、robot joint 名称嵌入以及由此产生的硬错误。RobotProfile 在当前
Pipeline 设计中不参与 tensor 接口推导。

### 3.3 Resolve Mappings

保留真正有执行或诊断意义的映射：

- `CameraMapping`：DataSchema camera → model vision slot；
- `StateMapping`：DataSchema state → model vector；
- `ActionMapping`：DataSchema action → model vector；
- `LanguageMapping`：DataSchema task → model prompt。

Mapping 不包含模型 padding 项，也不描述 robot-native 字段。

### 3.4 Build ModelIOSpec

继续直接从 `ModelMetadata`、model tunables 和 `DataSchema` 构建模型接口：

- vector width 来自模型维度策略；
- image size 来自模型显式事实；
- action horizon 来自模型事实/tunable；
- 不从 transform call 反向推导模型接口。

### 3.5 Plan Pipelines

```python
data_to_model = plan_data_to_model(declaration, context)
model_to_robot = plan_model_to_robot(data_to_model, context)
robot_to_model = data_to_model
```

任何声明但无法编译的 step 都不能得到一份声称完整的计划。若正向计划未解析完成，三条公开
路径均不得通过复制布尔值伪装成完整计划。

---

## 4. 实施工作包

### WP1：固定接口不变量

1. 在模型接口解析中补齐 `dim_policy_max` 校验：
   - `fixed` / `padded_to_max` 缺失或非正数 → `ValueError`；
   - `flexible` 不要求 `dim_policy_max`。
2. 增加 `robot_to_model == data_to_model` 的 resolver 测试。
3. 增加 Assembly JSON round-trip 后两条计划仍值相等的测试。

### WP2：收敛 RobotProfile 参与范围

1. `Check Pairs` 删除 robot camera slot 名称推断；
2. 删除 state/action joint name 的硬匹配；
3. 删除 action width 对 robot joint 数量的硬限制；
4. 保留显式 control mode 兼容检查，并确认它不依赖字段名推断；
5. 保留 `RobotProfile.validate()`，确保 profile 自身合法；
6. 清理只服务上述启发式检查的 helper、错误码和测试；
7. 删除无消费者的 `JointMapping`；若实施时发现仍有明确消费者，则先保留未解析空值，并在
   本计划中记录原因。

### WP3：生成三条语义 Pipeline

1. resolver 仍只调用 `plan_data_to_model()` 和 `plan_model_to_robot()`；
2. Emit 时令 `robot_to_model` 直接取 `data_to_model`；
3. 更新类型/docstring，删除“robot_to_model 尚不可推导”的旧说明；
4. CLI/inspect 摘要显示两条输入路径均 resolved，且明确它们复用同一计划；
5. 不新增 TransformStep 或 Planner hook。

### WP4：明确运行时 Adapter 边界

1. 训练继续执行 `assembly.data_to_model`；
2. 机器人平台 Adapter 输出 DataSchema-compatible observation；
3. 推理预处理可通过 `robot_to_model` 语义入口执行，但不得重新推导 calls；
4. 模型原始输出按 `model_output_dim` 校验；
5. 后处理结果按 `execution_action_dim` 校验；
6. Adapter 根据 DataSchema action keys/order 发送命令，不读取启发式 JointMapping；
7. Adapter 缺少 DataSchema 已声明的相机或 state 时立即报错，不生成 placeholder 掩盖输入
   缺失；`CameraMapping` 按模型策略显式留下的 unmapped vision slot 仍可使用既有 padding。

### WP5：文档和扩展指南同步

实施代码时同步修改：

- 中英文架构文档；
- assembly / inference / robot 模块文档；
- `.claude/CLAUDE.md`；
- 模型适配 skill；
- phase 2/3/4–5 计划中的后续决策说明；
- 示例中关于 `robot_to_model`、JointMapping 和 RobotProfile 跨字段验证的说明。

文档必须明确：RobotProfile 存在不代表 resolver 能证明 robot-native 字段与 DataSchema 字段
逐项相同。

---

## 5. 测试判据

至少覆盖：

1. `robot_to_model` 与 `data_to_model` 的 calls、resolved 值相等；
2. 无 RobotProfile 时训练/推理行为不变；
3. 有 RobotProfile 但相机、joint 命名不同，不因名字无法匹配而解析失败；
4. 非法 RobotProfile 仍在 Validate 阶段失败；
5. 双方显式声明的 control mode 冲突时仍失败；
6. ACT 使用相同的 data/robot 输入预处理；
7. PI0 的 data action 从 8 维 pad 到模型 32 维；
8. PI0 的模型输出从 32 维恢复到执行侧 8 维；
9. `model_output_dim` 与 `execution_action_dim` 分别校验对应位置；
10. `fixed` / `padded_to_max` 缺少 `dim_policy_max` 时失败；
11. 未注册或不可编译 step 不得让计划得到错误的 `resolved=True`；
12. Assembly artifact round-trip 后三条语义路径保持一致；
13. 全量测试与 `git diff --check` 通过。

---

## 6. 明确不做

- 跨机器人部署同一 checkpoint；
- 自动匹配 robot-native camera/joint 名；
- camera rename、state/action reorder；
- 单位转换；
- 绝对动作与增量动作转换；
- gripper 编码翻转；
- `RobotIOSpec`；
- canonical 子 Pipeline；
- 第四条 `model_to_data` Pipeline；
- 用启发式 JointMapping 发送命令或应用安全限位；
- 旧 checkpoint/assembly artifact 兼容层。

触发扩展设计的条件不是“理论上可能不同”，而是出现一份真实数据/机器人组合，Platform
Adapter 无法在不重复公共逻辑的前提下稳定输出 DataSchema。届时优先增加显式 binding；只有
多个平台反复需要同一种转换时，才把它升级成通用 TransformStep。

---

## 7. 建议提交切分

| # | 类型 | 内容 |
|---|---|---|
| 1 | `fix:` | `dim_policy_max` 不变量及测试 |
| 2 | `refactor:` | 收敛 RobotProfile 的 Check Pairs / Mapping 范围 |
| 3 | `feat:` | `robot_to_model` 复用 `data_to_model` + artifact/CLI 测试 |
| 4 | `refactor:` | 推理与 Platform Adapter 使用三条语义入口和两个 action width |
| 5 | `docs:` | 架构、模块文档、CLAUDE.md、skill 与示例同步 |

---

## 8. 实施结果

- resolver 只规划 `data_to_model` / `model_to_robot`，Emit 时令 `robot_to_model = data_to_model`；
- 删除 `JointMapping`、robot camera/joint 名称匹配及对应错误码；
- `fixed` / `padded_to_max` 的非正 `dim_policy_max` 在 Validate 阶段失败；
- RobotProfile 校验重复/空名、各关节数组长度、safety bounds 成对与长度、control mode 和频率；
- InferenceEngine 执行 `robot_to_model`，并在预处理前验证 DataSchema 必需相机/state；
- 原始模型输出按 `model_output_dim` 验证，后处理结果按 `execution_action_dim` 验证；
- assembly artifact 升级为 v2，要求三个语义入口 resolved 且两条输入计划值相等，不迁移旧 artifact；
- 全量回归：`351 passed, 38 warnings`；`git diff --check` 通过。

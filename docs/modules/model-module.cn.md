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

## 4. 模型描述（目标设计）

> **状态：目标设计，尚未实现。** 本章对齐架构 §4.1.2（ModelMetadata 与
> BaseContract）与 §3.5 的 `inspect` 能力，体例同
> [数据模块设计 §8](data-module.cn.md#8-数据集描述目标设计)。
> 落地前是设计评审对象，落地后按实现修订。

### 4.1 归属：三个载体，一个事实一个来源

与数据维度"全实测"不同，模型维度天然是「声明 × 实测」两层——语义与
策略只能随模型族声明，实例事实由 checkpoint 自述。字段先分归属再谈
内容：

| 载体 | 性质 | 拥有的事实 |
|---|---|---|
| `ModelMetadata`（registry entry 静态声明） | 每**模型族**一份，随代码发布 | 上游 `config.json` 表达不了的语义与策略：槽位接受什么语义、维度策略、归一化方法、缺槽位策略、控制模式偏好、微调挂载点 |
| `BaseContract`（读 checkpoint `config.json`） | 每**实例**一份，运行时实测 | checkpoint 能自述的事实：实际存在的视觉槽位及分辨率、实际 state/action 维度、model_type、checkpoint 路径 |
| baseline profile YAML（`recipe/model/*.yaml`） | 每模型族的运行**默认值** | transform 步骤清单等默认配置 |

合并规则沿用架构 §4.1.2（resolver 的 Materialize 阶段已实现
action_dim 一例）：**实例事实在声明能力边界内优先；越界即
`METADATA_CONTRACT_CONFLICT`；每项事实记来源**（`metadata` /
`base_contract`），inspect 与 `resolve --explain` 按来源展示。

两条归属纪律：

- **同一事实只能有一个来源。** 字段升入 ModelMetadata 后，baseline
  profile 里的对应配置退役（如 image normalization、prompt template
  目前活在 profile 的 transform 配置里）——组合解析层的方向是从声明
  推导 TransformPipelineSpec，profile 最终萎缩为纯默认值，但过渡期
  不允许两处都写。
- **实例身份不进族声明。** checkpoint 引用由 recipe `model.path` 选定、
  由 `BaseContract.repo_or_path` 记录；族声明里不写 `base_checkpoint`。

### 4.2 字段准入原则

同数据模块 §8.2：必须**有消费方**（组合解析检查、Mapping 生成、
transform 规划、训练/推理适配）才进第一版。声明侧字段"可产出"总是
成立（人写的），所以准入几乎只看消费方——没有消费方的声明是死重。

### 4.3 ModelMetadata 字段表（第一版）

**identity**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `name` | registry key，现有 | 全局 |
| `action_head` | 现有 `action_head_type`：`flow_matching` / `diffusion` / `autoregressive` / `regression` | 训练/推理适配 |

**vision —— 视觉契约（CameraMapping 的模型侧）**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `slots[]` | 每槽位一条：`{name, semantic_accepts, required, resolution, channels}` | CameraMapping：数据侧 `cameras[].semantic` 与 `semantic_accepts` 求交做槽位匹配 |
| `missing_slot_policy` | `zero_pad` / `drop` / `error`（模型训练时的约定） | 未映射槽位的 padding 规划（架构 §4.1.2） |
| `image_normalization` | `{method, values}`（自 profile transform 配置迁移） | normalize transform 规划 |

`semantic_accepts` 的取值域**就是**数据侧 `semantic` 受控词表（一处
定义、两处引用），允许泛化值（`third_person` 接受任意第三人称视角）。
BaseContract 实测面：槽位实际存在性 + 分辨率（`camera_roles` 现在
就在读 config.json 的 role + shape），实测槽位必须落在声明 slots 内。

**language —— 指令契约**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `template` | prompt 模板（如 `"{task}"`；自 profile 迁移） | `build_prompt` / `task_tokenize` |

tokenizer 随模型权重走（上游对象自带），不设 `tokenizer_ref`。

**proprio / action —— 维度与归一化契约**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `dim_policy` | `fixed: N` / `padded_to_max: N` / `flexible`（真投影可配） | 维度兼容检查分支 |
| `normalization` | `mean_std` / `quantile` / `min_max` | normalize 规划 + NormStats 满足性检查 |

三种 dim_policy 必须区分：ACT 是 `fixed`，pi0 是 `padded_to_max: 32`
（不足补零，不是 flexible），从零训练的投影层才是 `flexible`。
`normalization` 必须含 `quantile`——openpi/pi0 实际使用 q01/q99
（`UnnormalizeActionQuantileStep` 即为此存在）。

**action —— 动作契约（另含以下字段）**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `chunk` | `{predict, execute_recommended}`；`predict` 即现有 `action_horizon` | 采样窗口；推理层 receding_horizon 默认值 |
| `control_mode_pref[]` | 按优先级可接受的控制模式，词表与数据 `dims[].mode` / RobotProfile 共用（`joint_pos` / `joint_delta` / `joint_vel`） | 控制模式检查 |
| `segment_expectations[]` | 按段类别声明期望表示：`{class, repr, delta_ref?, convention?}`；第一版仅准入 gripper 段 `convention`（`1_is_open` 等）与 `repr`（absolute/delta） | gripper flip 检查的模型侧另一半；delta action 抽象（架构 §7.3 的标准抽象例子）的声明面 |
| `unification` | `{scheme: pad_to_max, pad_value}` | pad transform 配置（自隐式默认迁移） |

**temporal —— 时序契约**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `expected_hz` | 训练数据频率（如 pi0 = 50） | 频率检查的模型侧输入 |
| `history_frames` | 观测历史帧数 | 对账 sampler `n_obs_steps` |

**finetune —— 微调挂载点（对齐现有，不平行再造）**

草稿的 `parts` 即现有 `components`（组件名 → 参数名模式），
`adapter_mounts.lora` 即现有 `support_lora` + recipe
`lora.target_components`。第一版只做对齐，不新增字段。

### 4.4 不进第一版的字段

| 字段/块 | 原因 |
|---|---|
| `identity.family` | 信息性，无消费方 |
| `identity.base_checkpoint` / fingerprint / `schema_version` | 实例身份归 BaseContract；校验与版本机制同数据集侧一并删除 |
| `augmentation_trained_with`、`trained_languages` | 信息性；增广需求已有 `requires_augmentation` |
| `temporal.latency_assumption_s` | 无消费方 |
| `finetune.default_strategy`、`precision` | 弱消费（recipe 缺省提示 / 训练参数校验），按需准入 |
| `runtime` 块（`supports_sampling` / `likelihood_available` / `chunk_streaming`） | 消费方是 RL / TTS / 流式推理（架构 §7.1 对接 RLinf 阶段）；字段设计合理，届时原样准入 |
| `capability` 块（`params_b` / `min_train_vram_gb`） | 信息性；做资源预检/提示时准入 |
| `unification.embodiment_conditioning` / `trained_embodiments` | 软消费（分布内/OOD 警告），后续准入 |
| EEF 类 `segment_expectations`（`arm_eef` + `rotation_repr`） | 与数据侧 EEF 组（数据模块 §8.3）准入边界重合，随 EEF 模型适配一起进入 |

### 4.5 跨维度词表统一

三份词表必须单处定义、多处引用，避免每维度自带方言：

1. **camera semantic**：数据 `cameras[].semantic` = 模型
   `slots[].semantic_accepts` 取值域；
2. **control mode**：数据 `action.dims[].mode` = 模型
   `control_mode_pref` = RobotProfile `control_modes`
   （`joint_pos` / `joint_delta` / `joint_vel`；tokenized 是模型输出
   表示，属 `action_head`，不入此词表——现 `robot/profile.py`
   `_CONTROL_MODES` 含 `tokenized` 待修）;
3. **action head**：沿用现有 `flow_matching` / `diffusion` /
   `autoregressive` / `regression`（草稿的 `ar_token` 并入
   `autoregressive`）。

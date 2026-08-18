# 模型抽象（model）模块设计

> 文档状态：**已实现**。本文对齐当前 `model_interface.py`、`registry.py`
> 与 `adapters/` 的代码边界。
> 对应架构：见 [总体架构 § 4.1.2 VLA 模型](../architecture/vla-factory-architecture.cn.md) 与 [§ 2.2 目录结构 `model/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

模型抽象模块隔离上游模型实现差异，让训练和推理只依赖最小协议。它持有 `VLAModel` 接口、`Observation` 规范类型、`ModelMetadata`，以及模型注册表与上游模型的薄 adapter。模型接口（默认预处理、相机槽位、维度与推理步数等）只由 `ModelMetadata` 声明：具名字段是事实、不可在 recipe 里修改，`params` 是默认超参、可被 recipe `model.config` 覆盖。

## 1. 核心对象

- `model_interface.py`：模型层公共阅读入口，集中 `ModelMetadata`、
  `VisionSlot`、`Observation` 与 `VLAModel` backend 协议。
- `VLAModel` 协议：`compute_loss(observation, actions, ...)` 与 `predict_actions(observation, ...)`；PyTorch 模型另实现 `parameters()` / `named_parameters()` / `train()` / `to()`。
- `Observation`：跨维度共享的规范样本类型（归属本模块，被 composition / training / inference 共同消费）。
- `ModelMetadata`：模型族静态能力描述（视觉槽位、维度策略、动作 horizon、normalization、可训练组件、微调方式、install_hint 等）。
- checkpoint 可选一致性检查：读取外部 checkpoint 的 `config.json`，只验证其冗余形状信息是否与 `ModelMetadata` 相符，不参与事实解析。
- 注册表：`ModelRegistry`、`@register_vla` 与 `get_entry(name)`；内置实现
  lazy import `model/adapters/*`，外部包使用 `vla_factory.models` entry point。
- Thin Adapter：把 `Observation` 与上游模型 batch 互转，不复制上游模型代码。

## 2. 目录与阅读入口

```text
model/
├── model_interface.py       # 统一声明、运行时输入与行为协议
├── registry.py              # 内置发现、装饰器注册、外部插件发现
├── checkpoint_validation.py # checkpoint 的可选冗余事实检查
└── adapters/                # 上游模型薄适配与模型族声明
    ├── act.py
    ├── openpi.py            # PI0 / PI05 有意共享的上游胶水
    ├── pi0.py
    └── pi05.py
```

理解模型层从 `model_interface.py` 开始；新增模型从最接近的 adapter 开始；
checkpoint 检查保持独立，因为它涉及本地或外部仓库 I/O，而模型接口定义保持
纯净。

## 3. 扩展方式

仓库内 adapter 在 `model/adapters/<name>.py` 用
`@register_vla(ModelMetadata(...))` 注册。外部包通过标准 entry point 发布：

```toml
[project.entry-points."vla_factory.models"]
my-vla = "my_package.model:load_my_vla"
```

entry-point 名必须与 `ModelMetadata.name` 相同。无论内置还是外部 adapter，
optional dependency 都应延迟到 factory 调用时导入；adapter 不得根据数据集名
或机器人名加入隐藏分支，也不得重新推导 `ResolvedAssembly` 已确定的关系。

## 4. 模型描述（目标设计）

> **状态：已实现。** 本章对齐架构 §4.1.2（ModelMetadata）与 §3.5 的
> `inspect` 能力，体例同
> [数据模块设计 §8](data-module.cn.md#8-数据集描述目标设计)。
> 第一版字段表已落地（act / pi0 / pi05 三个 entry 已声明，profile 中重复
> 的事实已收敛到 ModelMetadata）。

### 4.1 归属：一份声明两个半区，一个事实一个来源

与数据维度「全实测」不同，模型接口必须在加载 checkpoint 之前就可解析，
因此由模型族声明完整给出。checkpoint 可以来自本地目录、权重文件旁的
`config.json` 或外部仓库；路径和权重内容是实例选择，不是新的接口事实层。

| 载体 | 性质 | 谁能改 | 拥有的事实 |
|---|---|---|---|
| `ModelMetadata` **具名字段**（registry entry 静态声明） | 每**模型族**一份，随代码发布 | 只能改模型声明本身；recipe 不可覆盖 | 完整接口事实：视觉槽位与尺寸、维度策略、action horizon、归一化方法、图像值域、缺槽位策略、控制模式偏好、微调挂载点 |
| `ModelMetadata.params`（同一份声明里的 dict） | 每模型族的运行**默认值** | recipe `model.config` 逐 run 覆盖 | 该模型自己的上游超参（层数、宽度、dropout、推理步数、compile 模式）；不包含 transform 步骤清单 |

**一个模型族 = 一个 adapter 声明文件。** 事实、默认值与 factory 同处一份声明，模型作者不需要先判断「这个键算事实还是算默认值」再决定写进哪个配置文件——**容器即属性**：框架级事实有具名字段和类型，其余一律放进 `params`。只有真正跨模型族共享的上游胶水才提取为独立模块，例如 `openpi.py` 被 PI0 与 PI05 共同使用。（早期版本把默认值放在独立的 `recipe/model/*.yaml`，那让扩展一个模型要写两个文件，还造成 model 叶子层反向依赖用户表达层，已取消。）

不存在 metadata 与 checkpoint 的合并优先级：resolver、ModelIOSpec、Mapping
和 TransformPipelinePlan 都只读 `ModelMetadata`。`checkpoint_validation.py`
可以在加载权重前对可读取的 `config.json` 做一致性检查；读取不到时跳过，读到
矛盾时失败，但无论哪种情况都不会把 checkpoint 值写回模型接口。

两条归属纪律：

- **同一事实只能有一个来源。** 事实升入具名字段后，`params` 里的
  transform 步骤配置不再重复它（image normalization、图像值域、
  归一化方法、pad 目标都已上提）——step 从 `ctx.metadata` 读取，
  且步骤配置里再出现该键即报错，不是覆盖。组合解析层的方向是从声明
  推导 TransformPipelinePlan，届时步骤清单本身也不再需要声明。
- **实例身份不进族声明。** checkpoint 引用由 recipe `model.path` 选定；同一
  模型族的不同外部 checkpoint 共用一个 entry，族声明里不写 `base_checkpoint`。

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
| `image_input_range` / `image_normalize_mode` | 图像值域与归一化方式 | resolver 推导 image_to_float / image_normalize call |
| `image_layout` | 模型输入采用 `CHW` 或 `HWC` | resolver 推导 image_layout call |
| `image_resize_mode` | 尺寸不一致时采用 `stretch` 或 `pad` | resolver 结合 ModelIOSpec 推导 resize_images call |

`semantic_accepts` 的取值域**就是**数据侧 `semantic` 受控词表（一处
定义、两处引用），允许泛化值（`third_person` 接受任意第三人称视角）。
若 checkpoint `config.json` 能报告 role + shape，可选校验会检查必需槽位、
额外槽位、通道数和分辨率是否与本表一致；这些观测值不参与 CameraMapping。

**language —— 指令契约**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `language_template` | prompt 模板（如 `"{task}"`） | `build_prompt` / `task_tokenize` |
| `tokenizer_repo` / `tokenizer_max_length` | tokenizer 来源与固定 token 长度 | resolver 推导 task_tokenize call |
| `prompt_includes_state` | 是否在 padding 前把归一化 state 编入 prompt | resolver 决定 tokenize/pad 依赖顺序 |

tokenizer 来源是模型族事实；若 family 未单独声明 repo，可由 recipe 的
`model.path` 指向带 tokenizer 的 checkpoint 作为运行时地址回退，但不能在
`model.config` 中逐 run 改写 tokenizer 语义。

**proprio / action —— 维度与归一化契约**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `dim_policy` | `fixed: N` / `padded_to_max: N` / `flexible`（真投影可配） | 维度兼容检查分支 |
| `normalization` | `mean_std` / `quantile` / `min_max` | normalize 规划 + NormStats 满足性检查 |

三种 dim_policy 必须区分：ACT 是 `fixed`，pi0 是 `padded_to_max: 32`
（不足补零，不是 flexible），从零训练的投影层才是 `flexible`。
`normalization` 必须含 `quantile`——openpi/pi0 实际使用 q01/q99
（`UnnormalizeActionQuantileStep` 即为此存在）。

resolver 先由这里的维度策略与 `DataSchema` 直接建立 `ModelIOSpec`，再让
`pad_dimensions` 消费源/目标宽度生成 call；禁止从 transform call fold 回模型宽度。
视觉尺寸同理：固定模型使用 `VisionSlot.resolution`，ACT 这类从头训练且输入尺寸可调的
模型使用显式 `params["input_image_size"]`（未设置时采用数据原生尺寸），不得把
`resize_images.height/width` 当作接口事实来源。

**action —— 动作契约（另含以下字段）**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `chunk` | `{predict, execute_recommended}`；`predict` 即现有 `action_horizon` | 采样窗口；推理层 receding_horizon 默认值 |
| `control_mode_pref[]` | 按优先级可接受的控制模式，词表与数据 `dims[].mode` / RobotProfile 共用（`joint_pos` / `joint_delta` / `joint_vel`） | 控制模式检查 |
| `segment_expectations[]` | 按段类别声明期望表示：`{class, repr, delta_ref?, convention?}`；第一版仅准入 gripper 段 `convention`（`1_is_open` 等）与 `repr`（absolute/delta） | gripper flip 检查的模型侧另一半；delta action 抽象（架构 §7.3 的标准抽象例子）的声明面 |
| `unification` | `{scheme: pad_to_max, pad_value}`；第一版由 `dim_policy` / `dim_policy_max` 承担 | pad transform 的目标维度（步骤配置里再写即报错） |

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
| `identity.base_checkpoint` / fingerprint / `schema_version` | checkpoint 路径归 recipe `model.path`；同族 checkpoint 不应复制接口声明，当前也没有版本迁移消费方 |
| `augmentation_trained_with`、`trained_languages` | 信息性，无消费方。（`requires_augmentation` 曾是它们的替代，但框架从未实现数据增广，该字段已随 recipe 的 `augmentation` 块一起删除） |
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

### 4.6 字段放哪：判定规则与配置面三道闸

#### 判定规则

一个键放哪，按顺序问两个问题：

	1. **组合解析要读它吗？**（兼容性检查 / Mapping 生成 / Pipeline 规划）
	   → 是：`ModelMetadata` **具名字段**。例：`vision_slots[].semantic_accepts`、
	   `dim_policy` / `dim_policy_max`、`vector_normalization`、`image_input_range`、
	   `image_normalize_mode`、`image_layout`、`image_resize_mode`、tokenizer 字段、
	   `control_mode_pref`、`missing_slot_policy`。
	2. **改了会改变模型对外的接口语义吗？**（槽位数量、pad 目标维度、动作表示）
	   → 是：具名字段；→ 否：**`params`**。例：`dim_model`、`n_heads`、`kl_weight`、
	   `dropout`、`paligemma_variant`、`pytorch_compile_mode`、`num_inference_steps`。

	Transform step 列表不属于第三类配置：resolver 根据上述具名事实推导并排序
	`TransformStepCall`，`model.config.transforms` 明确拒绝，不能逐 run 改写。

checkpoint 能自述的同名字段只是可选校验输入，不构成第三类事实来源。

两个集合**不相交**：解析器不消费 `dim_model`，用户怎么调都不影响组合是否成立；
反过来事实被逐 run 改会让 `ResolvedAssembly` 与实际运行的模型对不上。

#### 三道闸

自由 dict 的配置面会让三类错误全部静默通过，每一类都有真实案例：

| 闸 | 治什么 | 实现 |
|---|---|---|
| 1. 未声明的键即报错 | 键名拼错、写了早已删除的旧键。pi0 的 factory 逐个 `cfg.get()`，取不到的键凭空消失，从不报错 | `user_interface/recipe.py:merge_model_config()` 校验 `model.config` 的键 ⊆ `params` 的键；组合 override 只允许写入 `overrides`，报错时用 `difflib` 给候选 |
| 2. 未被读取的键即报错 | 声明了却无人消费——改了不生效且无提示。`num_inference_steps` 曾如此 | `utils/tracked_config.py:TrackedConfig` 记录读取，factory 末尾 `assert_all_consumed()`；框架在 factory 之外消费的键预先登记 |
| 3. Transform 配置面被移除 | recipe 改写 step、顺序或事实，导致训练/部署与模型契约漂移 | `merge_model_config()` 与 `resolve_from_facts()` 明确拒绝 `model.config.transforms`；pipeline 只由 resolver 从 `ModelMetadata` 事实推导 |

闸 2 的实现细节值得记一笔：`TrackedConfig` 是 `MutableMapping` 而不是 `dict` 子类——
CPython 对 `dict` 子类的 `**` 展开走的是具体类型快路径，不会调用被重写的
`__getitem__`，那样 ACT 透传给 `ACTConfig(**cfg)` 的键会全部被误判为未读。

配套的可见性由 `inspect model` 提供：输出每个可调键的**当前生效值**与**来源**
（`recipe` / `model default`）。用户的心智因此可以压缩成一句话——
**声明里有什么我就能调什么，写错启动就报错，改完 `inspect` 一看就知道生效没有。**

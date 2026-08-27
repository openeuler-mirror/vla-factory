# VLA Factory 架构设计 v2 —— 以 Embodiment Template 为核心

> **文档状态**：本文描述 v2 的**目标架构**，是对
> [v1 架构文档](vla-factory-architecture.md)（EN）/ `.cn.md` 的演进而非推翻。
> v1 中已实现且 v2 不改变的内容（协议、注册表机制、依赖管理、可靠性与测试
> 章节）不在此重复，只在相关处引用。当前实现进度见 [§7 迁移路线](#7-与-v1-的兼容与迁移路线)。
> 文中标注 `[planned]` 的能力为规划项，`[wip]` 为进行中——图和表同时承担
> 路线图职能，未实现的能力不冒充已实现。

---

## 0. 概览

VLA Factory 是配方（recipe）驱动的机器人 VLA 模型微调框架。v1 已经建立了
四根桩：统一 recipe、统一中间数据表示（IR）、模型注册表、训练产物契约
（`inference_metadata`）。v2 在此之上引入框架的核心抽象——
**Embodiment Template（具身模板）**，把 v1 中散落各处的隐式绑定知识提炼为
显式的、可校验的、随产物冻结的声明资产。

一句话总纲：

> **Embodiment Template 体系 = 三类按轴独立的声明资产 + 一个确定性的
> binding 推导器。用户在 YAML 里做选择，推导器算出组合方式，产物冻结
> 组合结果，部署执行冻结契约。**

三类声明资产按单轴定义、独立演进：

- `dataset_schema` —— 每数据集一份，声明"我有什么"；
- `embodiment_profile` —— 每机器人本体一份，声明"物理上是什么"；
- `model_interface` —— 每模型一份，声明"我需要什么"。

`BindingResolver` 对三份声明做确定性推导，产出 `BindingPlan`（有序变换
计划）。训练侧执行该计划，产物冻结该计划，部署侧逆向执行该计划。
N 个数据集 × M 个本体 × K 个模型只需要 N + M + K 份声明，而非
N × M × K 份适配代码。

### 目录

- [0. 概览](#0-概览)
- [1. 背景与动机](#1-背景与动机)
- [2. 设计原则](#2-设计原则)
- [3. 架构视图](#3-架构视图)
- [4. Embodiment Template 核心设计](#4-embodiment-template-核心设计)
- [5. 配置系统 v2](#5-配置系统-v2)
- [6. 核心模块的 v2 变化](#6-核心模块的-v2-变化)
- [7. 与 v1 的兼容与迁移路线](#7-与-v1-的兼容与迁移路线)
- [8. 可靠性与测试的 v2 增量](#8-可靠性与测试的-v2-增量)
- [9. 演进方向](#9-演进方向)
- [附录 A. 与 LlamaFactory 的概念对照](#附录-a-与-llamafactory-的概念对照)
- [附录 B. 术语表](#附录-b-术语表)

---

## 1. 背景与动机

### 1.1 v1 已经解决的问题

v1 完成了"工程胶水层"的骨架：

- **recipe 中枢**：一份 YAML 描述模型、数据、动作空间、微调策略、训练参数、
  输出位置；resolved recipe 冻结进产物。
- **IR 与格式读取器**：`FormatReader` 把 LeRobot v3 等外部格式统一为
  `DataSchema` / `NormStats` / episode 帧流。
- **注册表与薄适配器**：`@register_vla` + `ModelMetadata`，框架不拥有任何
  模型架构代码。
- **产物契约**：`inference_metadata/{recipe.yaml, schema.json,
  norm_stats.json}` + `final/model.pt`，部署只读快照、不回读数据集。
- **checkpoint 契约**：`base_contract.py` 从预训练 checkpoint 的
  `config.json` 读取真实输入契约（相机 role、维度、分辨率），而非硬编码
  查表。

### 1.2 v1 未解决的问题：绑定知识是隐式的

VLA 训练面对的是**三方绑定**问题，比 LLM 世界 chat template 解决的两方
绑定（数据格式 × 模型 token 约定）多出一条轴：

```text
dataset schema           embodiment schema        model interface
(数据里实际有什么)    ×   (机器人物理上是什么)   ×   (模型期望吃什么/吐什么)
LeRobot/HDF5 字段        自由度/相机/控制模式       pi0 要 3 路相机 + 50Hz chunk
                                                  OpenVLA 要单相机 + 7 维 token
```

同一份 ALOHA 数据要能喂 pi0 也能喂 ACT；同一个 pi0 要能适配不同本体。
三方任意换一个，绑定关系就变。这团绑定知识在 v1 中真实存在，但以隐式
形态散落四处：

| 绑定知识 | v1 中的藏身之处 | 问题 |
|---|---|---|
| 相机 → 模型槽位映射 | recipe 的 `model.config.camera_mapping`，用户手写，`check_camera_mapping()` 只校验不推导 | 放错抽屉（不是模型配置是绑定知识）；无法推导 |
| 变换链的组成 | model profile 的 `default_transforms` + 数据加载入口 `_build_transforms()` 组装 | 模型单方面说了算，数据集语义不参与；不会插夹爪翻转、不会转旋转表示、不会重采样 |
| 夹爪 0/1 语义、旋转表示、控制模式 | **无处声明**，靠数据集文档口口相传 | 社区排名第一的隐形 bug 来源 |
| 本体物理事实（限位、限幅、工作空间） | recipe 的 `action_spec.bounds_*`（每实验重写）+ deploy 平台适配器内的硬编码 | 本体事实混进实验意图；部署安全钳位无有主数据源 |
| 时序语义（采集频率 vs 模型频率 vs 控制频率） | 无处声明 | 部署插值靠约定俗成 |

后果是熟悉的 VLA 落地静默失败模式：模型没坏，是部署侧归一化统计、夹爪
约定或频率假设对不上。**把这团隐性知识提炼成声明式规范，就是
Embodiment Template 的全部使命。**

### 1.3 为什么是"升维"而不是重写

v1 的四根桩恰好是 Template 需要的地基：IR 是模板的物理载体，checkpoint
契约是 `model_interface` 的动态部分，resolved recipe 冻结机制是"模板随
产物走"的雏形。v2 缺的只是第三条轴（embodiment）、语义词表和推导器。
因此 v2 的全部改动是**做加法**：无 sidecar、无 `embodiment` 字段的旧
recipe 在 v2 下行为与 v1 完全一致（见 §7 退化规则）。

---

## 2. 设计原则

v1 的五条原则全部继承，不再展开（recipe 驱动、适配优先于重实现、协议不
假设模型结构、数据契约与模型解耦、依赖按需安装——见 v1 文档 §2）。v2
新增以下六条：

### 2.1 三层分离、绑定后置

声明资产按单轴定义：`dataset_schema` 随数据集、`embodiment_profile` 随
本体型号、`model_interface` 随模型。三类声明各自独立存在、独立演进——
就像插头和插座各自符合标准，而不是每对插头插座单独造一个转换头。
binding **不是第四种资产，而是一次计算的结果**：无损的语法规整可以急切
做、物化存储；有损的语义绑定（归一化、chunk 切分、槽位映射、指令渲染）
延迟到加载时按声明执行，绝不烧进存储层。

评审 schema 的最强判据：**每份声明只含自己轴上的事实**。dataset_schema
里出现任何模型知识、model_interface 里出现任何数据集知识，都是设计错误
的信号。

### 2.2 模板随产物走（单一事实源）

训练启动时解析出的完整绑定计划（含统计量指纹、所有变换链）必须冻结进
checkpoint 产物。部署侧只读快照、逆向执行（反归一化、约定还原、限幅
钳位、频率插值），**不读活注册表、不回读数据集**。训练-部署不对称是
VLA 落地静默失败的第一大来源，模板作为单一事实源从机制上消灭这类事故。

### 2.3 可校验、拒绝猜测

声明式的价值一半在于能静态检查：加载时验证维度、字段覆盖、指纹匹配；
推导遇到真歧义时**拒绝猜测、报可读错误**，报错指向模板字段
（"action.gripper：数据集声明 1_is_open，模型期望 0_is_open，将插入
flip 变换"），而不是让用户 debug 张量形状。

### 2.4 声明优先、代码逃生舱

注册表用纯 YAML 静态声明——可校验、可 diff、非程序员可读；推导逻辑全部
在代码里（与 v1 "配置 YAML 只有静态值"原则一致）。同时保留注册自定义
Python transform 挂进 binding 的逃生舱（v1 已有 `transform_imports`
机制），覆盖声明式表达不了的 10% 怪异场景，避免抽象变成牢笼。

### 2.5 声明在上、实现在下

三条轴每条在架构中出现两次——契约层一份声明（YAML/静态），执行层一份
实现（代码）：

| 轴 | 契约层（声明） | 执行层（实现） |
|---|---|---|
| 数据 | `dataset_schema` | `FormatReader` 读帧、codec、Manifest、Sampler |
| 模型 | `model_interface` | registry entry 薄适配器、`VLATrainer` |
| 本体 | `embodiment_profile` | deploy 的 platform adapter、transport |

声明层三者**只通过 BindingResolver 相遇**；实现层三者**只通过计划和产物
相遇**。任何跨轴直连（如模型适配器直接读数据集字段）都是架构违例——
这条判据使逻辑视图（§3.5）成为 PR 评审工具。

### 2.6 checkpoint 事实源与静态声明的合并规则

v1 的既有原则"契约从 checkpoint 读、绝不硬编码查表"与静态
`model_interface` 注册表并不冲突，按字段性质分家：

- **checkpoint 能自述的事实**（相机 role 名、state/action 维度、分辨率、
  `max_action_dim`）→ 继续由 `load_base_contract()` 动态读取，
  **checkpoint 赢**；
- **checkpoint 自述不了的语义**（role 的视角语义、夹爪内部约定、可接受
  控制模式优先级、归一化方法、缺槽策略、时序期望）→ 写进
  `model_interface` 静态声明，随框架版本发布。

冲突时（如声明 3 槽、checkpoint 只有 2 槽）以 checkpoint 为准并告警。
这样既获得静态语义的可推导性，又保住"新 checkpoint 免框架改动"。

---

## 3. 架构视图

单一视图无法同时回答"有什么能力""怎么运行""概念如何关联""谁写什么"。
v2 采用多视图体系，每张图有明确的读者和问题：

| 视图 | 回答的问题 | 读者 | 位置 |
|---|---|---|---|
| 上下文视图 | 上下游是谁、边界上流动什么、在用户场景中占哪一段 | 所有人（第一张图） | §3.1 |
| 角色视图 | 我是谁、写什么、不碰什么 | 新用户/贡献者 | §3.2 |
| 能力视图 | 有什么能力、插件插哪、什么是 planned | 选型者/评审 | §3.3（主图） |
| 运行视图 | 一次 train/deploy 怎么走 | debug 的人 | §3.4 |
| 逻辑视图 | 有哪些契约类型、静态关系 | 核心开发者/PR 评审 | §3.5 |
| 开发视图 | 目录/模块边界、v2 新增代码的落点 | 贡献者 | §3.6 + `CLAUDE.md` |

新读者按"上下文 → 角色 → 能力 → 运行 → 逻辑"的顺序渐进深入——先知道
系统站在生态的哪个位置，再看谁怎么用它、它有什么能力、它怎么运转、它由
哪些契约构成。

### 3.1 上下文视图：上下游生态位与场景位置

上下文视图把 VLA Factory 当作**黑盒**，只回答两个问题：它和哪些外部系统
相连（上下游关系），以及在用户的端到端工作流里它占据哪一段（场景位置）。
对一个定位为"生态胶水层"的框架，这是最重要的一张外部视图——它的价值
不在内部实现，而在它站的位置。

#### 3.1.1 系统上下文图

```text
                        UPSTREAM  (consumed through adapter boundaries)
  ┌────────────────────┐   ┌────────────────────┐   ┌─────────────────────────┐
  │  Model ecosystems  │   │ Pretrained ckpts   │   │  Data sources           │
  │  lerobot (ACT)     │   │ HuggingFace hub    │   │  teleop recording       │
  │  openpi (pi0/pi05) │   │ (lerobot/pi0_base) │   │  (lerobot record, ...)  │
  │  GR00T… [planned]  │   │                    │   │  public sets (OXE, ...) │
  └─────────┬──────────┘   └─────────┬──────────┘   └────────────┬────────────┘
            │ thin adapters          │ BaseContract              │ FormatReader
            │ (registry entries      │ (reads config.json)       │ + sidecar meta
            │  + optional extras)    │                           │
            ▼                        ▼                           ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                               VLA FACTORY                                 │
  │  owns:      semantic unification (Embodiment Template) · training         │
  │             orchestration · artifact contract · deploy serving            │
  │  not owns:  model architectures · teleop stack · simulators ·             │
  │             RL runtime · robot firmware                                   │
  └───────┬───────────────────────┬───────────────────────────┬───────────────┘
          │ job orchestration     │ artifact bundle           │ policy serving
          │ + ckpt round-trip     │ (weights + frozen         │ (ZMQ | TCP RPC
          │ + metric backflow     │  binding snapshot)        │  + connector)
          │ [planned]             ▼                           ▼
  ┌───────────────────┐   ┌───────────────────┐   ┌─────────────────────────┐
  │ External RL       │   │ Experiment mgmt / │   │ Robots & simulators     │
  │ engines           │   │ eval / repro      │   │ LeKiwi real robot (zmq) │
  │ (RLinf, ...)      │   │ tooling           │   │ RoboTwin sim (connector)│
  └───────────────────┘   └───────────────────┘   │ GR00T eval platform     │
                        DOWNSTREAM                └────────────┬────────────┘
                                                               │ rollout data ·
                                                               │ success/failure
             ┌─────────────────────────────────────────────────┘
             ▼
        data flywheel: back to Data sources / preference & RL stages  [planned]
```

这张图承载三条别的视图给不了的信息：

1. **每条边 = 仓库里的一族适配器边界。** 模型生态进来走 registry
   entries + optional extras；预训练 checkpoint 进来走 `BaseContract`；
   数据进来走 `FormatReader` + sidecar meta；机器人/仿真出去走
   platforms/transports/connectors；外部 RL 引擎走作业编排 adapter
   `[planned]`。上下文图的边和代码的扩展点一一对应——"胶水层"定位由此
   成为可检验的结构事实：**未来新增任何一条边，都必须能回答"它对应哪族
   adapter"**；答不上来的连接就是越过边界的耦合。
2. **"不拥有"清单与"拥有"清单同等重要。** 不做模型架构、不做遥操作
   采集栈、不做仿真器、不做 RL 运行时、不做机器人固件——每条都是刻意的
   边界决策，写在中心盒子里就是对"框架不会膨胀成什么都做"的公开承诺，
   也是 §2 各原则（适配优先于重实现、依赖按需）的黑盒表述。
3. **数据飞轮是图上唯一的环。** 部署侧 rollout 的成败数据回流到数据源与
   preference/rl stage（§6.3、§9），当前为 `[planned]`。这个环解释了
   为什么部署是一等模块：没有 policy serving 这条出边，飞轮就闭不上。

#### 3.1.2 场景旅程：端到端工作流中的覆盖段

上下文图回答"和谁相连"，场景旅程回答"用户完整工作流的时间轴上，哪段
是框架的"：

```text
  user journey:
  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌──────────┐  ┌────────┐  ┌──────────┐
  │ collect │─▶│ convert  │─▶│ describe │─▶│ train │─▶│ evaluate │─▶│ deploy │─▶│ feedback │
  │ teleop  │  │ to       │  │ recipe + │  │ sft   │  │ offline  │  │ sim /  │  │ rollout  │
  │ demos   │  │ lerobot  │  │ plan     │  │ (rl*) │  │ L1 · sim │  │ robot  │  │ data     │
  └─────────┘  └──────────┘  └──────────┘  └───────┘  └──────────┘  └────────┘  └──────────┘
   outside      boundary       CORE          CORE       CORE          CORE        planned
   (lerobot     (converter     (plan cmd)    (train)    (evaluate)    (deploy)    (flywheel →
    record)      + sidecar                                                         preference/
                 meta.yaml)                                                        rl stages)
```

每格下方的标注是覆盖度声明，含义如下：

- **CORE**：框架全权负责的段——从"拿到数据"到"机器人动起来"的中段
  全包，对应 CLI 的 plan/train/evaluate/deploy 命令。
- **outside**：刻意不做的段。遥操作采集属于 lerobot record 等上游工具，
  框架不重复建设。
- **boundary**：转换段标 boundary 而非 outside 是有意的——格式转换本身
  可以由外部工具完成，但**转换必须留痕**（sidecar `meta.yaml` 的
  `provenance` 段，§4.1.1）是框架规范的一部分。转换器从"只产数据"变成
  "产数据 + 产声明"，数据血统由此不断。
- **planned**：反馈段依赖 preference/rl stage 落地（§6.3、§7.3）。

一句话概括场景位置：**中段全包，两端刻意不做，但用契约把两端边界钉死**
——上游端用 sidecar meta 钉住（声明随数据走），下游端用冻结的 binding
snapshot 钉住（机器人侧只需机械执行契约）。

### 3.2 角色视图：角色 × 接触面

角色视图是经典用例视图（4+1 的 "+1"）的贡献者变体：不罗列"角色 × 用例"
，而是回答"角色 × 改什么文件"——与 §3.1 的黑盒外部视角互补，这里打开
盒子看接触面。

四类角色各有独立的、互不重叠的接触面，谁都不进核心：

```text
  Experiment user (fine-tune my robot)     Model integrator (add a new VLA)
  ─────────────────────────────────        ─────────────────────────────────
  writes: recipe.yaml (~10 lines)          writes: registry/entries/<m>.py
  runs:   plan -> train -> evaluate                config/model/<m>.yaml
          -> deploy                                (incl. interface section)
  never:  transforms, adapters,            runs:   pytest test_<m>_model.py
          normalization code               never:  data pipeline, trainer,
                                                   deploy internals
           │                                      │
           ▼                                      ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                          VLA Factory                            │
  │   stable core — no persona ever edits it:                       │
  │   BindingResolver · TransformPipeline · VLATrainer ·            │
  │   InferenceEngine · registry loading machinery                  │
  └─────────────────────────────────────────────────────────────────┘
           ▲                                      ▲
           │                                      │
  Data integrator (new dataset/format)     Platform integrator (new robot)
  ─────────────────────────────────        ─────────────────────────────────
  writes: formats/<f>.py reader            writes: config/embodiment/<e>.yaml
          sidecar meta.yaml                        platforms/<p>.py
  runs:   preprocess -> plan                       (+ transport if needed)
  never:  model code, trainer              runs:   deploy --platform <p>
                                           never:  model code, data pipeline
```

四个端口正好对应四条扩展轴（实验/模型/数据/本体），与能力视图的柱子、
逻辑视图的声明族一一对应。

配图：按**变化频率**分层的洋葱结构——

```text
  外环  recipe + sidecar meta        随实验/数据集变   实验用户、数据接入者
  中环  三类注册表 + reader/adapter  随资产变          接入者、社区 PR
  内环  核心契约与推导器             随框架版本变      框架维护者
```

三个环的变化频率相差一个量级。分离的判据正是：每次实验变的进实验
YAML，随资产变的进对应注册表，随组合变的进 binding 层（能推导就不
落盘），产物里冻结一切。

### 3.3 能力视图（主图）

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│         CLI / API   train · plan[planned] · preprocess · list ·              │
│                     evaluate · infer · deploy                                │
├──────────────────────────────────────────────────────────────────────────────┤
│   Recipe   stage · model + dataset + embodiment (by name) + hyperparams      │
│            + optional binding overrides                                      │
├────────────────────────┬─────────────────────────┬───────────────────────────┤
│      Model Layer       │     Training Layer      │       Deploy Layer        │
│  ────────────────────  │  ─────────────────────  │  ───────────────────────  │
│  registry entries      │  stages                 │  InferenceEngine          │
│   act · pi0 · pi05     │   sft (BC)              │  execution policies       │
│  protocols             │   preference  [planned] │   sync · ensembling ·     │
│   compute_loss         │   rl          [planned] │   receding_horizon        │
│   predict_actions      │  strategies             │  platforms                │
│  loading options       │   full · freeze ·       │   sim · lerobot ·         │
│   precision            │   selective · lora[wip] │   robotwin · groot        │
│   quantization [plan]  │  execution forms        │  transports  zmq · tcp    │
│   patching     [plan]  │   in-process (HF)       │  edge quantized           │
│                        │   orchestrated          │   runtime      [planned]  │
│                        │   (RLinf …)   [planned] │                           │
├────────────────────────┴─────────────────────────┴───────────────────────────┤
│    Embodiment Template  (cross-cutting: serves train · rollout · deploy)     │
│    dataset_schema × embodiment_profile × model_interface -> BindingResolver  │
│    -> BindingPlan (executed in training · consumed by rollout env [planned]  │
│    · frozen into artifacts · inversely executed at deploy)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│    Data Execution   readers (lerobot-v3 · hdf5[plan] · rlds[plan]) · codec · │
│    sampler · TransformRegistry (plan executor)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│    Artifacts   checkpoint + inference_metadata                               │
│    {recipe.yaml · schema.json · norm_stats.json · binding.json[planned]}     │
└──────────────────────────────────────────────────────────────────────────────┘
```

与 LlamaFactory 能力图的三处刻意不同：

1. **多一根 Deploy 柱。** LlamaFactory 的推理是导出给 vLLM；VLA 的部署是
   边缘闭环控制，是一等模块，也是本框架的生态位。量化在两侧的地位相应
   反转：LLM 世界训练侧量化（QLoRA）是刚需；VLA 模型规模较小，训练侧
   量化只是策略层普通插槽，**部署侧边缘量化才是差异化位置**。
2. **Template 画成横条不是竖柱。** 它是 Data Worker 的对应物，但服务范围
   横穿训练、rollout、部署三处（论据见 §6.5）。
3. **`[planned]` 显式标注。** 图同时承担路线图职能，避免能力视图变成
   期货清单。

### 3.4 运行视图

#### 3.4.1 分层运行图

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                           User Expression Layer                           │
│    vlafactory-cli | YAML recipe: model + dataset + embodiment (by name)   │
│                  + hyperparams + optional binding overrides               │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ resolves names against registries
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│               Semantic Contract Layer  (Embodiment Template)              │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐    │
│  │ dataset_schema   │  │ embodiment_      │  │ model_interface       │    │
│  │ "what I have"    │  │ profile          │  │ "what I need"         │    │
│  │                  │  │ "what I am"      │  │                       │    │
│  │ FormatReader     │  │ config/          │  │ interface decl in     │    │
│  │ + sidecar meta   │  │ embodiment/*.yaml│  │ model profile + base  │    │
│  │                  │  │                  │  │ contract (checkpoint) │    │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────┬───────────┘    │
│           └─────────────────────┼────────────────────────┘                │
│                                 ▼                                         │
│                BindingResolver (deterministic derivation)                 │
│        slot matching · convention reconciliation · norm resolution        │
│           · temporal alignment · ambiguity -> readable error              │
│                                 │                                         │
│                                 ▼                                         │
│      BindingPlan: ordered transform plan (`plan` dry-run, diffable)       │
└──────────────────┬──────────────────────────────────────┬─────────────────┘
                   │ executed by training path            │ frozen at train
                   ▼                                      │ start
┌─────────────────────────────────────────┐               │
│           Data Execution Layer          │               │
│  FormatReader (frames) | codec |        │               │
│  Manifest | Sampler |                   │               │
│  TransformPipeline (executes the plan)  │               │
└────────────────────┬────────────────────┘               │
                     ▼                                    │
┌─────────────────────────────────────────┐               │
│           Model Training Layer          │               │
│  registry entries (thin adapters) |     │               │
│  VLAModel protocol | finetuning         │               │
│  strategies | VLATrainer                │               │
└────────────────────┬────────────────────┘               │
                     ▼                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│           Artifact Layer  (single source of truth for deployment)         │
│   final/model.pt + inference_metadata/                                    │
│   {recipe.yaml, schema.json, norm_stats.json, binding.json}               │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ reads frozen snapshot only
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                             Deployment Layer                              │
│   InferenceEngine | inverse plan execution (denorm · convention restore   │
│   · clamp · resample) | execution policies | platforms | transports       │
└───────────────────────────────────────────────────────────────────────────┘
```

与 v1 分层图的三处拓扑变化：

1. **中枢从 recipe 换成 BindingPlan。** recipe 变薄（名字引用 + 超参），
   被训练与部署共同消费的是推导出的计划。
2. **流水线不再单出口。** 契约层两条出边——计划被训练路径执行、同时被
   冻结进产物——正是"训练部署同一份事实源"的图形化。
3. **Artifact 显式成层。** 部署层唯一入边来自它，"部署只读快照"的铁律
   在拓扑上成立。v1 的"外部数据解析层"与"IR 层"合并为 Data Execution
   Layer：schema/stats 提取上收进契约层后，剩下的帧读取、采样、变换执行
   职责同一——"按计划把字节变成 batch"。`FormatReader` 出现在两层不是
   重复：schema 提取属于契约层输入，帧读取属于执行层，这个双重角色正是
   语法与语义分离的落点。

#### 3.4.2 训练数据流（v2 增量）

在 v1 训练流"读 schema"之后插入推导节点，其余不变：

```text
  dataset path
      ▼
  FormatReader schema + sidecar meta ──┐
  embodiment profile ──────────────────┼──▶ BindingResolver ──▶ BindingPlan
  model interface + base contract ─────┘         │
                                                 ├──▶ freeze: inference_metadata/binding.json
                                                 ▼
  Manifest + Sampler ──▶ VLADataset ──▶ TransformPipeline(plan) ──▶ Trainer batch
      ──▶ model.compute_loss ──▶ loss / metrics / checkpoint
```

#### 3.4.3 部署推理流（v2 增量）

改两个节点，中间骨架（platform adapter → ObsDict → Observation →
predict_actions）不变：

```text
  checkpoint path
      ▼
  load frozen snapshot (recipe + schema + norm stats + binding.json)   ← 改
      ▼
  platform observation -> platform adapter -> ObsDict
      ▼
  TransformPipeline (forward plan) -> Observation
      ▼
  model.predict_actions -> normalized action chunk
      ▼
  inverse plan execution:                                              ← 改
    denormalize -> convention restore (gripper flip / rotation) ->
    safety clamp (from embodiment profile) -> temporal resample
      ▼
  execution policy + action adapter -> platform action command
```

反归一化、夹爪约定还原、限幅钳位、频率插值全部成为计划里逆序执行的
**显式步骤**，而不是散在 postprocessor 代码里的隐式逻辑。

### 3.5 逻辑视图：契约类型图

```mermaid
classDiagram
    direction TB

    class TrainRecipe {
        <<intent>>
        +stage: sft | preference* | rl*
        +model / dataset / embodiment (names)
        +hyperparams
        +binding overrides
    }

    class DataSchema {
        <<declaration>>
        +dims / cameras / keys
        +semantics: views · conventions · units
        +stats fingerprint
        +provenance
    }
    class EmbodimentProfile {
        <<declaration>>
        +kinematics / grippers
        +native control modes
        +safety limits
    }
    class ModelInterface {
        <<declaration>>
        +camera slots + view semantics
        +accepted control modes
        +normalization method
        +temporal expectations
    }
    class BaseContract {
        <<contract>>
        +camera roles / dims / resolution
        (read from checkpoint config.json)
    }

    class BindingResolver {
        <<service>>
        +derive(schema, profile, interface) BindingPlan
    }
    class BindingPlan {
        <<plan>>
        +ordered transform steps
        +declaration fingerprints
    }
    class TransformPipeline {
        <<executor>>
    }
    class ArtifactBundle {
        <<artifact>>
        +weights
        +inference_metadata incl. binding.json
    }
    class VLAModel {
        <<protocol>>
        +compute_loss()
        +predict_actions()
    }
    class InferenceEngine

    TrainRecipe ..> DataSchema : resolves by name
    TrainRecipe ..> EmbodimentProfile : resolves by name
    TrainRecipe ..> ModelInterface : resolves by name
    BaseContract --> ModelInterface : facts override declaration
    DataSchema --> BindingResolver
    EmbodimentProfile --> BindingResolver
    ModelInterface --> BindingResolver
    BindingResolver --> BindingPlan : derives, deterministic
    BindingPlan --> TransformPipeline : executed by
    BindingPlan --> ArtifactBundle : frozen into
    TransformPipeline --> VLAModel : Observation
    ArtifactBundle --> InferenceEngine : rebuilt from
    InferenceEngine --> VLAModel : predict_actions
```

（`*` 为 `[planned]`。）这张图上有三条只有逻辑视图能表达的架构主张：

1. **BindingPlan 是全图唯一的"汇点再分叉"**——三份声明汇入，执行与冻结
   分出。
2. **BaseContract 对 ModelInterface 的覆盖关系**（§2.6 的合并规则）是
   类型间关系。
3. **声明族三个盒子之间没有任何直连箭头**——图上"没有的边"就是架构禁令。
   评审新代码时可以直接问：你这个 import 对应图上哪条边？

### 3.6 开发视图：v2 代码目录结构

在 v1 目录（见 v1 文档 §3.3）基础上做加法。与 v1 相同的约定：本结构只
描述稳定的目录边界与模块职责，具体文件名随实现演进，架构文档不维护
文件级清单。`[v2]` 标注新增或职责变化的落点：

```text
vla_factory/
├── cli.py                       # [v2] + plan 子命令（§4.5）
├── config/
│   ├── recipe.py                # [v2] + stage 字段、顶层 binding override 段
│   ├── parser.py
│   ├── defaults.py
│   ├── model/                   # 模型 profile：act.yaml · pi0.yaml · pi05.yaml
│   │                            # [v2] 各 profile 增 interface: 段（§4.1.3）
│   ├── embodiment/              # [v2 新] 本体内置库（第三条轴，§4.1.2）
│   │   ├── aloha_2arm.yaml
│   │   ├── lekiwi.yaml
│   │   └── so100.yaml
│   └── dataset/                 # [v2 新] 常用公开数据集的 sidecar 注册表
│       └── <name>.meta.yaml     #        （用户私有数据集的 sidecar 放数据集旁）
├── template/                    # [v2 新] Embodiment Template 子系统（核心）
│   ├── vocab.py                 # 受控语义词表：枚举 + 包含关系（§4.2）
│   ├── declarations.py          # 三类声明的 dataclass + 加载时静态校验（§8.1）
│   ├── sidecar.py               # 数据集旁 meta.yaml 的读取与降级合并
│   ├── resolver.py              # BindingResolver：纯函数推导（§4.3）
│   └── plan.py                  # BindingPlan + binding.json 序列化（§4.4/§4.6）
├── model/
│   ├── protocols/               # ModelMetadata / VLAModel 协议（不变）
│   ├── base_contract.py         # [v2] 角色升级：向 resolver 提供 checkpoint
│   │                            #      事实，参与 §2.6 合并
│   └── registry/entries/        # 薄适配器：act.py · pi0.py · pi05.py（不变）
├── data/
│   ├── formats/                 # reader 接口不变；[v2] schema 提取多填语义字段
│   ├── codec/
│   ├── transforms/              # [v2] + 语义步骤：flip_gripper ·
│   │                            #      convert_rotation · resample_temporal
│   ├── sampling/
│   ├── manifest.py              # [v2] DataSchema 增语义/血统/指纹字段（全部可空）
│   ├── dataset.py
│   └── loader.py                # [v2] _build_transforms() 降级为计划执行（§6.1）
├── training/
│   ├── train.py                 # [v2] + 训练启动时推导计划、冻结 binding.json
│   ├── pytorch_trainer.py
│   └── strategies/
├── deploy/
│   ├── infer.py                 # [v2] postprocess 改为按计划逆行（§3.4.3）
│   ├── policy_runtime.py
│   ├── platforms/               # [v2] 安全钳位从冻结 embodiment 快照读
│   ├── transports/
│   └── connectors/
└── utils/
    └── constants.py             # [v2] + binding.json 等产物布局常量

test/
├── （现有测试不变）
├── test_template_vocab.py       # [v2] 词表取值与声明校验
├── test_binding_resolver.py     # [v2] 推导确定性 + §4.3 调和规则表逐行
├── test_binding_plan_golden.py  # [v2] 代表性组合的 golden 计划
└── test_binding_roundtrip.py    # [v2] 正向/逆向计划 roundtrip
```

三条目录级设计决策：

1. **`template/` 是独立顶层包，不塞进 `config/` 或 `data/`。** 它是 §3.4.1
   分层图里"语义契约层"的代码落点——核心设计配得上一等目录；同时它必须
   被 config（recipe 解析）、data（计划执行）、training（冻结）、deploy
   （逆行）四方消费，放进任何一方都会制造错误的从属关系。
2. **依赖方向单行道。** `template/` 只依赖标准库与轻量契约模块
   （`data/manifest.py` 的 `DataSchema`、`model/base_contract.py` 的
   `BaseContract`），**绝不 import 任何模型生态重依赖**；执行路径
   （loader、train、deploy）依赖 `template/`，反向禁止。这保证 `plan`
   命令在未安装任何模型 extra 的环境下可运行——与 v1 "registry 加载不
   依赖 optional extras"同一原则的延伸。
3. **声明资产与声明代码分离。** YAML 资产住 `config/`（embodiment/
   dataset/model 三个子目录，随包分发、走 `pyproject.toml`
   package-data）；解析与推导代码住 `template/`。资产可由非程序员 PR，
   代码由维护者演进，二者生命周期不同（§5.2 职责表）。

各目录的 `[v2]` 标注与 §7.2 五阶段一一对应：阶段 1 落
`template/vocab.py`、`declarations.py`、`sidecar.py`、`manifest.py`；
阶段 2 落 `config/embodiment/`；阶段 3 落 model profile 的 interface 段；
阶段 4 落 `resolver.py`、`plan.py`、`cli.py`；阶段 5 落 `loader.py`、
`train.py`、`deploy/`。

---

## 4. Embodiment Template 核心设计

### 4.1 三类声明资产

#### 4.1.1 dataset_schema——"我有什么"

物理容器不自研：LeRobot v3 已是社区事实标准，v1 押对了。dataset_schema
在 IR 之上做**语义 sidecar 加法**，来源两级：

- `FormatReader` 能从 `meta/info.json` 自动推导的（维度、相机列表、fps、
  state/action key 序、统计量）——v1 已实现，继续自动推；
- 推不出的语义（视角语义、夹爪约定、控制模式、旋转表示、单位/坐标系、
  转换血统）——从数据集旁的 **sidecar `meta.yaml`** 读取。无 sidecar 时
  语义字段为空，推导器降级为"只做语法绑定 + 显式告警"（v1 兼容）。

常用公开数据集的 sidecar 收入框架内置注册表，社区 PR 扩充。示例
（ALOHA 双臂数据集）：

```yaml
# datasets/aloha_transfer_cube/meta.yaml
schema_version: 1
identity:
  name: aloha_transfer_cube
  source_format: lerobot_v3          # 原始格式，转换血统的一部分
  fingerprint: sha256:8f3a...        # 数据内容指纹
  episodes: 50

observation:
  cameras:
    cam_high:
      resolution: [480, 640]
      semantic: third_person_top      # 语义标签，槽位匹配的依据（见 §4.2 词表）
    cam_left_wrist:
      resolution: [480, 640]
      semantic: wrist_left
    cam_right_wrist:
      resolution: [480, 640]
      semantic: wrist_right
  state:
    dim: 14
    layout:                           # 命名分段，不是裸向量
      - {name: left_arm_qpos,  dims: 6, unit: rad, frame: joint}
      - {name: left_gripper,   dims: 1, unit: normalized, convention: 1_is_open}
      - {name: right_arm_qpos, dims: 6, unit: rad, frame: joint}
      - {name: right_gripper,  dims: 1, unit: normalized, convention: 1_is_open}

action:
  dim: 14
  control_mode: joint_position        # 绝对关节角，不是 delta
  layout: same_as_state
  gripper_convention: 1_is_open       # 事故高发字段，必须显式
  rotation_repr: null                 # 关节空间，无 EEF 旋转

temporal:
  fps: 50
  alignment: action_t_follows_obs_t   # t 时刻动作对应 t 时刻观测

instruction:
  task_field: task
  language: en

stats:
  fingerprint: sha256:2c9b...         # 统计量指纹，绑定/复现的依据
  path: stats.parquet                 # mean/std/q01/q99/min/max

provenance:                           # 转换时做过什么，全部留痕
  - {op: rosbag_to_lerobot, version: 0.3.1}
  - {op: resample, from_hz: 60, to_hz: 50}
```

`provenance` 段消灭"急切转换的静默有损"：重采样、翻转、换表示等转换
决策全部留痕，数据血统不断。

#### 4.1.2 embodiment_profile——"物理上是什么"

v2 新增的第三条轴，目录与 `config/model/` 对称：
`vla_factory/config/embodiment/<name>.yaml`。内置库从 deploy 平台适配器
中已硬编码的本体反推最小字段集起步，社区 PR 扩充。示例：

```yaml
# vla_factory/config/embodiment/aloha_2arm.yaml
schema_version: 1
identity: {name: aloha_2arm, vendor: trossen}

kinematics:
  arms: 2
  dof_per_arm: 6
  urdf: builtin://aloha/vx300s_bimanual.urdf   # 控制模式换算的依据 [planned]
  joint_limits:
    position: from_urdf
    velocity_cap: [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]  # rad/s，部署钳位用

grippers:
  type: parallel_jaw
  native_convention: 1_is_open        # 本体的物理默认约定
  range_m: [0.002, 0.057]

control:
  native_modes: [joint_position]      # 本体只吃绝对关节角
  frequency_hz: {min: 25, max: 100, recommended: 50}

safety:
  workspace_aabb: [[-0.4, -0.6, 0.0], [0.4, 0.6, 0.5]]   # 米，基座系
  estop_behavior: hold_position
```

训练时它是校验依据，部署时它是安全钳位数据源（落掉 v1 §7.3 可靠性
TODO 的一块），跨本体研究时它是 embodiment adapter 的条件输入
`[planned]`。v1 recipe 中 `action_spec.bounds_*` 与 `action_type` 的
默认值迁移至此；recipe 保留同名字段仅作 override（§5.4）。

#### 4.1.3 model_interface——"我需要什么"

静态声明写在 model profile（`vla_factory/config/model/<name>.yaml`）新增
的 `interface:` 段，与动态 `BaseContract` 按 §2.6 规则合并。示例：

```yaml
# vla_factory/config/model/pi0.yaml 的 interface 段
interface:
  schema_version: 1

  vision:
    slots:
      - {name: base_0_rgb,        semantic: third_person, required: true}
      - {name: left_wrist_0_rgb,  semantic: wrist_left,   required: false}
      - {name: right_wrist_0_rgb, semantic: wrist_right,  required: false}
    missing_slot_policy: zero_pad     # 缺相机补零而非报错（pi0 训练约定）
    # 分辨率不在此声明——从 checkpoint BaseContract 读（§2.6）

  language:
    template: "{task}"                # pi0 直接吃裸指令文本

  proprio:
    accepts_dim: flexible             # 投影层可适配
    normalization: mean_std

  action:
    control_mode_pref: [joint_position, eef_delta]   # 按优先级可接受多种
    normalization: mean_std           # OpenVLA 会在自己的 interface 写 q01_q99
    gripper_convention: 1_is_open     # 模型训练时的内部约定
    # output_dim / chunk 不在此声明——从 checkpoint BaseContract 读

  temporal:
    expected_hz: 50
    history_frames: 1
```

原 `default_transforms` 列表退役为 interface 声明的推导依据："我需要
[-1,1] HWC float" 是声明，"插入 image_to_float" 是推导结果（§6.1）。

### 4.2 受控语义词表

三份声明能互相匹配的前提是语义标签来自**框架统一定义的受控词表**
（Python 枚举 + 文档），词表本身是规范的一部分、随框架版本演进。起步
集合（宁缺毋滥——加值容易，删值是 breaking change）：

| 词表 | 起步取值 |
|---|---|
| 相机视角 `CameraSemantic` | `third_person` · `third_person_top` · `wrist_left` · `wrist_right` |
| 控制模式 `ControlMode` | `joint_position` · `joint_delta` · `eef_delta` · `eef_absolute` |
| 旋转表示 `RotationRepr` | `euler_xyz` · `quaternion_wxyz` · `axis_angle` · `rot6d` |
| 夹爪约定 `GripperConvention` | `0_is_open` · `1_is_open` |
| 时序对齐 `TemporalAlignment` | `action_t_follows_obs_t` · `action_t_follows_obs_t_minus_1` |

视角语义支持包含关系（`third_person_top ⊂ third_person`），供槽位匹配
使用。落地时从已支持的三个模型、两三个本体反推最小字段集，再横向扩。

### 4.3 BindingResolver：确定性推导

推导器是纯函数：输入三份声明（+ BaseContract），输出确定性的
`BindingPlan`。同样输入永远产出同样计划——可打印、可 diff、可审计，
类似编译器 IR pass 或数据库查询计划。四类匹配：

1. **相机槽位匹配**：model_interface 的槽位带语义要求，dataset_schema 的
   相机带语义标签，做二分匹配。唯一解 → 自动接；有歧义（两路都匹配
   `primary`）→ 报错要求用户在 `binding:` 段显式指定，**绝不瞎猜**。
2. **约定调和**：逐字段比对数据集声明与模型期望——夹爪 0/1 不一致 →
   插入 `flip_gripper`；旋转表示不同 → 插入 `convert_rotation`；控制
   模式不同（数据是关节角、模型要 EEF delta）→ 查 embodiment_profile
   的 URDF 做正运动学换算 `[planned]`，第一版直接报可读错误。
3. **归一化解析**：方法从 model_interface 声明取（mean_std / q01_q99），
   统计量从 dataset_schema 的指纹取，缺失则现算并缓存回写。
4. **时序对齐**：数据集采集频率 vs 模型 chunk 规格，生成重采样/插值
   计划；对齐约定不一致 → 报错。

自动调和 vs 必须报错的规则表：

| 不一致类型 | 处理 |
|---|---|
| 夹爪约定相反 | 自动插入 flip，计划中显式列出 |
| 旋转表示不同（维度语义可转） | 自动插入转换 |
| 分辨率不同 | 自动插入 resize |
| 频率整数倍关系 | 自动插入重采样 |
| 归一化统计缺失 | 现算 + 回写缓存 + 告警 |
| 控制模式需 FK/IK 换算 | 报错（`[planned]`：URDF 齐备时自动换算） |
| 槽位匹配歧义 | 报错，要求 `binding:` 显式指定 |
| 臂数不匹配（单臂模型 × 双臂数据） | 报错，要求指定 `use_arm` |
| 时序对齐约定不一致 | 报错 |
| 声明语义字段缺失（无 sidecar） | 降级为语法绑定 + 告警（v1 兼容） |

### 4.4 BindingPlan：一物四态

同一份 binding 在生命周期里有四种形态，按优先级覆盖：

```text
自动推导（内存态，默认，覆盖约 90% 场景）
   ↓ 可视化：vlafactory-cli plan，训练前 dry-run 打印完整计划
     （terraform plan / SQL EXPLAIN 式，用户过目确认）
   ↓ 可覆盖：实验 YAML 的 binding: 段写几行 override（临时改动）
   ↓ 可固化：跨实验复用时存成 bindings/<name>.yaml，按名引用（具名态）
   ↓ 必冻结：训练启动时把解析后的完整计划快照写进 checkpoint（产物态）
```

binding 不是第四种资产，而是一次计算的结果；只有当用户需要覆盖或复用
时它才落盘。

### 4.5 plan 命令

`vlafactory-cli plan --config <recipe>` 是 Template 体系的第一个用户可见
功能，也是实现顺序上**最先做**的（它强迫三份 schema 的字段和推导规则
定义清楚，且为零风险只读功能）。v1 的 `list --config`
（`describe_model_config` 的 camera_mapping 报告）是它的前身，将被吸收。

正常推导的输出示例（aloha_transfer_cube × aloha_2arm × pi0）：

```text
BINDING PLAN (auto-derived, deterministic)
✔ cameras   cam_high -> base_0_rgb (third_person_top ⊂ third_person)
            cam_left_wrist -> left_wrist_0_rgb
            cam_right_wrist -> right_wrist_0_rgb
            + resize 480x640 -> 224x224 (x3)
✔ state     14-dim passthrough -> proprio projection dim=14
            normalize = mean_std @ stats:2c9b
✔ action    joint_position ∈ model accepted modes, no conversion
            gripper: dataset(1_is_open) == model(1_is_open) -> no flip
            normalize = mean_std @ stats:2c9b; chunk predict=50
✔ temporal  50Hz == 50Hz, no resample; alignment consistent
✔ safety    deploy manifest: velocity_cap, workspace_aabb (frozen, unused in training)
⚠ notes     none
```

独立性验证——只把 `model: pi0` 换成 `model: openvla`（单槽位、7 维
EEF delta、q01_q99、`0_is_open`），**三份资产文件一个字不改**，计划自动
变为：

```text
✔ cameras   cam_high -> primary; wrist cameras have no slot -> ignored (warn)
✘ action    control_mode: dataset(joint_position) vs model(eef_delta)
            -> needs forward kinematics; embodiment urdf found, but FK
               conversion is not implemented yet [planned] -> ERROR
            gripper: 1_is_open vs 0_is_open -> insert flip
            normalize: q01_q99 (computed now, cached back)
⚠ arms      single-arm interface vs bimanual dataset -> ambiguous
            specify in recipe binding section: use_arm: left | right
```

### 4.6 产物冻结与部署逆行

在 v1 三件套之外增加第四个文件：

```text
<output_dir>/inference_metadata/
├── recipe.yaml       # resolved recipe（v1 已有）
├── schema.json       # 数据 schema，v2 起含语义字段（v1 已有）
├── norm_stats.json   # 归一化统计（v1 已有）
└── binding.json      # [v2] 解析后的完整绑定计划 + 三份声明的指纹
                      #      + embodiment 安全钳位快照
```

部署规则不变且加强：**只读快照**。实验 YAML 引用的是"活"注册表（随
框架升级变化），但部署、复现、调试只认冻结快照——半年后复现实验不被
注册表漂移坑掉。部署侧的反归一化、维度裁剪、夹爪翻转、旋转还原、限幅
钳位从"按 recipe + stats 重建的隐式 postprocessor"改为"按计划逆序执行
的显式步骤"（§3.4.3）。

---

## 5. 配置系统 v2

### 5.1 recipe 变薄

v1 配置系统的合并机制（CLI > recipe > model profile > dataclass 默认，
OmegaConf 深合并，resolved recipe 冻结）全部保留。v2 的变化是 recipe
**内容**变薄——知识住进注册表，选择住在 recipe，两者靠名字绑定：

```yaml
# v2 实验 recipe（典型形态）
stage: sft                        # [v2] 训练阶段，当前仅接受 sft

model:
  name: pi0                       # -> model_interface + registry entry
  path: lerobot/pi0_base          # -> BaseContract 从此读取
embodiment: aloha_2arm            # [v2] -> config/embodiment/ 内置库
data:
  source:
    path: /data/aloha_transfer_cube   # sidecar meta.yaml 在数据集旁
    format: lerobot-v3

finetuning:
  strategy: lora
training:
  lr: 2.5e-5
  batch_size: 8
  total_steps: 20000
output:
  output_dir: outputs/aloha_pi0

# 90% 场景以下不用写；写了就是对推导结果的 override
binding:
  cameras:
    base_0_rgb: cam_high
```

### 5.2 四层职责表

| 文件 | LlamaFactory 对应物 | 谁写 | 生命周期 | 内容 |
|---|---|---|---|---|
| 实验 recipe | 训练 YAML | **用户**，每实验一份 | 一次运行 | stage、三个名字引用、超参、少量 binding override |
| dataset_schema | `dataset_info.json` | **转换器自动生成** + sidecar 补语义 | 随数据集 | 字段布局、单位、坐标系、约定、统计指纹、血统 |
| embodiment_profile | （无对应物，VLA 特有） | **框架内置库 + 社区 PR** | 随机器人型号 | 自由度、限位、默认约定、安全边界 |
| model_interface | `template.py` 注册表 | **框架维护者**，随模型支持发布 | 随框架版本 | 槽位语义、控制模式、归一化方法、时序期望 |

vla-factory 比 LlamaFactory 多出一层（embodiment_profile），是"三方绑定
比两方绑定难"在文件结构上的投影——不是复杂化，是把本来就存在的复杂度
放进正确的抽屉。

### 5.3 优先级链

```text
CLI 显式 override
  > 实验 recipe（含 binding: override 段）
    > BindingResolver 推导结果
      > 注册表声明（dataset_schema / embodiment_profile / model_interface）
        > dataclass 默认值
```

与 v1 的原则一致：离本次运行越近，优先级越高。推导结果插在 recipe 与
注册表之间——它是"注册表的组合物"，天然高于单份注册表、低于用户显式
意图。

### 5.4 v1 字段的去向

| v1 字段 | v2 去向 |
|---|---|
| `model.config.camera_mapping` | 迁移为顶层 `binding.cameras` override；迁移期两处都接受，`model.config` 位置标废弃 |
| `action_spec.action_dim` | 推导（数据集 dim × 模型 pad 目标），保留为 override |
| `action_spec.action_horizon` | 从 BaseContract / model_interface 读，保留为 override |
| `action_spec.action_type` | 由 dataset_schema `control_mode` 声明取代，保留为 override |
| `action_spec.bounds_*` | 迁入 embodiment_profile 安全段，保留为 override |
| model profile `default_transforms` | 退役为 interface 声明；变换链由推导器产出 |

所有迁移遵守退化规则：旧字段继续生效，缺 v2 声明时行为与 v1 一致。

---

## 6. 核心模块的 v2 变化

### 6.1 数据模块：TransformPipeline 从配置驱动变计划驱动

这是 v2 最关键的一步棋。v1 中变换链来自 model profile 的
`default_transforms` + norm stats，由 `_build_transforms()` 组装——模型
单方面说了算，数据集语义不参与。v2 中：

- `BindingResolver` 产出的 `BindingPlan` 是变换链的唯一来源；每步引用
  `TransformRegistry` 已注册步骤名（`resize_images`、`normalize`、
  `image_to_float`……），另新增少量语义步骤：`flip_gripper`、
  `convert_rotation`、`resample_temporal`；
- `_build_transforms()` 职责降级为"执行计划"；
- `TransformRegistry` 的注册机制、YAML 可声明性、自定义 transform 逃生舱
  全部不变。

数据模块其余部分（reader、codec、manifest、sampler、dataset、loader）
职责不变，详见 [数据模块设计](../modules/data-module.md)。

### 6.2 模型模块：适配器不变，profile 增 interface 段

registry、`ModelMetadata`、薄适配器、延迟导入规则全部不变。变化只有：

- model profile 新增 `interface:` 段（§4.1.3）；
- `base_contract.py` 的角色从"校验用户手写的 camera_mapping"升级为
  "向推导器提供 checkpoint 事实"（§2.6 合并规则）；
- 加载选项插槽：precision（已有）、quantization `[planned]`、
  patching（NPU 等）`[planned]`——均在适配器边界经上游机制（peft、
  bitsandbytes 等）实现，框架不拥有实现，与 v1 的 LoRA 设计
  （component → subtree → peft target_modules）同一模式。

### 6.3 训练模块：stage 轴与两种执行形态

v1 只有一个隐式 stage（SFT/BC）。v2 把 `stage` 提升为 recipe 一等字段
（当前仅接受 `sft`，先占住槽位），并确立训练层的两种执行形态：

```text
stage: sft            (BC)                         in-process   [现状]
stage: preference     (KTO × 执行反馈 / DPO 变体)   in-process   [planned]
stage: rl
  backend: builtin    (filtered BC / offline RL,    in-process   [planned]
                       无仿真器在环、NPU 友好)
  backend: rlinf      (PPO / GRPO 仿真在环)          orchestrated [planned]
```

- **in-process（进程内）**：复用 `VLATrainer`（HF Trainer），离线方法
  本质是带权重的监督循环，是普通训练负载；
- **orchestrated（作业编排）**：RLinf 这类拥有自己运行时的系统按
  **作业级编排**接入——配置翻译（保留 `rlinf_extra` 透传）、checkpoint
  双向互转、指标回流、环境资产管理四个接缝；核心框架不 import 任何
  RLinf 代码，版本 pin 死，上游 breaking change 的爆炸半径限制在一个
  adapter 内。接口设计预留其他引擎位置（后端插件架构，不是硬依赖）。

生态位一句话：**vla-factory 之于 RLinf，如 LlamaFactory 之于 TRL**——
但依赖形态是作业编排不是库嵌入（TRL 是可嵌入的库，RLinf 是拥有运行时
的框架，塞进自己的训练循环等于挖掉它的心脏）。

### 6.4 部署模块：逆向执行与安全钳位

deploy 模块的分层（infer 核心 / policy_runtime / platforms / transports /
connectors）不变，详见 [部署模块设计](../modules/deploy-module.md)。v2
变化：

- postprocessor 改为按 `binding.json` 逆序执行计划（§3.4.3）；
- 安全钳位（velocity_cap、workspace_aabb）从冻结的 embodiment 快照读，
  落掉 v1 §7.3 的部分 TODO；
- 边缘量化推理运行时 `[planned]`——VLA 部署是边缘闭环控制，这是量化
  能力的差异化位置（§3.3）。

### 6.5 Template 与多 stage 的关系

多 stage 扩展恰恰是 Embodiment Template 该成为核心的最强论据：RL
rollout 环境在仿真器边界要做的事——观测适配、动作反归一化、夹爪约定
还原、频率插值——**与部署侧逆向执行 BindingPlan 是同一件事**。没有
Template 层，每个 RL 后端、每个仿真环境都要重新手写一遍绑定；有了它，
`stage: rl` 的 rollout 消费的是与 `deploy` 同一份冻结计划。这也是本框架
与 LlamaFactory 的结构性差异：chat template 只服务训练，Embodiment
Template 服务训练、rollout、部署三处。

---

## 7. 与 v1 的兼容与迁移路线

### 7.1 退化规则

v2 的每一步都保持：**无 sidecar meta、无 `embodiment` 字段、无
`interface` 段的旧 recipe，行为与 v1 完全一致**。语义字段缺失时推导器
降级为语法绑定并显式告警，绝不静默改变已有训练结果。

### 7.2 五阶段路线

每阶段独立可合并、可测试，阶段 1–4 不改变任何现有训练行为：

| 阶段 | 内容 | 关键产出 | 状态 |
|---|---|---|---|
| 1 | 受控词表 + schema 扩展 | 语义枚举（§4.2）；`DataSchema` 增语义/血统/指纹字段（全部可空）；sidecar `meta.yaml` 读取 | 未开始 |
| 2 | embodiment 注册表 | `config/embodiment/` + 加载器；recipe `embodiment:` 字段；deploy 安全钳位接入 | 未开始 |
| 3 | model interface 段 | 三个已支持模型的 profile 增 `interface:`；与 BaseContract 的合并逻辑 | 未开始 |
| 4 | BindingResolver + `plan` 命令 | 推导器（纯函数）+ CLI dry-run；吸收 `list --config` 的 camera_mapping 报告 | 未开始 |
| 5 | 计划驱动执行 + 冻结 | `_build_transforms()` 改吃计划；`binding.json` 进产物；部署按计划逆行 | 未开始 |

另有一行低成本先手棋可随任一阶段合并：recipe 增 `stage:` 字段（默认
且仅接受 `sft`），把最重要的槽位先占住。

### 7.3 阶段边界外暂不做的事

- URDF 正/逆运动学换算（控制模式跨空间转换）——先立"报可读错误"的
  行为，FK 是后续增量；
- rollout 环境消费计划（依赖 stage: rl 落地）；
- 训练侧量化、NPU patching；
- 自研物理容器——LeRobot v3 作为 IR 容器,创新集中在语义声明层。若未来
  出现可测量的性能理由（如面向 NPU 的数据布局），再单独立项。

---

## 8. 可靠性与测试的 v2 增量

v1 §7/§8 的清单继续有效，以下为 Template 带来的增量。

### 8.1 校验器分层

| 时机 | 检查 | 形态 |
|---|---|---|
| 声明加载时 | 词表取值合法、维度自洽、layout 分段和等于 dim、指纹格式 | 静态校验，报错指向文件与字段 |
| 推导时（plan / train 启动） | 槽位可匹配、约定可调和、统计指纹匹配、频率可对齐 | §4.3 规则表；歧义即报错 |
| 训练启动时 | 冻结前复核：计划中每个步骤在 `TransformRegistry` 可解析 | fail-fast，先于写 metadata |
| 部署加载时 | 快照完整性（四文件齐、指纹一致）、计划可逆行 | 缺失即拒载，不回退到活数据源 |
| 运行时 | 观测分布漂移监测 | `[planned]`，可选 |

报错风格延续 §2.3：指向模板字段与两侧声明的具体取值，给出修复动作
（"在 binding 段指定 use_arm: left | right"），不让用户 debug 张量形状。

### 8.2 测试增量

- **推导器确定性**：同一输入三份声明，多次推导产出逐字节相同的计划；
- **plan golden 测试**：代表性组合（aloha × pi0、换 openvla 的冲突例）
  的计划输出作为 golden file，防推导规则静默漂移；
- **调和规则单测**：§4.3 规则表逐行——该 flip 的 flip、该报错的报错；
- **冻结/逆行 roundtrip**：正向计划处理样本 → 逆向计划还原 → 与原始
  值在容差内一致（夹爪、旋转、归一化各一例）；
- **退化兼容**：无 sidecar/无 embodiment 的 v1 recipe 在 v2 下产出与
  v1 相同的变换链（回归护栏）；
- 沿用 v1 规约：需要重依赖/GPU 的测试用 `find_spec` + `pytest.skip`
  守护，默认套件保持绿色。

---

## 9. 演进方向

v1 §9 的横向/纵向演进框架继续有效，Template 落地后各方向的依托点更新
如下：

- **横向（生态覆盖）**：新数据格式止步于 `FormatReader` + sidecar 语义
  声明；新模型止步于 entry + profile（含 interface 段）；新本体止步于
  embodiment profile + platform adapter。Day-0 支持新 VLA 的速度是社区
  心智的关键——资产按轴独立后，接入一个新模型不再牵动数据与本体两轴。
- **纵向（stage 谱系）**：SFT → verifier/偏好优化 → RL → 部署反馈闭环。
  轻量方法进程内（NPU 友好、无仿真器在环），重型 online RL 走编排后端；
  两层结构对冲上游成熟度风险（§6.3）。
- **词表治理**：受控词表是规范的一部分，新增取值走评审（语义是否与
  既有值正交、是否可被匹配规则消费），只加不删；删除/改义是 breaking
  change，需走版本迁移。
- **规范外溢**：一份设计良好、带校验器、带内置库的 embodiment template
  规范，配上"从数据到部署同一份事实源"的机制，具备成为社区事实标准的
  潜质——事实标准是框架最深的护城河。

---

## 附录 A. 与 LlamaFactory 的概念对照

| LlamaFactory | VLA Factory v2 | 差异要点 |
|---|---|---|
| Model Loader | model 层（registry entries + 加载选项插槽） | VLA 需统一异构动作头；量化重心在部署侧 |
| Data Worker + chat template | Embodiment Template（三声明 + 推导器） | 两方绑定 → 三方绑定；模板服务训练+rollout+部署 |
| Trainer（PT/SFT/RLHF/DPO 谱系） | 训练层 stage 轴（sft 今、preference/rl 规划） | 轻量进程内 / 重型作业编排两种形态 |
| TRL（进程内库依赖） | RLinf（作业级编排依赖）`[planned]` | 库嵌入 vs 编排，依赖形态不同 |
| `dataset_info.json` | dataset_schema（sidecar meta） | 增语义层：约定/单位/血统/指纹 |
| `template.py` 注册表 | model_interface（profile `interface:` 段） | 增 BaseContract 动态合并 |
| （无对应物） | embodiment_profile | VLA 特有的第三条轴 |
| vLLM/SGLang 导出 | deploy 层（边缘闭环控制） | VLA 部署是一等模块与生态位 |

## 附录 B. 术语表

| 术语 | 定义 |
|---|---|
| Embodiment Template | 三类声明资产 + BindingResolver + BindingPlan 组成的语义契约子系统，本框架核心设计 |
| dataset_schema | 数据轴声明："我有什么"。IR 语法层（自动推导）+ 语义 sidecar |
| embodiment_profile | 本体轴声明："物理上是什么"。框架内置库，每型号一份 |
| model_interface | 模型轴声明："我需要什么"。profile 静态段 + BaseContract 动态事实 |
| BaseContract | 从预训练 checkpoint `config.json` 读出的输入契约（v1 已有） |
| BindingResolver | 纯函数推导器：三份声明 → 确定性变换计划 |
| BindingPlan | 有序变换计划；可打印（plan 命令）、可 diff、可 override、必冻结 |
| binding snapshot | 冻结进 `inference_metadata/binding.json` 的计划产物态 |
| 受控词表 | 框架统一定义的语义枚举（视角/控制模式/旋转/夹爪/对齐），匹配的前提 |
| stage | 训练阶段轴：sft（现）、preference/rl（规划）；两种执行形态见 §6.3 |
| 退化规则 | 缺 v2 声明时行为与 v1 完全一致的兼容承诺 |

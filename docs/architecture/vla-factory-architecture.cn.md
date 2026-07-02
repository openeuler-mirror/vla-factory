# VLA Factory 架构设计

## 0. 总览

VLA Factory 是一个 recipe 驱动的机器人视觉-语言-动作模型训练与部署框架。它用一份 YAML recipe 描述模型、数据、微调策略、训练参数和输出目录，由框架完成从数据读取、样本构造、模型适配、训练、checkpoint 产物写入到在线推理服务的闭环。

系统的核心定位不是重新实现各类 VLA 或模仿学习模型，而是提供一层稳定的工程胶水：

- 对用户：用统一 recipe 启动训练、评估、推理和部署。
- 对数据：把不同数据格式转成统一的数据中间表示和训练样本。
- 对模型：通过薄适配器包装外部生态中的模型实现。
- 对训练：复用 PyTorch、HuggingFace Trainer、上游模型库的成熟能力。
- 对部署：通过统一推理引擎和平台 adapter 对接仿真器与真机。

当前实现的主链路是：

```text
YAML recipe
    -> TrainRecipe
    -> model registry
    -> data reader / codec / transforms / sampler
    -> VLADataset / DataLoader
    -> VLATrainer
    -> checkpoint + inference_metadata
    -> InferenceEngine
    -> ZMQ serve / dataset infer / evaluate
```

该文档描述的是 VLA Factory 的设计原则，架构边界，核心模块设计以及未来演进方向。

### 目录

- [0. 总览](#0-总览)
- [1. 背景与挑战](#1-背景与挑战)
  - [1.1 统一框架的核心诉求](#11-统一框架的核心诉求)
- [2. 设计原则](#2-设计原则)
- [3. 全局架构](#3-全局架构)
  - [3.1 分层架构图](#31-分层架构图)
  - [3.2 代码目录结构](#32-代码目录结构)
- [4. 配置系统](#4-配置系统)
- [5. 核心模块设计](#5-核心模块设计)
- [6. 依赖管理策略](#6-依赖管理策略)
- [7. 可靠性设计](#7-可靠性设计)
- [8. 测试策略](#8-测试策略)
- [9. 扩展与演进](#9-扩展与演进)

---

## 1. 背景与挑战

VLA Factory 的核心诉求是提供一个统一框架，解决机器人 VLA 模型工程链路高度碎片化的问题。这里的“统一”不是把所有模型、数据格式和运行平台改造成同一种实现，而是在数据、模型、训练、产物和验证之间建立稳定的工程契约，让不同来源的能力可以通过清晰边界接入同一条工作流。

### 1.1 统一框架的核心诉求

机器人策略模型从数据准备到验证通常不是单点任务，而是一条完整工作流：数据转换、训练配置、模型适配、checkpoint 管理、离线评估、仿真验证和真机验证。当前这些环节往往依赖大量人工脚本和临时约定，不同模型、不同数据集、不同实验之间难以复用。

具体问题包括：

- 数据格式不统一：LeRobot、HDF5、RLDS、ROS bags 等格式在图像、状态、动作、episode 边界和统计量表达上各不相同。
- 训练配置不统一：每个模型生态通常自带配置系统、训练入口、checkpoint 布局和超参命名，迁移实验时需要重新适配。
- 模型接口不统一：不同上游模型对 observation、action、loss 计算、动作预测和 checkpoint 组织方式的假设不同，很难直接复用同一套训练与评估代码。
- 产物契约不统一：训练完成后需要哪些 metadata、schema、norm stats 和配置快照，往往依赖项目约定，导致复现、调试和后续验证成本较高。

因此，VLA Factory 旨在提供一套统一的 recipe、数据中间表示、模型注册表、训练入口和产物契约，把人工适配沉淀为可复用的模块边界。框架的首要目标是让用户用同一种方式描述实验、接入数据、选择模型、启动训练、生成产物和复用结果；其他能力应建立在这套统一契约之上，而不是另起一套并行路径。

---

## 2. 设计原则

### 2.1 Recipe 驱动

一次训练应由 recipe 完整描述。模型选择、数据路径、采样窗口、动作空间、微调策略、训练步数、输出目录等都来自配置，而不是散落在脚本里。

authoring recipe 是用户最高优先级的配置入口。VLA Factory 暴露的所有可配置行为，都应能从 recipe 表达出来：通用能力放在顶层字段，模型专属能力放在 `model.config` 子树中。这样实验配置是可审计的，用户能在一份文件里看到本次实验主动覆盖了什么，而不是到脚本和隐式默认值里追踪行为来源。

`vla_factory/config/model/<name>.yaml` 这类模型默认 profile 是 `model.config` 子树的默认值来源，不是运行时并行存在的另一份配置。它承载模型相关的基线设置，例如上游模型超参和默认 transform pipeline。训练入口会把选中的模型 profile 与 recipe 中的 `model.config` 做深度合并；recipe 中显式写出的字段优先于 profile 默认值，如果 CLI 对相关字段提供临时覆盖，则 CLI 优先级最高。合并后的 `model.config` 是数据 transform、模型 adapter、训练和部署共同消费的唯一模型配置。

CLI 可以提供少量临时 override，例如 `--steps`、`--batch-size`、`--output-dir`，用于 smoke test 或调试，但 recipe 仍是主契约。

### 2.2 适配优于复现

VLA Factory 不持有上游模型架构代码。模型能力通过 registry entry 暴露，每个 entry 只负责：

- 声明 `ModelMetadata`。
- 解析 recipe 与 dataset schema。
- 构造上游模型对象。
- 在 VLA Factory 的 `Observation` / action tensor 与上游模型输入输出之间做格式转换。

这样可以减少自写模型引入的细微行为偏差，也能让上游生态更新时保持较低维护成本。

### 2.3 协议不假设模型结构

统一模型协议只要求两个核心能力：

- `compute_loss(observation, actions, ...)`
- `predict_actions(observation, ...)`

参数访问、设备迁移、训练模式等能力按 backend 扩展，例如 PyTorch 模型实现 `parameters()`、`named_parameters()`、`train()`、`to()`。框架不要求所有模型都暴露相同的内部模块。

### 2.4 数据契约与模型解耦

数据模块输出统一的 observation/action 样本，模型模块只消费抽象后的 `Observation` 和 action tensor。数据格式中的字段路径、视频编码、episode 索引、统计量、向量 key 顺序都不应泄漏到模型实现内部。

### 2.5 依赖按需安装

核心包保持轻量。ACT、OpenPI、GR00T 等上游生态依赖应通过 optional extras 引入。模型未被使用时，缺少该模型依赖不应影响其他模型的注册、训练和部署。

---

## 3. 全局架构

### 3.1 分层架构图

![VLA Factory 分层架构图，根据 ../graph/architecture-text.md 生成](../graph/vla-factory-layered-architecture.cn.svg)

五个分层：**用户表达层（recipe）→ 外部数据解析层 → 数据中间表示层 → 训练层 → 部署层**。
recipe 是中心枢纽，其余四层都消费它。

数据模块的训练数据流、部署推理流以及对应图示详见 [数据模块设计](../modules/data-module.cn.md#1-数据流全景)。

### 3.2 代码目录结构

当前核心代码位于 `vla_factory/`。该结构只描述相对稳定的目录边界和模块职责；具体文件名会随着实现演进新增或调整，架构文档不维护文件级清单。

```text
vla_factory/
├── examples/                      # recipe 示例和最小运行样例
├── docs/                          # 架构、使用说明和设计记录
├── config/                        # recipe 解析、默认值合并和运行时配置契约
│   └── ...
├── data/                          # 数据格式接入、中间表示、采样、transform 和 dataloader
│   ├── formats/                   # 外部数据格式 reader
│   ├── codec/                     # 视频解码与帧缓存
│   ├── transforms/                # 归一化、resize、padding 等 transform
│   ├── sampling/                  # 采样策略
│   └── ...
├── model/                         # 模型协议、metadata、registry 和上游模型 adapter
│   ├── protocols/
│   ├── registry/
│   └── ...
├── training/                      # 训练编排、Trainer 集成和微调策略
│   ├── strategies/
│   └── ...
├── deploy/                        # 推理引擎、平台 adapter、transport 和动作执行策略
│   └── ...
├── utils/                         # 跨模块共享的常量、工具函数和轻量辅助能力
│   └── ...
└── test/                          # 单元测试、契约测试和集成 smoke test
```

---

## 4. 配置系统

配置系统的职责是把用户可读的 YAML recipe 转成训练和部署都能消费的结构化对象。它同时服务两类需求：普通用户可以只写少量关键字段启动实验，资深用户也可以在 recipe 中覆盖更细粒度的模型、数据预处理和训练参数。

因此，VLA Factory 在概念上区分两种配置形态：

- authoring recipe：用户手写或维护的 recipe，只表达本次实验的显式意图。
- resolved recipe：运行时由 CLI、用户 recipe、模型默认 profile 和通用默认值合并后的完整配置，代表本次运行真正使用的最终配置。

训练产物中只保存 resolved recipe，并统一命名为 `recipe.yaml`。这样部署侧只需要读取一个配置文件，不需要判断“原始配置”和“最终配置”哪一个才是事实来源。

### 4.1 Recipe 结构

一个 recipe 通常包含以下顶层块：

```yaml
model:
  name: act
  path: null
  config:
    transforms:
      inputs:
        - type: image_to_float
          range: [0, 1]
        - type: image_layout
          to: CHW
        - type: image_normalize
          mode: imagenet

action_spec:
  action_dim: 6
  action_horizon: 100
  action_type: joint_pos

data:
  source:
    path: /path/to/dataset
    format: lerobot-v3
    video_codec: auto
  sampler:
    type: sliding_window
    n_obs_steps: 1
    action_horizon: 100
  split:
    strategy: episode
    train_ratio: 0.9
    val_ratio: 0.1
    seed: 42

finetuning:
  strategy: full

training:
  backend: pytorch
  lr: 1.0e-4
  batch_size: 8
  total_steps: 10000

output:
  output_dir: outputs/default
```

`vla_factory/config/recipe.py` 定义了 `TrainRecipe`、`ActionSpecConfig`、`DataConfig`、`SamplerConfig`、`SplitConfig`、`OutputConfig` 等 dataclass。`parser.py` 负责从 YAML 构造这些对象。

recipe schema 应覆盖框架允许定制的主要能力，包括模型选择、动作空间、数据源、采样策略、划分方式、微调策略、训练参数、输出目录，以及模型专属的 `model.config`。其中 `model.config` 是模型级扩展区，用于承载不同模型的专属超参和预处理配置。用户不需要在每个 recipe 中写满所有字段；只有当需要覆盖默认行为时，才写对应字段。

### 4.2 配置来源与优先级

配置系统需要同时支持通用训练字段和模型专属字段。配置合并遵循“越接近本次运行，优先级越高”的原则：

| 优先级 | 来源 | 作用范围 | 说明 |
|---|---|---|---|
| 1 | CLI 显式指定 | 本次运行的临时覆盖 | 最高优先级，用于 smoke test、调参和临时改输出目录。 |
| 2 | YAML recipe | 本次实验配置 | 用户主要配置入口，描述模型、数据、动作空间、训练策略和输出。 |
| 3 | 模型默认 profile | 模型专属默认值 | 位于 `vla_factory/config/model/<name>.yaml`，例如 `act.yaml`。用于承载不同模型的默认超参、transform pipeline、预处理偏好等。 |
| 4 | 通用默认值 | 框架级兜底 | 来自 `TrainRecipe` 及子 dataclass 的默认值，保证最小 recipe 仍可解析。 |

训练入口 `train()` 当前支持覆盖：

- `override_steps`
- `override_batch_size`
- `override_output_dir`

模型专属超参放在 `model.config` 字段中。不同模型可能有不同的默认值和处理细节，例如默认 transform 列表、动作 horizon、backbone 学习率或上游模型 config 参数。为了避免把这些差异塞进通用 recipe schema，模型 entry 可以读取对应的模型默认 profile，例如 `vla_factory/config/model/act.yaml`，再与 recipe 中的 `model.config` 做深度合并。

当前设计中，模型默认 profile 是“模型级实验基线”，不是不可变协议。它的作用是给某个模型提供合理默认值；一旦 YAML recipe 明确指定同名字段，应以 recipe 为准；如果 CLI 对相关字段提供覆盖，则以 CLI 为准。这样可以同时支持简单开箱配置和精确实验复现。

对于资深用户，定制路径不是复制整份模型 profile，而是在 recipe 中局部覆盖需要修改的字段。例如：

```yaml
model:
  name: act
  config:
    transforms:
      inputs:
        - type: image_to_float
          range: [0, 1]
        - type: image_layout
          to: CHW
        - type: image_normalize
          mode: imagenet
        - type: resize_images
          height: 320
          width: 240
```

这种方式保留了 authoring recipe 的可读性：用户能清楚看到本次实验真正改了什么；训练产物中的 `recipe.yaml` 会记录完整展开后的最终配置，满足复现和排查需求。

### 4.3 配置边界

配置系统只负责表达用户意图，不负责执行重型逻辑：

- 不在 parser 内创建模型。
- 不在 parser 内扫描数据集。
- 不在 parser 内导入可选模型依赖。
- recipe 支持细粒度覆盖，但不要求用户手写所有细节；模型默认 profile 提供模型相关默认值，训练产物中的 `recipe.yaml` 记录最终完整配置。

这种边界能让配置解析保持快速、可测试，也避免用户需要理解每个模型的内部预处理细节。

### 4.4 训练产物中的配置

训练开始前，`training/train.py` 会把部署需要的元数据写入输出目录的 `inference_metadata/`。这些文件用于推理、serve、复现和问题排查：

- `recipe.yaml`
- `schema.json`
- `norm_stats.json`

其中 `recipe.yaml` 保存的是 resolved recipe，也就是合并 CLI 覆盖、用户 recipe、模型默认 profile 和通用默认值之后的最终配置。部署、复现和问题排查都以这份 `recipe.yaml` 为准；用户原始 authoring recipe 可以由实验管理系统或版本控制保存，但不作为 checkpoint 的部署契约。这样 `InferenceEngine` 只需要读取一个配置文件，也保证了中间 checkpoint 能被直接加载，避免“只有训练结束后的 final checkpoint 才能部署”的限制。

---

## 5. 核心模块设计

### 5.1 数据模块

数据模块负责把外部数据集解析为 VLA Factory 的 Canonical IR，并进一步通过 transform pipeline、训练样本构建与批处理形成训练 batch；视频解码作为样本读取过程中的可替换能力使用。它同时为部署侧保存并复用 schema、norm stats 和 resolved recipe，保证训练与推理使用同一套数据契约。

详细设计见 [数据模块设计](../modules/data-module.cn.md)，其中展开说明：

- 外部数据解析层和数据中间表示层的职责边界。
- `FormatReader`、`DataSchema`、`NormStats`、`Episode`、`Frame`、`VideoRef`、`DatasetManifest` 等核心对象。
- 数据变换流水线、训练样本构建与批处理的设计，包括样本读取过程中的视频解码策略。
- 新增数据格式、视频解码策略和 transform step 的扩展方式。

### 5.2 模型抽象模块

模型抽象模块的目标是隔离模型实现差异，让训练和部署只依赖最小协议。

#### 5.2.1 ModelMetadata

`ModelMetadata` 是模型的静态能力描述，包括：

- 模型名称
- backend 类型
- action dim / horizon
- action head 类型
- architecture 类型
- training paradigm
- 可训练组件映射
- 是否需要 prompt
- image size
- 支持的微调方式
- 默认 transform 列表
- 安装提示

训练模块通过 metadata 做调度，不直接猜测模型内部结构。换句话说，VLA模型通过ModelMetadata做为描述，构造的输入是数据的schema以及训练的recipe。

#### 5.2.2 VLAModel Protocol

统一协议只定义训练和推理所需的最小方法：

```python
compute_loss(observation, actions, ...)
predict_actions(observation, **kwargs)
```

PyTorch 模型额外实现 `parameters()`、`named_parameters()`、`train()`、`to()`，用于优化器、冻结策略、设备迁移和 Trainer。

#### 5.2.3 Registry

模型通过装饰器注册：

```python
@register_vla(ModelMetadata(name="act", ...))
def load_act(recipe, schema):
    ...
```

`get_entry(name)` 在首次访问时 lazy import `model/registry/entries/*`，触发各 entry 的注册。registry loader 会把 entry 导入失败视为真实错误抛出，避免把语法错误或硬依赖缺失伪装成“模型未注册”。

可选依赖的缺失应在 factory 调用时给出清晰错误。例如 ACT 的 entry 可以被注册和列出，但真正创建 ACT 模型时，如果未安装 lerobot，应提示用户安装 `[act]` extra。

#### 5.2.4 Thin Adapter

每个模型 entry 应作为薄适配器存在。以 ACT 为例：

- 上游 `lerobot` 持有 ACTPolicy 和网络结构。
- VLA Factory 的 wrapper 只负责把 `Observation` 转成 lerobot batch dict。
- loss 和 action chunk 预测调用上游 policy。
- checkpoint 加载处理 wrapper 与上游模型 key prefix 差异。

这个边界要求 VLA Factory 不把上游模型代码复制进仓库，也不在 adapter 中重写模型细节。

### 5.3 训练模块

训练模块的入口是 `vla_factory/training/train.py` 的 `train()`。

训练流程：

```text
parse recipe
    -> prepare output_dir
    -> read schema / norm_stats
    -> resolve state/action vector keys
    -> save inference_metadata
    -> create model from registry
    -> apply fine-tuning strategy
    -> create dataloaders
    -> build TrainingArguments
    -> VLATrainer.train()
    -> save final/model.pt
```

#### 5.3.1 Fine-tuning Strategy

微调策略负责决定哪些参数可训练。它应基于 `ModelMetadata.components` 和 `named_parameters()` 操作参数，而不是依赖硬编码模型类型。

当前核心策略包括：

- `full`：全参数训练。
- `freeze`：冻结指定组件。
- `selective`：只训练指定组件。
- `lora`：面向支持 LoRA 的模型扩展。

ACT 从零训练通常使用 `full`；预训练 VLA 模型可使用 full、freeze、selective 或 LoRA。

#### 5.3.2 VLATrainer

`VLATrainer` 是 HuggingFace `Trainer` 的薄子类。它的职责是把 data pipeline 产出的 batch：

```python
{
    "observation": Observation,
    "actions": Tensor,
    "action_is_pad": Tensor | None,
}
```

桥接到：

```python
model.compute_loss(observation, actions, action_is_pad=...)
```

Trainer 生态提供混合精度、梯度累积、checkpoint、日志、优化器调度等能力。VLA Factory 只补充 VLA batch 适配、辅助 loss logging 和 `lr_backbone` 参数组。

#### 5.3.3 Checkpoint 与 Final Model

训练开始前，`training/train.py` 会把部署需要的元数据写入输出目录的 `inference_metadata/`。训练中间 checkpoint 由 HF Trainer 写入。训练结束后，框架额外写入：

```text
<output_dir>/final/model.pt
```

推理加载时会按优先级查找 final 权重、根目录权重、safetensors 或最近的 `checkpoint-*`。

### 5.4 部署模块

部署模块的目标是把训练产物转成平台可调用的实时策略服务。

#### 5.4.1 InferenceEngine

`InferenceEngine` 是部署核心。初始化时执行：

1. 从 checkpoint 目录加载 `inference_metadata`。
2. 根据 recipe 和 schema 创建模型。
3. 解析 checkpoint 权重路径并加载 state dict。
4. 构造与训练一致的 preprocessor 和 postprocessor。
5. 解析 camera key、state key、action key。
6. 初始化 action chunk 执行策略状态。

推理时，平台 observation 先转成 `ObsDict`：

```python
ObsDict(
    video={"front": np.ndarray, ...},
    state=np.ndarray | None,
    language=str | None,
)
```

然后进入：

```text
ObsDict
    -> Observation
    -> preprocessor
    -> model.predict_actions
    -> postprocessor
    -> raw action array
```

#### 5.4.2 Action Chunk 执行策略

当前支持三种策略：

- `synchronous`：一次返回完整 action chunk。
- `temporal_ensembling`：对重叠 chunk 做时间集成，返回单步动作。
- `receding_horizon`：预测一个 chunk，执行其中若干步后重新预测。

`receding_horizon` 对 ACT 这类 chunked policy 很重要，因为关键动作可能出现在 chunk 深处，不能每次只取第一步。

#### 5.4.3 Platform Adapter

部署 adapter 负责平台线协议与 `ObsDict` / action dict 的互转。

仿真器路径使用通用 ZMQ transport，约定 observation 中包含图像和 state 字段。

lerobot 真机路径使用 `LerobotHostObsAdapter` 和 `LerobotHostActionAdapter`：

- observation adapter 把逐电机 state 标量和 base64 图像转成 `ObsDict`。
- action adapter 把 action 向量按 `action_keys` 还原成逐电机命令。

state/action key 顺序来自训练时解析出的 schema 与 recipe 契约，不能在部署时临时排序生成。

---

## 6. 依赖管理策略

依赖管理遵循“核心轻量、生态按需”的原则。

### 6.1 Core Dependencies

核心依赖只覆盖配置解析、数据管线、PyTorch 训练基础、CLI 和通用部署能力。核心包不应默认安装所有模型生态依赖。

### 6.2 Optional Extras

模型依赖通过 optional extras 暴露，例如：

```bash
pip install -e ".[act]"
pip install -e ".[all]"
pip install -e ".[dev]"
```

`ModelMetadata.install_hint` 用于在缺少依赖时给出明确提示。CLI 的 `list` 命令应能列出已注册模型及其安装提示。

### 6.3 Adapter Dependency Boundary

模型 entry 模块可以被导入，不代表上游模型依赖必须已经安装。推荐做法是：

- entry 顶层只导入 VLA Factory 内部稳定模块。
- 上游模型库在 factory 内延迟导入。
- 缺失 optional dependency 时抛出清晰 ImportError。
- 真正的 entry 导入错误由 registry loader 显式报错。

### 6.4 不复制上游模型代码

VLA Factory 不维护 `vendor/` 模型实现。上游模型应来自 pip extra 或用户环境中的可安装包。adapter 中如果需要处理上游版本差异，应保持局部、可删除、可测试。

---

## 7. 可靠性设计

> TODO：本章描述的是后续需要实现和完善的可靠性设计，目前作为架构目标和实现检查清单保留。

可靠性设计围绕数据、训练和部署三条链路展开。目标不是只在最终失败时抛出异常，而是在数据准备、训练运行和部署推理过程中尽早发现异常，并把可定位的信息反馈给用户。

### 7.1 数据可靠性

数据可靠性关注“进入训练的数据是否结构正确、数值合理、语义一致”。这部分检查应尽量发生在 reader、manifest、dataset 和 transform 的边界上，避免明显异常的数据进入长时间训练。

主要检查包括：

- schema 检查：确认 state/action 维度、camera 列表、episode 数量、frame 数量符合 recipe 和模型要求。
- state/action key 顺序检查：确认 key 顺序可解析，并写入 schema。state/action 向量的维度顺序是数据与机器人之间的强契约，部署阶段应优先复用训练阶段解析出的顺序。
- episode split 检查：默认以 episode 为单位做 train/val split，降低相邻帧泄漏导致的虚假验证效果。
- 样本索引检查：确认滑窗采样不会产生越界样本，train/val split 不为空。
- 数值检查：确认 state/action 中不存在 NaN、Inf 或明显超出预期范围的值。
- 图像检查：确认图像能解码，shape、通道数和 dtype 符合 transform 预期。
- 统计量检查：确认 norm stats 与数据维度一致。

对于可自动修复的问题应记录 warning；对于会破坏训练语义的问题应直接失败。如果缺少 state/action key 信息，系统可以对不需要逐 key 还原的平台保持宽松，但对 lerobot 真机这类必须按电机名发命令的平台，应在 adapter 构造或发送前明确失败。

### 7.2 训练可靠性

训练可靠性关注“训练过程是否正常、产物是否可复现、checkpoint 是否能直接用于推理”。训练阶段不仅要跑完，还要持续暴露关键状态，避免用户训练数小时后才发现配置或数据问题。

训练过程需要监控：

- batch 检查：记录 batch 中 observation/action 的 shape、dtype、padding 比例等摘要。
- loss 检查：检测 NaN、Inf、异常尖峰或长期不下降。
- 梯度检查：可在 debug 或诊断模式下记录梯度范数、梯度裁剪情况。
- 指标日志：将 loss、辅助 loss、学习率和吞吐等指标输出到控制台或 TensorBoard/W&B。
- checkpoint 检查：确认 checkpoint 和 `inference_metadata` 写入成功。

训练产物应具备自描述能力。checkpoint 目录包含部署所需元数据，包括 `recipe.yaml`、`schema.json` 和 `norm_stats.json`。这样可以：

- 用中间 checkpoint 做推理。
- 离线评估 checkpoint。
- 避免用户手动记忆训练时的数据统计量。
- 减少部署时传错 recipe 的风险。

默认模式应避免过多日志影响训练速度；debug 模式可以开启更详细的 batch、transform 和模型调用信息。

### 7.3 部署可靠性

部署可靠性关注“训练产物能否被正确加载，推理链路是否和训练一致，模型输出是否能安全转换成平台动作命令”。

首先，训练和部署应共用 transform 注册与构建逻辑。部署时的 preprocessor/postprocessor 从 checkpoint 中的 recipe、schema、norm stats 和模型 metadata 构造，避免手写一套独立归一化逻辑。

部署阶段需要检查：

- shape 检查：确认模型输出 action shape 与 `action_horizon`、`action_dim` 一致。
- 数值检查：确认 action 中不存在 NaN、Inf。
- 范围检查：如果 `action_spec` 或平台 adapter 声明了动作上下界，应在发送前进行检查、裁剪或拒绝输出。
- key 映射检查：确认 action 向量能完整映射到平台要求的 action keys。
- 频率检查：记录推理耗时、控制循环频率和 observation 超时情况。
- 异常策略：当输出非法动作时，应明确失败或进入安全 fallback，而不是静默发送不可信命令。

这些检查的目标不是替代机器人底层安全控制，而是在 VLA Factory 这一层尽早发现模型、配置和协议适配问题。

### 7.4 日志与反馈机制

日志系统贯穿数据、训练和部署三条链路，用于记录一次运行中的关键事件、配置摘要、数据摘要、训练状态和部署状态。日志应支持不同级别，例如 `INFO`、`WARNING`、`ERROR` 和 `DEBUG`：

- `INFO`：记录正常运行进度，例如 recipe 解析结果、数据集 episode/frame 数量、模型名称、训练步数、checkpoint 保存位置。
- `WARNING`：记录可以继续运行但需要用户关注的问题，例如缺少可选统计量、某些日志后端未安装、数据字段存在轻微不一致。
- `ERROR`：记录导致当前任务失败的问题，例如 checkpoint 缺失、模型未注册、数据维度不匹配。
- `DEBUG`：记录更细粒度的调用细节和关键参数，例如 reader 解析到的字段、transform pipeline 组成、schema 内容、adapter 输入输出 shape。

日志应尽量包含“发生了什么、在哪个模块发生、当前输入摘要是什么、建议用户检查什么”。关键失败应尽早暴露：

- 未注册模型时列出可用模型。
- registry entry 导入失败时报告具体模块和异常。
- 缺少 optional dependency 时提示安装对应 extra。
- checkpoint 不存在或找不到权重时列出期望路径。
- 部署 observation 缺少 camera 时报告缺失 camera 和可用 camera。
- action/state 维度不匹配时在 transform、adapter 或 model wrapper 中显式报错。

视频帧缓存用于降低训练期间重复解码开销。`DataLoader` 当前 MVP 使用单进程 worker，后续可以在保证 codec 和 reader 线程/进程安全后扩展 worker 数量。

---

## 8. 测试策略

> TODO：本章描述的是后续需要补齐的测试策略，目前作为测试覆盖目标和回归检查清单保留。

测试应覆盖从配置解析到训练、推理和部署 adapter 的关键契约。

### 8.1 配置测试

配置测试关注：

- YAML 能解析成 `TrainRecipe`。
- 默认值符合预期。
- 嵌套配置结构稳定。
- CLI override 能正确影响训练参数。
- 顶层字段与嵌套字段的兼容策略明确。

### 8.2 数据管线测试

数据测试关注：

- reader 能读取 schema、norm stats 和 episode 信息。
- manifest 样本数、split 和索引范围正确。
- transform pipeline 的 normalize、resize、padding 行为正确。
- `VLADataset` 输出的 observation/action shape 符合模型预期。
- `collate_fn` 能处理 batch 聚合。

### 8.3 模型注册与 Adapter 测试

模型测试关注：

- registry 能发现模型 entry。
- 重复注册会失败。
- 缺少 optional dependency 时错误信息清楚。
- wrapper 能实现 `compute_loss` 和 `predict_actions`。
- state dict 保存和加载能 round trip。

### 8.4 训练 Smoke Test

训练测试不要求跑完整实验，但应覆盖最小步数训练：

- 小数据集。
- 小 batch。
- 少量 steps。
- 能写出 `inference_metadata`。
- 能写出 final 权重。
- loss logging 不保留 autograd graph。

### 8.5 推理与部署测试

推理测试关注：

- `InferenceEngine` 能从 checkpoint 加载 metadata 和权重。
- dataset sample inference 能输出正确 action shape。
- postprocessor 能还原原始动作尺度。
- `synchronous`、`temporal_ensembling`、`receding_horizon` 策略行为可预测。
- simulator 和 lerobot adapter 的输入输出 key 映射正确。

### 8.6 回归测试原则

凡是修复过的契约问题，都应固化为测试，尤其是：

- state/action key 顺序。
- action dim padding 与反向裁剪。
- checkpoint 路径解析。
- optional dependency 延迟导入。
- 训练与部署 transform 一致性。

---

## 9. 扩展与演进

VLA 是当前具身智能的主流路线之一，覆盖视觉、语言和动作三个模态。它未必是具身智能的最终模型形态，但在当前阶段具有很强代表性：学术界和工业界仍在围绕 VLA 的数据、模型、后训练和部署持续产生新方法。因此，VLA Factory 的演进目标不仅是做一个可用的微调工具，也是在 VLA 这条技术路线下探索基础软件应该如何设计。

换句话说，VLA Factory 是一个工程框架，也是一个研究载体。它借助统一的 recipe、数据契约、模型 adapter、训练引擎和部署引擎，持续研究以下问题：

- 数据如何被采集、清洗、标定、转换和复用。
- VLA 与模仿学习模型如何低成本微调、续训和后训练。
- 模型特有的技巧如何抽象成框架能力，让其他模型共享。
- 具身模型如何在端侧稳定、实时、安全地部署。
- 国产化软硬件环境下，训练和推理基础设施如何适配和优化。

### 9.1 横向演进：扩大生态覆盖

横向演进指的是扩大 VLA Factory 的生态适配范围。由于框架本身定位是胶水层，横向扩展的重点是接入更多数据格式、模型生态、训练策略和部署平台，让同一套 recipe 和部署接口覆盖更多真实场景。

横向扩展包括：

- 数据格式：从 LeRobot 扩展到 HDF5、RLDS、ROS bags、Zarr 和混合多源采样。
- 模型生态：从 ACT 扩展到 OpenPI、OpenVLA、GR00T、SmolVLA 等。
- 微调方式：从 full/freeze/selective 扩展到 LoRA、QLoRA、adapter tuning 和模型专属 tuning。
- 部署平台：从 ZMQ 仿真和 lerobot 真机扩展到更多机器人中间件、边缘设备和远程推理服务。
- 运行环境：从 CUDA 生态扩展到 OpenEuler + Ascend 等国产化环境。

横向扩展的工程量较高，适合借助 AI coding 和 loop engineering 等方式提高适配效率。但横向扩展不能以牺牲架构边界为代价：新增数据格式应停在 `FormatReader`，新增模型应停在 registry entry 和 adapter，新增平台应停在 deploy adapter 和 transport。

### 9.2 纵向演进：围绕真实场景做深

纵向演进指的是围绕一个真实需求或真实场景，把技术链路做深。具身智能的难点不只是“能不能接入某个模型”，而是从数据、微调、后训练到部署验证的完整闭环能否稳定工作。

纵向演进包括：

- 数据链路：录制数据自动标定、清洗、质量检查、统计量生成和格式转换。
- 微调链路：checkpoint 续训、不同模型的参数高效微调、跨数据集迁移和训练稳定性诊断。
- 后训练链路：从行为克隆扩展到 RL、偏好优化、失败样本挖掘和世界模型相关探索。
- 部署链路：端侧实时推理、动作 chunk 策略、频率控制、异常动作检测和安全 fallback。
- 评估链路：离线指标、仿真验证、真机验证和部署日志闭环。

纵向演进更强调技术积累，需要从实际场景不断迭代，而不是只做接口适配。尤其是部署方向，具身模型比 LLM/VLM 多了 action 层，并且常常运行在端侧闭环中，不能简单复用 LLM/VLM 的部署设施。动作输出的实时性、稳定性、合法性和安全边界，是具身基础设施需要单独研究的问题。

### 9.3 契约抽象：把模型特有技巧沉淀为框架能力

VLA Factory 的重要价值之一，是把某些模型特有的 trick 抽象成框架级能力，从而让其他模型复用。典型例子是 delta action：如果它最初只在某个模型中使用，但框架把动作变换、归一化、反归一化和部署还原抽象成统一 transform，那么其他 VLA 模型也可以基于同一套数据和训练契约尝试 delta action 微调。

这类抽象应遵循以下原则：

- trick 不直接写死在某个模型 adapter 内，而是沉淀到数据 transform、训练策略、action spec 或部署 postprocessor 中。
- 抽象后的能力应尽量跨模型复用，但允许模型通过 metadata 或 recipe 声明是否启用。
- 训练和部署必须共享同一套语义，不能只在训练侧生效。
- 每个抽象都应有可测试的输入输出契约，避免“看似通用，实际只服务一个模型”。

这种“契约抽象统一”是框架从胶水层走向基础设施的关键。它让 VLA Factory 不只是接模型，还能把新方法沉淀成可组合、可复用、可验证的基础模块。

### 9.4 部署推理演进

部署推理是统一框架向真实机器人闭环延伸后的重要演进方向，但它不是第一阶段构建框架的核心诉求。第一阶段应先保证训练产物、数据契约和模型协议稳定；在此基础上，部署推理可以围绕同一套 recipe、schema、norm stats 和 transform 继续深化。

该方向的重点包括：

- 推理一致性：训练和推理共用数据变换、归一化统计、camera/state/action key 顺序和 action spec。
- 实时控制：围绕端到端延迟、控制频率、动作 chunk 策略、缓存和异步执行做系统优化。
- 平台适配：通过 observation adapter、action adapter 和 transport 对接更多仿真器、机器人中间件和真机平台。
- 安全与可观测性：增加异常 observation 检查、动作合法性检查、频率监控、日志追踪和 fallback 策略。
- 部署评估：沉淀离线回放、仿真验证、真机验证和部署日志回流的统一评估方法。

这一方向的边界是：部署能力应复用训练阶段形成的统一契约，不应在部署侧重新定义一套独立的数据语义或模型输入输出协议。

### 9.5 国产化算力演进

国产化算力支持也是后续演进方向，而不是当前框架成立的前提。VLA Factory 的核心架构应先保持 backend、adapter 和 optional dependency 边界清晰，为后续在 OpenEuler + Ascend 等环境中验证训练、推理和部署链路预留空间。

该方向的重点包括：

- 训练 backend 适配：验证算子支持、混合精度、分布式训练、checkpoint 格式和性能调优方式。
- 上游模型兼容：识别 ACT、OpenPI、OpenVLA、GR00T 等模型生态中的 CUDA 隐式依赖，并通过 adapter 或依赖隔离降低迁移成本。
- 推理 runtime 验证：评估模型加载、数据预处理、动作后处理、通信协议和硬件 runtime 的端到端稳定性。
- 性能基线建设：建立 CUDA 与国产化环境下的数据加载、训练吞吐、推理延迟和控制频率对照。
- 工程经验沉淀：形成环境安装、问题定位、算子替代、精度差异和部署约束的可复用文档。

这一方向的边界是：国产化支持应通过 backend、adapter 和依赖管理逐步引入，不应让核心数据契约和模型协议绑定某一种硬件或系统环境。

### 9.6 扩展路径约束

无论横向还是纵向演进，都应遵守现有模块边界：

- 新增数据格式：实现新的 `FormatReader`，输出统一 schema、norm stats、episode 和 frame。
- 新增模型：添加 registry entry，声明 `ModelMetadata`，用薄 adapter 包装上游模型。
- 新增训练策略：通过 metadata components 或参数名规则选择参数，不写死模型内部结构。
- 新增部署平台：新增 observation adapter、action adapter 和必要 transport，不修改 `InferenceEngine` 核心预测逻辑。

演进过程中应坚持两个约束：主链路只依赖稳定契约，生态差异留在 adapter 内部。横向扩展负责扩大生态覆盖，纵向演进负责沉淀技术深度，两者互相垂直，可以并行推进。

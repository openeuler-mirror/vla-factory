# 用户表达层（recipe）模块设计

> 文档状态：**已对齐当前实现**（阶段 5 完成后）。架构文档描述目标态，可能超前于实现；
> 本文只描述**已经能跑**的行为。
> 对应架构：见 [总体架构 § 3 用户表达层](../architecture/vla-factory-architecture.cn.md#3-用户表达层) 与 [§ 2.2 目录结构 `recipe/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

用户表达层是 VLA Factory 的入口。它把用户可读的 YAML recipe 解析成训练和推理都能消费的
结构化对象（`TrainRecipe` 及子 dataclass），并提供 CLI 入口。

**recipe 只承载用户的选择，不承载三者之间的关系。** 哪个相机进模型的哪个视觉槽位、动作
向量怎么 padding、chunk 多长、图像 resize 到多大——这些都是数据、模型、机器人三份描述的
函数，由组合解析层推导（架构 §4.2）。recipe 里出现同样的字段，只会制造两个可能不一致的
答案。

## 1. 核心对象

| 对象 | 职责 |
|---|---|
| `TrainRecipe` 及子 dataclass | recipe 的结构化形式（`recipe.py`） |
| `parser.py` | YAML → `TrainRecipe`；未知键一律忽略 |
| `defaults.py` | `resolve_recipe()`：模型声明的 `params` 与 recipe 的 `model.config` 深合并（recipe 优先），并执行可调键 allow-list |
| `cli.py` | `vlafactory-cli`：train / preprocess / list / resolve / inspect / evaluate / infer / deploy |

## 2. 三个区

### 2.1 组合选择区

| 块 | 字段 | 说明 |
|---|---|---|
| `model` | `name`、`path` | 注册表中的模型名；`path` 微调必填，从零训练可省 |
| `data` | `path`、`format`、`video_codec` | 数据集位置与格式（`auto` 自动识别） |
| `robot` | `name` | 机器人本体 profile 名；留空表示不绑定本体。写了之后解析器会把数据的逐维关节名嵌入机器人关节表（产出 JointMapping），对不上直接失败——这正是它的价值 |

### 2.2 组合调整区（`assembly`，可选）

只在解析器无法唯一确定关系、或用户想用非默认策略时写。当前只有两个字段：

| 字段 | 说明 |
|---|---|
| `camera_mapping` | `{模型视觉槽位: 数据集相机}`。**给了就是完整声明**——没列出的槽位视为有意留空，走占位图 + zero mask，不再对它做自动推断（实测理由见 `phase3-mapping-and-t1-pipeline.cn.md` 修正 1） |
| `default_task` | 语言兜底。优先级：帧级 task 文本 > `default_task` > 空 prompt |

**本区只放解析器真正消费的 override**（`resolver.py:CONSUMED_OVERRIDES`，有一条测试守
着两者相等）。一个没有消费者的调整项等于「能写、但什么都不做」，所以频率与夹爪两项随
它们的兼容性检查一起推迟，不预留字段。

### 2.3 训练参数区

`finetuning`、`training`、`output`。与三者关系无关，改这里不会改变模型接口。

`data` 块下**只有数据集本身**（`path` / `format` / `video_codec`）：一个样本携带多少观测
帧、多少未来动作，两端都是模型的时序契约（`ModelMetadata.history_frames` → `ModelIOSpec.n_obs_steps`，以及 action horizon），
由解析器给出；训练/验证划分是框架固定策略（按 episode 9:1，见
`training/manifest.py:TRAIN_RATIO`）——训练期从不评估（`eval_strategy="no"`），这个旋钮
唯一的效果就是悄悄缩小训练集。

## 3. 模型可调项：`model.config`

模型声明一份 `ModelMetadata`：**具名字段是事实**（组合解析器读，recipe 永远不能覆盖），
**`params` 是这个模型的可调默认值**（recipe 的 `model.config` 深合并在上，recipe 优先）。
容器即属性——模型作者不需要给任何东西分类。

三道闸守着这个面（详见 `model-module.cn.md` §4.6）：

1. 模型没声明的键 → `resolve_recipe()` 报错并给出 `difflib` 候选；
2. 声明了却没人读的键 → 模型工厂报错（`utils/tracked_config.py`）；
3. 在 step config 里重复一个事实（图像值域、归一化方法、pad 目标、resize 尺寸）→
   `assembly/transforms/base.py` 报错。

`vlafactory-cli inspect model --name <model>` 打印每个可调项的生效值与来源。

## 4. 配置来源与优先级

| 优先级 | 来源 | 说明 |
|---|---|---|
| 1 | CLI 显式指定 | `--steps` / `--batch-size` / `--output-dir` |
| 2 | YAML recipe | 用户主要入口 |
| 3 | 框架默认值 | dataclass 默认值 + 模型声明的 `params` |

三者之间的**关系**不在这张表里——它由解析器从描述推导，不参与「后写覆盖前写」。

## 5. 没有兼容层

阶段 5 删掉的字段就是**过期配置**，不提供转换：仓库 0.1.0 早期、调用方全在仓内，维护
一层「旧拼法 → 新拼法」只会让两种写法长期并存，而它们随时可能给出不一致的答案。

`parser.py` 对未知键一律忽略，所以旧 recipe 不会崩——但那些键也不再有任何效果。已删除
的字段与它们如今的来源：

| 已删除 | 现在从哪来 |
|---|---|
| `action_spec.action_dim` | 数据集事实（`inspect data`）+ 模型 `dim_policy` |
| `action_spec.action_horizon` | 模型声明（from_scratch 走 `model.config.action_horizon`；预训练模型是家族事实） |
| `action_spec.action_type` | 数据侧 `action.dims[].mode` / 机器人 profile |
| `action_spec.bounds_low/high` | `RobotProfile` 安全边界 |
| `data.source.*` | 直接写在 `data` 下（少一层嵌套） |
| `data.sampler.*` | `ModelIOSpec.n_obs_steps` / `.action_horizon` |
| `data.split.*` | 框架固定策略（`training/manifest.py`） |
| `training.inference_steps` | `model.config.num_inference_steps` |
| `training.augmentation.*` | 无——声明了、被写进产物，却从未被任何 transform 应用 |
| `model.config.camera_mapping` / `default_task` | `assembly` 块 |
| `composition:` | `assembly:` |
| `assembly.accept_fps_mismatch` / `gripper_flip` | 无——对应检查未实现，从来没有效果 |

**迁移方式**：`vlafactory-cli resolve --config <recipe>` 打印解析器推出来的完整结果
（维度、chunk 长度、相机映射、两条 pipeline）。对着它把 recipe 里重复的字段删掉即可。

## 6. 训练产物中的 recipe

`train()` 把解析并合并后的 recipe 写进 `inference_metadata/recipe.yaml`。它在部署侧只提
供两件事：**模型名**（取 factory）与 **`model.config`**（可调项，如
`num_inference_steps`）。所有三者关系来自同目录的 `assembly.json`——recipe 不是组合事实
的来源（架构 §4.2.6）。

## 7. 扩展方式

- **新增训练参数**：加到 `training` / `output` 对应 dataclass，同步 `reference.yaml`。
- **新增模型可调项**：写进该模型的 `ModelMetadata.params`，并确保工厂或某个已注册的框架
  消费方真的读它（否则第 2 道闸会报错）。
- **新增受控 override**：同一个 commit 里加 `AssemblyConfig` 字段 **和**
  `CONSUMED_OVERRIDES` 条目，否则守护测试会红——这正是为了避免再出现「能写但没人读」的
  字段。
- **想加一个新的 recipe 字段之前先问**：它是「用户的选择」，还是「数据/模型/机器人三份
  描述的函数」？后者不属于 recipe——写进来就会有两个可能不一致的答案。
- **新增 CLI 子命令**：注意 `resolve` / `inspect` 必须在无 GPU、无可选 extras、无机器人
  连接的环境下可运行。

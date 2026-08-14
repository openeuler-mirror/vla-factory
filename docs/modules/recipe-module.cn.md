# 用户表达层（recipe）模块设计

> 文档状态：**已对齐当前实现**（阶段 5 完成后）。架构文档描述目标态，可能超前于实现；
> 本文只描述**已经能跑**的行为。
> 对应架构：见 [总体架构 § 3 用户表达层](../architecture/vla-factory-architecture.cn.md#3-用户表达层) 与 [§ 2.2 目录结构 `recipe/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

用户表达层是 VLA Factory 的入口。它把用户可读的 YAML recipe 解析成训练和推理都能消费的
结构化对象（`TrainRecipe` 及子 dataclass）。全局 CLI 位于 `vla_factory/cli.py`，
recipe 包只负责用户配置本身。

**recipe 只承载用户的选择，不承载三者之间的关系。** 哪个相机进模型的哪个视觉槽位、动作
向量怎么 padding、chunk 多长、图像 resize 到多大——这些都是数据、模型、机器人三份描述的
函数，由组合解析层推导（架构 §4.2）。recipe 里出现同样的字段，只会制造两个可能不一致的
答案。

## 1. 核心对象

| 对象 | 职责 |
|---|---|
| `train_recipe.py` | `TrainRecipe` 及子 dataclass；结构与公共 YAML 块一一对应 |
| `parser.py` | YAML → `TrainRecipe`；除 `model.config` / `finetuning.config` 外，未知键立即报错 |
| `model_config.py` | `merge_model_config()`：模型声明的 `params` 与 recipe 的 `model.config` 深合并（recipe 优先），并执行可调键 allow-list |
| `vla_factory/cli.py` | 全局命令入口：train / preprocess / list / resolve / inspect / evaluate / infer / deploy |

## 2. 三个区

### 2.1 组合选择区

| 块 | 字段 | 说明 |
|---|---|---|
| `model` | `name`、`path` | 注册表中的模型名；`path` 微调必填，从零训练可省 |
| `data` | `path`、`format`、`video_codec` | 数据集位置与格式（`auto` 自动识别） |
| `robot` | `name` | 可选机器人本体 profile 名。解析器校验 profile 自身及显式 control-mode 冲突，不用名称猜测 DataSchema 相机/关节对应 |

`model` 保留两种简写：`model: act` 等价于 `{name: act}`；带 `/` 的字符串同时表示
checkpoint 路径，最后一段作为默认模型名，例如 `model: lerobot/pi0` 等价于
`{name: pi0, path: lerobot/pi0}`。路径不能以 `/` 结尾；名称与路径不能按该规则对应时，
使用显式 mapping。

### 2.2 组合调整区（`overrides`，可选）

只在解析器无法唯一确定关系、或用户想用非默认策略时写。当前只有两个字段：

| 字段 | 说明 |
|---|---|
| `camera_mapping` | `{模型视觉槽位: 数据集相机}`。**给了就是完整声明**——没列出的槽位视为有意留空，走占位图 + zero mask，不再对它做自动推断（实测理由见 `phase3-mapping-and-t1-pipeline.cn.md` 修正 1） |
| `default_task` | 语言兜底。优先级：帧级 task 文本 > `default_task` > 空 prompt |

**本区只放解析器真正消费的 override**。parser 会根据 `AssemblyOverrides` 拒绝未知字段；
纯解析器也直接接收这个强类型对象，不再维护裸 dict、消费列表或平行枚举。一个没有消费者的调整项等于「能写、但什么都不做」，所以频率与夹爪两项随
它们的兼容性检查一起推迟，不预留字段。

### 2.3 训练参数区

`finetuning`、`training`、`output`。与三者关系无关，改这里不会改变模型接口。
`finetuning` 只有 `strategy` 与 `config`：前者选择已注册的
`FinetuningStrategy`，后者由该策略解析成自己的严格 config dataclass。LoRA、freeze
等参数不会继续堆进 `TrainRecipe`；新增策略只注册一个实现类。

`data` 块下**只有数据集本身**（`path` / `format` / `video_codec`）：一个样本携带多少观测
帧、多少未来动作，两端都是模型的时序契约（`ModelMetadata.history_frames` → `ModelIOSpec.n_obs_steps`，以及 action horizon），
由解析器给出。训练期目前不执行评估，因此所有 episode 都进入训练；等真实的 metric 与
evaluation loop 落地时，再一起引入 episode-level validation split，避免现在静默扣掉
一部分无人消费的数据。

## 3. 模型可调项：`model.config`

模型声明一份 `ModelMetadata`：**具名字段是事实**（组合解析器读，recipe 永远不能覆盖），
**`params` 是这个模型的可调默认值**（recipe 的 `model.config` 深合并在上，recipe 优先）。
容器即属性——模型作者不需要给任何东西分类。

三道闸守着这个面（详见 `model-module.cn.md` §4.6）：

1. 模型没声明的键 → `merge_model_config()` 报错并给出 `difflib` 候选；
2. 声明了却没人读的键 → 模型工厂报错（`utils/tracked_config.py`）；
3. 写入 `model.config.transforms` 试图改变 step 或顺序 → `merge_model_config()`
   明确报错；pipeline 由 resolver 从模型事实推导。

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

公共强类型块会拒绝未知键；旧 recipe 就是过期配置，不做字段翻译。已删除的字段与它们如今
的来源：

| 已删除 | 现在从哪来 |
|---|---|
| `action_spec.action_dim` | 数据集事实（`inspect data`）+ 模型 `dim_policy` |
| `action_spec.action_horizon` | 模型声明（from_scratch 走 `model.config.action_horizon`；预训练模型是家族事实） |
| `action_spec.action_type` | 数据侧 `action.dims[].mode` / 机器人 profile |
| `action_spec.bounds_low/high` | `RobotProfile` 安全边界 |
| `data.source.*` | 直接写在 `data` 下（少一层嵌套） |
| `data.sampler.*` | `ModelIOSpec.n_obs_steps` / `.action_horizon` |
| `data.split.*` | 当前无 split；全部 episode 用于训练，评估实现时再引入 |
| `training.inference_steps` | `model.config.num_inference_steps` |
| `training.augmentation.*` | 无——声明了、被写进产物，却从未被任何 transform 应用；`ModelMetadata.requires_augmentation` 也随之删除 |
| `model.config.camera_mapping` / `default_task` | `overrides` 块 |
| `composition:` / `assembly:` | `overrides:` |
| `overrides.accept_fps_mismatch` / `gripper_flip` | 无——对应检查未实现，从来没有效果 |

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
- **新增受控 override**：同一个 commit 里增加 `AssemblyOverrides` 字段和消费它的解析规则；
  resolver 直接把该属性交给真正负责它的规则，不要再增加允许字段列表。
- **想加一个新的 recipe 字段之前先问**：它是「用户的选择」，还是「数据/模型/机器人三份
  描述的函数」？后者不属于 recipe——写进来就会有两个可能不一致的答案。
- **新增 CLI 子命令**：注意 `resolve` / `inspect` 必须在无 GPU、无可选 extras、无机器人
  连接的环境下可运行。

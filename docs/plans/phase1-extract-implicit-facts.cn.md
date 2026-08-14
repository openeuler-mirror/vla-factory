# 开发计划：组合解析迁移阶段 1「抽取现有隐式事实」

> 状态：**WP0–WP7 已执行完毕**（工作树，未提交）。日期 2026-08-06 制定，2026-08-10 完成。
> **后续决策（2026-08-11）：** 本计划中 BaseContract / Materialize 的设计记录
> 已被撤销。当前实现以 ModelMetadata 为唯一模型接口事实源，checkpoint 只做
> 可选一致性检查；现状以架构文档和模型模块文档为准。
> **后续产物收敛：** 独立 `schema.json` / `norm_stats.json` 已删除；二者现在只作为
> `assembly.json` 的 `schema_ref` / `norm_stats_ref` 保存。正文中的旧文件名是阶段 1
> 当时的迁移记录，不代表当前产物布局。
> 依据：`docs/architecture/vla-factory-architecture.cn.md` §7.4 阶段1（ccb2ca8）；
> 字段级规范以 `feat_inspect` 分支 2223989 的
> [数据模块 §8](../modules/data-module.cn.md)、[模型模块 §4](../modules/model-module.cn.md)、
> 架构 §3.5（inspect）为准——见「前置决策 D1」。
>
> **执行中与原计划的三处偏差**（正文已就地更新）：
> 1. **D3 扩大为 D5**——不只是把重复事实从 profile 上提，而是整份 profile 并入
>    `ModelMetadata.params`，`recipe/model/` 目录删除，配套加了配置面三道闸。
>    起因是用户指出「扩展模型要写两个文件不够直接」，顺带发现 `adapters/act.py`
>    在模块顶层反向依赖 `recipe.defaults`（违反架构 §2.2）。
> 2. **多找到两个死配置**——`num_inference_steps` 与 `tokenizer_max_length` 写在
>    profile 里从来无人读取，改了不生效也无提示。闸2 正是为这类而设。
> 3. **发现并修复一处推理侧回归**——WP1 把 `state_keys` / `action_keys` 改成派生属性后，
>    `InferenceEngine.__init__` 仍在 `replace(schema, state_keys=...)`，导致**任何
>    checkpoint 都无法加载**，而全套测试保持绿色。已补
>    `test/test_train_infer_roundtrip.py`（唯一构造真实 `InferenceEngine` 的测试）。

## Context

阶段0（430e962）已落地：目录对齐目标架构、`RobotProfile` 与注册表、
`ResolvedAssembly` / `ResolutionError` 数据结构、`resolve_assembly()` 的
Load / Materialize / Validate 三阶段、`vlafactory-cli resolve` dry-run。
解析器目前只搬运事实——五类 Mapping 与三条 `TransformPipelineSpec` 都是
`resolved=False` 的空占位，`CanonicalInterface` 只有 5 个标量字段。

阶段1 的任务是**把事实喂饱**。§7.4 列了五件事：

1. `action_spec` 字段分流到 DataSchema / ModelMetadata / RobotProfile；
2. model config 中稳定的输入输出能力升入 ModelMetadata 接口部分；
3. 让 reader 补充可探测的数据语义；
4. 把 model adapter 中的关系假设提取为声明或解析规则；
5. 把 deploy adapter 中稳定的本体事实提取到 RobotProfile。

**完成判据**：阶段2 兼容性矩阵（架构 §4.2.2）的 11 行里，每行要比较的两/三侧事实，
在三份描述中都有确定的字段落点和来源标注。阶段2 只需写比较逻辑，不必再回头找事实。

**范围边界（本阶段不做）**：兼容性检查与 explain trace（阶段2）、Mapping 与
TransformPipelineSpec 生成（阶段3）、下游改吃具身组合（阶段4）、recipe 瘦身（阶段5）。
解析器只把新事实吸收进三者引用与 canonical interface，**不新增任何 pair 检查**。

**行为不变**：训练与部署产物与现在一致（新增日志 / warning 除外）。唯一例外是
WP4 的 ACT 相机错位修复——那是既有 bug，修复必然改变行为，单列一个 commit。

---

## 前置决策

### D1. 字段级规范先合回主线

data-module §8（DataSchema 字段表、准入原则、推断规则）、model-module §4
（ModelMetadata × BaseContract 归属划分、字段表、跨维度词表）、架构 §3.5（inspect）
是阶段1 的字段级规范，目前只在 `feat_inspect`（2223989）上，`ref_architecture` 上被
ccb2ca8 移除了。**实施前先把 feat_inspect 合回**，否则实现与文档分叉、评审无基准。
本计划的 WP1 / WP2 / WP6 直接实现这三章。

### D2. DataSchema 重构为逐条目表 + 派生属性兼容层

data-module §8.3 要求三类通道统一为逐条目表（cameras 逐相机、state/action 逐维），
取代现在靠下标对齐的平行数组。这会改变 `DataSchema` 结构，而它被序列化进
`inference_metadata/schema.json`，仓内有 12 个文件（8 个源码 + 4 个测试）读它的扁平字段。

采用：**新结构 + 只读派生属性**。`state_dim` / `action_dim` / `cameras` / `image_sizes` /
`state_keys` / `action_keys` / `has_language` / `robot_type` / `fps` / `total_episodes` /
`total_frames` 保留为从新结构算出的 `@property`，本阶段这些消费点一律不动
（符合「最小改动、触及范围内一致」）。兼容层的移除时点定在**阶段4**——下游改吃
`ResolvedAssembly` 时一并删除，届时不再有直接读 DataSchema 的调用方。

不引入 `schema_version` 字段（model-module §4.4 已否掉版本机制）：
`DataSchema.from_dict()` 以「顶层是否存在 `state_dim`」判定旧版扁平布局并升级，
判定确定、无需外部版本号。

### D3. 模型事实去重的落地模式（已被 D5 取代并扩大）

model-module §4.1 的纪律是「同一事实只能有一个来源」。
`image_to_float.range`、`image_normalize.mode`、`normalize_vector.method`、
`pad_dimensions.target_dim` 这些模型事实原本活在 baseline profile 的 transform 配置里。

落地形态：

```text
ModelMetadata 声明事实 → TransformContext 携带 → step.from_config 从 ctx.metadata 读取
                                                → 步骤配置里再出现该键即报错（不是覆盖）
```

**执行时扩大了范围（见 D5）**：不只是去掉重复键，而是整份 baseline profile
并入 `ModelMetadata.params`，`vla_factory/recipe/model/` 目录删除。

### D5. 载体边界规则：一个模型一个文件，容器即属性

D3 只搬事实、把默认值留在 profile YAML，结果是同一个模型的声明分居两个文件
（事实在 `.py`、默认值在 `.yaml`），模型作者还要先判断「这个键算事实还是算默认值」。
现有分层另带两处硬伤：`adapters/act.py` 在模块顶层 import `recipe.defaults`
（model 叶子层反向依赖用户表达层，违反架构 §2.2）；`resolve_recipe()` 与
`_resolve_*_config()` 各合并一次 profile。

因此：

- **默认超参并入同一份声明** —— `ModelMetadata.params`（模型专属超参 + 默认
  transform 步骤清单）。`recipe/model/` 目录、`load_model_defaults()`、
  `MODEL_CONFIG_DIR`、对应 package-data 全部删除。**一个模型 = 一个 entry 文件。**
- **容器即属性**，不需要逐参数标注：具名字段 = 事实（永不可覆盖），
  `params` 的键 = 超参（一律可被 `model.config` 覆盖）。
- **判定规则**：组合解析要读它 → 具名字段；改了会变接口语义 → 具名字段；
  其余 → `params`。checkpoint 自述的实例事实 → `BaseContract`。
- **配置面三道闸**（治「静默」，与分层正交）：未声明的键即报错（`resolve_recipe`
  用 difflib 给候选）、未被读取的键即报错（`utils/tracked_config.py`）、
  事实键被覆盖即报错（`assembly/transforms/base.py:reject_fact_override`）。
- `inference_num_steps` 具名字段下沉进 `params`（键名 `num_inference_steps`）——
  它名义上是框架级字段、实质上要可覆盖，容器归属与可覆盖性对不上。

放弃的两条：profile 外置的 diff-friendly 属性与 Hydra `defaults:` 后路。
详见 [模型模块 §4.6](../modules/model-module.cn.md)。

### D4. 跨维度词表收敛

model-module §4.5 要求三份词表单处定义。control mode 第一版只保留关节空间三值
`joint_pos` / `joint_delta` / `joint_vel`：

- `robot/profile.py:114` 的 `_CONTROL_MODES` 现含 `tokenized`（属模型 `action_head`，
  不是控制模式）与 `delta_joint`（命名与数据侧 `dims[].mode` 不一致）——本阶段修正；
- EEF 类（`delta_eef` / `se3`）与 `rotation_repr`、跨维分段**作为一组推迟**，随 EEF 模型
  适配一起进入（data-module §8.3 已说明准入边界重合）；
- 已确认全仓无使用者：`_CONTROL_MODES` 是唯一定义点，`lekiwi.yaml` 只用 `joint_pos`，
  recipe 的 `action_spec.action_type` 是独立字段（阶段5 才处理），不受影响。

---

## WP0：受控词表与来源标注的公共基础

三份词表和来源标注被三个维度共同引用，而 `data/` `model/` `robot/` 是叶子层、
不得依赖 `assembly/`（架构 §2.2 依赖方向）。落在 `vla_factory/utils/`。

- 新增 `vla_factory/utils/vocabulary.py`：
  - `CAMERA_SEMANTICS`：`third_person_front` / `third_person_top` / `third_person_side` /
    `wrist_left` / `wrist_right` / `wrist`，含泛化值 `third_person`（模型侧
    `semantic_accepts` 使用）；
  - `CONTROL_MODES`：`joint_pos` / `joint_delta` / `joint_vel`；
  - `ACTION_HEADS`：`flow_matching` / `diffusion` / `autoregressive` / `regression`。
- 来源标注类型：数据侧 `measured` / `inferred` / `undeclared`；模型侧
  `metadata` / `base_contract`。
- `robot/profile.py` 改为引用 `CONTROL_MODES`，删除本地 `_CONTROL_MODES`（D4）。

测试：词表被三个维度引用后仍单处定义（import 断言）；profile 校验对新旧值的接受/拒绝。

---

## WP1：DataSchema 逐条目重构 + reader 补齐可探测事实

实现 data-module §8。**这是阶段1 最大的一块，建议独立 PR。**

### 结构映射（旧 → 新）

| 现字段 | 新落点 |
|---|---|
| — | `identity: {name, source_format, episodes, total_frames}` |
| `robot_type` | `robot_ref: {name}`（字符串原样保留，**不解析**是否已注册） |
| `cameras` + `image_sizes` | `observation.cameras[]`：`{key, resolution, fps, encoding, semantic}` |
| `state_dim` + `state_keys` | `state.dims[]`：`{name, source_field}`，维数即 `len(dims)` |
| `action_dim` + `action_keys` | `action.dims[]`：`{name, source_field, mode}` + `action.frequency_hz` |
| `fps` | `temporal: {fps}` |
| `has_language` | `instruction: {task_field, granularity}` |

每项事实带 `source` 标注（measured / inferred / undeclared），随 `schema.json` 序列化。
`undeclared` 一律输出 null，**不是错误**——它是解析器保守失败、要求受控 override 的依据。

### 改动点

- `data/data_schema.py`：新 dataclass 结构 + D2 的派生属性 + `to_dict()` / `from_dict()`
  （`from_dict` 承担旧版扁平 `schema.json` 的升级）。`resolve_vector_keys()` 改为读
  `dims[]`（它现在校验的正是「每维恰好一个 key」，逐条目表天然满足，校验退化为存在性检查）。
- `training/train.py:440`：`json.dump(asdict(schema))` → `schema.to_dict()`。
- `inference/infer.py:315`：手工重建 DataSchema → `DataSchema.from_dict()`。
- `data/reader/lerobot_v3.py`：补 per-camera `resolution` / `fps` / `encoding`
  （`info.json` 的 `video_info`）、`instruction.granularity`、`identity.source_format`、
  `robot_ref`；`state.dims[].name` 保留原始后缀不剥离（如 `shoulder_pan.pos`）。
- `data/reader/robotwin.py`：`_JOINT_ORDER` 拼接布局（`left_arm` / `left_gripper` /
  `right_arm` / `right_gripper`）写进逐维 `source_field`——把 reader 代码里的隐式拼接
  变成 schema 里的显式事实；`action.dims[].mode` 按格式规范直接产出 measured
  （`/joint_action/*` 即 qpos 目标）。
- 新增推断规则模块（框架维护，非用户配置面，data-module §8.5）：
  相机 key → `semantic`（唯一命中才写，多候选/零候选一律 null）；
  action 维名后缀 `.pos` / `.vel` → `mode`（lerobot 这类通用容器格式**不设按格式的默认值**）。

### 测试

- 两个 reader 各一份 contract test：新字段产出、`source` 标注正确；
- 推断规则单测：唯一命中 / 多候选 / 零候选三种路径；
- **旧版 `schema.json` 升级测试**（用固定的旧格式 JSON 字面量，不依赖当前 writer）；
- `to_dict` / `from_dict` round-trip；派生属性与旧值逐一等价。

---

## WP2：ModelMetadata 接口区扩展 + BaseContract 归属划分

实现 model-module §4。可与 WP1 并行。

### ModelMetadata 新增（`model/model_interface.py`）

| 块 | 字段 |
|---|---|
| vision | `slots[]`：`{name, semantic_accepts, required, resolution, channels}`；`missing_slot_policy`（`zero_pad`/`drop`/`error`）；`image_normalization: {method, values}` |
| language | `template`（如 `"{task}"`） |
| proprio / action | `dim_policy`：`fixed: N` / `padded_to_max: N` / `flexible`；`normalization`：`mean_std` / `quantile` / `min_max` |
| action | `chunk: {predict, execute_recommended}`（`predict` 即现 `action_horizon`）；`control_mode_pref[]`；`segment_expectations[]`（第一版仅 gripper 段的 `convention` / `repr`）；`unification: {scheme: pad_to_max, pad_value}` |
| temporal | `expected_hz`、`history_frames` |

`slots[].semantic_accepts` 的取值域**就是** WP0 的 `CAMERA_SEMANTICS`（一处定义两处引用）。
`normalization` 必须含 `quantile`——openpi/pi0 实际用 q01/q99。
finetune 块只做对齐、不新增：草稿的 `parts` 即现有 `components`。

### 三个 entry 填充声明

- `adapters/act.py`：`dim_policy: flexible`（从零训练的投影层）、视觉槽位随数据、
  `image_normalization: imagenet`、`normalization: mean_std`、`requires_prompt: false`；
- `adapters/pi0.py` / `adapters/pi05.py`：3 个固定视觉槽位（224×224、`[-1,1]` HWC）、
  `dim_policy: padded_to_max: 32`、pi0 `mean_std` / pi05 `quantile`、`expected_hz: 50`。

### profile YAML 整体并入声明（按 D5）

`recipe/model/{act,pi0,pi05}.yaml` 的全部内容进入 `ModelMetadata.params`，
目录与 `load_model_defaults()` 一并删除；`TransformContext` 增加 `metadata` 字段，
事实类 step 参数从 `ctx.metadata` 读取，**步骤配置里再写该键即报错**。
当时暂把 transform 步骤清单保留在 `params["transforms"]`，供阶段 3 迁移规划；
后续已删除该配置面，当前由 resolver 从 `ModelMetadata` 具名事实直接推导 call。

顺带清掉两个死键：`num_inference_steps` 与 `tokenizer_max_length` 原本写在 profile
里但无人读取，改了不生效也无提示——前者接上消费链，后者删除（`task_tokenize`
步骤自带 `max_length`）。这两个正是闸2 存在的理由。

### 配置面三道闸（按 D5）

- 闸1 `recipe/defaults.py:resolve_recipe()`：`model.config` 的键 ⊆ `params` 的键
  （外加迁移期的 `camera_mapping` / `default_task`），否则报错 + difflib 候选；
- 闸2 `utils/tracked_config.py`：`TrackedConfig` 记录读取，factory 末尾
  `assert_all_consumed()`。必须是 `MutableMapping` 而非 `dict` 子类——CPython 对
  dict 子类的 `**` 展开走快路径、不调用重写的 `__getitem__`；
- 闸3 `assembly/transforms/base.py:reject_fact_override()`，接入
  `image_to_float` / `image_normalize` / `normalize_vector` / `pad_dimensions`。

### 推理步数三源收敛

`ModelMetadata.inference_num_steps`（唯一生效）、profile 的 `num_inference_steps`
（死配置）、`training.inference_steps`（死配置）三处合一：消费链改为
`model.config` > `params` 默认；`training.inference_steps` 由 parser 转发并打
deprecation warning；`infer.py` 的 ready 日志打印生效值与来源。

### BaseContract 与 Materialize

- `model/base_contract.py`：实测面按 §4.1 划定——实际存在的视觉槽位及分辨率、
  实际 state/action 维度、`model_type`、checkpoint 路径；**不声明超出 ModelMetadata
  能力边界的内容**，实测槽位必须落在声明 slots 内。
- `assembly/resolver/resolver.py:_materialize`：合并新字段，越界即
  `METADATA_CONTRACT_CONFLICT`，逐项记录来源（`metadata` / `base_contract`）——
  **已完成**：不只 `action_dim`，`dim_policy`/`dim_policy_max`/
  `vector_normalization`/`expected_hz`/`vision_slot_names` 均已纳入合并，
  外加 `vision_slots` 的能力边界检查（实测槽位必须落在声明内）。此前这里
  写的「现在只有 action_dim 一例」已过期，于阶段 2 摸底时更正。

测试：三个 entry 的 metadata contract test；BaseContract 越界冲突用例（槽位不在声明内、
分辨率超界）；`resolve` 输出中来源标注正确。

---

## WP3：`action_spec` 事实分流

依赖 WP1 + WP2。**recipe 字段本阶段不删、不 deprecate**（那是阶段5）；只把消费方的
事实来源改成三个维度，recipe 兜底。

| `action_spec` 字段 | 权威来源 | 兜底 |
|---|---|---|
| `action_dim` | 数据侧 `len(action.dims)`；模型上限 `ModelMetadata.dim_policy` | recipe |
| `action_horizon` | `BaseContract` > `ModelMetadata.chunk.predict` | recipe（见风险 R4） |
| `action_type` | 数据侧 `action.dims[].mode`；机器人侧 `RobotProfile.native_action_type` | recipe |
| `bounds_low/high` | `RobotProfile.safety_bounds_*` | recipe |

路由逻辑集中在 `assembly/action_facts.py`（`resolve_action_dim` /
`resolve_action_horizon`），四个消费点全部接入：`training/train.py`、
`adapters/act.py`（head 宽度 + chunk size）、`adapters/pi0.py`（horizon）、
`inference/infer.py`（引擎的 `action_dim` / `action_horizon`，以及 evaluate 的
ground-truth 窗口改用 `engine.action_horizon`）。统一规则：**维度事实优先、
recipe 兜底、两者不一致时 warning 并采用维度事实**（保守失败留给阶段2 升级为 error）。

接入时发现一处既有的自相矛盾：`train.py` 已按数据事实决定 pad 目标，而
`adapters/act.py` 仍按 recipe 建 action head——recipe 与数据集不一致时，头的宽度
和 dataloader 喂进来的宽度会对不上。现已统一。

测试：`test_action_facts.py`（路由单元）+ `test_act_model.py::TestActionFactRouting`
（消费点：head 宽度跟数据集、chunk size 跟 recipe）+ `test_train_infer_roundtrip.py`
（recipe 故意写错 `action_dim`，训练与推理两端都必须采用数据集的 8）。

---

## WP4：model adapter 关系假设显式化

### ACT 相机错位（bug 修复，本阶段唯一的行为变更）

`adapters/act.py:223` 用 `sorted(observation.images.keys())` 与 `self._image_keys` zip
建立相机对应，而 `_image_keys` 的顺序来自 `schema.cameras`（`info.json` 的 feature 顺序）。
**两者顺序不一致时相机会被静默交换**——腕部图像喂进第三人称槽位。当前测试数据
（`front` / `wrist`）恰好字典序一致，所以 bug 潜伏未爆。这正是架构 §4.2.3 明令禁止的
「靠字典序猜语义」。

改为：wrapper 持有显式的 `{dataset_camera: config_key}` 映射（构造期由 factory 依据
`schema.cameras` 与 `input_features` 的**同一份有序来源**建立），运行期按 key 取值，
缺失即报错。回归测试构造一份 `cameras=("wrist", "front")` 的 schema，断言映射正确。

### 其余关系假设

- `adapters/act.py:370` 的 `state_dim = schema.state_dim or action_spec.action_dim`
  兜底规则提取为独立函数并注释标注阶段3 的接管点；
- pi0 / pi05 的 `camera_mapping` 入口迁移：按架构 §3.1 三区划分它属于组合调整区，
  改为 `assembly.camera_mapping` 优先，`model.config.camera_mapping` 继续兼容并
  发迁移 warning；`base_contract.check_camera_mapping()` 的读取点同步（现在只读
  `model_config`，见 `base_contract.py:246`）。

---

## WP5：deploy adapter 本体事实 → RobotProfile

- **新增 RoboTwin 对应的机器人 profile**：`_JOINT_ORDER` 描述的双臂 + 双夹爪拓扑
  是本体事实，现在只活在 reader 常量里；提取为 `robot/profiles/*.yaml`。
- `lekiwi.yaml` 的 `limits_low/high` 与 `safety_bounds_*`：**查证后仍留空**。
  上游 URDF（[SIGRobotics-UIUC/LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/URDF/LeKiwi.urdf)，
  已填入 `urdf_ref`）里 `<limit>` 元素数为 **0**，9 个可动关节全部声明为
  `continuous`——连夹爪也是，显然是 OnShape 导出器缺限位数据的产物而非物理声明，
  所以既不能取限位、也不能据此改 `types`。真实范围需要 STS3215 servo 规格或
  逐台标定文件，钉住之前保持空——**不猜**（架构 §1.7）。
  URDF 另确认基座是三个全向轮（`base_left/right/back_wheel`），佐证 profile 里的
  `base_x/base_y/base_z` 是命令空间而非 URDF 关节；已在 profile 注释中写明。
- deploy 路径加对账：`LerobotHostObsAdapter` / `LerobotHostActionAdapter` 的 motor key
  来自 checkpoint schema（数据事实），profile 可用时增加「schema 逐维 `name` ↔
  profile `joints.names`」一致性检查，不一致 **warning**（阶段2 才升级为 error），
  且不改变实际使用的 key 顺序。
- 划线不动：wire format、`observation.images.X` 前缀、base64 JPEG 解码、transport、
  session config 属于推理模块，**不进 RobotProfile**（架构 §4.1.3）。

---

## WP6：inspect CLI + resolve 展示来源

实现架构 §3.5。这是阶段1 对用户可见的成果——把抽出来的事实和来源直接摆出来。

```bash
vlafactory-cli inspect data  --path <dataset> [--stats]
vlafactory-cli inspect model --name <model> [--path <checkpoint>]
vlafactory-cli inspect robot --name <robot>
vlafactory-cli inspect --config <recipe.yaml>    # 按 recipe 一次输出三份
```

- 统一信封 `{dimension, source, facts}`，默认人读 YAML，`--json` 供工具消费；
  **key 顺序确定、可 diff**（golden test 的前提）。
- 三条纪律：不猜语义（null 原样输出）、不触发重依赖（只读 registry metadata +
  checkpoint `config.json`，永不调用 factory；无 GPU / 无 extras / 无机器人连接可运行）、
  不解析跨维度引用（`robot_ref` 原样输出字符串）。
- `--stats` 是显式开销开关：统计量默认只出摘要。
- `resolve` 摘要同步展示新事实与来源标注（`resolve --explain` 属阶段2，本阶段不做）。

测试：三个子命令各一个 golden 输出（固定 key 顺序）；`inspect model` 在未安装
可选 extras 的环境中可运行（CI 默认环境即验证点）。

---

## WP7：文档与回归收口

- data-module §8 / model-module §4 状态标注从「目标设计，尚未实现」改为按实现描述；
- `robot-module.cn.md` 补 RobotProfile 第一版字段表（现为 TODO），与 WP0/WP5 对齐；
- `assembly-module.cn.md` 补 Materialize 的合并规则与来源记录（其余 TODO 留给阶段2/3）；
- `.claude/CLAUDE.md`：新增 `inspect` 子命令、DataSchema 新结构、词表位置；
  顺带修正 entries 清单（现文写 `act.py`、`pi0.py`，实际还有 `pi05.py`）；
- `examples/reference.yaml`：`assembly.camera_mapping` 作为首选入口的说明。

---

## 提交切分（建议 8 个 commit，可按 WP 拆 PR）

| # | 类型 | 内容 |
|---|---|---|
| 1 | `feat:` | WP0 词表与来源标注基础 + `_CONTROL_MODES` 收敛 |
| 2 | `refactor:` | WP1 DataSchema 逐条目重构 + 派生属性 + `to_dict`/`from_dict` + 新旧 schema.json 互通 |
| 3 | `feat:` | WP1 两个 reader 补齐可探测事实 + 推断规则 + contract test |
| 4 | `feat:` | WP2 ModelMetadata 扩展 + 三个 entry 声明 + profile 去重 + Materialize 扩展 |
| 5 | `refactor:` | WP3 `action_spec` 事实分流（行为不变，数值等价测试） |
| 6 | `fix:` | WP4 ACT 相机映射错位修复（**唯一行为变更**，独立 commit 便于回退） |
| 7 | `refactor:` | WP4 其余关系假设显式化 + WP5 本体事实提取与对账 |
| 8 | `feat:` + `docs:` | WP6 inspect CLI + WP7 文档收口 |

顺序依赖：1 → (2,3 ∥ 4) → 5 → (6 ∥ 7) → 8。

---

## 验证

- 每个 commit 后 `pytest` 全绿（无 extras 环境，heavy 用例按现有 skip 机制跳过）。
- **新旧 checkpoint 互通**：用改造前生成的 `inference_metadata/` 跑
  `vlafactory-cli infer` / `deploy`，确认旧 `schema.json` 仍能加载。
- **数值等价**：改造前后各跑一次短 train smoke（固定 seed、极少步数），
  比对 loss 序列与 `final/model.pt` 的 state_dict key 集合。
- `vlafactory-cli resolve --config examples/act_lekiwi.yaml` 输出新事实与来源；
  `inspect data/model/robot` 三个子命令在无 extras、无 GPU 环境下可运行。
- `grep` 确认词表单处定义、profile 中无与 metadata 重复的键。

---

## 风险

| # | 风险 | 应对 |
|---|---|---|
| R1 | DataSchema 结构变更破坏旧 checkpoint（`schema.json` 是训练产物，用户手上已有） | `from_dict()` 显式升级路径 + 用旧格式 JSON 字面量固化的回归测试；派生属性保证 12 个消费点零改动 |
| R2 | ACT 相机映射修复改变行为——对 `cameras` 顺序非字典序的数据集，旧行为是错的、新行为才对 | 独立 commit；PR 描述写清触发条件；若用户已用受影响数据集训练过，需在 README/发布说明提示重训 |
| R3 | profile 去重后 transform 拿不到值，静默退化（如 `normalize` 回退默认 zscore 而非声明的 quantile） | `from_config` 回退链上加显式断言：cfg 与 metadata 都没有该事实即报错，不静默用默认值 |
| R4 | `action_horizon` 来源切换踩到 ACT 与 pi0 的语义差异——ACT 从零训练时 horizon 确实由用户定，pi0 由 checkpoint 定 | 兜底顺序按 `training_paradigm` 分流：`from_scratch` 走 recipe 优先，`pretrained_finetune` 走 BaseContract > metadata > recipe |
| R5 | WP1 与 WP2 并行开发时在 `resolve` 输出上冲突 | golden 输出文件在 WP6 统一更新；前序 commit 的 golden 只断言各自新增字段 |
| R6 | 词表收敛移除 `delta_eef` / `se3`，未来 EEF 模型接入时需要回补 | 已在 D4 记录为「与 EEF 组一起准入」；`ResolutionError` 对未知控制模式给结构化报错，不静默通过 |

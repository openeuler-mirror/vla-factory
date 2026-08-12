# 开发计划：组合解析迁移阶段 3「生成 Mapping 与 T1 TransformPipelinePlan」

> **后续设计更正（阶段 4 review）**：本文记录的 `output_widths()` / pipeline fold 是
> 当时实现过程，不是当前架构。现实现先从 ModelMetadata、model tunables 与 DataSchema
> 直接建立 `ModelIOSpec`，再由 pipeline 消费目标接口；`output_widths`、
> `output_image_sizes`、`fold_widths` 均已删除。当前规则见阶段 4/5 计划 §2.8。

> 状态：**WP0–WP6 已执行完毕**（工作树，未提交），`pytest` 303 passed / 3 skipped。
> 依据：`docs/architecture/vla-factory-architecture.cn.md` §7.4 阶段3、
> §4.2.3（Mapping）、§4.2.4（Transform Pipeline）。
> 与阶段 1/2 同分支（`ref_transform`）：本阶段直接消费阶段 2 的候选推导逻辑。
>
> **实施中用真实数据验证/修正的四处**（正文已就地更新）：
> 1. **相机 override 是「完整声明」，不是「在推断结果上打补丁」**——实测
>    `examples/pi0_lora.yaml` 只映射 3 个槽位中的 2 个，若对剩下的
>    `right_wrist_0_rgb` 继续自动推断，数据集里唯一的 `wrist` 相机会同时喂给
>    左右两个腕部槽位；而 `entries/pi0.py` 运行时对未列出的角色发的是 -1 占位图
>    + zero mask（recipe 注释也写明「intentionally unmapped」）。推断会让具身组合
>    声称一个模型根本收不到的相机，因此：**只要给了 camera_mapping，数据侧的槽位
>    推断与歧义检查整体让位于它**。
> 2. **模型输入输出规格的宽度改由「pipeline 实际的 padding 决定」推导**——
>    原来 `state_dim` 直接取 `schema.state_dim`，于是 pi0 的 canonical 说 6、
>    pipeline 却把 state padding 到 32，同一个产物里两个不一致的「最终维度」。
>    现在 Build Interface 与 Mapping、Plan Pipeline 共用同一个 `_pad_plan()`，
>    三者不可能不一致（ACT 走 flexible 分支，仍是 6，无行为变化）。
> 3. **解析器需要 recipe 合并后的 `model.config`，不能只读 `metadata.params`**——
>    否则 recipe 里对 `transforms.inputs` 的逐 run 覆盖不会被看见。新增一个可选
>    kwarg `model_config`（CLI 传 `recipe.model_config`），不传时退回模型声明。
> 4. **`_plan_call_args` 里没用上的 `norm_stats` 形参已删**——统计量走
>    `stats_ref` 引用，规划期不读实际数值。
>
> **code review 后补的四处**（2026-08-10）：
> 5. **每条 `pad_dimensions` 按自己的 config 规划**——原来用「找第一条」的
>    `_declared_step()` 取参数，却遍历所有声明生成 call，声明里出现两条 fields
>    不同的 padding 时后一条会套用前一条的参数。改成把当前这条传进
>    `_pad_call()`，`_declared_step()` 随之删除。
> 6. **`resize_images` 补上非正数校验**、**`vector_normalization` 无对应实现时
>    给出可读报错**（原来 `min_max` 会 KeyError）——都是与运行时
>    `from_config` 的行为对齐。
> 7. **公开契约改为区分两类失败**：组合失败 → 结构化 `ResolutionError`；
>    registry entry 声明了不可能的东西（缺事实、尺寸非法、归一化无实现）→
>    朴素 `ValueError`，与 pipeline 构建期同款失败。原文档笼统写「任何失败都抛
>    `ResolutionError`」是过度承诺。
> 8. **`resolved` 的含义写进 `types.py` docstring**：它承诺的是「解析器规划过
>    这条路径」，**不是**「每个 call 都能无额外输入实例化」。
>
> 其中 5、6 提到的 `_pad_call()` / planner 内的 resize 校验，随后被 WP4.1 整体挪进
> 各 step 自己的 `compile_call()`；修正 2 的 `_pad_plan()` 又被 WP5.2 换成各 step
> 自报的 `output_widths()` fold——修正本身都仍然有效，只是归属换了地方。
>
> **一处按 review 建议改了、又按测试证据改回来的**：canonical 的 `action_dim`
> 一度改成「只认 pipeline 实际 padding 出来的宽度」，结果 phase-0 的三条测试立刻
> 挂掉——它们断言 checkpoint 自述的 `action_dim` 要refine 接口。那三条是对的：
> 没有 pad step 时按 pipeline 宽度上报，等于把 checkpoint 的自述维度整个丢掉。
> 最终保留「action 用模型自身宽度（含 contract 精化）、state 用 contract > pad >
> 数据」，并把真正的不变量与唯一的分歧场景写进 `_vector_widths` docstring。

## 范围：只填阶段 0 留下的空壳，不加新字段、不加新 API

阶段 0 已经把 `ResolvedAssembly` 的五类 Mapping 和三条 pipeline 规划的**形状定死了**
（`entries: tuple[dict]` + `resolved` 布尔）。本阶段的全部工作就是**把它们填上**，
不改形状：

- 不给 Mapping 的 entry 加类型（裸 dict 是阶段 0 定的协议，现在没有消费者要求收窄）；
- 不给任何既有 dataclass 加字段（**只删不加**，见 WP0）；
- 不动 `TransformRegistry` 的 API；
- 不新增 CLI flag。

> WP4 按 code review 反馈放宽了后两条中的第一条：`TransformStep` 上新增了三个
> 规划方法，`TransformRegistry` 本身的 API 仍未改动。见 WP4。

延续阶段 2 的两条判据：**架构没点名的不做**；**没有真实数据能验证的不做**（写出来
就是死代码）。据此，架构阶段 3 原文里点到的两项**本轮不做**，理由见下节。

### 不做「关节重排」与「夹爪 flip」两个 T1 step

架构原文举例是「normalize、padding、**关节重排**和**夹爪 flip** 等 T1 step」。前两项
的 step 都已存在（`normalize_vector` / `pad_dimensions`），后两项**在
`TransformRegistry` 里根本没有实现**——规划出引用未注册 step 的 plan，阶段 4 无法
实例化，等于产出一份假数据。要做就得先新写两个 step，那是新增特性，不是填空壳。

而且这两项都只在有机器人时才有意义，现状是：**四份 example recipe 没有一份声明
robot**，且唯一的真实 fixture 的 action 逐维名字是 `dim_0..dim_7`（阶段 2 已核实），
真跑必然是 `JOINT_ORDER_MISMATCH`。没有真实数据能验证 = 阶段 2 砍安全范围检查时用的
同一条理由。

**连锁简化**：`robot_to_model` 一并推迟（它就是「机器人相机 + 关节重排 + 同一条图像
链」，关节重排不做则它无从谈起），保持 `resolved=False`。本阶段落地的是
**data × model 这一半**：`data_to_model` 与 `model_to_robot`。后者在无 robot 时目标
空间就是数据动作空间——这正是今天 `InferenceEngine` 后处理在跑的链路，有真实覆盖。

---

## Context：两处必须复用、不能重写的既有逻辑

1. **阶段 2 已经算出了 Camera / Joint 两类 Mapping 的候选，算完就丢**。
   `_check_camera_slots_against()` 求出 `hits` 只用来判歧义；`_check_joint_order()`
   求出「剥后缀后的子集嵌入」也只用来判错。本阶段把这两处内联逻辑抽成纯函数，
   检查与建 Mapping 调同一个——**不允许出现第二份匹配实现**，否则会有「检查说能过、
   映射建不出来」的裂缝。
2. **今天真正跑的 transform 是模型声明列表的一个子集**：`PadDimensions` /
   `NormalizeVector` / `ResizeImages` 的 `from_config` 会按事实返回 `None`
   （pad 无意义 / 无 stats / 无 resize 目标）。这个跳过判定就是解析器该做的规划，
   前移即可，不是新逻辑。

两条实现约束（不是新功能，是 plan 的语义边界，写进注释即可）：

- **`TransformStepCall.args` 是已解析值，不回灌 `from_config`**。`transforms/base.py:
  reject_fact_override` 拒绝含 `method`/`target_dim` 的 config——那道闸防的是用户逐
  run 改事实，不是防解析器。已解析的 args 走 `from_call`（WP4.1），与用户/声明入口
  `from_config` 是两条路；把下游真正改成消费 plan 仍是阶段 4 的事。
- **统计量用引用不内联**：`{"stats_ref": "norm_stats"}`（步骤自己按 field 取
  state/action 那一半）。`ResolvedAssembly` 已带 `norm_stats_ref`，内联 32 维数组
  会让 golden 每次重算 stats 都冲突。

### 为什么产出的是数据（名字 + 参数），不是实例化好的 pipeline

`TransformStepCall` 就是两个字段——`type`（注册名）和 `args`（构造参数），没有第三层
抽象；`TransformRegistry` 把名字解析回类这条路今天就在跑。之所以不能让解析器直接返回
活的 `TransformPipeline`，只有一个硬理由：**它要跨进程**。

推理是在另一个进程里从 checkpoint 重建的（`inference/infer.py:563-580`）：读
`inference_metadata/` 里的 `recipe.yaml` + `schema.json` + `norm_stats.json`，重新造
`TransformContext`，重新跑一遍 `from_config`。训练进程里的活对象活不到那时候。

所以「名字 + 参数」的序列化形式**今天已经存在了**——就是保存在 recipe.yaml 里的
`model.config.transforms.inputs`。本阶段不是新增一层抽象，是把它挪进解析器产物，并
**把事实提前烤进去**：今天存的是候选列表，推理侧还要靠 `TransformContext` 重新推导一遍
pad target、normalize method 和那些 `from_config` 返回 `None` 的跳过判定；存成解析后的
plan，推理侧就只照着执行，不再独立推导第二遍。消灭「同一件事训练侧推一遍、推理侧再推
一遍」正是组合解析层存在的理由（架构 §4.2.6）。

（反过来说：如果训练和推理同进程、pipeline 不需要落盘，直接给实例就够了，plan 纯属
仪式。是那条 checkpoint → 部署进程的边界让它必须是数据。）

---

## WP0：清掉阶段 0 遗留的空转字段与不直观的命名（**已完成**）

纯清理，无行为变更，`pytest` 284 passed / 3 skipped 原样全绿。

- **删掉 `risk` / `reversible`**：阶段 0 加的，至今零消费者，本阶段规划的步骤里也没有
  需要标 lossy 的分支。等真要用它们做决策时，按那时的实际读法再加——现在留着只会让每份
  golden 多两行永远是默认值的噪音。保留 `resolved`（CLI 摘要在读）。
  架构 §4.2.4 仍保留「标记风险与可逆性」的目标描述（架构文档写目标态，可超前于实现），
  恢复路径见推迟表。
- **`Spec` → `Call` / `Plan`**，两级各用最贴的词：

  | 旧 | 新 | 字段 |
  |---|---|---|
  | `TransformStepSpec` | `TransformStepCall` | `type` + `args`（原 `config`） |
  | `TransformPipelineSpec` | `TransformPipelinePlan` | `calls` + `resolved` |

  step 级叫 `Call`——名字 + 参数就是一次调用，且不会被读成类（注册表里
  `type[TransformStep]` 才是类，`TransformStepCls` 会指错东西）。pipeline 级叫
  `Plan`——它带 `resolved` 状态和三条路径的方向语义，不是纯集合，`plan.calls` 读作
  「计划由若干次调用组成」；复数类名 `Calls` 被排除，因为 Python 里 `mock_calls`
  那种用法自带「已发生的调用记录」的读法。内层字段用 `args` 而非 `config`，因为
  `config` 在本仓专指 `model.config` 那种用户可覆盖项。两级与可执行体对称：
  `TransformStepCall → TransformStep`、`TransformPipelinePlan → TransformPipeline`。
- 已同步：`types.py`、`resolver/__init__.py` 导出、`resolver.py` docstring、
  `test_assembly_resolver.py`、架构文档中英两版对象表、`assembly-module.cn.md`、
  `model-module.cn.md`。历史计划文档（phase1、refactor-architecture-alignment）
  作为存档不动。
- `to_dict` / `from_dict` **不改**：它是全仓约定（4 个模块 33 处定义、44 处调用），
  只改这两个类会把一种约定变成两种；而且真正的序列化在调用点
  （`train.py:443` 的 `json.dump(schema.to_dict(), f)`），`to_dict` 是中间表示不是
  序列化器。若要统一改名，应是独立的机械 commit，不混进本阶段。

## WP1：Resolve Mapping（五类）

抽出两个纯函数到 `matching.py`（`camera_candidates()` / `embed_joints()`），Check Pairs
与 Mapping 生成共用同一结果。逐类：

- **Camera**：逐模型槽位一条 entry，`{model_slot, data_source, source}`；唯一命中
  → `source=inferred`；无命中 → `data_source=None, source=padding`（与
  `entries/pi0.py` 现网的 -1 图 + zero mask 行为一致，plan 只是把它写成声明）。
  消费 `assembly.camera_mapping` override：**给了 override 就以它为完整声明**，
  数据侧的推断与歧义检查整体让位（架构 §4.2.3「受控 override 直接产生最终
  Mapping」；实测理由见文首修正 1）；机器人侧的检查照跑，因为 override 名字是按
  `schema.cameras` 校验的，只约束数据侧。override 指向不存在的相机/槽位 → 新错误码
  `CAMERA_MAPPING_INVALID`（不加这条，写错的 override 会静默退化成 padding）。
- 模型没有声明 `vision_slots` 时（ACT）：模型的视觉输入**就是**数据集相机本身
  （`entries/act.py` 按 `schema.cameras` 建 input_features），逐相机一条恒等 entry。
- **State / Action**：逐模型槽位一条 entry，`{model_index, data_dim_index,
  data_name, padded}`（action 另带 `mode`）。今天 state/action 向量就是数据逐维顺序
  拼接，所以是恒等对应 + 超出部分 `padded=True`。不在这两类里重复机器人关节信息——
  那是 JointMapping 的职责。模型侧宽度取自 `pipeline_planner.vector_widths()`——直接
  从规划出的 calls 上读，与模型输入输出规格同一个来源（文首修正 2）。
- **Language**：一条 entry，`{model_field, data_field, template, fallback, source}`。
  来源链沿用 `task_tokenize` 现网行为：数据字段 → `default_task` → 空 prompt，
  **不硬失败**（阶段 2 已就此定调）。
- **Joint**：仅当 recipe 声明了 robot 时生成（子集唯一嵌入的结果，逻辑阶段 2 已有）；
  未声明则保持 `resolved=False`。它只是名字对应关系，不需要任何 step 存在。
  取 **action** 侧的逐维名字建映射——这份 Mapping 的消费者是 `model_to_robot` 上的
  关节重排，即下发给机器人的命令向量。

## WP2：Plan Pipeline（两条）

- **`data_to_model`**：读 `transforms.inputs`（recipe 覆盖后的最终值——解析器为此
  新增可选 kwarg `model_config`，见文首修正 3），
  **保留其顺序**（顺序是上游语义：pi05 的 `task_tokenize` 必须在 `pad_dimensions`
  之前，pi0 的 letterbox 必须在 `image_layout` 之后，框架不具备重新发明它的知识），
  逐步骤补全事实（pad target、normalize method、image range、resize 目标、
  `default_task`）并执行 Context 第 2 点的跳过判定。**plan 里出现的每一步都是真的会
  跑的步骤。**
- **`model_to_robot`**：对上面规划出的、影响 action 的两个步骤取逆并逆序——
  `pad_dimensions → unpad_action`、`normalize_vector → unnormalize_action` 或
  `unnormalize_action_quantile`（按 method）。没有逆的步骤直接消失，**不是把
  `data_to_model` 列表反转**（架构 §4.2.4 原文点了这个坑）。配对关系归各 step 自己的
  `inverse_call()`（WP4.1），planner 不认识任何 step 名。
- **`robot_to_model`** 保持 `resolved=False`（见范围一节）。

## WP3：展示、测试与文档收尾

- `cli.py:_print_assembly_summary()`：把 `mappings: 0/5 resolved (phase-0 skeleton)`
  换成五类 Mapping 的实际状态与要点（相机 `2/3 命中，1 走 padding`、language
  `来源=default_task`）+ 两条 pipeline 的步骤数。顺手补齐阶段 2 WP5 遗留没做的相机
  槽位摘要。不加 `--json`（golden 直接用 `to_dict()`，用户侧的 diff 需求等真提出来
  再说）。
- **Golden**：沿用阶段 2 的真实 fixture + 真实 registry entry，固化 3 份具身组合：
  ACT、pi0（带 camera override）、pi05（quantile 路径）。
- **等价性测试**（本阶段最关键的一条）：对 act / pi0，把现网 `create_dataloaders`
  构建出的 `TransformPipeline` 实例与规划出的 `data_to_model` plan 逐步骤比对
  （type 顺序 + `target_dim`/`method`/`height`/`width`/`max_length`）。比对的是**已
  构建实例的属性**，不是重跑 `from_config`。没有这条，规划出的 plan 就是一份没人执行、
  错了也不会红的死数据；有了它，阶段 4 切换下游才有判据。
- 文档：架构 §7.4 阶段 3 标注完成（并注明关节重排/夹爪 flip/`robot_to_model` 推迟）；
  本文档补「实施中修正」段。`assembly-module.cn.md` 保持 TODO——它要对齐的是完整实现，
  现在写等于写一半。

---

## 明确推迟（供下次启动参考）

| 项 | 现状 | 恢复时需要的前置工作 |
|---|---|---|
| 关节重排 step | `TransformRegistry` 无此实现；无 recipe 声明 robot | 先实现 `reorder_dims` step，再由 JointMapping 驱动规划 |
| 夹爪 flip step | 同上，且模型侧/数据侧 convention 事实都不存在 | `ModelMetadata.gripper_convention` + 数据侧 convention（`robot-module.cn.md §4.3`）+ step 实现 |
| `robot_to_model` | 依赖上面两项 | 两项落地后即可规划 |
| Mapping entry 类型化 | 裸 dict 是阶段 0 定的协议 | 等阶段 4 下游真正按字段读时，按实际读法一次收窄 |
| `TransformPipelinePlan.risk` / `.reversible` | WP0 已删（零消费者）；架构 §4.2.4 仍写着目标态 | 等真有消费者要按风险/可逆性分支时，按那时的读法重新加字段 |
| `resolved` 的强语义（「可直接实例化」） | 现在只承诺「解析器规划过这条路径」，已写进 `types.py` docstring | 有实例化侧（阶段 4 的下游接入）才可校验；届时可拆成两个布尔或一个枚举 |
| 模型声明自洽性检查（声明了 `action_dim` 却没声明 `pad_dimensions`） | 唯一能让 canonical 宽度与 pipeline 输出不一致的情形；无模型触发 | 需要新错误码，等真出现这种 entry 再加 |
| `resolve --json` / `--explain` | 摘要 + golden 已够 | 用户提出真实 diff 需求时 |

## WP4：code review 后的结构整改（**已完成**）

review 提出的三项结构问题，原计划推到阶段 4，按反馈改为本阶段落地。

### 4.1 每个 transform 一个规划入口，planner 不再认识任何 step 名

问题：planner 手工复刻了各 step 的 `from_config` 与 `inverse_for_output`，两套规则
已经出现分歧（`resize_images` 的非正数校验运行时有、planner 没有）。

改法是把规则挪到 step 自己身上，**而不是新起一套 compiler 注册表**——`TransformRegistry`
本来就把名字解析成类，规则挂在类上即可，不需要第二套注册机制：

| 方法 | 职责 |
|---|---|
| `compile_call(cfg, ctx) -> args \| None` | 声明 + 事实 → 这次调用的参数；`None` 表示该步在这组事实下是空操作，从 pipeline 里消失 |
| `from_call(args, ctx) -> TransformStep` | 参数（+ 运行时上下文，用于参数带不动的活对象，如统计量）→ 可执行步骤 |
| `inverse_call(args, ctx) -> (name, args) \| None` | 正向/逆向配对的唯一归属；有损步骤必须返回 `None` 而不是找个近似的 |

关键收益是 **`from_config` 变成 `compile_call + from_call` 的组合**——所以这不是加了
第三份实现，而是把原来的两份合成一份：planner 与构建路径跑的是同一段代码，物理上无法
再分歧。`inverse_for_output` 同样落到 `inverse_call` 上（`NormalizeVector` 保留一层薄
覆写，因为它能从实例自身判断有没有 action 统计量，而 ctx 可能没传）。

新增 `PlanContext`（`transforms/base.py`）承载两侧共用的事实；`TransformContext.plan()`
从运行时侧构造同一个类型。副作用是 **step 不再需要认识 recipe**：tokenizer repo 与
default_task 的兜底由调用方解析进 context。

回归护栏：`test_planner_holds_no_step_names` 直接扫 `pipeline_planner.py` 源码，出现任何
已注册 step 名就失败。

### 4.2 删除中间模型事实层

后续简化删除了 `BaseContract`、`materialize.py` 和 `ModelFacts`。resolver、兼容性
检查与 pipeline planner 直接读取 `ModelMetadata`；checkpoint 的冗余配置只进入
可选一致性检查，不能精化或覆盖解析结果。这样同一事实只有一个结构和一个来源。

### 4.3 按职责拆文件

| 模块 | 行数 | 职责 |
|---|---|---|
| `resolver.py` | 229 | 八阶段编排 + Load/Validate |
| `compatibility.py` | 278 | Check Pairs 六行 |
| `mappings.py` | 210 | 五类 Mapping |
| `pipeline_planner.py` | 174 | 声明 → calls、逆向 calls、宽度 |
| `checkpoint_validation.py` | 可选 | checkpoint config 与 ModelMetadata 一致性诊断，不参与解析 |
| `matching.py` | 74 | 相机候选、关节嵌入（Check Pairs 与 Mapping 共用） |

**Build Interface 与 Plan Pipeline 的顺序与架构编号相反**（先规划再定接口）：canonical
宽度直接从规划出的 calls 上读，接口就不可能报出 pipeline 不产生的宽度。模块 docstring
里写明了这个取舍。

---

## WP5：第二轮 code review 的整改（**已完成**）

### 5.1 删除「树外自定义 transform」特性

`transforms.imports`（recipe 字段 + `loader.py` / `infer.py` 的 import 回放 +
`train.py` 的产物序列化）整条链路删除。四份 example、全部测试零使用。

删的理由不是「用不上」，是**它和纯逻辑解析层在设计上冲突**：这个特性的机制本身就是
副作用式 import 用户模块，而解析器不做副作用、也不收 recipe。冲突的后果实测可见——
未注册的 step 会让 `model_to_robot` **静默漏掉它的逆向调用，却仍标 `resolved=True`**：

```
运行时 postprocessor : ['undelta_action', 'unnormalize_action']
计划 model_to_robot  : ['unnormalize_action']       resolved = True
```

（用一个 delta action 自定义 step 复现；架构 §7.3 正是拿 delta action 举例。）下游照
这份计划走会把增量当绝对关节位置下发。

删掉之后，未注册的 step type 只剩一种可能——声明写错名字，直接由
`TransformRegistry.get` 报错（自带候选列表）。planner 里的 pass-through 分支消失。
新增 transform 的方式变成「在 `assembly/transforms/` 加文件 + `@register`」，与
FormatReader / model entry / RobotProfile 三个扩展点一致（原先它是唯一允许树外注册的）。

老 checkpoint 的 `inference_metadata/recipe.yaml` 里残留的 `transforms.imports` 不影响
解析——`parser.py` 对未知 key 是忽略。

### 5.2 宽度由 step 自报，声明只做约束

`TransformStep.output_widths(args, input_widths) -> dict` —— **fold 形式**，不是返回绝对
宽度：padding 的输出是 `max(input, target)`，未来的 crop / projection 同样需要输入宽度；
若返回绝对值，planner 又得知道「这个数是下限、上限还是增量」，等于把刚消掉的名字泄漏
换成参数泄漏。

`vector_widths()` 于是变成纯 fold：

```python
widths = {"state": schema.state_dim, "actions": schema.action_dim}
for call in plan.calls:
    widths = TransformRegistry.get(call.type).output_widths(call.args, widths)
```

canonical、mapping、pipeline 输出从此是同一个数。**模型声明不再提供宽度，只约束宽度**：
声明了 `action_dim` 却与 fold 结果不符 → 新错误码 `PIPELINE_WIDTH_MISMATCH`。
没有 plan 时（未声明步骤列表）不做这个判断——没有 pipeline 就没有可矛盾的对象。

### 5.3 空步骤列表 = 未配置

原来 `loader.py:53` 判的是 `is None`，所以显式 `inputs: []` 会静默造出一条空 pipeline。
三处（loader ×2、infer ×1）统一改成 `if not transform_items`，与 resolver 的
`if not declaration` 对齐。

### 5.4 `call_args` 盖章移到不可绕过的外层路径

实测陷阱：只实现 `inverse_call` 而忘了 `call_args` 的 step，**规划侧说有逆、运行时说
没有**。盖章不能放在 `from_call()` 里——`NormalizeVector` 系列覆写了它，会被绕过。改为
在 `from_config()` 与 `inverse_for_output()` 两条外层路径上调用 `stamp_call_args()`，
基类 `call_args()` 默认读这个章。扩展者要写的规划方法回到三个。

### 5.5 删掉阶段考古

模块 docstring 里的 "Stage 7" / "phase-0 placeholder" / "later phases will…" 全部删除，
只留职责名——阶段编号在 `resolver.py` 的表和这份计划文档里就够了，散在各处只会互相矛盾
（实际已经矛盾：`resolver.py` 说 Plan Pipeline 是第 5 阶段，`pipeline_planner.py` 说是
第 7）。保留的是**只看代码看不出、删掉会重新踩坑**的结论：步骤顺序是上游语义、override
是完整声明、保守失败。

---

## WP6：`CanonicalInterface` → `ModelIOSpec`（**已完成**）

旧名两处不准：`Interface` 在本仓已被 `model/interfaces/` 的 `VLAModel` 协议占用，
而这个类是五个标量、零个方法；`Canonical` 也没回答「相对谁而言规范」。

新名按字段的实际含义取：输入侧是相机 key、状态宽度、是否需要 prompt，输出侧是动作宽度
与 horizon —— `IO` 两个方向都盖住。（一度考虑过 `SampleContract`，但 `action_dim` /
`action_horizon` 描述的是模型**输出**，用「样本」框不住。）

| 旧 | 新 |
|---|---|
| `CanonicalInterface` | `ModelIOSpec` |
| `assembly.canonical_interface` | `assembly.model_io_spec` |
| 解析阶段 `Build Interface` | `Build IO Spec` |

这是**概念改名**，不只是类名：架构文档 §4.2.1 的具身组合结构图、§4.2.2 的八阶段列表、
§4.2.6 的上下游边界规则中英两版都同步了，`assembly-module.cn.md` 同上。历史计划文档
（phase1/2、refactor-alignment）保留原始实施记录，并在开头标注后续取代决策。

---

## 提交切分

| # | 类型 | 内容 |
|---|---|---|
| 0 | `refactor:` | WP0：删 `risk`/`reversible` + `Spec` → `Call`/`Plan` 改名 |
| 1 | `feat:` | WP1：五类 Mapping（含候选纯函数抽取与阶段 2 检查改调用） |
| 2 | `feat:` | WP2：`data_to_model` + `model_to_robot` 规划 |
| 3 | `feat:` + `test:` + `docs:` | WP3：resolve 摘要 + golden + 等价性测试 + 文档 |
| 4 | `refactor:` | WP4.1：transform 单一规划入口（`compile_call`/`from_call`/`inverse_call`） |
| 5 | `refactor:` | WP4.2 + 4.3：删除中间模型事实层 + resolver 按职责拆分模块 |
| 6 | `refactor:` | WP5.1：删除树外自定义 transform 特性 |
| 7 | `feat:` | WP5.2 + 5.3：`output_widths` fold + `PIPELINE_WIDTH_MISMATCH` + 空列表语义 |
| 8 | `refactor:` + `docs:` | WP5.4 + 5.5：`call_args` 盖章 + 注释清理 |
| 9 | `refactor:` + `docs:` | WP6：`CanonicalInterface` → `ModelIOSpec` 概念改名 |

顺序依赖：0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9。

实际落地为**单个 commit**，与阶段 1（`ffbc71c`）、阶段 2（`e8a66fc`）一致——上表是开工前的拆分设想，WP4–WP6 是评审反馈驱动的整改，与前面几个 WP 改动同一批文件，事后拆分只会产出跑不过测试的中间态。

## 验证（实测结果）

- `pytest`：**296 passed / 3 skipped**（新增 11 条，见 `test/test_resolve_mapping.py`）。
- `resolve` 对 act / pi0 recipe 均输出完整摘要；两次运行 `to_dict()` 逐字节一致
  （`test_resolution_is_deterministic`，§1.7）。
  注：`examples/*.yaml` 指向的数据集不在仓库里，冒烟时把 `data.source.path` 换成
  `test/data/lerobot_train_data_3_episodes` 即可。
- **行为不变**：`test_train_infer_roundtrip.py` 原样通过——本阶段 training/inference
  一行不改，解析器只在 `resolve` 路径触发。
- **等价性测试绿灯**（act + pi0 两份 recipe）= 阶段 4 可以开工的判据。
- 失败用例两条：override 指向不存在的槽位 / 不存在的相机 →
  `CAMERA_MAPPING_INVALID`，断言 `code`/`path`/`params`，不匹配完整文案（§4.2.5）。

## 风险

| # | 风险 | 应对 |
|---|---|---|
| R1 | Check Pairs 与 Mapping 各写一份匹配逻辑后行为分叉 | WP1 抽纯函数，重构与新功能同一 commit，不留两份实现的中间态 |
| R2 | 规划出的 plan 没有执行者，错了也没人发现，到阶段 4 才爆 | WP3 等价性测试；这是本阶段唯一不能砍的测试 |
| R3 | 「保留模型声明顺序」被实现成「照抄声明列表」，跳过判定漏做，plan 里出现实际不跑的步骤 | golden 里放一份 ACT（其 `pad_dimensions` 在 8 维数据上会被跳过），plan 中不应出现该步骤 |

# 开发计划：组合解析迁移阶段 4「下游接入」+ 阶段 5「Recipe 瘦身」

> 状态：**阶段 4（WP1–WP7 + 一轮 code review 整改）已执行完毕**（工作树，未提交），
> `pytest` **351 passed**（openpi 环境另跑 330 passed / 21 skipped）；
> 阶段 5 待执行。分支 `ref_train_infer`（阶段 0–3 已提交至 `a50e83e`）。
>
> **实施中与本文档不同 / 需要补记的七处**（正文已就地更新）：
> 1. **`_action_horizon` 无 `model_config` 时回退到 `metadata.params`**——与
>    `transform_declaration` 同一条来源规则。否则任何不传 recipe 的解析（golden 测试、
>    `make_assembly` 这类夹具）都会因为「没有 tunable」而误判成声明不完整。
> 2. **「两边都没有 horizon」也是声明错误**，实施后波及全部 stub metadata：仓库里所有
>    直接构造 `ModelMetadata` 喂给解析器的测试都补了 `action_horizon=50`。这是规则的
>    正确代价，不是意外——一个连 chunk 长度都不声明的 entry 无法建模型。
> 3. **WP6 顺带删掉了 `stamp_call_args` / `call_args()`**：它们存在的唯一理由是
>    `inverse_for_output` 要从**已构建**的正向步骤反推参数；后处理改由 `model_to_robot`
>    计划给出后，这条链整段消失（原文只列了四组，实际是四组连带的一整族）。
> 4. **`TransformContext` 只剩 `norm_stats` 一个字段**，`plan()` 一并删除。已核实无任何
>    step 读它的 recipe / schema / model_config / split 字段。
> 5. **`test_data_pipeline.py::TestEndToEnd` 从「永远 skip」变成真的在跑**——它原先指向
>    一个不存在的 `vla_factory/examples/act_aloha.yaml`；改成 `examples/act_lekiwi.yaml`
>    + 真实 fixture 后，它正好覆盖 `resolve_from_recipe` → `create_dataloaders` 这条新链路。
> 6. **`resolve` 的 CLI 输出对老 checkpoint 少了一行**：`(skipped checkpoint compatibility
>    check: ...)` 改成 `logger.info`，因为这行提示搬进了 `resolve_from_recipe`，而它同时
>    服务 train / infer，print 到 stdout 会污染训练日志。
> 7. **阶段 4 review 后的 shape fold 方案已再次重构**：Mapping 先统一解析，`ModelIOSpec` 在 pipeline
>    之前从 `ModelMetadata` / model tunables / `DataSchema` 直接解析，pipeline 只消费目标
>    接口。已删除 `output_widths`、`output_image_sizes`、`fold_widths`；ACT 使用显式
>    `input_image_size`，pi0/pi05 的 224 只来自 `VisionSlot.resolution`。推理侧字段最终命名为
>    `model_output_dim` / `execution_action_dim`。
> 依据：`docs/architecture/vla-factory-architecture.cn.md` §7.4 阶段 4/5、§4.2.6（上下游边界）、
> §3.1（recipe 三个区）、§4.2.1（具身组合是下游唯一入口）。
>
> **基线**：`pytest` **306 passed / 3 skipped / 26 warnings**（完整 dev 环境）。
>
> **本计划改写了架构 §7.4 阶段 4 的一条原文**：原文写「保留显式兼容层并提供迁移告警」，
> 本计划对**旧训练产物**取消兼容层（缺 `assembly.json` 直接失败），理由见 §2.3；WP7 负责
> 把架构文档中英两版同步改掉，不留「文档说要兼容、代码直接报错」的分歧。
>
> **为什么两个阶段写在一份文档里**：阶段 5 要删的 recipe 字段，正是阶段 4 删掉读者之后
> 剩下的空壳（`action_spec` 四个字段在阶段 4 结束时会变成零消费者）。拆成两份会让
> 「先删读者、再删字段」这条唯一的顺序依赖散开。

---

## 0. 开工前用真实数据确认的五件事

用仓库唯一的真实 fixture（`test/data/lerobot_train_data_3_episodes`）替换掉两份 example
recipe 的 `data.source.path` 后实跑 `vlafactory-cli resolve`：

```text
# act_lekiwi.yaml                          # pi0_lora.yaml
model:      act                            model:      pi0
cameras:    ['front', 'wrist']             cameras:    ['front', 'wrist']
action:     dim=8 horizon=0                action:     dim=32 horizon=50
state:      dim=6                          state:      dim=32
mappings:                                  mappings:
  camera:   2/2 slots mapped                 camera:   2/3 slots mapped; 2 via override;
  state:    6/6 dims from data                         padding: right_wrist_0_rgb
  action:   8/8 dims from data               state:    6/32 dims from data, 26 padded
pipelines:                                   action:   8/32 dims from data, 24 padded
  data_to_model:  4 steps                    language: data field 'task'
  model_to_robot: 1 step                   pipelines:
  robot_to_model: unresolved                 data_to_model:  6 steps
                                             model_to_robot: 2 steps
```

**1. 计划齐了，缺的只是执行者。** 两条真实路径的 `data_to_model` / `model_to_robot` 都
`resolved=True`，逐步骤参数与现网构建出的 pipeline 已由阶段 3 的等价性测试逐字段比对过
（`test_resolve_mapping.py:226`）。阶段 4 不补规划能力，只把执行侧的输入从「声明列表 +
运行时重新推导」换成「计划」。

**2. ACT 的 `horizon=0` 是接入前必须先补的洞。** `ModelIOSpec.action_horizon` 直接取
`metadata.action_horizon`，而 ACT 是 `from_scratch`、chunk size 由用户选（R4，见
`assembly/action_facts.py:53`），它的具名字段是 0。**IO spec 目前没有任何渠道拿到
from_scratch 模型的 horizon**——训练一旦改成消费 IO spec，ACT 会拿到 0（WP2）。

**3. 重复推导的确切位置**（阶段 4 要删的就是这张表的右列）：

| 现在在哪里推导 | 推的是什么 | 接入后由谁提供 |
|---|---|---|
| `assembly/action_facts.py`（整个模块，72 行） | action_dim / action_horizon 的三方路由 | `assembly.model_io_spec` |
| `training/train.py:175`、`inference/infer.py:534-541` | 同上，各调一次 | 同上 |
| `entries/act.py:436-447` | action_dim / horizon / state_dim / camera 列表 | `model_io_spec` + `camera_mapping` |
| `entries/pi0.py:296-297,317-319` | `get_camera_mapping(recipe)`、horizon | `assembly.camera_mapping` + `model_io_spec` |
| `training/loader.py:63-72,119-131` | 从 recipe 取 `transforms.inputs`，再 `from_config` 重推事实 | `assembly.data_to_model` |
| `inference/infer.py:560-578` | 同上，在**另一个进程**里第二次重推 | `assembly.data_to_model` + `model_to_robot` |
| `training/loader.py:139-140` | 采样窗口长度取自 `recipe.data.sampler.action_horizon` | `model_io_spec.action_horizon` |

**4. `action_spec` 的四个字段，两个已经是死字段。** 全仓 grep：`action_type` 与
`bounds_low/high` **零消费者**；`action_dim` / `action_horizon` 只被 `action_facts.py` 读。
阶段 4 删掉 `action_facts.py` 之后，整块 `action_spec` 零读者——阶段 5 删字段才有干净前提。

**5. tokenizer 地址的兜底只存在于运行时，解析器看不见。**
`transforms/pipeline.py:69-88`（`TransformContext.plan()`）把 `recipe.model_path` 当作
`tokenizer_repo` 兜底传给
`task_tokenize`（`task_tokenize.py:_ensure_tokenizer` 没有它就报错），而
`pipeline_planner.plan_context()` 显式把 `tokenizer_repo` 置 `None`。今天没暴露，是因为
pi0/pi05 两份声明都写死了 `tokenizer_repo`；但只要运行时 context 按 WP6 缩成统计量载体，
一个依赖外部 checkpoint 自带 tokenizer 的模型，落盘的 call 里就没有 tokenizer 地址，执行
必炸。**兜底必须搬进解析入口，不能跟着 context 一起删**（WP1）。

---

## 1. 判据（沿用阶段 2/3）

- **架构点名的才做。** §7.4 阶段 4 点名四件事：训练消费「数据 × 模型」、推理消费
  「模型 × 机器人」、删除 adapter 中重复的关系推导、以及兼容层——最后一项本计划按 §2.3
  的理由改写为「旧训练产物明确不支持」，并同步架构文档。阶段 5 点名五件事：标记重复的
  action/state 事实 deprecated、把旧 recipe 自动转换成受控 override、新 recipe 只留三者
  选择与必要 override、提供迁移命令**或**可读提示、为兼容层设定明确移除周期。
- **没有真实数据能验证的不做**（写出来就是死代码）。

---

## 2. 阶段 4：下游接入

### 2.1 WP 一览

```text
WP1  统一解析入口          resolve_from_recipe()：registry → checkpoint 检查 → 描述 → resolve_assembly
WP2  horizon 归位          from_scratch 模型的 horizon 有声明来源 + 按 paradigm 互斥校验
WP3  训练接入              train() / create_dataloaders() 消费 ResolvedAssembly
WP4  工厂接入              factory(recipe, assembly)；删 assembly/action_facts.py
WP5  产物 + 推理接入        assembly.json（带 format_version）；缺失即失败；快照一致性校验
WP6  清理                  删掉运行时二次推导的四组代码
WP7  文档同步              架构 §7.4 兼容层原文改写 + CLAUDE.md + skill
```

顺序依赖：WP1 → WP2 → WP3 → WP4 → WP5 → **WP6 必须最后**（风险 R1：阶段 3 的等价性测试
是唯一护栏，它的对照物要等两个消费者都切完才能拆）→ WP7。

### 2.2 事实边界：三条不可越界的规则

这三条是本阶段最容易糊掉的地方，写在最前面，后面每个 WP 都受它约束。

**① 解析时，`ModelMetadata` 是模型接口事实的唯一来源。**
checkpoint 的 `config.json` 只进可选一致性检查（`model/checkpoint_validation.py`，阶段 3
已定），不提供也不精化任何事实。不重新引入 `BaseContract`。

**② `assembly.json` 是解析结果的「版本化执行快照」，不是新的事实来源。**
它由 ①（当时的 ModelMetadata）+ 数据描述生成，落盘之后就是这个 checkpoint 的接口契约。
推理侧**只**从它读：`schema_ref`、`norm_stats_ref`、`model_io_spec`、五类 Mapping、两条
plan。不允许「一半读快照、一半读当前声明」——那正是组合解析层要消灭的双来源。

**③ 快照与当前声明的漂移必须显式校验，不能指望权重加载捕获。**
原计划写过「声明漂移会被 `load_state_dict(strict=True)` 挡住」——**这条不成立**：
`image_input_range`、`vector_normalization`、`vision_slots[].semantic_accepts`、
`requires_prompt`、`language_template` 这些字段改了，权重 shape 一个都不变，模型照常加载，
只是喂进去的像素范围/归一化/prompt 全错，表现为「精度莫名其妙掉了」。所以 WP5 要做
一次显式的接口事实子集比对（做法见 WP5，不需要 `ModelMetadata.from_dict`）。

**推论**：推理侧对 registry entry 的使用收敛到两处——**取 factory（那是代码，无法序列化）**
与 **做 ③ 的一致性校验（那是事实）**。除此之外，推理侧的模型工厂不得再从
`entry.metadata` 推导任何 I/O 关系。

### 2.3 旧训练产物：明确不支持（取消兼容层）

要区分两类 checkpoint，它们的兼容承诺完全不同：

| 类别 | 承诺 |
|---|---|
| **外部基础模型 checkpoint**（HF repo、本地路径，经 `model.path`） | 继续支持，且继续做可选一致性检查（`validate_checkpoint_if_available`）。这是框架的入口能力，不受本次影响 |
| **旧版 VLA Factory 训练产物**（无 `inference_metadata/assembly.json`） | **不支持**。`InferenceEngine` 直接报错，提示该 checkpoint 由旧版本训练、需用当前版本重训 |

理由：所谓「回退」只有一个可行实现——在部署进程里用当时的模型声明重解析一遍。但那份
声明可能已经变了（见 §2.2 ③），重解析出来的 pipeline 未必是这个 checkpoint 训练时用的
那条，而**它会静默地跑起来**。一个静默用错归一化的推理服务，比一句「这个 checkpoint 太
旧，请重训」危险得多。仓库版本 0.1.0、训练产物的唯一生产者就是本仓库，重训成本可控。

据此删除原计划里的：重解析 fallback、`schema.json`/`norm_stats.json` 与快照混读、旧
checkpoint 兼容测试、fallback 的 0.3.0 移除周期、以及风险 R2。`resolve_from_recipe()` 也
因此不需要为部署接受外部传入的描述（WP1）。

`schema.json` / `norm_stats.json` **继续写出**——它们是人可读的产物、`inspect` 与外部工具
的输入；但**执行路径不读它们**（引擎只读 `recipe.yaml` + `assembly.json`）。

### 2.4 明确不做（逐条理由）

- **`robot_to_model` 与 JointMapping 的消费者**。阶段 3 已推迟规划侧（关节重排 / 夹爪
  flip 两个 T1 step 在 `TransformRegistry` 里没有实现），没有计划就没有可接入的东西。
  本阶段的「推理消费模型 × 机器人组合结果」只能落地 `model_to_robot`（= 今天
  `InferenceEngine` 后处理跑的链路，有真实覆盖）与 `robot_ref` 的透传；平台适配器
  （`platforms/lerobot.py` 的 motor key 顺序）继续按 `schema` 的逐维名字走。四份 example
  无一绑定真实 robot（`reference.yaml` 的 `robot.name` 是空串），唯一 fixture 的 action
  逐维名是 `dim_0..dim_7`（阶段 2 已核实必然 `JOINT_ORDER_MISMATCH`）——没有真实数据可验证。
- **`ModelMetadata.from_dict`**。§2.2 ③ 的校验是「快照 dict vs 当前声明 dict」的子集比对，
  不需要把快照反序列化回 dataclass，也不需要 `BaseContract`。
- **`resolve --json` / assembly diff**：阶段 3 已推迟，用户侧无真实需求。
- **`n_obs_steps` 与 `metadata.history_frames` 合并**：两个已适配模型都是 1，合并前后
  逐字节相同 = 无法验证。
- **LoRA / 训练策略走 assembly**：`apply_strategy` 消费的是 components 与微调能力，属于
  「模型自身事实」，不是三者关系，§4.2.6 没有点名。不动。

### WP1：`resolve_from_recipe()`——统一解析入口（orchestration adapter）

今天「recipe → 三份描述 → `resolve_assembly()`」这段胶水只存在于 `cli.py:591-689`，阶段 4
会新增两个调用方（train、InferenceEngine），照抄就是三份；`_run_resolve` 里的 override
拼装已经在 `test_resolve_mapping.py:232` 被复制了第四次。

- 新增 `vla_factory/assembly/from_recipe.py`：

  ```python
  def resolve_from_recipe(recipe: TrainRecipe) -> ResolvedAssembly:
      """Orchestration adapter — NOT a pure function.

      It touches the registry, the filesystem (dataset meta, checkpoint config)
      and the robot profile registry; ``resolve_assembly()`` stays pure behind it.
      """
  ```

  职责顺序固定为：
  1. `list_entries()` 取 `ModelMetadata`（未注册 → `UNKNOWN_MODEL`）；
  2. **可选 checkpoint 一致性检查**（`validate_checkpoint_if_available`）——这是本阶段
     把它收进统一入口的原因：现在这道检查只在 `resolve`/`inspect` 路径上跑，`train()`
     里那次（`train.py:158-164`）发生在 **`output_dir` 已经被 `rmtree` + `mkdir` 之后**，
     用户会看到「实验目录被清空了，然后告诉我 checkpoint 和模型对不上」；
  3. 取 `RobotProfile`（`recipe.robot.name` 非空时）；
  4. reader 读 `DataSchema` / `NormStats`；
  5. 拼 overrides（`recipe.assembly` 四个字段 → dict，与 `CONSUMED_OVERRIDES` 的对应关系
     收在这一处）；
  6. 调 `resolve_assembly(..., model_config=recipe.model_config, model_path=recipe.model_path)`。
- **`resolve_assembly()` 新增一个可选 kwarg `model_path`**（第 0 节第 5 点）：它只作为
  `PlanContext.tokenizer_repo` 的兜底值进入规划，**解析器不打开这个路径下的任何文件**
  （checkpoint 的读取是上面第 2 步的事）。这样落盘的每个 call 参数都是完整的，执行侧不
  需要任何隐式兜底。`pipeline_planner.plan_context()` 里那段「tokenizer_repo 恒为 None」
  的 docstring 同步改写。
  语义上这也站得住：`model.path` 属于 recipe 的**组合选择区**（架构 §3.4 明确「checkpoint
  实例：recipe `model.path` 选择」），本来就是解析器的合法输入。
- **描述必须成对**：签名里没有 `schema` / `norm_stats` 参数——§2.3 取消了部署回退之后，
  再没有调用方需要注入描述。若将来确有需要，必须**成对**注入（一个参数带两者，而不是
  两个各自可选的参数），避免「schema 来自 checkpoint、norm_stats 来自数据集」这种半截
  组合。这条写进函数 docstring。
- `cli.py:_run_resolve` 改为调用它，只保留 CLI 特有的失败打印与 exit code。

### WP2：`action_horizon` 归位 + 按 paradigm 互斥校验

规则：**horizon 对 `pretrained_finetune` 是家族事实，对 `from_scratch` 是模型 tunable。**
两者都在模型声明里，只是分属「具名字段」和「`params`」两个容器——CLAUDE.md「facts vs
tunables，容器即属性」那条规则的直接应用。

- `entries/act.py`：`params` 增加 `"action_horizon": 100`（与现有 example 一致）。
  pi0/pi05 **不加**——它们的 horizon 是具名事实，加了等于允许 recipe 覆盖预训练固定的
  chunk 长度。
- **互斥校验（不能只靠 allow-list）**。recipe 侧确实被 `resolve_recipe()` 的 tunable
  allow-list 挡住了（pi0 没声明该 key，recipe 写了直接报错），但那**约束不了 registry
  entry 自己**：一个新 entry 完全可能同时写 `action_horizon=50` 和
  `params["action_horizon"]=100`。所以在 Build IO Spec 前显式校验：

  | `training_paradigm` | 允许的来源 | 违规 |
  |---|---|---|
  | `pretrained_finetune` | 只有具名 `metadata.action_horizon` | `params` 里也有 → 声明错误 |
  | `from_scratch` | 只有 resolved `model.config.action_horizon` | 具名字段非 0 → 声明错误 |
  | 任一 | —— | 两边都没有 → 声明错误 |

  三种违规都抛朴素 `ValueError`，不是 `ResolutionError`——与 `resolve_assembly` docstring
  已定的分界一致：组合失败给结构化错误，registry entry 声明了不可能的东西给普通异常。
- `ActionSpecConfig` 四个字段默认值改成 `None`，让「用户写没写」可观测（现在
  `action_horizon` 默认 50，无法区分「用户要 50」和「用户没写」）。
- `recipe/defaults.py:resolve_recipe()` 增加迁移转发（必须放在 defaults 里——只有这里
  拿得到 `metadata.params`，知道该模型是否接受这个 tunable）：

  | 情形 | 行为 |
  |---|---|
  | recipe 显式写了 `action_spec.action_horizon`，模型声明了该 tunable（ACT），`model.config` 没写 | 转发 + 一次性 deprecation warning |
  | 同上，但模型没声明该 tunable（pi0） | 只警告「该字段已废弃，模型自带 horizon 事实，本次忽略」，**不转发**（转发会被 allow-list 拒掉，让老 recipe 直接跑不起来） |
  | recipe 没写 | 静默，用模型声明 |

- **两处配套改动**（漏了会在构造 `ACTConfig` 时炸）：
  - `utils/tracked_config.py:FRAMEWORK_CONSUMED_KEYS` 加 `action_horizon`——它的消费者是
    **解析器**（Build IO Spec），永远不会被模型工厂读到；
  - `entries/act.py:461` 的 framework-managed pop 列表加 `action_horizon`——它不是
    `ACTConfig` 的字段，留在 cfg 里会被 `**cfg` 带进去抛 `TypeError`。
- **这是一次真实的行为变化，不只是摘要数字**：一份没写 `action_spec` 的最小 ACT recipe，
  以前拿到全局默认 50，之后拿到 ACT 声明的 100。补一条测试
  （`test_act_model.py`：最小 recipe → `chunk_size == 100`）并在 commit message 里写明。
- 验证：`resolve --config act_lekiwi.yaml` 的 `horizon` 从 `0` 变成 `100`——这是本阶段
  **唯一**预期的 resolve 输出差异，其余逐字节不变。

### WP3：训练接入

- `train.py`：把 `resolve_from_recipe(recipe)` 提到**创建 output_dir 之前**（现在
  `train.py:98-101` 先 `rmtree`+`mkdir` 再读数据）。架构 §4.2.2 要求解析器在完成校验前不
  创建下游输出目录；组合不成立时不该先毁掉上一次的实验目录（与 WP1 第 2 步同一个理由）。
- `create_dataloaders(recipe, assembly)`：
  - transform pipeline ← `assembly.data_to_model`；
  - 采样窗口 ← `assembly.model_io_spec.action_horizon`（删掉 `sampler_cfg.action_horizon`）；
  - 手拼的 `model_metadata` dict 参数删除。
- 从计划实例化 pipeline 的入口（`assembly/transforms/pipeline.py`）：

  ```python
  def build_pipeline(plan: TransformPipelinePlan, ctx) -> TransformPipeline:
      return TransformPipeline([
          TransformRegistry.get(c.type).from_call(c.args, ctx) for c in plan.calls
      ])
  ```

  已核实：`from_call` 需要的运行时上下文**只有统计量**（`_stats_from(ctx)`），没有任何
  step 读 `TransformContext` 的 recipe / schema / model_config / split 字段（tokenizer 兜底
  由 WP1 在规划期解决）。ctx 因此退化成统计量载体，WP6 收尾时再瘦身。

### WP4：工厂签名 `factory(recipe, assembly)`，删掉 `action_facts.py`

adapter 里的重复推导全部来自它只拿到 `(recipe, schema)`——两个都是原料，关系只能自己再
推一遍。给它成品即可。

- `model/registry/registry.py`：factory 协议与 docstring 改为 `(recipe, assembly)`。
  `recipe` 仍要传：工厂需要 `model.path`（checkpoint 选择）与 `model_config`（tunables），
  这两样不是三者关系，不在 assembly 里。
- `ResolvedAssembly` 增加两个反序列化访问器（`functools.cached_property`，frozen dataclass
  可用，写的是实例 `__dict__`）：
  - `schema` → `DataSchema.from_dict(self.schema_ref)`，给 ACT 取 `image_sizes`（逐相机
    图像尺寸是数据侧事实，不属于 IO spec；§4.2.6 允许下游读 assembly 里保留的 DataSchema）；
  - `norm_stats` → 需要新增 `NormStats.from_dict()`（`data/manifest.py`；今天
    `infer.py:349-359` 手写了一份 `_parse_feature_stats`，一并收敛掉）。
- `entries/act.py`：`action_dim` / `action_horizon` / `state_dim` / `camera_names` 改读
  `assembly.model_io_spec`（相机取 `camera_mapping` 的恒等条目，等价于今天的
  `schema.cameras`，但来源是组合结果而非自查数据）；删 `resolve_action_*` 的 import。
- `entries/pi0.py`：`camera_mapping` 改读 `assembly.camera_mapping`（折成
  `{model_slot: data_source}`，未映射槽位不进 dict——与 wrapper 现在「未列出的角色发 -1
  占位图 + zero mask」逐字一致）；`action_dim` 改读 `model_io_spec.action_dim`（今天是
  `metadata.dim_policy_max`，阶段 3 已证明 pi0 上两者都是 32，且 IO spec 那个是 fold 出来
  的、与 pipeline 不可能矛盾）；horizon 同理。
- **删除 `vla_factory/assembly/action_facts.py`**（整个模块）及其 4 处调用。
- `cli.py:_describe_model_config`（`inspect model`）继续调 `get_camera_mapping(recipe)`：
  它读的是「recipe 写了什么」而不是「解析出什么」，属于 inspect 的职责（§3.5：inspect
  不做跨维度解析）。阶段 5 删 legacy 分支时再改一次读法。

### WP5：`assembly.json`（带 `format_version`）+ 推理接入 + 快照一致性校验

- **产物与版本**。新增 `vla_factory/assembly/artifact.py`，把版本信封关在一处：

  ```json
  {"format_version": 1, "assembly": { ...ResolvedAssembly.to_dict() }}
  ```

  `save_assembly_artifact(path, assembly)` / `load_assembly_artifact(path) -> ResolvedAssembly`。
  **第一次落盘就带版本号**：字段形状「已经定死」只是当下的事实，产物一旦分发出去就要
  长期共存；没有版本号时，未来任何形状变更都只能靠猜。未知或更高的 `format_version` →
  保守失败（§1.7），提示用对应版本的框架或重训。`ResolvedAssembly.to_dict()` 的形状本身
  **不变**（阶段 3 的约定），版本号住在信封上。
  `utils/constants.py` 增加 `ASSEMBLY_FILE = "assembly.json"`；`train.py` 在
  `_save_inference_metadata` 里与另外三件套同批写出（训练开始前写，中途 checkpoint 可用）。
- **`InferenceEngine` 只读两个文件**：`recipe.yaml`（模型名 + tunables，训练时已是
  `resolve_recipe()` 合并后的结果）与 `assembly.json`。
  - 缺 `assembly.json` → **直接失败**（§2.3），错误信息说明：该 checkpoint 由旧版本训练，
    请用当前版本重训；不提供回退。
  - `schema` / `norm_stats` ← 快照的 `schema_ref` / `norm_stats_ref`，**不再读**
    `schema.json` / `norm_stats.json`。
  - `preprocessor` ← `build_pipeline(assembly.data_to_model, ctx)`；
    `postprocessor` ← `build_pipeline(assembly.model_to_robot, ctx)`。
  - `camera_keys` / `action_dim` / `action_horizon` ← `model_io_spec`；删两处
    `resolve_action_*`，删 `num_inference_steps` 回退读 `entry.metadata.params` 的分支
    （保存的 recipe 里已经是合并后的值）。
  - `state_keys` / `action_keys` 继续用 `resolve_vector_keys(schema)`——那是数据描述的
    自校验，不是三者关系（见推迟表）。
- **快照一致性校验**（§2.2 ③）。加载时把 `assembly.metadata_ref` 与
  `_to_jsonable(entry.metadata)` 的**接口事实子集**做比对，不一致 → 失败，错误里列出
  漂移的字段名与两侧的值：

  ```text
  接口事实子集 = 解析器读过的具名字段：
    action_dim, action_horizon, dim_policy, dim_policy_max, vision_slots,
    missing_slot_policy, image_input_range, image_normalize_mode,
    vector_normalization, requires_prompt, language_template,
    control_mode_pref, expected_hz, history_frames
  ```

  子集里**不含** `install_hint` / `components` / `support_*` / `params`——它们不影响这个
  checkpoint 的接口契约，改了不该拦住部署。用一条测试守住这份清单不漂移（做法照
  `CONSUMED_OVERRIDES` 的 `test_every_assembly_override_is_accounted_for`：遍历
  `ModelMetadata` 的字段，要求每个字段要么在子集里、要么在显式的豁免列表里）。

### WP6：清理运行时二次推导（四组代码）

两个消费者都切完之后，下面这些的存在理由（「运行时从声明重新推导一遍」）就没了。逐组
删除前先 grep 确认无生产调用方：

| 删什么 | 它原来的存在理由 | 为什么现在没理由了 |
|---|---|---|
| `build_preprocessor` / `build_transforms`（`pipeline.py:91-115`） | 声明列表 → pipeline | 输入换成计划 |
| `TransformStep.from_config` / `TransformRegistry.create_from_config` | `compile_call + from_call` 的组合入口 | 规划期已 `compile_call`，执行期只 `from_call` |
| `TransformContext.plan()` + 除 `norm_stats` 外的全部字段 | 从运行时侧构造 `PlanContext` | 运行时侧不再规划；tokenizer 兜底已在 WP1 移入解析入口 |
| `inverse_for_output` + `call_args()` / `stamp_call_args` / `_compiled_call_args` | 从**已构建**的正向步骤反推后处理 | 后处理由 `model_to_robot` 计划给出 |

- 三处事实闸测试（`test_model_config_surface.py:172-207`）改为直接调
  `compile_call(cfg, PlanContext(...))`——`reject_fact_override` 本来就住在那里，闸门一寸
  没松，只是不再经过 `from_config` 这层壳。
- 阶段 3 的等价性测试（`test_resolve_mapping.py:226`）的对照物消失，改写成：把
  `data_to_model` 计划实例化出来跑一个真实 sample，断言输出 shape / dtype / 数值范围
  （state 被 pad 到 32、图像落在模型声明的 range 内）。它从「两份实现是否一致」变成
  「唯一那份实现是否真的能跑」——阶段 3 R2 期待的接班人。
- `resolver/__init__.py` docstring 里残留的 `Build Interface` 顺手改成 `Build IO Spec`。

### WP7：文档同步

- **架构 §7.4 阶段 4 中英两版**：「保留显式兼容层并提供迁移告警」→ 按 §2.3 改写为
  「外部基础 checkpoint 继续经 `model.path` 支持并做可选一致性检查；旧版训练产物（无
  `assembly.json`）明确不支持，缺失即保守失败」。同时标注阶段 4 完成范围与
  `robot_to_model` / 平台适配器仍在推迟。
- **§4.2.1 `ModelIOSpec` 一句澄清**（`types.py` docstring 同步）：`cameras` 是**框架
  Observation 使用的数据侧 canonical camera key**，**不是**模型的 `vision_slots`。pi0 上
  两者不同（数据侧 `front`/`wrist`，模型侧三个 openpi 角色），连接它们的是 `CameraMapping`。
  现在这一点只能从代码推出来，读文档的人极易误解。
- CLAUDE.md：Code structure（新增 `assembly/from_recipe.py`、`assembly/artifact.py`，删掉
  `action_facts.py`）、How it runs（train/deploy 两条链路都经过 assembly；部署产物多一件
  `assembly.json`）、Key ideas 增加「具身组合是下游唯一入口」与 §2.2 三条边界。
- `adapt_new_model` skill：工厂签名 `(recipe, schema)` → `(recipe, assembly)`；「从 schema
  自己推 action_dim」的写法改成读 IO spec；新模型的 horizon 按 paradigm 放对容器（WP2 的
  互斥表）。这是适配新模型时第一件会踩的事。

### 2.5 阶段 4 验证（**实测结果**）

- `pytest`：**351 passed**（基线 306；净增来自新增的
  `test_assembly_artifact.py` / `test_action_horizon_source.py` /
  `test_resolve_from_recipe.py`、以及原先永远 skip 的 `TestEndToEnd` 开始真跑，
  减去删掉的 `test_action_facts.py`）。
- `resolve --config examples/act_lekiwi.yaml`（数据指向真实 fixture）：与阶段 3 输出
  逐行一致，**唯一差异** `horizon: 0 → 100`，如计划所述。
- 真实 2-step 训练（act + fixture，CUDA）：产出
  `inference_metadata/{assembly.json,recipe.yaml,schema.json,norm_stats.json}`，
  `assembly.json` 首行即 `"format_version": 1`。
- 同一 checkpoint 上 `infer`（`action_shape (100, 8)`）与
  `evaluate`（1 episode / 414 frames，平均 L1 0.1973）均正常。
- 失败路径：删掉 `assembly.json` → `FileNotFoundError` 指明「旧版本训练的 checkpoint，
  请重训」（`test_train_infer_roundtrip.py`）；`format_version=999` → 保守失败；
  篡改 `image_input_range` / `vector_normalization` / `requires_prompt` / `dim_policy`
  任一 → `AssemblyDeclarationDrift` 并列出漂移字段（`test_assembly_artifact.py`）。
- 顺序：组合失败时 `train()` 不再动 output_dir
  （`test_failed_resolution_leaves_the_output_directory_untouched`）。
- pi0 侧仍**未**在 uv/openpi 环境实跑（风险 R4 未关闭），见下方「遗留」。

### 2.6 阶段 4 原计划验证项

- 每个 commit 后 `pytest` 全绿（基线 306 passed / 3 skipped）。
- `resolve` 输出与阶段 3 逐字节一致，**唯一预期差异**：act 的 `horizon` 0 → 100。
- 最小 ACT recipe（不写 `action_spec`）→ `chunk_size == 100`（WP2 的行为变化）。
- act + 真实 fixture 跑 `train --steps 2`：产出 `inference_metadata/assembly.json`，
  `{"format_version": 1, ...}`，内层与阶段 3 的 act 组合断言一致（落盘 → load → `to_dict`
  round-trip）。
- 用该 checkpoint 跑 `infer` / `evaluate`；`deploy --platform simulator` 能起。
- **失败路径三条**（都要断言错误信息可读、不是深层异常）：
  1. 删掉 `assembly.json` → 报「checkpoint 由旧版本训练，请重训」，**不回退**；
  2. 把 `format_version` 改成 `999` → 保守失败；
  3. 篡改 `metadata_ref.image_input_range` → 快照一致性校验失败并列出漂移字段
     （这条正是「权重能加载但语义已错」的回归用例）。
- `test_train_infer_roundtrip.py` 每个 WP 后都必须绿——端到端护栏。
- grep 无残留：`action_facts`、`build_transforms`、`build_preprocessor`、`from_config(`、
  `inverse_for_output`、`resolve_action_dim`。
- pi0 侧：在 uv 装的 pi0 环境跑一次 `test_pi0_model.py` + 200 step 训练冒烟。默认环境
  跳过 heavy extras，**act 全绿不等于 pi0 没坏**（风险 R4）。

### 2.7 code review 后的整改（**已执行**；shape 方案随后按 2.8 收敛）

六条评审意见全部复现属实，其中两条是本阶段引入的 blocker：

**1. padding 模型的推理必炸（引入的回归）。** `InferenceEngine.action_dim` 取了
`io_spec.action_dim`（pi0 = 32），而 `model_to_robot` 的 `unpad_action` 把动作还原成
8 维，`_predict_chunk` 又按 32 校验——每次预测都会失败；`LerobotHostActionAdapter`
也会因为「32 维 vs 8 个 motor key」在初始化就拒绝启动。阶段 4 之前这里是
`resolve_action_dim(schema, ...)` = 8，旧注释写明了这一点，被我一并删掉。
漏网原因：唯一跑真实 `_predict_chunk` 的是 ACT，而它没有 pad step。
**第一版改法（已由 2.8 取代）**：曾给 `UnpadAction` 增加 `output_widths` 并用
`fold_widths()` 反推执行宽度。最终保留“两端分别校验”的结论，但删除反推 hook：引擎用
`model_output_dim` 校验模型原始输出，用 `execution_action_dim` 校验后处理结果并供平台
adapter 使用。新增
`test_padded_model_inference.py`：用真实解析出的 padding 组合 + stub 网络，让一个
padding 模型完整走完预测——这条测试是这个 bug 唯一的结构性护栏。

**2. `assembly.json` 读取过松。** 实测：从合法产物里删掉 `model_to_robot`，
`from_dict` 的空默认值让它变成 unresolved 空计划，`build_pipeline` 照样构造出**空后
处理器**——ACT 会把归一化空间里的动作直接下发，shape 校验还照过；删掉
`metadata_ref` 里的接口事实，漂移检查直接通过（`if key in stored` 把「缺字段」当成
「无需比较」）。**改法**：`load_assembly_artifact` 增加 v1 结构校验（三份描述 +
`model_io_spec` 必须非空、两条 plan 必须 `resolved=True`、`INTERFACE_FACTS` 必须齐
全）；漂移检查把缺字段计为漂移；`build_pipeline` 拒绝未解析计划。

**3. `--camera-names` 绕过 CameraMapping。** ACT 抛 `KeyError`（响亮），pi0 走
`camera_mapping.get(role)` 找不到就发 -1 占位图 + zero mask —— **模型全盲但继续推理**。
`robot_to_model` 落地前这个入口没有存在理由，直接删除（CLI flag 一并删）；平台自己的
相机命名归 PlatformAdapter。

**4. 副作用屏障不完整。** 原计划只把「解析」提到了 mkdir 之前，但
`resolve_vector_keys`、finetune-only 的 `model.path` 检查、`data_to_model.resolved`
检查都还在后面——实测 pi0 不写 `model.path` 会先清空上一次实验目录再报错。**改法**：
vector keys 校验移入 resolver 的 Validate 阶段（失败给结构化 `INVALID_DESCRIPTION`，
与上一轮推迟表里的说法相反，评审判断更对：它属于描述自校验，正是 Validate 的职责），
另两项提到 mkdir 之前。连带效果：`make_schema` 现在给每个维度生成默认名字——真实
reader 永远会填，无名维度是「坏 reader」的产物，专门测它的用例改为直接构造 DataSchema。

**5. ACT 二次推导图像尺寸。** `ModelIOSpec` 新增 `camera_shapes`。第一版曾由
`TransformStep.output_image_sizes` fold 得出，最终按 2.8 改为从模型/数据事实直接解析。
ACT 工厂改读它，`_resolve_resize_image_size` /
`_schema_image_size` 两个函数删除。三处臆造默认值（`state_dim or action_dim`、
`or ["top"]`、引擎的 `("front",)`）一律改成显式失败。趁 artifact v1 未发布把字段补齐，
不需要动 `format_version`。

**6. 小清理。** `pipeline.py` 补回 `Iterable` 导入（`typing.get_type_hints()` 实测
`NameError`）；`PlanContext.of()` 零调用方，删除。

### 2.8 shape 事实源收敛（**已执行**）

确认 transform 不是模型接口事实源：未来删除 `model.transforms` 时，模型所需尺寸仍必须
成立。因此解析顺序改为 Resolve Mappings → Build IO Spec → Plan Pipeline：

- `StateMapping` / `ActionMapping` 只记录真实维度对应关系，不再按模型宽度生成无来源的
  padded entries；padding 数量由 `ModelIOSpec` 目标宽度减去真实 mapping 数量得到。五类
  Mapping 因此可以在同一阶段解析，`CameraMapping` 随后作为 IO 图像尺寸投影的输入。
- 向量目标宽度直接来自 `ModelMetadata.dim_policy` / `action_dim` / `dim_policy_max`，
  flexible 模型回退到 `DataSchema`；缺少 pad placeholder 时 planner 从 IO 差异补 call。
- 固定视觉尺寸来自 `VisionSlot.resolution`；ACT 暴露 `input_image_size`，未设置时采用
  `DataSchema` 原生尺寸。`resize_images` 声明只保留 mode/interpolation 等执行策略。
- 删除 `output_widths` / `output_image_sizes` / `vector_widths` / `camera_shapes` /
  `fold_widths`，避免 step 参数语义泄漏回 resolver。
- 推理宽度命名为 `model_output_dim` / `execution_action_dim`；前者取
  `ModelIOSpec.action_dim`，后者当前取 schema 动作空间，待 robot-side IO 落地后再切换来源。

### 2.9 openpi 侧验证（**R4 已关闭**）

在装了 openpi 的环境（`envs/vla_factory_pi`）实测，全程**不加载预训练权重**——接线是否
正确与权重无关，而权重加载慢且验证的是上游模型本身：

- 全量 `pytest`：**330 passed / 21 skipped**（lerobot 侧用例在该环境跳过）。
- **真实 `Pi0Config` + 真实 `PI0Pytorch`**（`gemma_2b` + `gemma_300m`，`model.path=None`
  只构造结构）经 `resolve_from_recipe` → `factory(recipe, assembly)` 建成：
  `config.action_dim=32` / `action_horizon=50` 均来自 `model_io_spec`；wrapper 拿到的
  `camera_mapping` 正是 `{base_0_rgb: front, left_wrist_0_rgb: wrist}`，未映射的
  `right_wrist_0_rgb` 不进 dict（走占位图 + zero mask）。
- 真实 `compute_loss`（2.49）与 `predict_actions` → `(1, 50, 32)`，即
  `io_spec.action_horizon × io_spec.action_dim`。
- **训练侧数据链路，零模型依赖**：pi0 的 `resolve_from_recipe` → `create_dataloaders`
  取一个 batch —— images `(2, 3, 224, 224)` float32 落在 `[-1, 1]`、state `(2, 32)`、
  actions `(2, 50, 32)`、prompt `(2, 48)`，与 `ModelIOSpec` 逐项一致。

仍未做、也**不打算在本阶段做**的只剩一件：用真实预训练权重跑一次 pi0 微调。它验证的是
收敛性与数值精度，不是本阶段改动的接线；需要时单独安排。

---

## 3. 阶段 5：Recipe 瘦身

阶段 4 结束时下面这些字段已经零读者。阶段 5 只做两件事：**删字段**、**给用户一条可读的
迁移路径**。

### WP1：删 `action_spec` 块

- `recipe.py` 删 `ActionSpecConfig`；`parser.py` 删解析；四份 example 删该块；
  `defaults.py` 里 WP2 那条转发一并删除。
- 迁移：读到 `action_spec` 就发 deprecation warning，逐字段说明去处——`action_horizon`
  → 模型声明（ACT 走 `model.config.action_horizon`，pi0 无需配置）；`action_dim` → 数据集
  事实（`inspect data` 可看）；`action_type` → 数据侧 `action.dims[].mode` / 机器人侧
  `native_action_type`；`bounds_*` → `RobotProfile.safety_bounds`。后三个从来没被读过，
  「忽略 + 告知去处」不改变任何行为。
- 不需要为旧 checkpoint 的 ACT horizon 做任何处理：带 `assembly.json` 的 checkpoint 里
  horizon 已经烤进 `model_io_spec`，不带的本来就不支持（§2.3）。

### WP2：删 `data.sampler.action_horizon`

采样窗口长度必须等于模型的 action horizon（今天靠用户在两处写同一个数保证）。阶段 4 已让
`create_dataloaders` 改读 `model_io_spec.action_horizon`，这里删字段即可。`SamplerConfig`
只留 `type` / `n_obs_steps`。

### WP3：删 `training.inference_steps`

`parser.py:119-132` 已有转发 + 告警（去 `model.config.num_inference_steps`），
`TrainRecipe.inference_steps` 字段零读者。删字段，保留读到即告警。

### WP4：删 legacy 组合入口

| 删什么 | 现状 |
|---|---|
| `model.config.camera_mapping` / `model.config.default_task` 两条 legacy 分支 + `get_camera_mapping` / `get_default_task` 两个 helper | 已带 deprecation warning；阶段 4 后 entries 不再调，只剩 `inspect model` 一个读法，改为直接读 `recipe.assembly` |
| `composition:` 旧块名（`parser.py:87-89`） | 阶段 0 改名时的过渡，仓内零使用 |
| `AssemblyConfig.accept_fps_mismatch` / `gripper_flip` | 解析器 `CONSUMED_OVERRIDES` 不含它们——**写了就报 `UNSUPPORTED_OVERRIDE`**，即一个永远失败的字段。随对应检查（频率 / 夹爪，均在阶段 2 推迟表里）一起恢复 |
| `utils/tracked_config.py:FRAMEWORK_CONSUMED_KEYS` 里的 `camera_mapping` / `default_task` | 随上面的 legacy 分支一起删 |

### WP5：迁移提示与移除周期

- **迁移提示，不新增命令**。架构原文是「迁移命令**或**可读提示」，二选一。选提示：
  parser/defaults 的所有 deprecation 收敛到一个出口 `_warn_deprecated(field, replacement)`，
  并在 `resolve` 摘要末尾列出本次 recipe 命中的废弃字段。新增 `migrate` 子命令要处理 YAML
  注释保留、原地改写、备份三件事，收益是省用户几行手改——不成比例。
- **移除周期只有一档**（§2.3 取消了训练产物兼容层，checkpoint 内 `recipe.yaml` 的废弃字段
  由 `parser` 的「未知 key 忽略」天然消化，不需要单独条目）：

  | 兼容对象 | 周期 |
  |---|---|
  | 用户手写 recipe 的废弃字段（`action_spec`、`sampler.action_horizon`、`training.inference_steps`、两条 legacy 组合入口） | **0.2.0 告警 → 0.3.0 删除** |

  理由：仓库 0.1.0 早期、调用方全在仓内，长兼容期只会让两种写法长期并存。周期写进
  CLAUDE.md 与 `examples/reference.yaml` 抬头。

### WP6：示例与文档

- `examples/reference.yaml` 重写（它是「每个字段都有注释」的模板，字段删了必须同步）；
  另外三份 example 删对应块。
- 架构 §3.2 字段概览表中英两版同步；§7.4 阶段 5 标注完成。
- `docs/modules/recipe-module.cn.md` 从 TODO 补出「三个区 + 字段来源 + 废弃周期」一节
  ——这是本次唯一有资格脱离 TODO 的模块文档，因为 recipe 的形状到这一步才定下来。
  `assembly-module.cn.md` 继续 TODO（要对齐完整实现，`robot_to_model` 还缺）。

### 3.1 阶段 5 验证

- `pytest` 全绿；四份 example `resolve` 成功且摘要与阶段 4 一致。
- 构造一份「全是废弃字段」的旧 recipe：逐条告警、不失败、解析结果与新写法等价。
- act 真实 fixture 上再跑一次 `train --steps 2` + `infer`，动作数值与阶段 4 结束时一致
  （recipe 瘦身不应改变任何数值行为）。

---

## 4. 明确推迟（供下次启动参考）

| 项 | 现状 | 恢复时需要的前置工作 |
|---|---|---|
| `robot_to_model` 规划与消费 | 阶段 3 推迟；`TransformRegistry` 无关节重排 / 夹爪 flip step | 实现两个 T1 step + 一份绑定真实 robot 的 recipe |
| 平台适配器改读 JointMapping | 现在按 `schema` 逐维名字取 motor key | 同上；且需要 action 侧带真实关节名的数据集（现 fixture 是 `dim_0..7`） |
| `n_obs_steps` 与 `metadata.history_frames` 合并 | 两个已适配模型都是 1，合并前后逐字节相同 | 适配一个 history > 1 的模型 |
| `resolve_vector_keys` 并入解析器 Validate 阶段 | 它是数据描述自校验（每维一个名字），不是三者关系 | 需要时随 Validate 阶段的其他数据校验一起搬，别单独动 |
| `assembly.json` 的 `format_version` 2 及迁移器 | 本阶段只落地 v1 + 未知版本保守失败 | 第一次真要改形状时，按那次的改动写迁移，而不是现在预设一套 |
| `resolve --json` / assembly diff | 摘要 + 组合断言已够 | 用户提出真实 diff 需求 |
| `TransformPipelinePlan.risk` / `.reversible` | 阶段 3 WP0 删除（零消费者） | 有消费者要按风险/可逆性分支时 |
| 训练策略（LoRA / freeze）走 assembly | 消费的是模型自身事实，不是三者关系 | §4.2.6 没点名，除非出现跨维度的策略决策 |

---

## 5. 提交切分

| # | 阶段 | 类型 | 内容 |
|---|---|---|---|
| 1 | 4 | `refactor:` | WP1：`resolve_from_recipe()`（含 checkpoint 检查前置、`model_path` → tokenizer 兜底），CLI 改为调用它 |
| 2 | 4 | `feat:` | WP2：`action_horizon` 归位 + paradigm 互斥校验 + defaults 转发 + ACT 默认值行为变化测试 |
| 3 | 4 | `feat:` | WP3：训练接入（`build_pipeline` + `create_dataloaders(recipe, assembly)` + 解析前置于 mkdir） |
| 4 | 4 | `refactor:` | WP4：工厂签名 `(recipe, assembly)`；删 `action_facts.py`；`ResolvedAssembly` 访问器 + `NormStats.from_dict` |
| 5 | 4 | `feat:` | WP5：`assembly.json` + `format_version` + 推理接入 + 快照一致性校验 + 三条失败路径测试 |
| 6 | 4 | `refactor:` + `test:` | WP6：删四组运行时二次推导 + 测试改写 |
| 7 | 4 | `docs:` | WP7：架构 §7.4 兼容层原文改写、`ModelIOSpec.cameras` 澄清、CLAUDE.md、skill |
| 8 | 5 | `refactor:` | 阶段 5 WP1–WP4：删废弃字段与 legacy 入口 |
| 9 | 5 | `feat:` + `docs:` | 阶段 5 WP5–WP6：迁移提示 + 移除周期 + 示例与文档 |

顺序依赖：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9，无并行分支。

> 阶段 1/2/3 最终都落成单个 commit（评审整改与主改动落在同一批文件，事后拆只会产出跑不过
> 测试的中间态）。本阶段刻意让 1–6 各自可独立通过 `pytest`——它们改的是不同文件、不同
> 消费者。若评审整改再次跨 WP，按同样理由合并，不要制造跑不过的中间态。

---

## 6. 风险

| # | 风险 | 应对 |
|---|---|---|
| R1 | 阶段 3 的等价性测试是「计划正确」的唯一护栏；WP6 删掉它的对照物（声明式构建）之后，若先删再切，两侧都没护栏 | WP6 强制排最后；WP3/WP5 切换期间等价性测试持续绿，切完才改写它 |
| R2 | 快照一致性校验的字段子集漏一项（例如将来新增一个影响像素范围的具名事实却忘了加进子集），漂移重新变回静默 | 用遍历 `ModelMetadata` 字段的清单测试守护：每个字段要么在子集里、要么在显式豁免列表里，新增字段不表态就红 |
| R3 | 工厂签名改动牵连 3 个 entry + 2 个调用方 + 6 个测试 + skill/docs，中途留两种签名会让 registry 出现「有的 entry 收 schema、有的收 assembly」 | 单个 commit 改完（提交 4），registry 协议同批更新；不提供兼容重载 |
| R4 | pi0/pi05 用例在默认环境全部 skip，act 全绿不代表 openpi 侧没坏（工厂改动同时影响三个 entry） | 阶段 4 结束前在 uv pi0 环境跑 `test_pi0_model.py` + 200 step 训练冒烟；结果写回本文档的实施记录段 |
| R5 | ACT 默认 horizon 50 → 100 是真实行为变化，未声明 horizon 的存量 recipe 训出来的 chunk 长度会变 | WP2 补最小 recipe 测试 + commit message 显式声明；四份 example 都写了 100，受影响的只有未显式声明的自定义 recipe |
| R6 | 阶段 5 删字段时误删仍被读的字段（例如 `sampler.n_obs_steps`） | 每删一个字段先 grep 全仓读点；删完跑一次 act 的 `train --steps 2` + `infer`，数值必须与阶段 4 结束时一致 |

# 开发计划：组合解析迁移阶段 2「解析诊断」

> 状态：**WP1–WP5 已执行完毕**（工作树，未提交）。2026-08-10 按范围反馈重新
> 收窄后同日实施完毕。**与阶段 1 同分支**（`feat_inspect`）——两阶段关联紧密，
> 阶段 2 的检查逻辑直接消费阶段 1 刚落地的字段，拆分支只会制造合并冲突，不
> 产生真实隔离价值。
> **后续决策（2026-08-11）：** 本计划中从 BaseContract/Materialize 读取精化
> 维度的方案已撤销。Check Pairs 现直接读取 ModelMetadata；checkpoint 只能
> 可选校验，不能改变解析结果。下文相关内容仅保留为历史实施记录。
> **后续决策（2026-08-13）：** RobotProfile 与 DataSchema 的相机名/关节名不再
> 通过字符串启发式做硬兼容检查。Platform Adapter 负责输出 DataSchema 接口；
> RobotProfile 本阶段只校验自身结构。本文关于 robot camera candidates、关节名称
> 子集嵌入和对应错误码的内容是历史实现记录，后续方案见
> `runtime-pipeline-convergence.cn.md`。
> 依据：`docs/architecture/vla-factory-architecture.cn.md` §7.4 阶段2、
> §4.2.2（兼容性检查矩阵）、§4.2.5（解析失败处理）。
>
> **实施中用真实数据验证/修正的三处**（正文已就地更新）：
> 1. **动作维度的机器人侧参与，从"精确子集嵌入"简化为"粗粒度数量守卫"**——
>    实测 LeKiwi 数据集 action_dim=8、机器人声明 9 个关节，若按名字精确匹配
>    会与关节顺序检查重复且规则冲突；改为只check `data_dim <= len(robot.joints)`
>    的必要条件，精确的按名匹配完全交给关节顺序检查（WP4）负责，两处不重叠。
> 2. **状态/动作维度检查改读 `merged["state_dim_from_contract"]` /
>    `merged["action_dim"]`（Materialize 阶段已产出的精化值），不是重新读
>    `metadata.dim_policy_max`**——否则 checkpoint 通过 BaseContract 收窄的
>    真实 pad target 会被忽略，Check Pairs 可能和 `canonical_interface` 算出
>    两个不一致的"最终维度"。
> 3. **关节顺序检查在真实数据上发现一个组合解析层面的真实发现，不是 bug**：
>    仓库唯一的真实测试数据集 `meta/info.json` 把 action 特征的逐维名字写成
>    `dim_0`..`dim_7`（而非真实关节名），state 侧则有真实名字（带 `.pos` 后缀）。
>    这让 ACT/pi0 + LeKiwi 机器人的组合在 action 侧必然报
>    `JOINT_ORDER_MISMATCH`——按 data-module §8.1「undeclared 是受控 override
>    的触发依据」的既定哲学，这是正确行为，已经写成 golden test，不是需要绕过
>    的障碍。

## Context

阶段 1（commit `ffbc71c`）把兼容性矩阵 11 项检查中大部分要用的事实喂饱了：
`DataSchema` 的逐条目表、`ModelMetadata` 的具名事实字段、`RobotProfile` 的
关节/控制模式/安全边界。`resolver.py` 的 Materialize 阶段实际已经比阶段 1
计划文档记录的更完整——不只 `action_dim` 一例，`dim_policy`、
`vector_normalization`、`expected_hz`、`vision_slots` 的能力边界检查都已落地
（该文档那句「现在只有 action_dim 一例」已过期，一并在本次收尾时更正）。

### 范围收窄：只做架构文档点名的 5 类，不是矩阵全部 11 行

初版计划把 §4.2.2 的完整目标态矩阵（11 行）当成了阶段 2 的范围。重新核对
§7.4 对阶段 2 的原文描述后发现这是范围过大——原文只点了五类：

> 「对**维度、相机、统计量、控制模式和字段顺序**提前报错」

对应矩阵：状态维度、动作维度、相机槽位、控制模式、归一化统计、关节顺序——
**6 行**。矩阵里另外 5 行（语言输入、夹爪约定、旋转表示、频率、安全范围）
架构文档没有把它们划进阶段 2，本轮**明确不做**，理由逐条见下：

- **语言输入**：`task_tokenize` 已有 `sample task > default_task > 空 prompt`
  的运行时兜底链，从不硬失败；矩阵「无兜底则报错」的行为和现有运行时行为
  冲突，值得做但不是现在，且不在架构点名范围内。
- **夹爪约定**：三侧事实都缺（模型侧的 `gripper_convention`、机器人侧的
  `grippers[]` 复数、数据侧的显式 convention 都没有），是矩阵里最重的一行，
  且不在架构点名范围内——设计已经写进 `robot-module.cn.md §4.3`，代码本轮
  不动，留给需要时再启动。
- **旋转表示**：`rotation_repr` 在任何维度都不存在，随 EEF 组一起推迟
  （data-module §8.3 已有此安排），无争议。
- **频率**：矩阵「默认 warning，不隐式重采样」——这是 6 行必做检查里唯一
  可能需要「成功但带提示」语义的一类，砍掉这行之后**不需要再为阶段 2 设计
  explain trace 载体**（见下）。
- **安全范围**：当前所有已知 `RobotProfile`（LeKiwi、RoboTwin）的
  `safety_bounds_*`/`limits_*` 都是空的，这行检查写出来也大概率跑不出非空
  分支——现在实现属于没有真实数据能验证的死代码，等有真实限位再做。

**连锁简化**：砍掉频率和安全范围后，剩下 6 行全部是二元判断（过/不过），
没有一行需要「警告不失败」的中间态——不需要新设计 explain trace 载体
（`ExplainEntry`），继续用现成的 `ResolutionError`（成功就是成功，失败就是
结构化失败，不引入第三种状态）。

**连锁简化**：相机槽位检查不需要给 `RobotProfile.cameras` 新增
`{key, semantic}` 结构。已验证：把阶段 1 写好的 `infer_camera_semantic()`
直接套在机器人的相机名字上就够用——`head_camera` 能唯一命中
`third_person_front`（函数本身识别 "head" 关键词），`left_camera`/
`right_camera` 推不出来时返回 `None`，作为「不参与自动匹配」的安全默认值，
不算错误也不算命中。这和之前设计的「留空」效果一致，但不需要新建任何
dataclass 字段，也不需要猜 RoboTwin 相机的物理朝向。

**结果**：收窄后的阶段 2 **不需要在任何维度新增 dataclass 字段**——阶段 1
留下的字段直接够用。工作全部落在 `resolver.py`（新增 Check Pairs 阶段）和
`errors.py`（新增错误码），没有 breaking change，没有新 API 形状。

### 关节顺序检查踩到的真实坑（唯一一处需要非平凡逻辑的地方）

6 行里 5 行是直接的字段比较，只有关节顺序需要一点设计——我用真实测试数据
验证过，裸字符串比较会立刻在仓库唯一一份 golden fixture 上产生假阳性：

- **后缀不能裸比**：LeKiwi 数据集的 `state_dims[].name` 是
  `shoulder_pan.pos`（data-module §8.3 规定「保留原始后缀不剥离」），机器人
  侧 `joints.names` 是 `shoulder_pan`（无后缀）。裸比较会误判为「无法重排」。
- **不是集合相等，是子集嵌入**：LeKiwi 数据集只有 6 维（6-DoF 臂），机器人
  声明 9 个关节（3 base + 6 arm）。检查要做的是「数据维度能否唯一嵌入机器人
  关节集合的某个子集」，机器人关节比数据多是合法的（base 没被训练数据记录），
  不该报错。

规则：两侧都按 `data/semantics.py:infer_action_mode` 已经在用的同一张后缀表
（`.pos`/`.vel`/`.delta`）剥离后缀，再做子集唯一嵌入——唯一命中才生成
JointMapping，命不中或多命中才是错误。不新起一份后缀规则。

---

## WP1：Check Pairs 骨架 + 维度检查（矩阵行 1-2）

- `resolver.py` 新增 `_check_pairs()` 阶段，插在 Materialize 和 Build
  Interface 之间；返回一组错误而不是即时抛出——同一次解析里独立的问题要能
  一起收集后统一报（架构 §4.2.5 已有此要求，阶段 0 未实现）；本阶段涉及的
  6 类检查全部二元，收集到的要么是空列表（成功），要么是一个或多个
  `ResolutionError` 候选，取第一个/合并抛出。
- 状态维度：`schema.state_dim` vs `metadata.dim_policy`（`fixed` 不等则
  错误；`padded_to_max`/`flexible` 放行）。
- 动作维度：同上 + 机器人侧 `len(joints.names)`（仅当
  `native_action_type` 直接等于关节堆叠时参与比较）。
- 新错误码：`STATE_DIM_INCOMPATIBLE`、`ACTION_DIM_INCOMPATIBLE`。

## WP2：相机槽位检查（矩阵行 3）

- 数据 × 模型：`schema.cameras_entries[].semantic`（已推断）与
  `metadata.vision_slots[].semantic_accepts` 求交，唯一命中 → 待生成
  CameraMapping 条目（阶段 3 才真正生成 Mapping 本体，本阶段只诊断该不该报
  歧义）；零命中且槽位 `required` → 错误；多候选 → `CAMERA_SLOT_AMBIGUOUS`。
- 机器人 × 模型：对 `robot.cameras` 里每个字符串复用
  `infer_camera_semantic()`，推不出来的按「不参与自动匹配」处理，不报错、
  不算命中——`RobotProfile` 不改动，`data/semantics.py` 的推断函数原样复用。
- 未映射的必需槽位 → padding 规划留给阶段 3，本阶段只确认「有解」。

## WP3：控制模式 + 归一化统计检查（矩阵行 5、8）

- 控制模式：`schema.action_dims[].mode` 集合 vs `metadata.control_mode_pref`
  vs `robot.control_modes` 三者交集（`robot.control_modes` 阶段 1 已有，
  不需要新字段）；空交集 → `CONTROL_MODE_INCOMPATIBLE`。
- 归一化统计：`metadata.vector_normalization == "quantile"` 时校验
  `NormStats` 是否带 q01/q99；`mean_std` 校验 mean/std 非空；
  `NORM_STATS_INSUFFICIENT`。

## WP4：关节顺序检查（矩阵行 10）

- 实现上面「后缀剥离 + 子集唯一嵌入」的规则；唯一嵌入 → 待生成
  JointMapping 条目（同 WP2，本阶段只诊断不产出 Mapping 本体）；
  `JOINT_ORDER_AMBIGUOUS`（多命中）/`JOINT_ORDER_MISMATCH`（零命中）。

## WP5：`resolve` 摘要展示 + golden tests

- `cli.py`：`resolve` 子命令的成功摘要里补上本阶段新增的诊断信息（如「相机
  槽位：2/3 命中，1 个走 padding」），不需要新增 `--explain` flag——没有
  警告态，成功输出和失败输出保持现有的二元形状。
- golden tests：为「成功」「结构化失败」两类各挑 2-3 个代表性组合（用现有
  LeKiwi + 3-episode 测试数据集 + act/pi0 metadata 拼），固定输出，回归比对。
- 阶段 1 计划文档的 WP2 那句「现在只有 action_dim 一例」顺手更正为
  「Materialize 已覆盖 dim_policy/vector_normalization/expected_hz/
  vision_slots」。

---

## 明确推迟、不在本阶段范围内的工作（供下次启动时参考）

| 项 | 现状 | 恢复时需要的前置工作 |
|---|---|---|
| 语言输入检查 | `task_tokenize` 运行时兜底已覆盖，架构未点名阶段 2 覆盖 | 无新字段需求，直接是一行 explain/warning 级别检查，可随时补 |
| 夹爪约定检查 | 设计已写入 `robot-module.cn.md §4.3`（`grippers[]`），代码未动 | 落地 `RobotProfile.grippers`、`ModelMetadata.gripper_convention`（详见该文档） |
| 旋转表示检查 | 无任何维度有 `rotation_repr` 事实 | 随 EEF 模型适配一起启动（data-module §8.3） |
| 频率检查 | 无字段缺口，纯粹因为需要「警告不失败」语义被本轮排除 | 需要先设计 explain trace 载体（本轮评估过，为它单独引入过重） |
| 安全范围检查 | 无字段缺口，但所有已知 profile 的限位字段都是空的 | 等任一 `RobotProfile` 填上真实 `limits_low/high` 后再实现，否则是死代码 |

---

## 提交切分

| # | 类型 | 内容 |
|---|---|---|
| 1 | `feat:` | WP1：Check Pairs 骨架 + 维度检查 |
| 2 | `feat:` | WP2：相机槽位检查 |
| 3 | `feat:` | WP3：控制模式 + 归一化统计检查 |
| 4 | `feat:` | WP4：关节顺序检查 |
| 5 | `feat:` + `docs:` | WP5：resolve 摘要 + golden tests + 阶段1文档更正 |

顺序依赖：1 → (2, 3 可并行) → 4 → 5。

## 验证

- 每个 commit 后 `pytest` 全绿。
- `vlafactory-cli resolve --config examples/act_lekiwi.yaml` 成功路径不报错
  （回归基线）；构造一个 action_dim 故意不匹配的 recipe，确认
  `ACTION_DIM_INCOMPATIBLE` 结构化报错、`code`/`path`/`params` 符合专用构造
  入口。
- **行为不变**：`train`/`infer` 路径本阶段不改，跑一次已有的
  `test_train_infer_roundtrip.py` 确认零影响（Check Pairs 只在 `resolve`/
  `inspect` 路径触发，训练/推理入口不调用它）。

## 风险

| # | 风险 | 应对 |
|---|---|---|
| R1 | Check Pairs 收集多个独立错误后统一抛出（§4.2.5 要求），但阶段 0 的 `_load`/`_materialize`/`_validate` 都是「遇错即抛」的老写法，混用两种风格容易出 bug | WP1 明确 `_check_pairs()` 用「收集后统一抛」，前三个阶段维持现状不改，两种风格在 `resolve_assembly()` 里衔接处写清楚注释 |
| R2 | 相机槽位检查复用 `infer_camera_semantic()` 时，机器人侧字符串（如 `left_camera`）和数据侧字符串的关键词分布可能不同，函数是为数据侧措辞调的，直接套用到机器人名字上可能命中率偏低 | 命中率低的后果只是「不参与自动匹配」（安全默认），不是误判——golden test 里用 RoboTwin 的三个相机名验证这条「推不出来」路径确实按预期返回 `None` 而非抛错 |
| R3 | WP4 关节顺序检查是本阶段唯一涉及"多对多匹配"逻辑的检查，比其余 5 行复杂，容易在切分维度（子集而非全集）上出边界错误 | 用 LeKiwi（6 维数据 ⊂ 9 关节机器人）和一个"数据维度在机器人里零命中"的构造用例各写一个 golden test，覆盖唯一嵌入与不匹配两条路径 |

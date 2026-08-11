# 机器人（robot）模块设计

> 文档状态：**TODO** —— 本文档待补充。完成后对齐**当前已实现**的行为（架构文档描述目标架构，可能超前于实现），供读者参照学习。
> 对应架构：见 [总体架构 § 4.1.3 机器人：RobotProfile](../architecture/vla-factory-architecture.cn.md) 与 [§ 2.2 目录结构 `robot/`](../architecture/vla-factory-architecture.cn.md)。

## 0. 职责

机器人模块负责 `RobotProfile` 的注册与校验，描述机器人本体：自由度、关节名称与顺序、单位与限位、控制模式、夹爪约定、坐标系与 URDF 引用、静态安全边界、推荐控制频率。`RobotProfile` **不**描述连接到哪个进程、使用何种 transport、ROS topic / IP / 端口等运行时信息——这些由推理模块管理。

## 1. 核心对象

- `RobotProfile`：机器人本体声明，组合解析层只消费它，不连接机器人平台。
- 注册表：机器人本体 profile 的注册与校验。

## 2. 详细设计

TODO，后续补充：

- `RobotProfile` 字段全集与校验（关节拓扑、控制模式枚举、夹爪约定、安全边界）。
- 从 URDF / 厂商描述 / 现有 adapter 自动导入可确定字段的规则。
- profile contract test。

## 3. 扩展方式

TODO：新增机器人的标准步骤（声明 profile + 引用 URDF + 补充 VLA 语义）；运行平台、connector、transport 不在本模块范围。

## 4. 机器人描述（目标设计）

> **状态：目标设计，部分已实现。** 本章对齐架构 §4.1.3（RobotProfile）与 §4.2.2
> 兼容性矩阵，体例同 [数据模块设计 §8](data-module.cn.md#8-数据集描述目标设计)、
> [模型模块设计 §4](model-module.cn.md#4-模型描述目标设计)。当前 `RobotProfile`
> dataclass（`robot/profile.py`）已覆盖本章大部分字段；`grippers[]`（原单数
> `gripper`）是本章相对当前实现新增的设计。

### 4.1 归属：单层声明，无实例契约

数据维度是「全实测」（§8.1），模型维度由族声明 `ModelMetadata` 唯一
定义接口，checkpoint 只可选校验（§4.1）。机器人维度同样只有
**一层**：`RobotProfile` 是随框架发布的声明文件，字段来自 URDF、厂商资料或
现有 adapter 在**编写时**确定，没有运行时自述的「实例契约」——机器人不像
checkpoint 那样能在连接时上报自己的事实。

这个单层设计是刻意的，不是偷懒：草案曾提议 `hardware_revision`（同型号不同批次
的硬件差异）和逐机标定 `calibration_note`，两者都指向「需要一个实例级覆盖层」。
但目前没有任何消费方需要区分同一 `name` 下的不同批次或标定状态——引入这层机制
之前，先确认有真实场景需要它（§4.4）。

### 4.2 字段准入原则

同数据模块 §8.2 / 模型模块 §4.2：一个字段进入第一版必须**可产出**（能从
URDF / 厂商资料 / 现有 adapter 确定性导出）且**有消费方**（兼容性矩阵
某行检查、Mapping 生成、已实现的 T1 transform）。机器人维度字段"可产出"
门槛通常不难满足——难的是消费方，本章大部分被推迟的字段都卡在这里。

### 4.3 RobotProfile 字段表（第一版，对齐当前实现 + grippers[] 新设计）

**identity —— 机器人身份**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `name` | registry key | 全局：`get_robot_profile()` 查找 |
| `urdf_ref` | URDF 权威来源的 URI/路径 | provenance；`limits_low/high` 等数值字段的 authoring 时来源标注 |

`variant`/`manufacturer` 已移除（原设计里的展示信息，无消费方）——同 §4.4 的
准入原则，纯信息性字段在有真实展示/消费需求前不进声明。

**joints —— 逐关节事实**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `names` | 机器人原生关节顺序（有序） | JointMapping（架构 §4.2.3）：与数据侧逐维 `name` 对账 |
| `units` | 关节单位（`radian`/`meter`） | 单位转换检查 |
| `types` | 逐关节类型（`revolute`/`prismatic`/`continuous`/`fixed`） | 关节拓扑校验 |
| `limits_low` / `limits_high` | 逐关节**位置**限位，与 `names` 等长 | 矩阵「安全范围」检查的机器人侧上限（关节空间原生限位，与下面 `safety_bounds_*` 是不同维度的量，见 §4.3 附注） |

`limits_low/high` 与顶层 `safety_bounds_low/high`（下方 frame/safety 块）**不是
重复字段**：前者永远是关节空间的物理硬限位（弧度/米），后者是当前
`native_action_type` 表示下、动作向量层面的运行边界。对 `joint_pos` 机器人
（当前全部已知 profile）两者数值上会重合，但语义不同——一旦引入 delta 或 EEF
类动作表示（架构 §7.3 的标准抽象方向），动作向量就不再是关节位置的直接堆叠，
两者会分叉。保持两个字段是为那一天打基础，不是当前的冗余。

**grippers[] —— 末端执行器（新设计，替换单数 `gripper`）**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `name` | 该末端执行器的稳定语义名（如 `left_gripper`） | 与数据/模型侧的夹爪语义对账（多末端时按名匹配，不靠下标） |
| `open_value` / `close_value` | 完全张开/闭合对应的动作值 | 矩阵「夹爪约定」检查：与数据 `action.dims[].mode` 及 recipe `assembly.gripper_flip` 比对，决定是否需要 flip transform |
| `joint_index` | 该夹爪在 `joints.names` 中的位置（0-based），非独立执行器时为 `None` | 定位 gripper_flip transform 作用的具体维度 |

**为什么要改**：现有 `gripper: GripperConvention`（单数）假设机器人只有一个
夹爪，`joint_index: int | None` 只能存一个整数。这个假设在双臂机器人上已经
不成立——本轮刚提交的 `robot/profiles/robotwin.yaml` 就是实例，它有两个夹爪
（`joint_index` 6 和 13），现在被迫写 `joint_index: null` 并在注释里承认
「单一 convention 字段共享 open/close 值，逐夹爪索引留给结构化 action 段」。
这不是假设性需求，是已经落地的 profile 在打补丁绕过限制。

**control —— 控制接口事实**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `control_modes` | 支持的控制模式集合 | 矩阵「控制模式」检查的候选集 |
| `native_action_type` | 默认/规范控制模式 | 未显式指定时的兜底选择 |
| `recommended_control_hz` | 推荐控制频率 | 矩阵「频率」检查的机器人侧输入（默认 warning，不强制重采样） |

**frame/safety**

| 字段 | 说明 | 消费方 |
|---|---|---|
| `coordinate_frame` | 坐标系标签 | 目前只做标签透传；坐标转换是 T2，见 §4.4 |
| `safety_bounds_low` / `safety_bounds_high` | 动作向量层面的运行边界 | 矩阵「安全范围」检查（见 `limits_low/high` 附注） |

### 4.4 不进第一版的字段

| 字段/块 | 原因 |
|---|---|
| `morphology.*`（`arms`/`dof_per_arm`/`has_mobile_base`/`has_torso`/`kinematic_chains`） | `kinematic_chains` 的 `base_link`/`tip_link` 是为 FK/IK 存在，FK/IK 属架构 §4.2.2 的 T2（「依赖机器人模型或运行条件」），架构明确「不把 T2 作为组合解析成立的前提」；其余布尔标志无消费方，机器人拓扑目前从 `joints.names` 的实际内容（如 LeKiwi 的 `base_x/base_y/base_z`）隐式表达即可 |
| `joints.velocity_limits` / `effort_limits` | 当前兼容性矩阵与已实现 T1 transform 均不涉及速度/力矩控制，无消费方 |
| `joints.sign_convention` | 矩阵「关节顺序」检查只处理顺序重排，不含符号翻转；这与夹爪 `open_value`/`close_value` 的 flip 机制是两回事，不能合并复用；当前无消费方 |
| `joints.zero_pose` / `calibration_note` | 信息性，无消费方（同模型模块 §4.4 的 `identity.family` 一类：写了但没人读的声明是死重） |
| `identity.hardware_revision` | 同名机器人的批次差异；无实例级校验/合并机制消费它——引入前先确认有真实场景需要区分批次（§4.1） |
| `identity.urdf.fingerprint`、顶层 `schema_version` | **明确不引入**，为保持三个维度一致对待：模型模块 §4.4 已否决 `identity.base_checkpoint`/`fingerprint`/`schema_version`（checkpoint 路径归 recipe，当前无版本迁移消费方），理由同样适用于机器人——且更弱：`RobotProfile` 随框架发布、跟框架版本一起升级，不是像 `DataSchema` 那样活在用户 checkpoint 里、需要独立的向后兼容升级路径。没有版本迁移的现实场景就不引入版本机制 |
| `sensors_mounts`（相机外参 `camera_mounts`、`ft_sensor_mounts`、`tactile_mounts`） | 相机外参的消费方是 T2 坐标转换；当前相机映射只按语义名匹配（架构 §4.2.3），不需要外参。`ft_sensor_mounts`/`tactile_mounts` 无 Reader 生产方，与数据模块 §8.3 `extra_modalities` 被推迟的理由对称 |
| `frames`（命名坐标系定义） | 同上，T2 territory；当前 `coordinate_frame` 只是标签，足够现有消费方使用 |
| `safety.workspace_aabb` / `self_collision_pairs` | 规划/防碰撞的消费方不在当前范围 |
| `safety.estop_behavior` | 不是"字段暂缓"而是"字段放错模块"——这是运行时安全执行行为，架构 §4.1.3 已明确 `RobotProfile` 只装静态本体事实，运行时安全执行归推理模块 |
| `capability.payload_kg` / `reach_m` | 信息性，处理方式对齐模型模块 §4.4 的 `capability` 块（`params_b`/`min_train_vram_gb`）：做资源预检/提示时再准入 |
| `joints.position_limits` 的 `from_urdf` 符号引用机制 | 不引入运行时 URDF 解析依赖。`limits_low/high` 继续是就地数值数组；数值由 authoring 时的离线脚本读取 URDF 后铺进 YAML（与现状一致的方式），`RobotProfile` 本身保持纯声明、无解析器依赖。这也是为什么 LeKiwi 的 URDF 查证后仍留空——上游 URDF 零个 `<limit>` 元素，`from_urdf` 指针在这类真实 URDF 上一样解不出东西 |

### 4.5 消费方对应表

| 字段 | 兼容性矩阵行（架构 §4.2.2） |
|---|---|
| `joints.names` | 关节顺序 |
| `limits_low/high` / `safety_bounds_low/high` | 安全范围 |
| `cameras` | 相机槽位（机器人侧候选） |
| `control_modes` / `native_action_type` | 控制模式 |
| `grippers[]` | 夹爪约定 |
| `recommended_control_hz` | 频率 |

### 4.6 与阶段2的关系

本章只是把机器人维度已有的事实写成可评审的字段表，并新增 `grippers[]`
设计——不是一次大重构。与数据/模型维度的阶段1不同，机器人维度当前实现已经
覆盖了"必要"字段的绝大部分，唯一的真实缺口是双臂夹爪的复数表示。阶段2的
Check Pairs 会是本章字段表第一次被真正消费——「夹爪约定」「关节顺序」「安全
范围」等检查落地前，`grippers[]`/`limits_low/high` 目前都只是声明，没有比较
逻辑读它们。

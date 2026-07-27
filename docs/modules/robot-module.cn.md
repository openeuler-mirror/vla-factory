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

# SimplerEnv 平台接入

> English: [simplerenv.md](./simplerenv.md)

VLA Factory 通过客户端/服务器边界接入
[SimplerEnv](https://github.com/simpler-env/SimplerEnv) 基准，架构与 RoboTwin、
RoboCasa 接入一致。模型及其依赖运行在 VLA Factory 环境中，SimplerEnv 仿真器
（ManiSkill / SAPIEN）运行在独立环境中——SimplerEnv 固定的 SAPIEN、ManiSkill
版本与框架的 torch 依赖线冲突。两个进程通过与 RoboTwin / RoboCasa 相同的
长度前缀 JSON-RPC 协议通信。免依赖的外部连接器从观测 dict 读取相机画面和末端
状态，连同语言指令与步数计数一起转发；维度校验与模型预处理由 VLA Factory
服务器依据 checkpoint 元数据完成。

## 1. 准备两个环境

按 [官方安装指南](https://github.com/simpler-env/SimplerEnv#installation)
在独立环境中安装 SimplerEnv（Python ≥3.9 的 conda 环境，`pip install -e .`
及其 ManiSkill/SAPIEN 附加依赖）。VLA Factory 与 SimplerEnv 必须分环境，避免
OpenPI、LeRobot 与 SAPIEN 依赖冲突。

在 VLA Factory 环境安装所需模型依赖（如 `scripts/install.sh` 的 pi0 或
openvla 环境）。该环境无需安装 `simpler_env`——连接器只在仿真侧运行，缺失时
会给出可执行的安装提示。

## 2. 启动模型服务器（VLA Factory 环境）

```bash
vlafactory-cli deploy \
  --checkpoint outputs/<checkpoint> \
  --platform simplerenv \
  --host 0.0.0.0 \
  --port 9999
```

服务器启动时打印 checkpoint 要求的相机列表、`state_dim` 与 `action_dim`，随后
监听 TCP 端口。服务器使用 `SimplerEnvAdapter` 把连接器观测翻译为 checkpoint
的 `DataSchema` 键。默认执行策略为 `receding_horizon`（按动作块重规划）。

## 3. 运行基准（SimplerEnv 环境）

闭环基准对每个任务创建一个环境（`simpler_env.make(task)`）、在其上跑
`--trials` 个回合，把第三视角相机画面与 `obs["agent"]["eef_pos"]` 转发给模型
服务器，通过 `env.step` 执行返回的动作块，并汇总二值成功标志为按任务的成功率
报告：

```bash
export VLA_FACTORY_PATH=/path/to/vla-factory

PYTHONPATH="$VLA_FACTORY_PATH${PYTHONPATH:+:$PYTHONPATH}" \
python -m vla_factory.inference.connectors.simplerenv \
  --port 9999 \
  --tasks google_robot_pick_coke_can widowx_put_eggplant_in_basket \
  --trials 25 \
  --report simplerenv_report.json
```

常用参数：`--cameras`（单相机；默认按任务前缀推断——`google_robot_*` 用
`overhead_camera`，`widowx_*` 用 `3rd_view_camera`）、`--instruction`（覆盖每
回合的 `get_language_instruction()`）、`--gripper-mode`（见下）、`--max-steps`
（默认由环境注册的 TimeLimit 结束回合）、`--seed`。

## 运行时契约

- **相机**：SimplerEnv 每个 embodiment 只有一个第三视角相机——Google robot
  为 `overhead_camera`、WidowX 为 `3rd_view_camera`——没有腕内相机。单相机
  checkpoint 按名称（含别名）或按位置与之配对；多相机 checkpoint 无法在
  SimplerEnv 上评测，服务器会在推理前报错说明。画面按环境原生分辨率转发，
  由 checkpoint 保存的 transform 管线在服务端完成 resize。
- **状态**：`obs["agent"]["eef_pos"]`，8 维 = 平移 (3) + 四元数 (4, wxyz) +
  夹爪 (1)。服务器在推理前按 checkpoint 的 `state_dim` 校验，不匹配时报出
  期望与实际值。
- **动作**：两个 embodiment 都消费 7 维扁平向量
  `[dx, dy, dz, droll, dpitch, dyaw, gripper]`。7 维 checkpoint 直接透传；
  4 维 checkpoint（xyz + 夹爪）补零旋转扩展；其他维度报错。由于不同训练管线
  的夹爪约定不同，`--gripper-mode` 负责适配最后一维：`raw`（默认，透传）、
  `google`（[0,1] → [-1,1] 并附带 sticky 连续闭合，匹配 Google robot 任务的
  夹爪动力学）、`widowx`（按 0.5 二值化）。结果截断到 [-1, 1]。
- **指令**：每回合读 `env.unwrapped.get_language_instruction()`（如 "pick up
  the coke can"）；`--instruction` 可覆盖。Diffusion Policy 等模型忽略它。
- **成功判定**：取自回合终止步——`info["success"]`、回合终止或包装器锁存的
  `success_once`。仅 TimeLimit 截断不算成功。
- 连接器位于 `vla_factory.inference.connectors.simplerenv`，不依赖 torch、
  Transformers、OpenPI 或 LeRobot，SimplerEnv 环境可直接导入。

## 协议说明

本运行器用 `env.reset(seed=seed + trial)` 重置——即预置初始状态协议。
SimplerEnv 官方基准还会通过 `options` 变化机器人/物体初始状态（variant
aggregation），并支持带 RGB overlay 的 visual-matching 设定。因此本运行器的
成功率适合做相对比较（消融、checkpoint 对比），**不能直接与官方发表的
SimplerEnv 基准数字对表**。

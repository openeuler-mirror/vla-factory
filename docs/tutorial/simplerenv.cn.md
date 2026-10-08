# SimplerEnv 平台接入

> English: [simplerenv.md](./simplerenv.md)

VLA Factory 通过客户端/服务器边界接入
[SimplerEnv](https://github.com/simpler-env/SimplerEnv) 基准，架构与 RoboTwin、
RoboCasa 接入一致。模型及其依赖运行在 VLA Factory 环境中，SimplerEnv 仿真器
（ManiSkill / SAPIEN）运行在独立环境中——SimplerEnv 固定的 SAPIEN、ManiSkill
和数据服务版本与框架的 torch 依赖线冲突。两个进程通过与 RoboTwin / RoboCasa
相同的长度前缀 JSON-RPC 协议通信。免依赖的外部连接器从环境读取相机画面和机器
人状态，连同语言指令与步数计数一起转发；维度校验与模型预处理由 VLA Factory
服务器依据 checkpoint 元数据完成。

## 1. 准备两个环境

按 [官方安装指南](https://github.com/simpler-env/SimplerEnv#installation)
在独立环境中安装 SimplerEnv（Python ≥3.9 的 conda 环境，`pip install -e .`
及其 ManiSkill/SAPIEN 附加依赖）。VLA Factory 与 SimplerEnv 必须分环境，避免
OpenPI、LeRobot 与 SAPIEN 依赖冲突。

在 VLA Factory 环境安装所需模型依赖（如 `scripts/install.sh` 的 pi0 或
openvla 环境）。该环境无需安装 `simpler_env`——部署命令只在仿真侧探测，缺失时
连接器会给出可执行的安装提示。

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
的 `DataSchema` 键。默认执行策略为 `receding_horizon`（每 5 步重规划）。

## 3. 运行基准（SimplerEnv 环境）

闭环基准对每个任务创建 `--trials` 个 SimplerEnv 环境，把 agentview/wrist 相机
画面与 `env.get_robot_state()` 转发给模型服务器，通过 `env.step` 执行返回的动作
块，并汇总二值成功标志为按任务的成功率报告：

```bash
export VLA_FACTORY_PATH=/path/to/vla-factory

PYTHONPATH="$VLA_FACTORY_PATH${PYTHONPATH:+:$PYTHONPATH}" \
python -m vla_factory.inference.connectors.simplerenv \
  --port 9999 \
  --tasks google_robot_pick_coke_can widowx_put_eggplant_in_basket \
  --trials 25 \
  --report simplerenv_report.json
```

常用参数：`--cameras`（默认 `agentview robot0_eye_in_hand`）、`--image-size W H`
（默认 224 224，须与 checkpoint 训练分辨率一致）、`--max-steps`（默认使用
SimplerEnv 按任务步数表：Google robot 任务 280、WidowX 任务 150）、`--seed`。

## 运行时契约

- SimplerEnv 消费其原生动作空间的扁平向量：Google robot 为 4 维（xyz 增量 +
  夹爪），WidowX 为 7 维（末端增量 + 夹爪）。连接器按 `env.action_space` 校验
  模型动作宽度并截断到 [-1, 1]；checkpoint 的动作语义必须与本 embodiment 一致。
- 机器人状态为 `env.get_robot_state()`：每臂 8 维（夹爪开合 + 末端位姿 pos/quat）。
  服务器在推理前按 checkpoint 的 `state_dim` 校验，不匹配时报出期望与实际值。
- 运行时任务必须暴露 checkpoint schema 中记录的全部相机。checkpoint 相机
  `wrist` 映射到 SimplerEnv 的 `robot0_eye_in_hand`。
- 语言条件模型以任务名作为指令；Diffusion Policy 等模型忽略它。
- 成功标志取自回合终止步（终止、`success` info 键或包装器锁存的
  `success_once`）。
- 连接器位于 `vla_factory.inference.connectors.simplerenv`，不依赖 torch、
  Transformers、OpenPI 或 LeRobot，SimplerEnv 环境可直接导入。

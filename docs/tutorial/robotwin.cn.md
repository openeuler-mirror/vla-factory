# RoboTwin 平台接入

> English: [robotwin.md](./robotwin.md)

VLA Factory 通过客户端/服务端方式接入 RoboTwin：模型及其依赖运行在
VLA Factory 环境，SAPIEN 仿真运行在独立的 RoboTwin 环境，两端使用
RoboTwin 原生 TCP 模型协议通信。外置 connector 只转发原始 observation、
语言指令和执行步数；相机选择、关节状态解析、维度校验和模型预处理均由
VLA Factory 服务端根据 checkpoint metadata 完成。因此不需要向 RoboTwin
仓库复制适配代码，也不需要维护模型相关的 `CAMERAS` 列表。

## 1. 准备两个环境

按照 [RoboTwin 官方文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
安装 RoboTwin。VLA Factory 与 RoboTwin 建议使用两个独立的 Python/Conda
环境，避免 OpenPI、LeRobot、SAPIEN 等依赖互相冲突。

在 VLA Factory 环境中安装所需模型依赖；读取 RoboTwin 原生 HDF5 数据时
额外安装 `robotwin` extra：

```bash
cd /path/to/vla-factory
pip install -e ".[robotwin]"
```

## 2. 配置 RoboTwin 原生训练数据

训练 recipe 的数据源应指向包含 `data/episode*.hdf5` 和
`instructions/episode*.json` 的 RoboTwin 任务配置目录：

```yaml
data:
  source:
    path: /path/to/dataset/<task>/<embodiment>_clean_50
    format: robotwin
    video_codec: hdf5_jpeg
```

## 3. 启动模型服务（VLA Factory 环境）

```bash
vlafactory-cli serve \
  --checkpoint outputs/<checkpoint> \
  --platform robotwin \
  --host 0.0.0.0 \
  --port 9999
```

服务启动时会打印 checkpoint 要求的相机列表、`state_dim` 和
`action_dim`，随后监听 TCP 端口。不要同时启动 RoboTwin 自带的
`policy_model_server.py`。

## 4. 启动评测客户端（RoboTwin 环境）

从 RoboTwin 仓库根目录运行：

```bash
export VLA_FACTORY_PATH=/path/to/vla-factory

PYTHONPATH="$VLA_FACTORY_PATH${PYTHONPATH:+:$PYTHONPATH}" \
python script/eval_policy_client.py \
  --port 9999 \
  --config "$VLA_FACTORY_PATH/vla_factory/deploy/configs/robotwin.yml" \
  --overrides \
    task_name beat_block_hammer \
    task_config demo_randomized \
    ckpt_setting vla_factory \
    instruction_type unseen \
    seed 0
```

`robotwin.yml` 是 RoboTwin `eval_policy_client.py` 要求的最小启动配置，
只声明外置 connector；真正的模型和 checkpoint 位于服务端。命令中的端口
必须与服务端一致。评测结果由 RoboTwin 写入其 `eval_result/` 目录。

## 运行时契约

- Aloha-AgileX 动作按 qpos 执行，典型维度为 14：左右各 6 个手臂关节和
  1 个夹爪关节。
- 运行时任务必须提供 checkpoint schema 中记录的全部相机。相机或状态维度
  不匹配时，服务端会在推理前报出所需值和实际值。
- PI0/PI0.5 等语言条件模型会收到 `TASK_ENV.get_instruction()` 返回的指令；
  ACT 等不使用语言的模型会忽略该字段。
- connector 位于 `vla_factory.deploy.robotwin_connector`，自身不依赖 torch、
  Transformers、OpenPI 或 LeRobot，因此可以由 RoboTwin 环境直接导入。

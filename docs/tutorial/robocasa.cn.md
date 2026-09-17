# RoboCasa365 平台接入

> English: [robocasa.md](./robocasa.md)

VLA Factory 通过客户端/服务端方式接入 RoboCasa365 benchmark：模型及其依赖
运行在 VLA Factory 环境，RoboCasa gymnasium 仿真运行在独立的 RoboCasa 环境
（robocasa 拉入 robosuite master、mujoco、约 10GB 厨房 3D 资产，以及一个与
框架 torch 线冲突的 numba pin），两端使用与 RoboTwin 相同的 length-prefixed
JSON-RPC 协议通信。外置 connector 只转发原始 gym observation、语言指令和执行
步数；相机选择、状态向量解析、维度校验和模型预处理均由 VLA Factory 服务端
根据 checkpoint metadata 完成。

RoboCasa365 数据集为 LeRobot 格式（parquet + MP4 + v3 风格 `meta/`），标注
`codebase_version: v2.1`。`lerobot-v3` reader 按结构识别（一个可解析的
`meta/info.json` 含 features map，外加 `data/` 下的 parquet 分片），无需新增
reader，也不做版本字符串比较。

## 1. 准备两个环境

按照
[RoboCasa 官方文档](https://robocasa.ai/docs/introduction/overview.html)
安装 RoboCasa。VLA Factory 与 RoboCasa 必须使用独立的 Python/Conda 环境，
避免 OpenPI、LeRobot、mujoco/robosuite 和 numba 依赖互相冲突。

在 VLA Factory 环境中安装所需模型依赖（例如通过 `scripts/install.sh` 安装
diffusion_policy 或 pi0 模型环境）。该环境**不需要**安装 `robocasa` extra——
它只是一个 metadata-only extra；download、deploy、evaluate 命令在运行时探测
该包，缺失时给出可执行的安装提示。

## 2. 下载 benchmark 数据

RoboCasa365 数据从 Box（UT Austin 托管）获取，落在 robocasa 安装目录下的
`datasets/v1.0/`。如需重定向，在 `robocasa/macros_private.py` 中设置
`DATASET_BASE_PATH`。请直接在 robocasa 环境中使用上游 robocasa 脚本（VLA
Factory 不封装下载——robocasa 的 asset/mujoco 依赖属于该环境）：

```bash
cd /path/to/robocasa && python -m robocasa.scripts.download_datasets \
  --split target --tasks PickPlaceCounterToCabinet

# 预览而不下载（列出所有将被下载的任务）：
python -m robocasa.scripts.download_datasets --split target --dryrun

# 厨房 3D 资产（约 10GB，首次构建 gym env 时需要）：
python -m robocasa.scripts.download_kitchen_assets
```

`--split` 接受多个值；`--source human|mimicgen` 过滤 pretrain 集合。完整参数
见 [robocasa 文档](https://robocasa.ai/docs/introduction/overview.html)。
下载后的 LeRobot 数据由 VLA Factory 的 `lerobot-v3` reader 读取，无需额外配置。

## 3. 在 RoboCasa365 数据上训练 / 微调

训练 recipe 的数据源指向下载的 LeRobot 根目录，并绑定 Panda+Omron 体型：

```yaml
data:
  path: /path/to/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
  format: lerobot-v3
  video_codec: pyav
robot:
  name: panda_omron
```

完整 recipe 见 `examples/robocasa365_diffusion_policy.yaml`。动作为 12-D EEF delta（OSC 末端执行器
+ 夹爪 + 底盘），因此 robot profile 声明 `native_action_type: eef_delta`，
相机语义规则将 `robot0_eye_in_hand` → `wrist`、
`robot0_agentview_left/right` → `third_person_front`。

## 4. 启动模型服务（VLA Factory 环境）

```bash
vlafactory-cli deploy \
  --checkpoint outputs/<checkpoint> \
  --platform robocasa \
  --host 0.0.0.0 \
  --port 9999
```

服务启动时会打印 checkpoint 要求的相机列表、`state_dim` 和 `action_dim`，
随后监听 TCP 端口。服务端使用 `RoboCasaAdapter` 将 gym observation 转换为
checkpoint 的 `DataSchema` 键。

## 5. 运行 benchmark（RoboCasa 环境）

闭环 benchmark 对每个任务执行 `gym.make("robocasa/<task>", split=...)` × N
个场景，将每个 observation 转发给模型服务端，执行返回的 action chunk，并收集
终止 `info` 中的二值 success 标志。从 RoboCasa 环境运行：

```bash
export VLA_FACTORY_PATH=/path/to/vla-factory

PYTHONPATH="$VLA_FACTORY_PATH${PYTHONPATH:+:$PYTHONPATH}" \
  python -m vla_factory.inference.connectors.robocasa \
  --port 9999 \
  --tasks PickPlaceCounterToCabinet \
  --trials 50 \
  --report robocasa_report.json
```

connector 驱动 gymnasium env，连接 `--port` 指定的模型服务端，并写入 JSON
成功率报告。客户端与服务端端口必须一致。推理时服务端通过 `PolicyExecutor`
执行 action chunk；在 `deploy` 上使用 `--strategy receding_horizon
--n-action-steps 5` 以对齐上游评测协议（每 5 步重新规划）。

每个 trial 完成后，connector 通过 tqdm 进度条实时刷新累计成功率（RoboCasa
环境未装 tqdm 时自动降级为逐行 print）。运行结束后打印汇总：总成功数 / 总
trial 数、总体成功率、总耗时和每集均耗时，并写入 `--report` 指定的 JSON。

离线评估（预测动作 vs 录制动作的 L1，不跑仿真）使用标准 `evaluate` 命令——
它直接读取 LeRobot 数据集，与格式无关：

```bash
vlafactory-cli evaluate \
  --checkpoint outputs/<checkpoint> \
  --dataset /path/to/robocasa/datasets/.../PickPlaceCounterToCabinet/20250811/lerobot
```

## 运行时标准

- 训练和模型输出保留数据集 12-D `modality.json` 顺序：base motion (4)、
  control mode (1)、EEF translation (3)、EEF rotation (3)、gripper (1)。
  connector 在构造 RoboCasa 的 `action.*` gym 字典时按此分段命名。
- 运行时任务必须提供 checkpoint schema 中记录的全部相机。相机或状态维度
  不匹配时，服务端会在推理前报出所需值和实际值。
- PI0/PI0.5 等语言条件模型会收到任务指令；Diffusion Policy 等不使用语言的
  模型会忽略该字段。
- connector 位于 `vla_factory.inference.connectors.robocasa`，自身不依赖 torch、
  Transformers、OpenPI 或 LeRobot，因此可以由 RoboCasa 环境直接导入。

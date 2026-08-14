# 微调层（training）模块设计

> 文档状态：**已对齐当前实现**（2026-08-13）。

## 0. 职责

`training/` 消费 `ResolvedAssembly`，把 Canonical IR 物化为训练样本并驱动监督训练。
它不重新推导数据、模型和机器人的关系，也不把参数选择策略与训练算法混为一层。

## 1. 目录与阅读入口

```text
training/
├── train.py          # 公开编排入口，按顺序组织完整训练生命周期
├── dataset.py        # SampleWindow、窗口构建、VLADataset、collate_fn
├── dataloader.py     # recipe + assembly → 单一训练 DataLoader
├── trainer.py        # VLATrainer 与 HuggingFace TrainingArguments 映射
├── checkpoint.py     # 训练契约与最终 inference state_dict 持久化
└── strategies/
    ├── base.py       # FinetuningStrategy 生命周期
    ├── registry.py   # @register_strategy / get_strategy
    ├── basic.py      # full / freeze / selective
    └── lora.py       # LoRA 注入、合并与 state_dict 清理
```

推荐按 `train.py → dataloader.py → dataset.py → trainer.py → strategies/ →
checkpoint.py` 阅读。

## 2. 训练生命周期

```text
recipe
  → merge_model_config
  → resolve_assembly                 # 在输出目录副作用之前
  → get_strategy + parse_config      # 严格校验 strategy config
  → 保存 immutable training contract
  → model factory
  → strategy.prepare_model
  → create_dataloader
  → VLATrainer.train
  → strategy.finalize_model/state_dict
  → 保存 final/model.pt
```

`train.py` 只决定这些操作的先后，不包含窗口算法、PEFT 合并或文件格式细节。

## 3. 样本与 DataLoader

`SampleWindow` 是一个训练样本的延迟定位信息，不是额外 Manifest。它记录 episode、
起始 frame、观测步数和 action horizon。`build_sample_windows()` 按确定顺序使用全部
episode；当前 `eval_strategy="no"`，所以不保留一份无人消费的 validation split。
加入真实 evaluation loop 时，再按 episode 粒度恢复 split。

`VLADataset` 负责按窗口读取 Frame、解码 VideoRef、repeat-last padding，并执行
assembly 保存的 `data_to_model` pipeline。`collate_fn` 才把 flat numpy sample 组装成
`Observation + actions + action_is_pad`。

## 4. 微调策略扩展点

微调策略只负责“哪些参数训练、模型是否需要 adapter 包装以及保存前如何收口”：

```python
@register_strategy("my_strategy")
class MyStrategy(FinetuningStrategy[MyConfig]):
    config_type = MyConfig

    def prepare_model(self, model, config, metadata):
        return model

    def finalize_model(self, model):
        return model
```

Recipe 统一写成：

```yaml
finetuning:
  strategy: my_strategy
  config:
    some_option: true
```

基类会拒绝未知或缺失字段。新增策略不修改 `TrainRecipe`、`train.py`、`trainer.py` 或
`checkpoint.py`。LoRA 额外覆写 `state_dict()`，用于清理 PEFT 合并后残留的 key 前缀。

如果一个新方法改变 loss、teacher/student、rollout、replay buffer、optimizer step 或
训练循环，它不是 `FinetuningStrategy`。等第一个真实需求出现时，应建立独立的
`training/methods/`，而不是继续给 strategy 增加 hook。

## 5. 持久化

`checkpoint.py` 写出：

- `inference_metadata/assembly.json`：唯一执行契约；
- `inference_metadata/recipe.yaml`：resolved 选择与训练配置；
- `final/model.pt`：strategy finalize 后的单一 inference state dict。

训练入口不根据策略名分支。是否合并 adapter、如何规范 state key，由策略生命周期负责。

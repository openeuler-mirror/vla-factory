# VLA 自动适配执行架构（审核稿）

## 1. 目标与原则

输入为 `upstream repo + VLA 名称 + checkpoint`，输出为 VLA-Factory 的适配补丁、训练 Recipe，以及一个可复现的确定性裁决。

核心原则只有两条：

1. Reference / Adapter Agent 可以说明或实现“模型如何调用”，不能控制“如何被评测”。
2. 所有影响 PASS/FAIL 的输入、执行器、指标和阈值都由 Judge 持有并固定。

因此，上游没有统一 benchmark 命令不构成问题：Reference Agent 只需把上游模型包装为统一的 Policy Endpoint；Judge 用同一 Benchmark Controller 调用两个 Endpoint。

## 2. 总体执行流

```text
                    Job Manifest（Judge 创建、版本固定）
       repo + vla + checkpoint + factory base + benchmark revision
                                   │
           ┌───────────────────────┴───────────────────────┐
           │                                               │
           v                                               v
  Reference Agent                                    Adapter Agent
  理解上游推理路径                                   改造 VLA-Factory
           │                                               │
           v                                               v
  Reference Bundle                                  Adapter Patch + Recipe
  env lock / bridge / runtime spec                  model adapter / transform / metadata
           │                                               │
           └───────────────────────┬───────────────────────┘
                                   v
                         Judge（唯一裁决与执行者）
                 ┌─────────────────┼─────────────────┐
                 v                 v                 v
          Bundle preflight    两个隔离运行时     架构合规检查
                 │                 │                 │
                 └─────────────────┴────────┬────────┘
                                             v
                              同一 Benchmark Controller
                                ┌────────────┴────────────┐
                                v                         v
                       Reference Policy Endpoint   Factory Policy Endpoint
                                │                         │
                           upstream VLA           VLA-Factory adapter
                                └────────────┬────────────┘
                                             v
                           结果比较 + 可复现性复跑 + Final Verdict
```

## 3. 角色、产物与权限

| 角色 | 只负责 | 必须交付 | 不得控制 |
|---|---|---|---|
| Reference Agent | 还原上游的加载、预处理、推理和动作语义 | `reference_bundle/`：依赖锁定、启动说明、Policy Bridge、运行时声明 | benchmark 版本、任务、seed、初始状态、指标、阈值 |
| Adapter Agent | 将同一语义接入 VLA-Factory | 最小代码补丁、模型元数据、Transform、Recipe | Judge 的测试集、benchmark runner、结果比较逻辑 |
| Judge | 执行实验并裁决 | 所有日志、trace、标准化结果、机器可读 verdict | 不修改候选实现以使其通过 |

`Job Manifest` 是任务的唯一事实来源，由 Judge 在开始时冻结：

```yaml
job_id: ...
upstream:
  repo: ...
  commit: ...
  model: ...
  checkpoint: ...
factory:
  base_commit: ...
benchmark:
  repo: ...
  commit: ...
  dataset_revision: ...
  tasks: [...]
  episodes_per_task: 20
  seeds: [...]
  timeout_seconds: 120
comparison_spec: comparison.v1
architecture_contract: architecture.v1
```

任何 Manifest 的变更生成新的 job，不能在同一 job 内覆盖。

## 4. 统一 Policy Endpoint

Judge 不调用上游各自的 benchmark 命令；它只调用统一策略接口。Factory 一侧使用既有确定命令启动服务，上游一侧由 Reference Bridge 启动服务。

```text
POST /health
POST /reset  {task_context, seed}
POST /act    {canonical_observation} -> {canonical_action}
POST /close
```

Bridge 的职责严格是：

```text
Canonical Observation -> upstream preprocess -> upstream VLA
                      -> upstream postprocess -> Canonical Action
```

Endpoint 只能读取 observation、指令、机器人状态、模型与 checkpoint；不能拿到 reward、success、future done、ground-truth action、隐藏环境状态或 metric。这样它可以适配模型，不能参与评分。

## 5. Judge 的分阶段执行

### Phase 0：冻结输入

Judge 拉取并记录 upstream、VLA-Factory、benchmark、数据集和 checkpoint 的版本；准备只读的 benchmark 资产及隐藏测试任务。此后 Agent 只能向自己的产物目录写入。

### Phase 1：生成候选产物

Reference Agent 产出 Reference Bundle；Adapter Agent 产出补丁和 Recipe。两者可并行，但都不运行正式 benchmark，也不产出可被当作裁决依据的结果。

### Phase 2：预检

Judge 在干净环境中分别构建上游与 Factory runtime，并检查：

```text
bridge 可导入 -> 模型可加载 -> reset 可调用 -> act 可调用
-> canonical action schema / shape / range 合法 -> 固定 seed 下行为可重复
```

预检失败即停止，避免将“启动失败”伪装为 benchmark 失败。

### Phase 3：受控基准评测

Judge 的单个 Benchmark Controller 在相同的 task、dataset、seed、初始状态、episode 数、超时和 metric 实现下，分别驱动两个 Policy Endpoint。模型运行时可隔离（例如上游 JAX 与 Factory PyTorch 分容器），但 benchmark 本身只能有一份。

### Phase 4：架构合规

Judge 对 Adapter Patch 执行静态规则和运行时 trace 检查。运行训练/推理的受控 smoke path 时，框架必须产生下列有序事件：

```text
训练：recipe.parsed -> assembly.resolved -> data_ir.created
      -> transform.applied -> model_contract.bound -> trainer.started

推理：observation.received -> transform.enter -> contract.preprocess
      -> policy.forward -> contract.postprocess -> action_compiler.run
      -> action.returned
```

Judge 检查的是必经节点和偏序，不要求 trace 文本完全一致。

### Phase 5：复现与裁决

Judge 在新的干净 runtime 复跑通过候选，比较规范化结果、配置指纹、版本锁定和 trace。所有产物与证据按 job 保存，供复核。

## 6. 确定性规则与门禁

`architecture_contract` 编译为下列机器检查，而非让 LLM 打分：

| 检查 | 证据 | 是否硬门禁 |
|---|---|---|
| Policy / Model Contract | 动态接口和 schema 测试 | 是 |
| Dependency Boundary | Python AST import graph | 是 |
| Diff / Extension Boundary | 相对 base commit 的文件清单 | 否；未声明的 core 改动失败 |
| Model-specific Leakage | AST symbol、import、模型标识扫描 | 是 |
| Training Dataflow | 运行时 trace 偏序 | 是 |
| Inference Execution Flow | 运行时 trace 偏序 | 是 |
| Config Ownership | resolved config provenance、未知/未使用 override | 否 |
| Reproducibility | 清洁环境复跑及结果指纹 | 是 |

默认策略是候选只能改动模型扩展点、Transform、配置和注册处。若必须变更 core，候选须携带 `architecture_exception.yaml`，注明影响的公共契约与原因；该例外需要人工审核，不能自动放行。

## 7. 比较与最终判定

`comparison_spec` 由 Judge 版本化，明确每项 metric、规范化方式与容差。若模型在固定 seed 下可确定，则可额外比较 canonical action；否则只比较同一 episode 集合的 benchmark 结果。容差绝不由 Agent 提供。

```text
Functional PASS
  = 所有必需 benchmark metric 满足 comparison_spec

Architecture PASS
  = 所有硬门禁通过
    AND 所有非豁免架构规则通过

Reproducibility PASS
  = 清洁环境复跑满足 comparison_spec
    AND manifest / config / artifact fingerprint 一致

FINAL PASS
  = Functional PASS AND Architecture PASS AND Reproducibility PASS
```

建议 Judge 只输出离散 verdict 和证据，不输出会掩盖硬失败的“综合分数”：

```json
{
  "verdict": "FAIL",
  "functional": {"status": "PASS"},
  "architecture": {
    "status": "FAIL",
    "failed_rules": ["MODEL_SPECIFIC_LEAKAGE"],
    "evidence": ["vla_factory/core/train.py: import openpi"]
  },
  "reproducibility": {"status": "NOT_RUN"},
  "manifest_digest": "...",
  "artifacts": {"trace": "...", "results": "..."}
}
```

LLM 可以作为只读的 Architecture Reviewer，提出重复实现、职责不清等建议；它的结论不进入 PASS/FAIL。

## 8. 最小落地顺序

第一期只搭执行闭环，不加入架构门禁：

1. Judge 创建并冻结 `Job Manifest`，实现统一 Policy Endpoint。
2. 实现 Reference Bridge 预检：能加载上游模型，并完成 `reset`、`act` 与 canonical action 转换。
3. 让 Judge 创建两个隔离 runtime，以同一 Benchmark Controller 分别驱动 Reference 与 Factory Endpoint。
4. 固定结果采集、标准化比较、日志和 artifact 保存，输出 Functional PASS / FAIL。

完成以上四步，就已经消除“双方各跑一套脚本、Judge 只比两个自报结果”的不确定性。架构门禁（diff scope、依赖图、Policy Contract、trace）是第二期的验证能力：它的作用是防止功能通过但绕过 VLA-Factory 架构，不是搭起双端评测执行架子的前置条件。

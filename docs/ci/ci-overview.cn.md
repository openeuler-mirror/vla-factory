# vla-factory CI 系统

> 关联 Issue: #7（测试流水线构建）

---

## 一、背景与约束

### 为什么不用 GitCode 原生 CI

2026-07-29 在 fork 上推送了一个带 `on: push` 触发器的 `.gitcode/workflows/l0.yml`
进行探测，**没有任何 job 运行**。进一步排查发现：openeuler 组织下**没有任何仓库**
携带 `.gitcode/workflows/` 或 `.github/workflows/`，其 CI 由仓库根 `Jenkinsfile` +
共享 `ci-scripts/` 驱动（Jenkins 在仓库外部配置）。接入该体系需要向
sig-infrastructure 申请 onboarding，不是提交一个文件就能完成的。

### GitCode API 能力（实测确认）

| 能力 | 状态 | 备注 |
|------|------|------|
| PR 列表 / 详情 API | ✅ | `GET /pulls?state=open`，返回所有作者的 PR |
| PR 评论 API | ✅ | 创建返回 `note_id`（数字，编辑用）和 `id`（hash） |
| commit status API | ❌ 404 | `/statuses/{sha}`、`/commits/{sha}/statuses`、`/check-runs` 全部不存在 |
| fork 上创建 webhook | ✅ | `push_events` / `merge_requests_events`（boolean flag），仅 basic auth 鉴权 |
| upstream 上创建 webhook | ❌ 403 | 非 admin 无法操作 |

---

## 二、方案演进

### 尝试 1：webhook on fork → 放弃

最初设计在 fork 上挂 `push_events` webhook，VPS 接收后查 PR 列表入队。
实测后发现根本限制：**webhook 挂在 fork 上只能捕获 fork owner 的 push**，
其他贡献者从自己的 fork 提 PR 时不会触发。团队有多名贡献者（hezhenhao2、
leningchen_admin 等），漏掉谁的 PR 都不可接受。

### 尝试 2：VPS + webhook + 轮询兜底 → 简化

为了覆盖所有作者，在 webhook 基础上加了轮询线程（每 30s 扫全部 open PR）。
但既然轮询已经能覆盖所有 PR，webhook 的秒级低延迟优势变得鸡肋——多了一个
公网端口、一份 basic auth、一个 payload 格式未知的依赖，得不偿失。

### 最终方案：纯轮询，无 VPS，无 webhook

去掉 webhook 和 VPS 中间层。一个 Python 进程跑在本地 GPU 机器上，直接轮询
GitCode PR API，覆盖**所有贡献者**的 PR，只需要出站 HTTPS。

---

## 三、最终架构

```
┌─ 本地 GPU 机器 (单进程 daemon) ──────────────────────────┐
│                                                          │
│  每 30s:                                                 │
│    1. GET GitCode /pulls?state=open                     │
│       → 返回所有作者的 open PR                            │
│    2. 比对本地 SQLite，发现新 head SHA → 处理             │
│    3. git fetch PR head → checkout --detach              │
│    4. 按环境 × tier 跑 pytest，产出 junit XML            │
│    5. 解析结果 → 格式化 markdown 表格                     │
│    6. POST / 编辑 PR 评论                                │
│    7. SHA 记入 SQLite（去重）                             │
│                                                          │
└──────────────────────────────────────────────────────────┘
         │  出站 HTTPS
         ▼
    GitCode API
```

### 设计要点

| 选择 | 理由 |
|------|------|
| 纯轮询，无 webhook | webhook 只覆盖 fork owner；轮询覆盖**所有贡献者** |
| 无 VPS 中间层 | 没有 webhook 就不需要公网端口；本地直连 GitCode API |
| 本地 SQLite 去重 | `UNIQUE(pr_number, head_sha)` 保证同一 PR 同一 SHA 只跑一次 |
| 结果走 PR 评论 | GitCode 没有 commit status API，别无选择 |
| 编辑同一条评论 | 不追加新评论，避免 push 一次刷一条 |

### 环境 × tier 分配

每个测试环境只跑自己能覆盖的 tier，不重复：

| 环境 | 依赖 | 跑哪些 tier | 覆盖的用例 |
|------|------|------------|-----------|
| **base** | core + `[dev]` | L0 | 全部框架契约测试 |
| **act** | + lerobot | L1 + L2 | lerobot parity + 过拟合冒烟（计划中，见 §4） |
| **pi** | + openpi | L1 | openpi parity（计划中，见 §4） |

L1 测试通过 `pytest.importorskip` 自动分流：act 环境跑 lerobot 相关用例，
pi 环境跑 openpi 相关用例，互不干扰。缺依赖时自动 skip，不报错。

> openpi 和 lerobot 无法共存于同一环境（openpi 通过 uv git source pin 了
> 旧版 lerobot），所以必须分环境。见 `scripts/ci/build_ci_envs.sh`。

---

## 四、测试分层

测试按「验证对象」分三层，用 pytest marker 隔离。

| 层 | marker | 验证对象 | 成本 | 触发 |
|----|--------|---------|------|------|
| **L0** 单元 | 无（`not l1 and not l2 and not l3`） | 我们自己的代码：模块功能、边界条件、报错路径 | 秒级 CPU | 每次 PR |
| **L1** parity | `@pytest.mark.l1` | 引进的上游语义：transform 链、归一化公式、PEFT 挂载 | 秒级 CPU | 每次 PR |
| **L2** 冒烟 | `@pytest.mark.l2` | 端到端连通性：单条 episode 过拟合 | 分钟级 CPU/GPU | 每次 PR |

marker 已在 `pyproject.toml` `[tool.pytest.ini_options] markers` 注册。当前
master 上尚无任何测试携带 `l1`/`l2` 标记（parity / 冒烟测试在 `dev_ci-backup`
分支，见下），因此**现阶段 L0 = 全套件**，L1/L2 tier 收集 0 例。tier 收集
0 例（junit `tests=0`，pytest exit 5）视为 **skip** 而非 FAIL；只有某环境的
junit 报告目录整个缺失（环境根本没跑，如崩溃/配置错误）才判 FAIL。

### L0 — 框架测试

验证**我们自己的代码**。基础环境运行通用测试；需要可选模型依赖的用例在
依赖缺失时明确 skip。测试按稳定的行为契约组织，不再为每个开发阶段保留
独立的 phase 验证脚本，也不在文档中维护容易漂移的逐文件例数清单。

| 子系统 | 主要契约 |
|--------|----------|
| Data | reader / codec 注册与发现、语义推断、真实数据读取、transform、sample window 和 DataLoader |
| Model | ModelMetadata 字段分类、内置声明、外部插件、可选 checkpoint 一致性检查、ACT / PI0 / PI05 adapter |
| Assembly | 兼容性诊断、mapping、ModelIOSpec、TransformPipelinePlan、序列化和接口漂移拒绝 |
| Training | strategy 注册与严格配置、LoRA 挂载、真实 train → checkpoint 产物 |
| Inference / Deployment | 执行策略、双 action width、train → infer round trip、平台 adapter、PolicyRunner 和 RPC transport |
| User Interface | Recipe 解析与拒绝路径、inspect 输出、CLI 命令注册和失败副作用边界 |

当前重构分支有 22 个测试模块；在完整开发环境中收集 365 例并全部通过。
收集数会随 parametrization 和可选依赖变化，CI 以 pytest 结果而不是本文数字
作为最终依据。

### L1 — parity 测试（计划中，尚未合入）

> L1 parity 测试文件（`test/parity/*.py`）目前在 `dev_ci-backup` 分支，
> **尚未合入 master**。daemon 在 master 上跑 `pytest -m l1` 收集 0 例，
> 该 tier 显示 `— (skip)`、环境整体不因此判 FAIL（见 §3）；但为了让 tier
> 真正发挥防回归作用，建议在 parity 文件合入后再给 daemon 配置 act/pi 环境。
> 以下为合入后生效的计划清单。

验证**引进的上游语义**与官方实现一致。golden 值内嵌在测试代码中（常量/参考实现），
不依赖外部 `.npz`。每个上游契约 pin 到源码 commit，缺依赖时 `importorskip` 自动 skip。

| 文件（计划） | 例数 | 对照上游 | 验证的契约 |
|------|------|---------|-----------|
| `test/parity/utils.py` | — (helper) | — | `assert_tensor_parity`：报告首个不匹配元素位置/双方值/shape/dtype |
| `test_normalize_parity.py` | 10 | openpi (eps 1e-6) + lerobot (eps 1e-8) | eps 是 per-model 上游契约；config eps 到达算术；两个数量级差异；openpi pin 未漂移 |
| `test_openpi_pipeline_parity.py` | ~10 | openpi (`PI0Pytorch`) | pi0/pi05 全链 parity：state/actions 逐元素相等、图像角色匹配、letterbox padding、prompt token 对齐 |
| `test_act_pipeline_parity.py` | 6 | lerobot (`processor_act`) | ACT 全链 parity：state/actions/images 逐元素相等、channels-first layout、ImageNet 归一化等价 |
| `test_peft_parity.py` | 10 | peft (张量级) + openpi (契约级) | LoRA 挂载面张量一致、scaling 公式 == openpi、adapter 保持 float32 on bf16 base、merge 写入 delta |

### L2 — 端到端冒烟（计划中，尚未合入）

> 同 L1：L2 文件目前在 `dev_ci-backup` 分支，尚未合入 master。
> 需要 `[act]` extra（ACT 可跑 CPU），pi0/pi05 需要 GPU。

验证**端到端连通性**：单条 episode 训到近零 loss。联合断言——数据、归一化、
模型输入契约、可训练参数集合，任何一环错了都过不去。

| 文件（计划） | 覆盖 |
|------|------|
| `test/integration/test_overfit_smoke.py` | loss 数量级下降（vs degenerate baseline）、训练产物走推理路径重建同一条 episode、实际可训练参数集合 == `ModelMetadata.components` 声明集合、语言条件通路存活 |
| `test/integration/thresholds.yaml` | per-model 过拟合判据阈值 |

---

## 五、使用方式

### 首次准备

```bash
# 1. 准备测试环境（至少 base；L1 需要额外 act/pi）
#    daemon 首次启动时会自动 clone 仓库，不需要手动 clone
bash scripts/ci/build_ci_envs.sh base          # 最低要求：L0
# bash scripts/ci/build_ci_envs.sh base act pi # 完整覆盖：L0 + L1 + L2
```

### 日常启动

```bash
python3 scripts/run_ci.sh
```

交互式配置（首次运行，之后存到 `~/.vlaf_ci.conf` 自动复用）：

```
============================================================
  vla-factory CI daemon 配置
  (方括号内为默认值, 直接回车采用)
============================================================

  GitCode token [13rYhn...]:              ← 从 config.json 自动读
  CI 目录 (不存在会自动 clone) [~/vla-factory-ci]:  ← 默认值
  轮询间隔 (秒) [30]:

  测试环境 (act/pi 留空则跳过, 仅跑 L0):

  base python (L0) [/.../python3.12]:    ← 当前解释器
  act python (L1+L2, 留空跳过) []:
  pi python (L1, 留空跳过) []:
```

配置完成后 daemon 开始轮询，看到新 PR 自动跑测试并发评论。

### 后台常驻（systemd）

```ini
# /etc/systemd/system/vlaf-ci.service
[Unit]
Description=vla-factory CI daemon
After=network.target

[Service]
Type=simple
EnvironmentFile=/home/you/.vlaf_ci.conf
WorkingDirectory=/home/you/vla-factory-ci
ExecStart=/usr/bin/python3 scripts/ci/daemon.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vlaf-ci
```

### 配置项参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLAF_GITCODE_TOKEN` | GitCode access token（必填） | 从 config.json 读 |
| `VLAF_BASE_DIR` | CI 目录（不存在自动 clone） | `~/vla-factory-ci` |
| `VLAF_ENV_BASE` | base 环境 python（必填） | 当前解释器 |
| `VLAF_ENV_ACT` | act 环境 python（可选） | 空 |
| `VLAF_ENV_PI` | pi 环境 python（可选） | 空 |
| `VLAF_POLL_INTERVAL` | 轮询间隔秒 | 30 |
| `VLAF_DB_PATH` | 去重 DB 路径 | `~/.vlaf_ci.db` |
| `VLAF_UPSTREAM` | upstream 仓库 | `openeuler/vla-factory` |
| `VLAF_ENV_BASE_TIERS` | base 环境跑哪些 tier（覆盖） | `l0` |
| `VLAF_ENV_ACT_TIERS` | act 环境跑哪些 tier（覆盖） | `l1,l2` |
| `VLAF_ENV_PI_TIERS` | pi 环境跑哪些 tier（覆盖） | `l1` |
| `VLAF_CI_WORKERS` | CI 任务线程池并发数 | 5 |
| `VLAF_CMD_WORKERS` | 评论命令任务线程池并发数 | 5 |
| `VLAF_TIER_TIMEOUT` | 单 tier pytest 超时（秒） | 1200 |
| `VLAF_MAX_RETRIES` | 同一 (pr, sha) crash 重试上限 | 3 |
| `VLAF_CHECKOUT_TTL_DAYS` | per-PR 检出目录 GC 天数 | 7 |

### /vla-factory 评论命令（review 相关配置）

任何人在 open PR 评论整行输入即可触发（作者鉴权与频控见下）：

- `/vla-factory help` — 列出命令
- `/vla-factory retest` — 对当前 head 重新触发 CI（运行中会拒绝）
- `/vla-factory review` — 调用 skills 子模块的 vlafactory-code-review
  技能检视本 PR，问题以行内评论贴出

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLAF_AGENT_CMD` | 无头 agent 命令模板（`{prompt}`/`{output}` 经 shell 引用代入） | claude 无头模板（禁用工具与斜杠命令） |
| `VLAF_AGENT_TIMEOUT` | 单个 reviewer 执行超时（秒） | 600 |
| `VLAF_REVIEW_STEP_TIMEOUT` | 生成 prompts / 汇总发布步骤超时（秒） | 180 |
| `VLAF_REVIEW_WORKERS` | 单 PR 内 reviewer 并发 | 1 |
| `VLAF_REVIEW_GLOBAL_WORKERS` | 全 daemon reviewer 并发上限 | 4 |
| `VLAF_REVIEW_COOLDOWN_MIN` | 同一 PR 两次 review 的冷却分钟数（0 关闭） | 30 |
| `VLAF_REVIEW_ALLOWLIST` | 允许触发 review 的登录名白名单（逗号分隔，空 = 任何人） | 空 |
| `VLAF_REVIEW_LANG` | 检视评论语言（en/zh） | zh |
| `VLAF_SKILL_DIR` | 技能目录（缺省用 skills 子模块并同步最新 main） | 空 |
| `VLAF_NODE_BIN` | 显式指定 node 路径 | 自动探测 |

### PR 评论格式

daemon 对每个 PR 先发一条「运行中」评论，跑完后编辑为结果表格：

当前（仅 base 环境、L0 = 全套件）的实际输出形如：

```markdown
## CI 测试报告 — pass

branch: `dev_ci` · commit: `517028f8f6fe` · all tests passed · 32s

| 环境 | L0 单元 | L1 parity | L2 冒烟 | 耗时 |
|------|---------|-----------|---------|------|
| base | 159 passed, 3 skipped | — (skip) | — (skip) | 21s |
```

parity / 冒烟测试合入并配置 act/pi 环境后，表格会扩展为多环境多 tier。

---

## 六、组件与文件结构

```
scripts/
  install.sh               模型环境安装（--model 可选；缺省装全部并需确认）
  run_ci.sh                CI daemon 入口（薄壳, 转发到 ci/run_ci.py）
  ci/
    run_ci.py              交互式启动器（配置 → 存盘 → 启动 daemon）
    daemon.py              核心 daemon（轮询 GitCode → 跑测试 → 发评论）
    pr_reporter.py         GitCode PR 评论工具（创建/编辑/格式化）
    parse_results.py       JUnit XML → 结构化结果摘要
    build_ci_envs.sh       一条命令建 base/act/pi 三个 venv
```

**核心 daemon 循环**（主循环只轮询与派发，任务在双线程池上异步执行）：

```python
while True:
    gc_old_checkouts()                          # TTL 回收已合并 PR 的检出目录
    prs = fetch_open_prs()                      # GET /pulls?state=open（所有作者）
    for pr in prs:
        poll_commands(pr.number)                # 评论命令 → cmd 线程池（水位去重）
    for pr in prs:
        if is_seen(pr.number, pr.sha):          # done/failed/running 均视为已处理
            continue
        mark_seen(pr.number, pr.sha, "running") # 派发前占位，防止重复提交
        ci_pool.submit(ci_task, pr)             # fetch → pytest → comment
    sleep(30)
```

`ci_task` 在独立检出目录（`pr-<N>-<sha8>`，互不干扰）按环境 × tier 执行：
base 跑 L0，act 跑 L1+L2，pi 跑 L1。每个 tier 直接调 `pytest --junitxml`，
不依赖 PR 分支上的脚本（版本可能不一致）。单个 tier exit 5（无测试收集）不
算失败，tier 收集 0 例视为 skip；**某环境的报告目录整个缺失（一个 junit 都
没产出）才判 FAIL**。tier 超时写一条合成失败用例（`tier-timeout`），不会
无限重试；同一 (pr, sha) 连续 crash 超过重试上限后放弃。

---

## 七、安全与风险

| 威胁 / 风险 | 对策 |
|------------|------|
| token 泄露 | 只存环境变量 / systemd unit，不进仓库；API 请求经 `Authorization: Bearer` 头携带，不进 URL（避免代理/日志泄露） |
| 执行恶意 PR 代码 | `checkout --detach` + `git clean -qfd`，不碰工作区 |
| PR 评论 API 限流 | SHA 去重 + 编辑同一条评论（不追加） |
| GitCode PR API 变更 | 只用标准 `GET /pulls?state=open`，最稳定的端点 |
| merge ref 不存在（冲突） | 捕获 → 评论报 "fetch failed" |
| daemon 挂了漏掉 PR | SQLite 只把 `done`/`failed` 视为已完成；`running`/`crashed` 状态重启后自动重试，并复用记录的 comment_id 编辑原「运行中」评论（不留残骸） |
| openpi + lerobot 环境冲突 | 三环境隔离（`build_ci_envs.sh`） |

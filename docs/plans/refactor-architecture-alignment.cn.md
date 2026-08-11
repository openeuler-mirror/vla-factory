# 重构计划：代码布局对齐新架构 + 组合解析层（assembly）阶段0骨架

> 状态：待执行。基于 `docs/architecture/vla-factory-architecture.cn.md`（d3b0e86）
> 制定，日期 2026-08-06。执行时按「提交切分」一节逐 commit 推进，每步保持 pytest 全绿。
> **后续决策（2026-08-11）：** 本计划里的 BaseContract / Materialize 骨架已在
> 后续简化中删除。当前 resolver 直接消费 ModelMetadata，checkpoint 只参与可选
> 一致性检查；下文相关段落是历史计划，不代表当前接口。

## Context

`docs/architecture/vla-factory-architecture.cn.md`（d3b0e86 更新）描述了目标架构：
`recipe/ · data/ · composition/ · model/ · robot/ · training/ · inference/` 七个模块，
以及「数据 × 模型 × 机器人」组合解析层（具身组合 / `resolve` / 结构化解析错误 /
`RobotProfile`）。但 d3b0e86 只改了文档，代码仍是旧布局
（`config/`、`model/protocols/`、`data/transforms/`、`data/dataset.py+loader.py+sampling/`、
`deploy/`），组合解析层代码完全不存在。

已确认范围：**结构重组 + §7.4 阶段0 骨架**——目录对齐新架构，引入组合解析层的
术语与数据结构 + `resolve` dry-run，**不接管下游执行，现有训练/部署行为不变，测试全绿**。

决策：
- 不留旧路径兼容 shim（仓库早期、所有调用方均在仓内），一次性改全部 import。
- **组合解析层目录命名为 `assembly/`（已选定，替代文档中的 `composition/`）**，
  英文术语同步改名：`ResolvedComposition` → `ResolvedAssembly`、
  `resolve_composition()` → `resolve_assembly()`、recipe 的组合调整区块
  `composition:` → `assembly:`；`Resolver` / `ResolutionError` 保留（"解析"语义不变）。
  中文术语「组合解析层」「具身组合」在中文文档中保留，仅代码符号与英文术语更新
  （见 Part C 文档同步）。

---

## Part A：结构重组（纯移动 + 改 import，无行为变化）

### A1. 目录移动（git mv）

| 现路径 | 新路径 | 说明 |
|---|---|---|
| `vla_factory/config/` | `vla_factory/recipe/` | recipe.py / parser.py / defaults.py / model/*.yaml 文件名不变 |
| `vla_factory/cli.py` | `vla_factory/recipe/cli.py` | 文档把 CLI 入口归入用户表达层；`__main__.py` 与 pyproject 同步改 |
| `vla_factory/model/protocols/` | `vla_factory/model/interfaces/` | Observation / VLAModel / ModelMetadata |
| `vla_factory/data/transforms/` | `vla_factory/assembly/transforms/` | TransformStep / Pipeline / Registry / 各 step |
| `vla_factory/data/dataset.py`、`loader.py`、`sampling/` | `vla_factory/training/` 下同名文件 | 文档规定「data/ 不构建样本」，样本构建归微调层 |
| `vla_factory/deploy/` | `vla_factory/inference/` | infer.py / policy_runtime.py / platforms/ / transports/ / connectors/ 整体平移 |

`data/` 保留：`formats/`、`codec/`、`manifest.py`（Canonical IR）。

### A2. 同步修改点（已探明的全部引用位）

- **pyproject.toml**：
  - `[project.scripts] vlafactory-cli = "vla_factory.recipe.cli:main"`
  - package-data：`"vla_factory.config.model"` → `"vla_factory.recipe.model"`；
    `"vla_factory.deploy.connectors"` → `"vla_factory.inference.connectors"`
- **包内 import**（约 20 处）：`training/train.py`、
  `training/strategies/{full,lora}.py`、`inference/infer.py`（改 `data.transforms`→
  `assembly.transforms`、`model.protocols`→`model.interfaces`）、`inference/platforms/*`、
  原 `data/dataset.py`（Observation import）、`model/registry/entries/{act,pi0,pi05}.py`、
  `model/base_contract.py`、`recipe/cli.py` 内全部子命令的懒 import。
- **`vla_factory/utils/constants.py`**：`MODEL_CONFIG_DIR` 相对 `recipe/` 解析（defaults.py 的
  `Path(__file__).parent` 随目录移动自动生效，仅核对）。
- **原 `data/loader.py:59,74`** 错误提示字符串中的 `vla_factory/config/model/<name>.yaml` 路径。
- **`inference/connectors/robotwin.yml`** 及 `docs/tutorial/robotwin*.md`：connector 被外部机器人
  环境按模块路径导入，`vla_factory.deploy.connectors` → `vla_factory.inference.connectors`。
- **test/ 13 个文件**：逐一改 import（多为函数内局部 import，需逐行改）。
- **scripts/**（install.sh、run_ci.sh、scripts/ci/）：grep 旧路径核对，预计无引用但需确认。

---

## Part B：组合解析层阶段0骨架（新代码，不接管下游）

按 §7.4 阶段0：「确定术语和数据结构；引入 BaseContract、RobotProfile、解析器、具身组合、
ResolutionError；保持现有训练和部署行为不变；新增 resolve dry-run」。

### B1. `vla_factory/robot/`

- `profile.py`：`RobotProfile` frozen dataclass（identity/本体变体、相机语义名、关节
  名称/顺序/单位/类型/限位、控制模式、夹爪约定、坐标系与 URDF 引用、静态安全边界、
  推荐控制频率）——字段按 robot-module.cn.md 与架构 §4.1.3。
- `registry.py` + `profiles/*.yaml`：YAML 声明 + `get_robot_profile(name)` /
  `list_robot_profiles()`；先提供 1 个示例 profile（`lekiwi.yaml`，字段从现有
  `examples/act_lekiwi.yaml` 与 lerobot adapter 中的稳定事实提取）。
  pyproject package-data 增加 `"vla_factory.robot.profiles" = ["*.yaml"]`。

### B2. `vla_factory/assembly/resolver/`

- `errors.py`：`ResolutionError(code, path, params)`——稳定错误码 + 专用构造入口
  （每个 code 定义允许的 params 集合，§4.2.5）。
- `types.py`：可序列化数据结构——`TransformStepSpec`、`TransformPipelineSpec`、
  `CameraMapping` / `StateMapping` / `ActionMapping` / `LanguageMapping` / `JointMapping`
  （阶段0 只定义结构）、`ResolvedAssembly`（三者描述引用 + canonical interface +
  五类 Mapping + `data_to_model` / `robot_to_model` / `model_to_robot` 三条
  PipelineSpec；`to_dict()` / `from_dict()` round-trip）。
- `resolver.py`：`resolve_assembly(schema, norm_stats, metadata, *, base_contract=None,
  robot_profile=None, overrides=None) -> ResolvedAssembly`。阶段0 只实现
  Load / Materialize（ModelMetadata+BaseContract 合并，冲突即失败）/ Validate 三个阶段；
  Mapping 与 PipelineSpec 输出空占位。确定性纯逻辑：不建模型、不建 DataLoader、
  不依赖 GPU、结果可序列化。

### B3. Recipe 与 CLI

- `recipe/recipe.py` + `parser.py`：新增可选块 `robot`（`RobotConfig(name)`）与
  `assembly`（`AssemblyConfig`：camera_mapping / accept_fps_mismatch /
  gripper_flip / default_task 等受控 override，均可空）。仅供 resolve dry-run 消费，
  train() 行为不变；`_recipe_to_yaml_dict()` 同步补这两个区。
- `recipe/cli.py`：新增 `resolve` 子命令——parse recipe → reader 读 schema/norm_stats →
  registry 取 ModelMetadata（不触发 factory，无重依赖）→ 可选 `load_base_contract` /
  `get_robot_profile` → `resolve_assembly()` → 成功打印摘要，失败打印结构化
  `code/path/params`。全程无 GPU / 无 optional extras 可运行。

### B4. 新增测试

- `test/test_assembly_resolver.py`：相同输入结果相同（确定性）、`to_dict/from_dict`
  round-trip、ModelMetadata×BaseContract 冲突失败、缺输入的 `ResolutionError` 断言
  code/path/params（不匹配完整文案）。
- `test/test_robot_profile.py`：profile 加载、必填字段校验、未知 profile 报错。

---

## Part C：文档同步（含 assembly 术语改名）

- **架构文档术语同步**：`docs/architecture/vla-factory-architecture.cn.md` / `.md` 中的
  英文符号 `composition/` 目录、`ResolvedComposition`、`resolve_composition()`、
  recipe `composition:` 区 → `assembly/`、`ResolvedAssembly`、`resolve_assembly()`、
  `assembly:`；中文「组合解析层」「具身组合」措辞保留。
- `docs/modules/composition-module.cn.md` → 重命名为 `assembly-module.cn.md`，
  内部符号同步；TODO 正文本次不写。
- `.claude/CLAUDE.md`：Code structure / How it runs 段落更新为新目录（config→recipe、
  protocols→interfaces、deploy→inference、transforms 归 assembly、样本构建归 training）。
- `docs/modules/data-module*.md`、`deploy-module*.md` 中的旧路径引用只做路径替换
  （模块文档「对齐当前已实现」）。
- README 若引用旧路径则同步。

---

## 提交切分（同一分支 `ref_architecture`，4 个 commit，便于 review 与回退）

1. `ref:` 目录重命名（recipe/、interfaces/、inference/、assembly/transforms/）+ 全量 import/pyproject/测试修正 → pytest 绿。
2. `ref:` 样本构建（dataset/loader/sampling）data → training → pytest 绿。
3. `feat:` robot/ + assembly/resolver 阶段0 + recipe robot/assembly 区 + CLI `resolve` + 新测试。
4. `docs:` 架构/模块文档 assembly 术语同步 + CLAUDE.md 路径同步。

## 验证

- 每个 commit 后：`pytest`（无 extras 环境全绿，heavy 用例按现有 skip 机制跳过）。
- `vlafactory-cli list`、`python -m vla_factory --help` 正常。
- `vlafactory-cli resolve --config examples/act_lekiwi.yaml` dry-run 输出摘要；
  构造一个歧义/缺字段 recipe 验证结构化 ResolutionError。
- `uv pip install -e ".[dev]"` 重装后确认 `vla_factory/recipe/model/*.yaml`、
  `inference/connectors/robotwin.yml`、`robot/profiles/*.yaml` 均随包分发
  （`vlafactory-cli list` 能读到 model defaults 即证）。
- `grep -rn "vla_factory\.\(config\|deploy\)\|model\.protocols\|data\.transforms"` 全仓无残留。

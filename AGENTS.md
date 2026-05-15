# AGENTS.md

> 本文件供 AI 编码助手（Codex CLI 等）启动时阅读，目的是把 AI 引到正确的项目文档与约定，避免重复探索。

---

## 沟通约定

- **始终使用中文与用户沟通**（与项目既有中文注释、文档保持一致）。
- 回复风格：简洁、直接，给结论与依据；避免冗长复述代码。

---

## 必读文档（按顺序）

开始任何编码工作前，请依次读完以下文件：

1. [CLAUDE.md](./CLAUDE.md) — 项目整体架构、两阶段训练、PatchCore 记忆库、命名规则、容易踩的坑。
2. [app/REQUIREMENTS.md](./app/REQUIREMENTS.md) — 当前任务（PySide6 推理 GUI）的需求规约：功能 / 非功能 / 验收标准 AC1–AC8。
3. [app/IMPLEMENTATION.md](./app/IMPLEMENTATION.md) — 当前任务的执行文档：模块划分、关键代码骨架、§5 的 7 步实施计划、§6 的验证方案。

---

## 当前任务上下文

**任务**：基于现有 DeSTSeg ROD 缺陷分割模型，开发一个 PySide6 桌面推理工具（CPU 推理，纯展示用）。

**目标设备**：Windows 11 + 无 GPU 笔记本；权重在 GPU 服务器训练得到。

**核心约束（已与用户确认，勿擅自改动）**：

- 框架 PySide6 ≥ 6.5
- 强制 CPU 路径：`device = torch.device("cpu")`，**不**根据 `torch.cuda.is_available()` 走分支
- 复用现有 `destseg` conda 环境，**不**重装 CPU 版 torch（CUDA 版 torch 在无 GPU 机器上 CPU 推理完全可用）
- 仅新增依赖：PySide6（写入 `app/requirements_app.txt`）
- 不修改 [train.py](train.py) / [eval.py](eval.py) / [model/](model/) / [data/](data/) 任何既有文件
- 默认权重：`saved_model/D2T_MemB_Dynamic_0.7_10000_202601161613.pckl` + 同名 `_memory_bank_prebuilt.pt`
- 模型配置硬编码：`num_classes=4 / use_d2t=True / use_patchcore=True / use_afs=False / use_rrs=False`

**目录结构（已规划，未实现）**：

```
app/
├── REQUIREMENTS.md     # 已存在
├── IMPLEMENTATION.md   # 已存在
├── requirements_app.txt
├── __init__.py
├── main.py
├── inference.py        # 纯计算层，无 Qt 依赖
├── workers.py          # QThread Worker
└── main_window.py      # MainWindow + 三视图
```

---

## 实施进度

按 [app/IMPLEMENTATION.md §5](./app/IMPLEMENTATION.md) 的 7 步推进。当前状态：

- [ ] 步骤 1：创建 `app/__init__.py` + `app/requirements_app.txt`
- [ ] 步骤 2：实现 `app/inference.py`（含 CLI 自检）
- [ ] 步骤 3：实现 `app/workers.py`
- [ ] 步骤 4：实现 `app/main_window.py`
- [ ] 步骤 5：实现 `app/main.py`
- [ ] 步骤 6：端到端联调（AC1–AC7）
- [ ] 步骤 7：最终回归（含 AC8 调试项）

> **接力规则**：每完成一步，就把对应方框打勾并提交。如果某步发现需求文档或实施文档需修订，先改文档再改代码——文档是接力棒。
>
> **细粒度日志**：本节只标粗进度。每次会话的决策、踩坑、偏离记录追加到 [app/PROCESS.md](./app/PROCESS.md)（倒序，最新在最上面）。新工具接手时建议先看 PROCESS.md 的最近一节再动手。

---

## 代码与提交约定

- **Python 注释 / docstring**：后续新增或重写的 `.py` 文件必须包含详细中文 docstring 与必要的中文行内注释。模块、类、关键函数都应说明职责、输入输出、异常、关键约束和容易踩坑的实现原因；注释要解释“为什么这样做”，避免只复述代码。
- **Git commit 信息**：提交信息必须尽可能详细描述“对哪些文件做了哪些改动”。标题沿用现有格式 `type(scope): 中文简短说明`；正文按文件或模块列出关键变更、验证命令和特殊兼容处理。不要只写笼统的 `update` / `fix` / `misc`。

---

## 行为准则

- **先读文档再动手**：在尚未读完上述三份文件前，不要写代码、不要改文件。
- **改动前确认**：对超出 `app/` 范围的修改、删除文件、强制覆盖等动作，先与用户确认。
- **遵守现有踩坑点**：尤其是 [CLAUDE.md](./CLAUDE.md) 末尾「与代码交互时容易踩的坑」一节里列举的。
- **首版克制**：[app/REQUIREMENTS.md §1.3](./app/REQUIREMENTS.md) 标记为「非目标」的功能（训练 / 评估指标 / 批量推理 / GPU 加速 / GUI 暴露开关）一律不做，记录到 §8 后续版本。
- **保留可回滚**：所有产物只放在 `app/` 子目录下。

---

## 提示首轮 prompt 模板（给用户参考）

如果是新会话/新工具首次接手，建议第一句话是：

```
先依次读完 AGENTS.md、CLAUDE.md、app/REQUIREMENTS.md、app/IMPLEMENTATION.md。
读完后简述你打算如何执行 IMPLEMENTATION.md §5 的下一个未完成步骤。
不要直接动手，等我确认。
```

# 实施进展日志

> 与 [../AGENTS.md](../AGENTS.md) 的「实施进度」勾选框配合使用。
>
> **角色分工**：
> - `AGENTS.md` 的勾选框 = **粗粒度**进度（7 步走到哪了），新工具一眼看到。
> - 本文件 = **细粒度**记录（决策、坑、偏离），写文档里没说的部分。
>
> **更新规则**：
> - 每次会话结束前追加一节，**倒序**（最新在最上面）。
> - 写「为什么这样做 / 踩了什么坑 / 偏离了什么 / 下一步交接」。
> - **不**写「读了哪些文件」「完成了 1/7」这种 git log 或 AGENTS.md 已能体现的信息。

---

## 2026-05-15 · 步骤 2：CPU 推理计算层

**工具**：Codex CLI
**状态**：已完成

### 决策与偏离
- 在 `app/inference.py` 中同时提供 `InferenceEngine` 与 `python -m app.inference <png>` CLI 自检入口；仍保持纯计算层，不引入 Qt / PySide6。
- 默认 checkpoint 未包含当前模型定义新增的 `channel_mask` buffer；加载侧仅允许缺失这一项，沿用模型初始化的全 1 buffer，其他 state_dict 差异仍然报错。
- 未修改 `AGENTS.md` 的粗粒度勾选框，因为该文件位于 `app/` 外；如需同步勾选，后续单独确认后再改。

### 遇到的坑
- `DeSTSeg` 当前分割输出为 `64×64`，而 GUI 首版展示固定 `256×256`；计算层按 `eval.py` 的保险逻辑先插值回输入尺寸再 `argmax`，避免叠加图尺寸不一致。
- 本地 git 写 `.git/index.lock` 需要提升权限；提交阶段需用已授权的 git 前缀或重新批准。

### 下一步
- 从 IMPLEMENTATION.md §5 **步骤 3** 开始：实现 `app/workers.py`，用 QThread Worker 包装 `InferenceEngine.load()` 与 `InferenceEngine.predict()`。

## 2026-05-15 · 步骤 1：应用包与 GUI 依赖入口

**工具**：Codex CLI
**状态**：已完成

### 决策与偏离
- 无。按 IMPLEMENTATION.md §5 步骤 1 创建空包文件与独立 GUI 依赖清单。

### 遇到的坑
- PowerShell 启动时尝试加载未签名 profile，输出执行策略报错；后续命令改用 `login=false`，不影响项目文件。
- 首次读取 `AGENTS.md` 时终端中文编码显示异常；后续读取项目文档改用 `-Encoding utf8`。

### 下一步
- 从 IMPLEMENTATION.md §5 **步骤 2** 开始：实现 `app/inference.py`，保持纯计算层、无 Qt 依赖，并先用命令行自检模型加载与单图推理。

## 模板（复制下面这段开新会话节）

```markdown
## YYYY-MM-DD · 步骤 N：<标题>

**工具**：Codex CLI / Claude Code / 其他
**状态**：进行中 / 已完成 / 阻塞

### 决策与偏离
- 偏离 IMPLEMENTATION.md 的地方 + 原因。如果没有偏离写「无」。

### 遇到的坑
- 报错、权宜之计、未来要注意的点。如果没有写「无」。

### 下一步
- 给下一个接手者的明确指引：从哪一步开始 / 有什么前置条件。
```

---

## 2026-05-14 · 步骤 0：文档与跨工具交接基建

**工具**：Claude Code
**状态**：已完成

### 决策与偏离
- 新增 [../AGENTS.md](../AGENTS.md) 作为 Codex CLI 的入口文档，把 AI 引到 [CLAUDE.md](../CLAUDE.md) + [REQUIREMENTS.md](./REQUIREMENTS.md) + [IMPLEMENTATION.md](./IMPLEMENTATION.md)。原 IMPLEMENTATION.md §5 表格未含「跨工具交接」步骤，这是工程外补加的一步，不算进 7 步实施。
- 进度跟踪一分为二：AGENTS.md 勾选框管粗粒度，本文件管细粒度。避免两份文档重复维护。

### 遇到的坑
- 无。

### 下一步
- 由用户指定工具（Codex / CC）接手 [IMPLEMENTATION.md §5](./IMPLEMENTATION.md) **步骤 1**：创建 `app/__init__.py`（空文件）+ `app/requirements_app.txt`（一行 `PySide6>=6.5`）。
- 接手时第一句话建议用 [AGENTS.md 末尾的首轮 prompt 模板](../AGENTS.md)。

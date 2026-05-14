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

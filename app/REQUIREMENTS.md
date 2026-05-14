# DeSTSeg 推理桌面工具 — 需求文档

> 本文档面向「基于现有 DeSTSeg ROD 缺陷分割模型的 PySide6 桌面推理工具」的首版需求规约。
> 实现计划（架构、文件拆分、步骤、风险）将在本需求确认后另行产出，本文不涉及实现细节。

---

## 1. 背景与目标

### 1.1 背景
仓库 [ml-destseg](..) 已基于 DeSTSeg 改造为 ROD 燃料棒缺陷的多类分割模型，训练/评估流程完整但**只能通过命令行使用**（[train.py](../train.py) / [eval.py](../eval.py)），不便于答辩演示与外部用户体验。

**关键环境差异**：[saved_model/](../saved_model/) 下的权重是在配 GPU 的服务器上训练得到的（训练日志保存在 [logs/](../logs/)），而**当前部署目标设备是无 GPU 的笔记本**——因此本工具必须在纯 CPU 环境下完成推理。

### 1.2 目标
交付一个独立的 PySide6 桌面应用：选一张图 → 一键推理（**CPU 推理**）→ 直观看到分割结果。**完全去除训练流程**，只读现有 checkpoint 与记忆库。

### 1.3 非目标（首版明确不做）
- 模型训练 / 微调 / 记忆库重建
- 评估指标计算（mIoU / Dice / AUPRO）
- 数据增强、合成缺陷
- 任何 GPU 加速路径（即使运行环境恰好有 CUDA 也不启用，避免增加分支与首版风险）
- 批量推理（保留为后续扩展）

---

## 2. 用户与典型场景

**主用户**：论文作者本人（开发者）。

**典型场景**：
- S1 答辩 / 演示：现场对 [dataset/eval/](../dataset/eval/) 下任意图片做实时推理展示。
- S2 个例调试：检查模型在某张特定样本上的预测掩码与 GT 的视觉差异。
- S3 移交他人：把整个 `app/` 目录加权重交给他人，对方装好依赖即可运行。

---

## 3. 功能需求

### 3.1 启动 & 模型加载
| 编号 | 需求 |
|------|------|
| F1.1 | 启动时自动扫描 [saved_model/](../saved_model/) 下的 `*.pckl` 文件 |
| F1.2 | 仅有一个 `.pckl` 时直接采用；存在多个时弹出选择对话框 |
| F1.3 | 自动配对同名 `_memory_bank_prebuilt.pt`（PatchCore 必需，找不到 → 报错并阻止进入主界面） |
| F1.4 | 模型加载在后台线程进行，主界面显示「加载中…」 |
| F1.5 | 加载失败给出明确错误（缺文件 / state_dict key 不匹配 / num_classes 不匹配）|

### 3.2 图片选择
| 编号 | 需求 |
|------|------|
| F2.1 | 提供「选择图片」按钮 → `QFileDialog.getOpenFileName` |
| F2.2 | QFileDialog 默认目录指向 [dataset/eval/](../dataset/eval/) |
| F2.3 | 过滤器 `Images (*.png)`（首版仅 PNG，与训练时保持一致）|
| F2.4 | 选中后立即在 UI 上显示完整路径与文件名 |

### 3.3 推理执行
| 编号 | 需求 |
|------|------|
| F3.1 | 「执行推理」按钮触发，**绝不阻塞 UI 主线程**（QThread / `concurrent.futures`）|
| F3.2 | 状态栏显示阶段：加载图片 → 预处理 → 前向 → 渲染 |
| F3.3 | 推理过程中按钮禁用，避免重入 |
| F3.4 | 推理失败 → 错误对话框 + traceback 摘要 + UI 回到可重试状态 |

### 3.4 结果展示（核心）
**三视图横向并排**（用户已确认）：

1. **原图**：单通道灰度 resize 到 `256×256` 后的可视化
2. **叠加图**：原图 + 彩色掩码（α=0.5），复用 [draw.py:42](../draw.py#L42) `overlay_mask`
3. **纯掩码**：argmax 后用 [draw.py:9](../draw.py#L9) `COLORS` 直接索引上色

**类别图例**（颜色与 [draw.py:9-15](../draw.py#L9-L15) 完全一致）：

| 类别 ID | 名称 | 颜色 |
|---------|------|------|
| 0 | 背景 | ■ 黑 |
| 1 | Scratch | ■ 红 |
| 2 | Dent | ■ 绿 |
| 3 | Dotted | ■ 蓝 |

| 编号 | 需求 |
|------|------|
| F4.1 | 三视图同步等比缩放，窗口 resize 不变形 |
| F4.2 | 图例固定显示在底部 / 侧栏 |
| F4.3 | 显示一行元信息：模型名 / 设备（GPU/CPU）/ 推理耗时 |

### 3.5 保存结果（可选，加分项）
| 编号 | 需求 |
|------|------|
| F5.1 | 「保存结果」按钮 → 选目录后导出三张 PNG |
| F5.2 | 命名：`<原文件名>_raw.png` / `<原文件名>_overlay.png` / `<原文件名>_mask.png` |

---

## 4. 非功能需求

### 4.1 性能（CPU 演示场景）
- N1 单图推理（含 PatchCore 最近邻搜索）：CPU 下**实际耗时不严格要求**，演示能跑即可。预估典型值：冷启动（含模型 + 记忆库加载）20–30 s，热推理（已加载）1–3 s
- N2 模型与记忆库**只加载一次**，切换图片不重新加载——这是 CPU 环境下最重要的性能保证

### 4.2 兼容性
- N3 **强制 CPU 路径**：`device = torch.device("cpu")`，**不**根据 `torch.cuda.is_available()` 走分支。理由：避免在不同设备间的代码路径差异，简化首版调试
- N4 目标平台：Windows 11 + Python 3.10+
- N5 框架：**PySide6**（用户已确认），版本 ≥ 6.5

### 4.3 工程约束
- N6 新代码统一放在新建目录 `app/` 下，**不修改** [train.py](../train.py) / [eval.py](../eval.py) / [model/](../model/) / [data/](../data/) 任何既有文件
- N7 仅复用以下既有模块（避免重复造轮子）：
  - [model/destseg.py](../model/destseg.py) `DeSTSeg`
  - [model/patchcore_mem.py](../model/patchcore_mem.py) `MemoryBank`
  - [draw.py](../draw.py) `overlay_mask` / `COLORS` / `denormalize`
  - [constant.py](../constant.py) `RESIZE_SHAPE` / `NORMALIZE_MEAN_*` / `NORMALIZE_STD_*`
- N8 **沿用当前 conda 环境 `destseg`**，仅新增 PySide6 一个依赖（`pip install PySide6`）。**不重装 CPU 版 torch**——CUDA 版 PyTorch 在无 GPU 设备上完全可用：
  - `import torch` 正常
  - `torch.cuda.is_available()` 返回 `False`，CPU 推理走 MKL-DNN 后端，性能与 `torch+cpu` 版本几乎一致
  - 唯一代价：磁盘多占几百 MB CUDA 库（闲置），无运行时影响
- N9 加载侧的两条强制要求（CPU 跨设备加载权重）：
  - 加载 checkpoint：`torch.load(ckpt_path, map_location="cpu")`，不显式指定会尝试还原训练时的 `cuda:0` 并报错
  - 加载记忆库：[model/patchcore_mem.py](../model/patchcore_mem.py) 的 `MemoryBank.load()` 内部已用 `map_location=self.device`，只要构造时传 `device=torch.device("cpu")` 即可
- N10 `app/` 内提供 `requirements_app.txt`（仅列 `PySide6>=6.5` 一行），项目根 [requirements.txt](../requirements.txt) **保持不动**

### 4.4 模型配置（首版硬编码，用户已确认）
- N11 `num_classes = 4`
- N12 `use_d2t = True`，`use_patchcore = True`
- N13 `use_afs = False`，`use_rrs = False`（这两个开关只影响 `channel_mask` 是否生效；当前 checkpoint 的 buffer 默认全 1，不会引起逻辑差异）
- N14 默认 checkpoint：`saved_model/D2T_MemB_Dynamic_0.7_10000_202601161613.pckl`
- N15 默认记忆库：`saved_model/D2T_MemB_Dynamic_0.7_10000_202601161613_memory_bank_prebuilt.pt`
- N16 首版**不**通过 GUI 暴露 D2T / PatchCore / num_classes 等开关

---

## 5. 推理流水线规范（实现合同）

### 5.1 设备初始化（强制 CPU）
```python
import torch
device = torch.device("cpu")  # 强制 CPU，不查 cuda.is_available()
torch.set_num_threads(max(1, os.cpu_count() - 1))  # 留 1 核给 UI 线程
```

### 5.2 模型加载（跨设备）
```python
# checkpoint 必须 map_location 到 CPU，否则会找不到 cuda:0
state_dict = torch.load(ckpt_path, map_location="cpu")
# 防御性 strip "module." 前缀（DataParallel 训练时会带）
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

model = DeSTSeg(num_classes=4, ED=True, use_d2t=True,
                use_patchcore=True, use_afs=False, use_rrs=False).to(device)
model.load_state_dict(state_dict)  # channel_mask buffer 形状会做天然校验
model.eval()

# MemoryBank 内部用 self.device，构造时传入 device 即可
mem_bank = MemoryBank(device=device, ...)
mem_bank.load(memory_bank_path)
```

### 5.3 预处理（与 [data/rod_dataset.py](../data/rod_dataset.py) 对齐）
1. `PIL.Image.open(path).convert("L")` —— 强制灰度
2. `image.resize(RESIZE_SHAPE, Image.BILINEAR)`，`RESIZE_SHAPE = [256, 256]`
3. 灰度分支：`ToTensor` → `Normalize(mean=[0.5], std=[0.5])` → `img_l`，形状 `(1, 1, 256, 256)`
4. RGB 分支：`image.convert("RGB")` → `ToTensor` → `Normalize(ImageNet mean/std)` → `img_rgb`，形状 `(1, 3, 256, 256)`
5. 两路 tensor 同时 `.to(device)`

### 5.4 前向
```python
# memory_bank 必须传入（与 N12 保持一致）
with torch.no_grad():  # CPU 推理务必关闭 autograd
    output_seg, _, _, _ = model(img_l, img_rgb, img_l, img_rgb, memory_bank=mem_bank)
pred_mask = output_seg.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # (256, 256)
```

### 5.5 渲染
- 灰度可视化：`draw.denormalize(img_l[0])` → `(H, W, 3)` float ∈ [0, 1]
- 叠加：`draw.overlay_mask(gray_np, pred_mask, alpha=0.5)`
- 纯掩码：`draw.COLORS[pred_mask] / 255.0`

### 5.6 PySide6 显示
- numpy `(H, W, 3)` float ∈ [0, 1] → `*255 → uint8 → QImage(..., Format_RGB888)` → `QPixmap` → `QLabel.setPixmap`
- 三个 QLabel 设置 `setScaledContents(True)` 或自定义 `resizeEvent` 保持纵横比

### 5.7 关键陷阱（已识别，实现时务必注意）
- **map_location**：不显式指定，加载在 GPU 上保存的 state_dict 时会触发 `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`
- **state_dict key 前缀**：训练时若用 DataParallel，权重 key 带 `module.` 前缀；[train.py](../train.py) 已用 `real_model.state_dict()` 规避，但加载侧仍需做防御性 strip
- **`channel_mask` buffer 形状**：随 `use_d2t` / `use_patchcore` 变化（448 / 896 / +1），错配会在 `load_state_dict` 时直接报错——这正好作为「checkpoint 与开关不匹配」的天然校验
- **PatchCore 强制启用**：当前 checkpoint 训练时 `use_patchcore=True`，推理时 `memory_bank` 必须非 `None`，否则 [model/destseg.py](../model/destseg.py) 走 Phase 1 分支，`patchcore_map` 是全零占位符，结果会偏差
- **CPU 下 `torch.cdist` 性能**：[model/patchcore_mem.py](../model/patchcore_mem.py) 的最近邻搜索是单图推理 CPU 下的主要瓶颈；记忆库约 5MB（~1 万 patch），`batch_size=1024` 在 CPU 仍可接受。**绝对不能**在 GUI 主线程执行
- **autograd**：CPU 推理必须包 `torch.no_grad()`，否则会构建反传图、占内存、拖慢约 2x

---

## 6. UI 草图

```
┌──────────────────────────────────────────────────────────────────┐
│ DeSTSeg ROD 缺陷分割 - 推理工具         [模型: D2T_MemB_…✓ CPU] │
├──────────────────────────────────────────────────────────────────┤
│ [选择图片]  [执行推理]  [保存结果]    当前: synthetic_0009_03.png│
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────┐   ┌────────────┐   ┌────────────┐                │
│  │            │   │            │   │            │                │
│  │   原图     │   │  叠加图    │   │  纯掩码    │                │
│  │            │   │            │   │            │                │
│  └────────────┘   └────────────┘   └────────────┘                │
├──────────────────────────────────────────────────────────────────┤
│ 图例:  ■ 背景  ■ Scratch  ■ Dent  ■ Dotted    状态: 推理完成 1.8s│
└──────────────────────────────────────────────────────────────────┘
```

> 设备标签固定显示 `CPU`（不显示 GPU/CUDA），与 N3 强制 CPU 路径保持一致。

---

## 7. 验收标准

| 编号 | 测试场景 | 预期结果 |
|------|----------|----------|
| AC1 | 启动应用（默认权重 + 记忆库齐全，**无 GPU 笔记本**）| 自动加载完成（冷启动 ≤ 30 s 视为通过），按钮可点击，无报错弹窗 |
| AC2 | 选 [dataset/eval/images/](../dataset/eval/images/) 下含缺陷图，执行推理 | 三视图显示，叠加图可见红/绿/蓝彩色区域，与 [dataset/eval/labels/](../dataset/eval/labels/) 同名 GT 视觉吻合 |
| AC3 | 选 [dataset/eval/good_900/images/](../dataset/eval/good_900/images/) 下无缺陷图 | 掩码图基本全黑，叠加图与原图视觉无明显差异 |
| AC4 | 拖拽窗口改大小 | 三视图等比缩放、不变形、不闪烁 |
| AC5 | 启动前删除 `_memory_bank_prebuilt.pt` | 加载阶段弹错，提示「记忆库文件缺失」，不崩溃 |
| AC6 | 模型加载阶段（CPU 推理）| 不阻塞 UI，主窗口可拖动；状态栏正确显示 `CPU` |
| AC7 | 连续切换 10 张图分别推理 | 每次都成功且无内存泄漏（任务管理器 RAM 稳定，单图热推理 ≤ 5 s 视为通过）|
| AC8 | 故意把 `torch.load` 的 `map_location` 删除并跑一次 | 复现 `RuntimeError: Attempting to deserialize ... CUDA device`——用于验证 N9 的必要性（不作为发布检查项，仅开发调试用）|

---

## 8. 待澄清 / 后续版本候选

以下条目首版**不做**，但记录在此供后续讨论：

- Q1 推理结果是否需要回插值到图片原始分辨率？（首版固定 256×256）
- Q2 批量推理 + 文件夹输出（参考 [eval.py](../eval.py) 的 `save_visual_comparison`）
- Q3 GUI 暴露 `--use_d2t` / `--use_patchcore` / 多 checkpoint 切换
- Q4 当所选图片在 `labels/` 下有对应 GT 时，加第四视图显示 GT 用于对照
- Q5 推理热力图（直接显示 `output_de_st`，不做 argmax）

---

## 9. 用户已确认事项（来源：本轮 + 上轮 AskUserQuestion）

1. **框架版本**：PySide6
2. **输出形式**：原图 + 彩色掩码叠加 + 纯掩码 三视图
3. **模型配置策略**：硬编码当前模型配置（D2T=True / PatchCore=True / num_classes=4）
4. **图片来源**：QFileDialog，默认目录指向 [dataset/eval/](../dataset/eval/)
5. **目标设备**：**无 GPU 笔记本**，权重来自有 GPU 服务器训练（[logs/](../logs/) 留有训练日志）
6. **性能期望**：演示能跑就行，CPU 实际耗时不严格要求
7. **依赖策略**：沿用现有 `destseg` conda 环境（已装 CUDA 版 torch），仅新增 `PySide6`，**不重装 CPU 版 torch**

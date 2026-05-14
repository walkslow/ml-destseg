# DeSTSeg 推理桌面工具 — 执行文档

> 本文档是 [REQUIREMENTS.md](REQUIREMENTS.md) 的实施伙伴。需求文档回答「做什么 / 为什么」，本文档回答「怎么做 / 分几步 / 怎么验证」。

---

## 1. 上下文

需求已锁定（见 [REQUIREMENTS.md](REQUIREMENTS.md)）：
- 框架 PySide6，目标 Windows 11 + 无 GPU 笔记本，纯 CPU 推理
- 复用既有 [model/destseg.py](../model/destseg.py) / [model/patchcore_mem.py](../model/patchcore_mem.py) / [draw.py](../draw.py) / [constant.py](../constant.py)
- 默认权重：`saved_model/D2T_MemB_Dynamic_0.7_10000_202601161613.pckl` + 同名 `_memory_bank_prebuilt.pt`
- 硬编码：`num_classes=4` / `use_d2t=True` / `use_patchcore=True` / `use_afs=False` / `use_rrs=False`

用户在执行文档撰写前的决策（直接驱动本文档形状）：
- **后台并发**：QThread + Signal/Slot（Qt 主流方案）
- **保存结果**：首版包含（F5.1 / F5.2）
- **代码拆分**：4 模块（main / inference / workers / main_window）

---

## 2. 目录结构

```
app/
├── REQUIREMENTS.md                # 需求文档（已落）
├── IMPLEMENTATION.md              # 本文档
├── requirements_app.txt           # 仅一行 PySide6>=6.5
├── __init__.py                    # 空文件，让 app 成为包
├── main.py                        # 启动入口：QApplication + 启动流程
├── inference.py                   # 纯计算层：模型加载 + 预处理 + 前向（无 Qt 依赖）
├── workers.py                     # QThread Worker：ModelLoadWorker / InferenceWorker
└── main_window.py                 # MainWindow + 三视图布局 + 信号绑定
```

**严格分层**：
- `inference.py` **不得 import PySide6**——可单独跑命令行单元测试
- `workers.py` 是 Qt ↔ inference 的薄包装层
- `main_window.py` 只负责 UI 与信号路由，**不直接调推理**

---

## 3. 依赖与运行环境

### 3.1 `app/requirements_app.txt`
```
PySide6>=6.5
```

### 3.2 安装
```bash
# 在仓库根（已激活 destseg conda env）
pip install -r app/requirements_app.txt
```

### 3.3 运行入口
```bash
# 在仓库根
python -m app.main
```

> 用 `python -m app.main` 而非 `python app/main.py`：前者会把仓库根加入 `sys.path`，让 `from model.destseg import DeSTSeg` 直接能找到，**不需要在代码里 hack `sys.path.insert`**。

---

## 4. 模块实现细节

### 4.1 `app/inference.py`（纯计算层）

#### 职责
- 加载 checkpoint 与记忆库
- 单图预处理（与 [data/rod_dataset.py:62-72, 121-122, 228-229](../data/rod_dataset.py) 对齐）
- 前向 + argmax + 渲染三视图

#### 类设计
```python
# app/inference.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 复用既有模块（不改动它们）
from model.destseg import DeSTSeg
from model.patchcore_mem import MemoryBank
from constant import (
    RESIZE_SHAPE,
    NORMALIZE_MEAN_L, NORMALIZE_STD_L,
    NORMALIZE_MEAN_RGB, NORMALIZE_STD_RGB,
)
from draw import COLORS, denormalize, overlay_mask


class InferenceEngine:
    """模型 + 记忆库的封装，单图推理用。

    线程安全约定：load() 与 predict() 不可并发；
    GUI 端通过 QThread 串行调用即可。
    """

    NUM_CLASSES = 4  # 与 N11 对齐
    USE_D2T = True
    USE_PATCHCORE = True

    def __init__(self):
        self.device = torch.device("cpu")  # N3：强制 CPU
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # 留 1 核给 UI
        self.model: DeSTSeg | None = None
        self.memory_bank: MemoryBank | None = None
        self.ckpt_path: str | None = None
        self._tx_l = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN_L, NORMALIZE_STD_L),
        ])
        self._tx_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN_RGB, NORMALIZE_STD_RGB),
        ])

    # ------ 加载 ------
    def load(self, ckpt_path: str, mem_bank_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
        if not os.path.exists(mem_bank_path):
            raise FileNotFoundError(f"记忆库不存在: {mem_bank_path}（PatchCore 必需）")

        # N9：跨设备加载，必须 map_location="cpu"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # 防御性 strip "module." 前缀（DataParallel 训练时会带）
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        # 注意签名是 ed=（小写），不是 ED=
        model = DeSTSeg(
            dest=True, ed=True,
            num_classes=self.NUM_CLASSES,
            use_d2t=self.USE_D2T,
            use_patchcore=self.USE_PATCHCORE,
            use_afs=False, use_rrs=False,
        ).to(self.device)
        # channel_mask buffer 形状会在此处做天然校验，不匹配会直接 raise
        model.load_state_dict(state_dict)
        model.eval()

        mem_bank = MemoryBank(device=self.device)
        mem_bank.load(mem_bank_path)
        if mem_bank.memory_bank is None:
            raise RuntimeError(f"记忆库加载后为空: {mem_bank_path}")

        self.model = model
        self.memory_bank = mem_bank
        self.ckpt_path = ckpt_path

    # ------ 预处理（与 RodDataset 对齐）------
    def preprocess(self, img_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(img_path).convert("L")
        image = image.resize(RESIZE_SHAPE, Image.BILINEAR)
        img_l = self._tx_l(image).unsqueeze(0).to(self.device)             # (1,1,256,256)
        img_rgb = self._tx_rgb(image.convert("RGB")).unsqueeze(0).to(self.device)  # (1,3,256,256)
        return img_l, img_rgb

    # ------ 推理 ------
    @torch.no_grad()
    def predict(self, img_path: str) -> dict:
        if self.model is None or self.memory_bank is None:
            raise RuntimeError("模型未加载，先调用 load()")
        img_l, img_rgb = self.preprocess(img_path)
        # 与 eval.py:106-108 一致：img_aug 与 img_origin 在测试模式下相同
        out_seg, _, _, _ = self.model(
            img_l, img_rgb, img_l, img_rgb, memory_bank=self.memory_bank
        )
        pred_mask = out_seg.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # (256,256)

        # 渲染三视图（draw.denormalize 接受 CHW tensor）
        gray_np = denormalize(img_l[0])                       # (H,W,3) float[0,1]
        overlay_np = overlay_mask(gray_np, pred_mask, alpha=0.5)
        mask_np = (COLORS[pred_mask] / 255.0).astype(np.float32)  # (H,W,3) float[0,1]

        return {
            "raw": gray_np,        # 用于 UI 显示
            "overlay": overlay_np,
            "mask": mask_np,
            "pred_mask_int": pred_mask,  # 备用：保存调试 / 后续 GT 对照
        }
```

#### 自检（不需 Qt）
```bash
cd D:/Code/Trae/DeSTSeg/ml-destseg
python -c "
from app.inference import InferenceEngine
eng = InferenceEngine()
eng.load(
    'saved_model/D2T_MemB_Dynamic_0.7_10000_202601161613.pckl',
    'saved_model/D2T_MemB_Dynamic_0.7_10000_202601161613_memory_bank_prebuilt.pt',
)
res = eng.predict(r'dataset/eval/images/<选一张>.png')
print('mask classes:', set(res['pred_mask_int'].ravel().tolist()))
print('shapes:', res['raw'].shape, res['overlay'].shape, res['mask'].shape)
"
```
预期：3 视图均为 `(256, 256, 3)`，缺陷图的 `mask classes` 含非 0 值。

---

### 4.2 `app/workers.py`（QThread 包装层）

#### 职责
- 把 `InferenceEngine.load()` / `predict()` 的阻塞调用搬到子线程
- 通过 PySide6 信号机制把结果 / 异常推回主线程

```python
# app/workers.py
from __future__ import annotations
import traceback
from PySide6.QtCore import QObject, Signal, Slot
from app.inference import InferenceEngine


class ModelLoadWorker(QObject):
    """加载模型 + 记忆库；耗时 20–30 s，必须在子线程执行。"""
    finished = Signal(object)        # payload: InferenceEngine 实例
    failed = Signal(str)             # payload: 错误信息

    def __init__(self, ckpt_path: str, mem_bank_path: str):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.mem_bank_path = mem_bank_path

    @Slot()
    def run(self):
        try:
            eng = InferenceEngine()
            eng.load(self.ckpt_path, self.mem_bank_path)
            self.finished.emit(eng)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


class InferenceWorker(QObject):
    """单次单图推理；耗时 1–3 s。"""
    finished = Signal(dict, float)   # (result_dict, elapsed_seconds)
    failed = Signal(str)

    def __init__(self, engine: InferenceEngine, img_path: str):
        super().__init__()
        self.engine = engine
        self.img_path = img_path

    @Slot()
    def run(self):
        import time
        try:
            t0 = time.perf_counter()
            res = self.engine.predict(self.img_path)
            self.finished.emit(res, time.perf_counter() - t0)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
```

#### 主线程调度模板（写在 `main_window.py` 中）
```python
from PySide6.QtCore import QThread
self._thread = QThread(self)
self._worker = InferenceWorker(self.engine, img_path)
self._worker.moveToThread(self._thread)
self._thread.started.connect(self._worker.run)
self._worker.finished.connect(self._on_inference_done)
self._worker.failed.connect(self._on_inference_failed)
# 完成后自动清理
self._worker.finished.connect(self._thread.quit)
self._worker.failed.connect(self._thread.quit)
self._thread.finished.connect(self._worker.deleteLater)
self._thread.finished.connect(self._thread.deleteLater)
self._thread.start()
```

> **必须**让 `_thread` 与 `_worker` 是 `self.` 属性，否则会被 Python GC 提前回收引发崩溃。

---

### 4.3 `app/main_window.py`（UI 主体）

#### 布局（与需求文档 §6 草图一致）
```
QMainWindow
└── 中心 Widget (QVBoxLayout)
    ├── 顶部 QHBoxLayout: [选择图片] [执行推理] [保存结果] | QLabel(当前文件名)
    ├── 中部 QHBoxLayout: 三个 ImageView (QLabel 子类，等比缩放)
    └── 底部 QHBoxLayout: 图例(4 个色块 QLabel) | QStatusBar 信息
```

#### 关键控件 / 信号
```python
class MainWindow(QMainWindow):
    def __init__(self, engine: InferenceEngine):
        super().__init__()
        self.engine = engine
        self.current_img_path: str | None = None
        self.current_result: dict | None = None
        # 三个 ImageView 用 QLabel + setAlignment(AlignCenter) + 自定义 resizeEvent
        # 不用 setScaledContents=True（会变形）
        self.view_raw = ImageView()
        self.view_overlay = ImageView()
        self.view_mask = ImageView()
        # 按钮信号
        self.btn_pick.clicked.connect(self._pick_image)
        self.btn_infer.clicked.connect(self._run_inference)
        self.btn_save.clicked.connect(self._save_results)
```

#### `numpy → QPixmap` 工具
```python
def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """arr: (H, W, 3) float[0,1] → QPixmap（深拷贝避免 numpy buffer 失效）"""
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    arr_u8 = np.ascontiguousarray(arr_u8)  # QImage 要求 C-contiguous
    h, w, _ = arr_u8.shape
    qimg = QImage(arr_u8.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)
```

> `.copy()` 不可省：`QImage` 不会持有 numpy 的内存，函数返回后 `arr_u8` 释放会导致花屏。

#### `ImageView` 自定义子类（保纵横比）
```python
class ImageView(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #888;")
        self._pix: QPixmap | None = None

    def setImage(self, pix: QPixmap):
        self._pix = pix
        self._update_scaled()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_scaled()

    def _update_scaled(self):
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        ))
```

#### 槽函数关键逻辑

**选图（F2.1–F2.4）**：
```python
def _pick_image(self):
    default_dir = str(Path(__file__).resolve().parents[1] / "dataset" / "eval")
    path, _ = QFileDialog.getOpenFileName(
        self, "选择测试图片", default_dir, "Images (*.png)",
    )
    if path:
        self.current_img_path = path
        self.lbl_filename.setText(Path(path).name)
        self.btn_infer.setEnabled(True)
```

**执行推理（F3.1–F3.4）**：见 §4.2 主线程模板，加上 `setEnabled(False)` 防重入。

**结果回调**：
```python
def _on_inference_done(self, result: dict, elapsed: float):
    self.current_result = result
    self.view_raw.setImage(numpy_to_qpixmap(result["raw"]))
    self.view_overlay.setImage(numpy_to_qpixmap(result["overlay"]))
    self.view_mask.setImage(numpy_to_qpixmap(result["mask"]))
    self.statusBar().showMessage(f"推理完成 {elapsed:.2f} s")
    self.btn_pick.setEnabled(True)
    self.btn_infer.setEnabled(True)
    self.btn_save.setEnabled(True)
```

**保存结果（F5.1 / F5.2）**：
```python
def _save_results(self):
    if self.current_result is None or self.current_img_path is None:
        return
    out_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
    if not out_dir:
        return
    stem = Path(self.current_img_path).stem
    for suffix, key in [("raw", "raw"), ("overlay", "overlay"), ("mask", "mask")]:
        arr = (np.clip(self.current_result[key], 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(arr).save(Path(out_dir) / f"{stem}_{suffix}.png")
    self.statusBar().showMessage(f"已保存到 {out_dir}")
```

---

### 4.4 `app/main.py`（启动入口）

```python
# app/main.py
import sys
from pathlib import Path
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication, QMessageBox, QProgressDialog, QInputDialog,
)
from app.workers import ModelLoadWorker
from app.main_window import MainWindow

REPO_ROOT = Path(__file__).resolve().parents[1]
SAVED_MODEL_DIR = REPO_ROOT / "saved_model"


def discover_checkpoint() -> tuple[str, str]:
    """F1.1–F1.3：扫描 saved_model/，返回 (ckpt, mem_bank) 路径。"""
    pckls = sorted(SAVED_MODEL_DIR.glob("*.pckl"))
    # 排除 *_best_st.pckl / *_best_seg.pckl 等中间文件（train.py 训练完会清理，但留心）
    pckls = [p for p in pckls if not p.stem.endswith(("_best_st", "_best_seg"))]
    if not pckls:
        raise FileNotFoundError(f"未在 {SAVED_MODEL_DIR} 找到 *.pckl")
    if len(pckls) == 1:
        ckpt = pckls[0]
    else:
        # F1.2：多 checkpoint 时弹下拉选择
        names = [p.name for p in pckls]
        sel, ok = QInputDialog.getItem(None, "选择模型", "存在多个权重，请选择：", names, 0, False)
        if not ok:
            sys.exit(0)
        ckpt = next(p for p in pckls if p.name == sel)
    mem_bank = ckpt.with_name(ckpt.stem + "_memory_bank_prebuilt.pt")
    if not mem_bank.exists():
        raise FileNotFoundError(f"找不到配套记忆库: {mem_bank}（PatchCore 必需）")
    return str(ckpt), str(mem_bank)


def main():
    app = QApplication(sys.argv)
    try:
        ckpt, mem_bank = discover_checkpoint()
    except FileNotFoundError as e:
        QMessageBox.critical(None, "启动失败", str(e))
        return 1

    # 加载阶段：模态进度对话框 + 子线程加载
    progress = QProgressDialog("正在加载模型与记忆库…", None, 0, 0)
    progress.setWindowTitle("DeSTSeg 启动中")
    progress.setCancelButton(None)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    thread = QThread()
    worker = ModelLoadWorker(ckpt, mem_bank)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    holder = {"engine": None, "error": None}

    def on_done(eng):
        holder["engine"] = eng
        thread.quit()

    def on_fail(msg):
        holder["error"] = msg
        thread.quit()

    worker.finished.connect(on_done)
    worker.failed.connect(on_fail)
    thread.start()

    # 阻塞事件循环直到 thread.finished
    thread.finished.connect(progress.close)
    while thread.isRunning():
        app.processEvents()

    if holder["error"]:
        QMessageBox.critical(None, "加载失败", holder["error"])
        return 2

    win = MainWindow(holder["engine"])
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

> 加载阶段为何要走 `processEvents` 阻塞而不是异步：启动期没有主窗口可承载信号，最简单的做法是等加载完再 `MainWindow(engine)`。这段是首版唯一一处 "伪同步"，工程上可接受。

---

## 5. 实施步骤（推荐顺序，每步可独立提交）

| # | 步骤 | 产物 | 自检方式 |
|---|------|------|----------|
| 1 | 创建 `app/__init__.py`（空文件）+ `app/requirements_app.txt`（一行 `PySide6>=6.5`） | 包结构成型 | `pip install -r app/requirements_app.txt` 成功，`python -c "import PySide6"` 无错 |
| 2 | 实现 `app/inference.py`（仅 `InferenceEngine`，不依赖 Qt） | 纯计算层 | 跑 §4.1 末尾的 `python -c` 自检脚本，输出三视图 shape 与含非 0 类别 |
| 3 | 实现 `app/workers.py`（两个 Worker 类） | 线程包装层 | `python -c "from app.workers import ModelLoadWorker, InferenceWorker"` 无错 |
| 4 | 实现 `app/main_window.py`（MainWindow + ImageView + numpy_to_qpixmap） | UI 主体 | 临时在文件末尾加 `if __name__ == "__main__"` 用桩 engine 起空窗口确认布局 |
| 5 | 实现 `app/main.py`（discover_checkpoint + 启动流程） | 完整应用 | `python -m app.main` 能启动并加载默认模型 |
| 6 | 端到端联调：选含缺陷图、无缺陷图、保存结果 | 通过 AC1–AC7 | 见 §6 |

> **不建议跳步**——尤其第 2 步必须先于第 4 步通过自检，否则 GUI 报错时分不清是推理 bug 还是 UI bug。

---

## 6. 验证（与需求文档 AC1–AC8 对照）

| AC | 测试操作 | 预期 |
|----|----------|------|
| AC1 | `python -m app.main` | 30 s 内进入主界面，状态栏显示模型名 + `CPU` |
| AC2 | 选 [dataset/eval/images/](../dataset/eval/images/) 任一含缺陷图 → 执行推理 | 中部叠加图见红/绿/蓝区域；右部纯掩码非全黑；与 [dataset/eval/labels/](../dataset/eval/labels/) 同名 GT 视觉吻合 |
| AC3 | 选 [dataset/eval/good_900/images/](../dataset/eval/good_900/images/) 任一图 → 执行推理 | 纯掩码近全黑，叠加图与原图视觉无显著差异 |
| AC4 | 拖大 / 拖小窗口 | 三视图等比缩放，无变形、无闪烁 |
| AC5 | 启动前临时把 `*_memory_bank_prebuilt.pt` 重命名 | `discover_checkpoint` 抛 `FileNotFoundError`，`QMessageBox.critical` 弹错并退出，**不**进入主界面 |
| AC6 | 加载阶段拖动启动期对话框 / 切换窗口焦点 | UI 不卡死（依赖 `processEvents`） |
| AC7 | 连续切换 10 张图各推理一次 | 每次成功；任务管理器内存平稳；单图热推理 ≤ 5 s |
| AC8 | 调试用：临时把 `inference.py` 的 `map_location="cpu"` 删除并起一次 | 复现 `RuntimeError: Attempting to deserialize ... CUDA device`，验证 N9 必要性 |

### 额外回归
- 故意把 `InferenceEngine.USE_D2T` 改成 `False`：应在 `model.load_state_dict()` 处直接抛 `RuntimeError`（`channel_mask` buffer 形状 896→448 不匹配）——这是 N12 的天然校验，符合预期。
- 关闭主窗口 → 进程应立即退出（无残留 QThread）。

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `numpy → QImage` 内存悬空导致花屏 | 中 | 高 | `numpy_to_qpixmap` 内 `.copy()` + `np.ascontiguousarray` |
| `QThread` 与 `Worker` 被 Python GC 提前回收 | 中 | 高 | 主窗口持有为 `self._thread` / `self._worker`；用完 `deleteLater` 链清理 |
| 启动加载期主窗口未创建，信号无处转发 | 低 | 中 | `main()` 中用 `processEvents` 伪同步等待；首版接受 |
| `RodDataset` 未来变更 `RESIZE_SHAPE` 或归一化常量 | 低 | 中 | `app/inference.py` 始终从 [constant.py](../constant.py) import，不硬编码 |
| 用户给的 `.pckl` 与 `.pt` 不匹配（不同训练 run） | 低 | 高 | `discover_checkpoint` 严格按 `<stem>_memory_bank_prebuilt.pt` 配对，不允许跨配对 |
| 用户机器无 PySide6 ≥ 6.5（版本太旧） | 低 | 中 | `requirements_app.txt` 锁 `>=6.5`；首次 `import PySide6` 失败时给出 `pip install` 提示 |
| Windows 路径含中文 / 空格 | 低 | 中 | 全程 `pathlib.Path`，避免字符串拼接；不用 `os.system` |

### 回滚
所有产物都在 `app/` 子目录下，**未触碰**任何既有文件。回滚等于 `rm -rf app/`，零副作用。

---

## 8. 后续版本（首版不做，记录于此）

承袭需求文档 §8 的 Q1–Q5：原分辨率回插值、批量推理、GUI 暴露开关、GT 第四视图、热力图。在首版稳定后，优先级排序建议为 Q4（GT 对照，最直接强化演示效果）→ Q1（原分辨率，做答辩截图用）→ 其余。

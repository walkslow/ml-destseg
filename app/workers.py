"""PySide6 后台 Worker 层。

本模块只做一件事：把 `app.inference.InferenceEngine` 的阻塞调用包装成
可放入 `QThread` 的 `QObject`，让 GUI 主线程能够保持响应。

设计原则：
- 这里不创建窗口，不访问按钮，不管理状态栏。
- 这里只负责把“耗时的推理动作”搬到子线程，并把成功/失败结果通过信号抛回去。
- 线程生命周期由 `main_window.py` 维护；Worker 只负责执行一次任务。
"""

from __future__ import annotations

import time
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from app.inference import InferenceEngine


class ModelLoadWorker(QObject):
    """在子线程里加载 DeSTSeg checkpoint 与 PatchCore 记忆库。

    这个 Worker 用在应用启动阶段。它的职责非常单一：
    1. 创建 `InferenceEngine`。
    2. 调用 `InferenceEngine.load()`。
    3. 成功时把可继续推理的 engine 交回主线程。
    4. 失败时把详细错误字符串交回主线程，让 GUI 弹窗提示用户。

    信号约定：
    - `finished(object)`：payload 是已经加载完成的 `InferenceEngine` 实例。
    - `failed(str)`：payload 是格式化后的异常信息，包含异常类型、信息和 traceback。
    """

    # 用 object 作为载荷类型，避免把具体类与 Qt 信号系统绑死。
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, ckpt_path: str, mem_bank_path: str) -> None:
        """保存加载所需的两个路径。

        Args:
            ckpt_path: DeSTSeg checkpoint 路径。
            mem_bank_path: 与 checkpoint 同名的 PatchCore 预构建记忆库路径。
        """
        super().__init__()
        self.ckpt_path = ckpt_path
        self.mem_bank_path = mem_bank_path

    @Slot()
    def run(self) -> None:
        """执行一次后台加载任务。

        这里故意不捕获并吞掉异常，只把异常转成字符串通过 `failed` 传回去。
        这样主线程可以统一决定是弹窗、记录日志还是重新启用按钮。
        """
        try:
            engine = InferenceEngine()
            engine.load(self.ckpt_path, self.mem_bank_path)
            self.finished.emit(engine)
        except Exception as exc:  # noqa: BLE001 - 这里需要捕获所有异常再回传给 GUI
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


class InferenceWorker(QObject):
    """在子线程里执行单张图片推理。

    这个 Worker 只负责调用 `InferenceEngine.predict()` 并统计耗时。
    它不会决定如何显示结果，也不会保存图片；这些都属于主线程 GUI 逻辑。

    信号约定：
    - `finished(dict, float)`：第一个参数是结果字典，第二个参数是耗时（秒）。
    - `failed(str)`：payload 是格式化后的异常信息，包含异常类型、信息和 traceback。
    """

    finished = Signal(dict, float)
    failed = Signal(str)

    def __init__(self, engine: InferenceEngine, img_path: str) -> None:
        """保存已加载的推理引擎和待推理图片路径。

        Args:
            engine: 已经完成 `load()` 的 `InferenceEngine`。
            img_path: 待推理图片路径。
        """
        super().__init__()
        self.engine = engine
        self.img_path = img_path

    @Slot()
    def run(self) -> None:
        """执行一次单图推理任务并回传结果。

        这里使用 `time.perf_counter()` 统计墙钟时间，便于 GUI 在状态栏里展示
        “推理完成 1.23 s” 这类信息。
        """
        try:
            started_at = time.perf_counter()
            result = self.engine.predict(self.img_path)
            elapsed_seconds = time.perf_counter() - started_at
            self.finished.emit(result, elapsed_seconds)
        except Exception as exc:  # noqa: BLE001 - 需要把任意异常完整回传给 GUI
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

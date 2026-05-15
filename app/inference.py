"""DeSTSeg 桌面推理工具的纯计算层。

本模块只负责“模型相关”的工作：

1. 在 CPU 上加载 DeSTSeg checkpoint。
2. 加载与 checkpoint 同名的 PatchCore 记忆库。
3. 将单张 PNG 图片预处理成模型需要的灰度分支与 RGB 分支输入。
4. 执行一次前向推理，得到类别预测 mask。
5. 把预测结果渲染成 GUI 可直接显示的三张 RGB 图。

重要约束：
- 本文件不得 import PySide6，避免计算层与 GUI 层耦合。
- 推理设备固定为 CPU，不根据 `torch.cuda.is_available()` 做分支。
- 首版只服务当前训练好的 ROD 缺陷分割模型，模型开关在类常量中硬编码。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from constant import (
    NORMALIZE_MEAN_L,
    NORMALIZE_MEAN_RGB,
    NORMALIZE_STD_L,
    NORMALIZE_STD_RGB,
    RESIZE_SHAPE,
)
from draw import COLORS, denormalize, overlay_mask
from model.destseg import DeSTSeg
from model.patchcore_mem import MemoryBank


REPO_ROOT = Path(__file__).resolve().parents[1]

# 首版默认模型由 REQUIREMENTS.md / IMPLEMENTATION.md 锁定。
# 这里集中定义默认 stem，避免 checkpoint 与记忆库路径拼接处重复写字符串。
DEFAULT_MODEL_STEM = "D2T_MemB_Dynamic_0.7_10000_202601161613"
DEFAULT_CKPT_PATH = REPO_ROOT / "saved_model" / f"{DEFAULT_MODEL_STEM}.pckl"
DEFAULT_MEMORY_BANK_PATH = (
    REPO_ROOT / "saved_model" / f"{DEFAULT_MODEL_STEM}_memory_bank_prebuilt.pt"
)


class InferenceEngine:
    """DeSTSeg + PatchCore 单图 CPU 推理封装。

    这个类是 GUI 层唯一需要持有的推理对象。它把模型、PatchCore 记忆库、
    预处理 transform 和默认设备封装在一起，避免 MainWindow 或 Worker 直接接触
    `model/` 里的训练实现细节。

    首版固定配置：
    - `num_classes=4`：背景 + scratch / dent / dotted。
    - `use_d2t=True`：与默认 checkpoint 的 D2T 结构匹配。
    - `use_patchcore=True`：推理时必须传入 memory bank。
    - `use_afs=False` / `use_rrs=False`：首版不启用特征选择开关。

    线程安全约定：load() 与 predict() 不可并发；GUI 层通过 QThread 串行调用。
    """

    NUM_CLASSES = 4
    USE_D2T = True
    USE_PATCHCORE = True
    USE_AFS = False
    USE_RRS = False

    def __init__(self) -> None:
        """初始化 CPU 设备、线程数和图像预处理管道。

        注意这里故意不检查 CUDA。目标设备是无 GPU 笔记本，即使运行环境中
        恰好有 CUDA，也要走完全一致的 CPU 路径，减少演示环境差异。
        """
        self.device = torch.device("cpu")

        # CPU 推理时 PatchCore 最近邻搜索和 backbone 前向都会占用 CPU。
        # 留出 1 个逻辑核心给 GUI 主线程，避免推理期间窗口完全失去响应。
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

        # load() 成功后才会填充模型与记忆库。这样可以让 GUI 明确区分
        # “对象已创建”和“模型已可推理”两个状态。
        self.model: DeSTSeg | None = None
        self.memory_bank: MemoryBank | None = None
        self.ckpt_path: str | None = None
        self.memory_bank_path: str | None = None

        # 与 data/rod_dataset.py 保持一致：
        # - 学生网络吃单通道灰度图，归一化到 [-1, 1]。
        # - 教师网络吃三通道 RGB 图，使用 ImageNet 均值/方差。
        self._tx_l = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN_L, NORMALIZE_STD_L),
            ]
        )
        self._tx_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN_RGB, NORMALIZE_STD_RGB),
            ]
        )

    def load(self, ckpt_path: str | os.PathLike[str], mem_bank_path: str | os.PathLike[str]) -> None:
        """加载 checkpoint 与同名 PatchCore 记忆库。

        Args:
            ckpt_path: DeSTSeg `state_dict` 文件路径，通常是 `saved_model/*.pckl`。
            mem_bank_path: PatchCore 预构建记忆库路径，必须与 checkpoint 同 stem。

        Raises:
            FileNotFoundError: checkpoint 或记忆库文件不存在。
            TypeError: checkpoint 不是可直接加载的 `state_dict` 字典。
            RuntimeError: checkpoint 与当前硬编码模型结构不匹配，或记忆库加载为空。

        关键点：
        - `torch.load(..., map_location="cpu")` 必须保留，否则无 GPU 笔记本加载
          GPU 服务器保存的权重时会尝试还原 `cuda:0` 并报错。
        - 当前 checkpoint 可能来自 DataParallel，也可能来自 `real_model.state_dict()`；
          因此这里防御性移除 `module.` 前缀。
        """
        ckpt = Path(ckpt_path)
        mem_bank_file = Path(mem_bank_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt}")
        if not mem_bank_file.exists():
            raise FileNotFoundError(f"记忆库不存在: {mem_bank_file}（PatchCore 必需）")

        # 跨设备加载的核心约束：始终映射到 CPU。
        # 这也是 REQUIREMENTS.md N9 明确要求的行为。
        state_dict = torch.load(ckpt, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint 格式不支持，期望 state_dict 字典: {ckpt}")

        # 训练脚本通常保存 real_model.state_dict()，不会带 module.；
        # 但如果用户换成 DataParallel 外层保存的权重，这里也能兼容。
        state_dict = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

        # 这些开关必须与 checkpoint 训练时的结构一致。
        # 特别是 use_d2t / use_patchcore 会改变 segmentation_net 输入通道数。
        model = DeSTSeg(
            dest=True,
            ed=True,
            num_classes=self.NUM_CLASSES,
            use_d2t=self.USE_D2T,
            use_patchcore=self.USE_PATCHCORE,
            use_afs=self.USE_AFS,
            use_rrs=self.USE_RRS,
        ).to(self.device)

        # 当前仓库模型定义新增了 channel_mask buffer；默认 checkpoint 可能是在该
        # buffer 保存前训练得到的。AFS/RRS 关闭时，初始化的全 1 channel_mask
        # 与“无筛选”行为等价，因此只允许缺失这一项。
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"channel_mask"}
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if unexpected or missing - allowed_missing:
            raise RuntimeError(
                "Checkpoint 与模型结构不匹配: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        model.eval()

        # MemoryBank.load() 内部已经使用 self.device 作为 map_location；
        # 因此只要构造 MemoryBank 时传 CPU，就能满足跨设备加载要求。
        memory_bank = MemoryBank(device=self.device)
        memory_bank.load(str(mem_bank_file))
        if memory_bank.memory_bank is None:
            raise RuntimeError(f"记忆库加载后为空: {mem_bank_file}")

        self.model = model
        self.memory_bank = memory_bank
        self.ckpt_path = str(ckpt)
        self.memory_bank_path = str(mem_bank_file)

    def preprocess(self, img_path: str | os.PathLike[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """读取 PNG 并生成灰度学生分支与 RGB 教师分支输入。

        Args:
            img_path: 待推理图片路径。首版 GUI 只选择 PNG，本函数本身不强制后缀。

        Returns:
            `(img_l, img_rgb)`：
            - `img_l` 形状为 `(1, 1, 256, 256)`，供 student_net 使用。
            - `img_rgb` 形状为 `(1, 3, 256, 256)`，供 teacher_net 使用。

        Raises:
            FileNotFoundError: 图片路径不存在。

        这里先统一转为灰度 `L`，再复制成 RGB 分支。这样与 ROD 数据集推理路径
        一致：原始输入是单通道燃料棒图像，但教师网络仍需要三通道张量。
        """
        path = Path(img_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")

        # PIL 文件句柄只在 with 块内持有；convert() 会复制像素数据，
        # 后续 resize 不依赖原始文件句柄。
        with Image.open(path) as opened:
            image_l = opened.convert("L")

        # RESIZE_SHAPE 在 constant.py 中定义为 [256, 256]，语义是 width, height；
        # PIL.resize() 也要求 (width, height)，因此这里直接 tuple() 即可。
        image_l = image_l.resize(tuple(RESIZE_SHAPE), Image.BILINEAR)

        img_l = self._tx_l(image_l).unsqueeze(0).to(self.device)
        img_rgb = self._tx_rgb(image_l.convert("RGB")).unsqueeze(0).to(self.device)
        return img_l, img_rgb

    @torch.no_grad()
    def predict(self, img_path: str | os.PathLike[str]) -> dict[str, Any]:
        """对单张图片执行推理，返回三视图和整型预测掩码。

        Args:
            img_path: 待推理图片路径。

        Returns:
            一个字典，供 GUI Worker 直接通过 Qt Signal 传回主线程：
            - `raw`: `(256, 256, 3)`，float32，范围 `[0, 1]`，灰度原图可视化。
            - `overlay`: `(256, 256, 3)`，float32，范围 `[0, 1]`，原图叠加彩色 mask。
            - `mask`: `(256, 256, 3)`，float32，范围 `[0, 1]`，纯彩色 mask。
            - `pred_mask_int`: `(256, 256)`，uint8，类别 ID 掩码。

        Raises:
            RuntimeError: 还未成功调用 `load()`。

        推理阶段必须传入 `memory_bank`。如果传 `None`，DeSTSeg 会走训练 Phase 1
        的 PatchCore 占位路径，得到的分割结果会偏离默认 checkpoint 的真实配置。
        """
        if self.model is None or self.memory_bank is None:
            raise RuntimeError("模型未加载，先调用 load()")

        img_l, img_rgb = self.preprocess(img_path)

        # eval.py 的测试路径中，img_aug 与 img_origin 在推理时相同。
        # 这里保持相同调用约定，避免改动 model.forward 的公共接口。
        out_seg, _, _, _ = self.model(
            img_l,
            img_rgb,
            img_l,
            img_rgb,
            memory_bank=self.memory_bank,
        )

        # segmentation_net 当前输出空间尺寸为 64x64；GUI 首版展示固定 256x256。
        # 与 eval.py 的保险逻辑一致，先把 logits 双线性插值回输入尺寸，再 argmax。
        # 先插值 logits 而不是插值类别 mask，可以避免类别边界出现硬块状放大。
        if out_seg.shape[-2:] != img_l.shape[-2:]:
            out_seg = F.interpolate(
                out_seg,
                size=img_l.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pred_mask = out_seg.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        # draw.denormalize() 接受 CHW tensor 并输出 HWC RGB float 图；
        # 后续 main_window.py 会把这些数组转换成 QImage/QPixmap。
        raw = _as_float_image(denormalize(img_l[0]))
        overlay = _as_float_image(overlay_mask(raw, pred_mask, alpha=0.5))
        mask = _as_float_image(COLORS[pred_mask] / 255.0)

        return {
            "raw": raw,
            "overlay": overlay,
            "mask": mask,
            "pred_mask_int": pred_mask,
        }


def _as_float_image(arr: np.ndarray) -> np.ndarray:
    """规范化 GUI 侧预期的 HWC float32 RGB 图像数组。

    Args:
        arr: 任意可转为 numpy array 的 RGB 图像数组，通常已经是 `[0, 1]`。

    Returns:
        `float32` 数组，数值裁剪到 `[0, 1]`。这里不负责检查 shape，
        因为调用方只会传入本模块内部生成的三视图。
    """
    return np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行自检入口的参数解析器。

    CLI 入口主要给步骤 2 的纯计算层自检使用，不承担 GUI 功能。
    后续如果 GUI 报错，可以先用 `python -m app.inference <png>` 判断问题
    是否出在模型加载/推理层。
    """
    parser = argparse.ArgumentParser(description="DeSTSeg CPU 单图推理自检")
    parser.add_argument("image", help="待推理 PNG 图片路径")
    parser.add_argument(
        "--ckpt",
        default=str(DEFAULT_CKPT_PATH),
        help=f"checkpoint 路径，默认: {DEFAULT_CKPT_PATH}",
    )
    parser.add_argument(
        "--memory-bank",
        default=str(DEFAULT_MEMORY_BANK_PATH),
        help=f"PatchCore 记忆库路径，默认: {DEFAULT_MEMORY_BANK_PATH}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行一次命令行单图推理自检。

    Args:
        argv: 可选参数列表。传 `None` 时使用真实命令行参数，便于人工执行；
            测试或调试时也可以传入列表复用本函数。

    Returns:
        进程退出码。当前只要没有异常就返回 0；异常交给 Python 打印 traceback，
        方便定位模型加载或推理层问题。
    """
    args = _build_parser().parse_args(argv)
    engine = InferenceEngine()
    engine.load(args.ckpt, args.memory_bank)
    result = engine.predict(args.image)

    classes = sorted(set(result["pred_mask_int"].ravel().tolist()))
    print("device:", engine.device)
    print("classes:", classes)
    print(
        "shapes:",
        result["raw"].shape,
        result["overlay"].shape,
        result["mask"].shape,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

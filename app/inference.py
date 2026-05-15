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
DEFAULT_MODEL_STEM = "D2T_MemB_Dynamic_0.7_10000_202601161613"
DEFAULT_CKPT_PATH = REPO_ROOT / "saved_model" / f"{DEFAULT_MODEL_STEM}.pckl"
DEFAULT_MEMORY_BANK_PATH = (
    REPO_ROOT / "saved_model" / f"{DEFAULT_MODEL_STEM}_memory_bank_prebuilt.pt"
)


class InferenceEngine:
    """DeSTSeg + PatchCore 单图 CPU 推理封装。

    线程安全约定：load() 与 predict() 不可并发；GUI 层通过 QThread 串行调用。
    """

    NUM_CLASSES = 4
    USE_D2T = True
    USE_PATCHCORE = True
    USE_AFS = False
    USE_RRS = False

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

        self.model: DeSTSeg | None = None
        self.memory_bank: MemoryBank | None = None
        self.ckpt_path: str | None = None
        self.memory_bank_path: str | None = None

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
        """加载 checkpoint 与同名 PatchCore 记忆库。"""
        ckpt = Path(ckpt_path)
        mem_bank_file = Path(mem_bank_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt}")
        if not mem_bank_file.exists():
            raise FileNotFoundError(f"记忆库不存在: {mem_bank_file}（PatchCore 必需）")

        state_dict = torch.load(ckpt, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint 格式不支持，期望 state_dict 字典: {ckpt}")
        state_dict = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

        model = DeSTSeg(
            dest=True,
            ed=True,
            num_classes=self.NUM_CLASSES,
            use_d2t=self.USE_D2T,
            use_patchcore=self.USE_PATCHCORE,
            use_afs=self.USE_AFS,
            use_rrs=self.USE_RRS,
        ).to(self.device)
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

        memory_bank = MemoryBank(device=self.device)
        memory_bank.load(str(mem_bank_file))
        if memory_bank.memory_bank is None:
            raise RuntimeError(f"记忆库加载后为空: {mem_bank_file}")

        self.model = model
        self.memory_bank = memory_bank
        self.ckpt_path = str(ckpt)
        self.memory_bank_path = str(mem_bank_file)

    def preprocess(self, img_path: str | os.PathLike[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """读取 PNG 并生成灰度学生分支与 RGB 教师分支输入。"""
        path = Path(img_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")

        with Image.open(path) as opened:
            image_l = opened.convert("L")
        image_l = image_l.resize(tuple(RESIZE_SHAPE), Image.BILINEAR)

        img_l = self._tx_l(image_l).unsqueeze(0).to(self.device)
        img_rgb = self._tx_rgb(image_l.convert("RGB")).unsqueeze(0).to(self.device)
        return img_l, img_rgb

    @torch.no_grad()
    def predict(self, img_path: str | os.PathLike[str]) -> dict[str, Any]:
        """对单张图片执行推理，返回三视图和整型预测掩码。"""
        if self.model is None or self.memory_bank is None:
            raise RuntimeError("模型未加载，先调用 load()")

        img_l, img_rgb = self.preprocess(img_path)
        out_seg, _, _, _ = self.model(
            img_l,
            img_rgb,
            img_l,
            img_rgb,
            memory_bank=self.memory_bank,
        )
        if out_seg.shape[-2:] != img_l.shape[-2:]:
            out_seg = F.interpolate(
                out_seg,
                size=img_l.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pred_mask = out_seg.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

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
    """规范化 GUI 侧预期的 HWC float32 RGB 图像数组。"""
    return np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)


def _build_parser() -> argparse.ArgumentParser:
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

import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data.data_utils import perlin_noise


class MVTecDataset(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir, # MVTec数据集目录路径
        resize_shape=[256, 256], # 图像调整大小的目标尺寸，用 [width, height] 顺序（PIL图像标准）
        normalize_mean=[0.485, 0.456, 0.406], # 归一化使用的均值，是 ImageNet 数据集在 RGB 三个通道上的均值
        normalize_std=[0.229, 0.224, 0.225], # 归一化使用的标准差，是 ImageNet 数据集在 RGB 三个通道上的标准差
        dtd_dir=None, # DTD纹理数据集目录（仅训练时使用）
        rotate_90=False, # 是否进行90度倍数的随机旋转
        random_rotate=0, # 随机旋转角度范围
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        # transforms.Resize 的 size 参数采用 (height, width) 顺序
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        # 返回数据集中图像的总数
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            # (1,): 生成一个随机数的tensor；.item() 将单元素tensor转换为Python标量值
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            fill_color = (114, 114, 114) # 114是一个常用的经验值，接近ImageNet数据集的平均像素值
            # rotate_90
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )
            # random_rotate
            if self.random_rotate > 0:
                # 从均匀分布中抽取随机样本
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )

            # 使用Perlin噪声算法将 dtd_image 纹理合成到原始 image 上
            # 生成增强图像 aug_image 和对应的掩码 aug_mask，aug_prob=1.0 表示100%执行增强操作
            aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=1.0)
            aug_image = self.final_preprocessing(aug_image)

            image = self.final_preprocessing(image)
            # 训练模式：返回添加DTD纹理后的图像img_aug，原始图像(可选增强)img_origin，img_aug对应的掩码mask
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else: # 测试过程
            image = self.final_preprocessing(image)
            dir_path, file_name = os.path.split(self.mvtec_paths[index])
            base_dir = os.path.basename(dir_path)
            if base_dir == "good":
                # 创建与图像相同尺寸的全零掩码，表示没有异常区域
                # image[:1]相当于image[:1, :, :]和image[:1, ...]
                # 选取image第0个元素，即第0个通道，其形状为[1,H,W]
                mask = torch.zeros_like(image[:1])
            else:
                mask_path = os.path.join(dir_path, "../../ground_truth/")
                mask_path = os.path.join(mask_path, base_dir)
                # 掩码的文件名 = 图片名 + _mask.png
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = Image.open(mask_path)
                mask = self.mask_preprocessing(mask)
                # 将掩码二值化：像素值<0.5设为0，≥0.5设为1
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )
            # 测试模式：返回原始图像img和对应的异常区域掩码mask
            return {"img": image, "mask": mask}

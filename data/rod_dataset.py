import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data.data_utils import cut_paste


class PILToLongTensor:
    """
    将一个像素值本身就是类别ID的PIL图像，直接转换为LongTensor。
    """
    def __call__(self, pic):
        # 将PIL图像转换为numpy数组，数据类型为无符号8位整数
        img_np = np.array(pic, dtype=np.uint8)
        # 从numpy数组创建Tensor，并转换为LongTensor
        return torch.from_numpy(img_np).long()


class RodDataset(Dataset):
    def __init__(
        self,
        is_train,
        rod_dir,  # 燃料棒数据集目录路径
        resize_shape=[256, 256],  # 图像调整大小的目标尺寸，用 [width, height] 顺序（PIL图像标准）
        normalize_mean=[0.5],  # 归一化使用的均值（单通道）
        normalize_std=[0.5],  # 归一化使用的标准差（单通道）
        scratch_dir=None,  # 划伤数据集目录（仅训练时使用）
        dent_dir=None,  # 磕伤数据集目录（仅训练时使用）
        dotted_dir=None,  # 异物数据集目录（仅训练时使用）
        rotate_90=False,  # 是否进行90度倍数的随机旋转
        random_rotate=0,  # 随机旋转角度范围
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train

        # 定义图像的预处理
        self.image_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

        if is_train:
            self.rod_paths = sorted(glob.glob(rod_dir + "/*.png"))
            self.scratch_dir = scratch_dir
            self.scratch_img_paths = sorted(glob.glob(scratch_dir + "/images" +"/*.png")) if scratch_dir else None
            self.scratch_msk_paths = sorted(glob.glob(scratch_dir + "/labels" +"/*.png")) if scratch_dir else None
            self.dent_dir = dent_dir
            self.dent_img_paths = sorted(glob.glob(dent_dir + "/images" +"/*.png")) if dent_dir else None
            self.dent_msk_paths = sorted(glob.glob(dent_dir + "/labels" +"/*.png")) if dent_dir else None
            self.dotted_dir = dotted_dir
            self.dotted_img_paths = sorted(glob.glob(dotted_dir + "/images" +"/*.png")) if dotted_dir else None
            self.dotted_msk_paths = sorted(glob.glob(dotted_dir + "/labels" +"/*.png")) if dotted_dir else None
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
            
            # 训练掩码预处理器：cut_paste返回的PIL掩码，其像素值就是类别ID
            # 它已经是正确的尺寸，所以只需要直接转换为LongTensor
            self.mask_preprocessing = PILToLongTensor()
        else:
            self.rod_paths = sorted(glob.glob(rod_dir + "/*.png"))
            # 测试掩码预处理器：从文件加载的PIL掩码，其像素值也是类别ID
            # 需要先进行Resize，然后再转换为LongTensor
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.NEAREST,
                    ),
                    PILToLongTensor(),
                ]
            )

    def __len__(self):
        # 返回数据集中图像的总数
        return len(self.rod_paths)

    def __getitem__(self, index):
        image = Image.open(self.rod_paths[index]).convert("L")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            if self.scratch_dir:
                defect_idx = torch.randint(0, len(self.scratch_img_paths), (1,)).item()
                scratch_img = Image.open(self.scratch_img_paths[defect_idx]).convert("L")
                scratch_img = scratch_img.resize(self.resize_shape, Image.BILINEAR)
                scratch_msk = Image.open(self.scratch_msk_paths[defect_idx]).convert("L")   
                scratch_msk = scratch_msk.resize(self.resize_shape, Image.NEAREST)
            else:
                scratch_img = None
                scratch_msk = None
            if self.dent_dir:
                defect_idx = torch.randint(0, len(self.dent_img_paths), (1,)).item()
                dent_img = Image.open(self.dent_img_paths[defect_idx]).convert("L")
                dent_img = dent_img.resize(self.resize_shape, Image.BILINEAR)
                dent_msk = Image.open(self.dent_msk_paths[defect_idx]).convert("L")   
                dent_msk = dent_msk.resize(self.resize_shape, Image.NEAREST)
            else:
                dent_img = None
                dent_msk = None
            if self.dotted_dir:
                defect_idx = torch.randint(0, len(self.dotted_img_paths), (1,)).item()
                dotted_img = Image.open(self.dotted_img_paths[defect_idx]).convert("L")
                dotted_img = dotted_img.resize(self.resize_shape, Image.BILINEAR)
                dotted_msk = Image.open(self.dotted_msk_paths[defect_idx]).convert("L")   
                dotted_msk = dotted_msk.resize(self.resize_shape, Image.NEAREST)
            else:
                dotted_img = None
                dotted_msk = None
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

            
            # 生成增强图像 aug_image 和对应的掩码 aug_mask，表示100%执行增强操作
            aug_image, aug_mask = cut_paste(image, scratch_img, scratch_msk, dent_img, dent_msk, dotted_img, dotted_msk)

            aug_mask = self.mask_preprocessing(aug_mask)
            aug_image = self.image_preprocessing(aug_image)
            image = self.image_preprocessing(image)
            
            # 训练模式：返回添加DTD纹理后的图像img_aug，原始图像(可选增强)img_origin，img_aug对应的掩码mask
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else: # 测试过程
            image = self.image_preprocessing(image)
            dir_path, file_name = os.path.split(self.rod_paths[index])
            mask_path = os.path.join(dir_path, "../labels/")
            mask_path = os.path.join(mask_path, file_name)
            mask = Image.open(mask_path)
            mask = self.mask_preprocessing(mask)
            # 测试模式：返回原始图像img和对应的异常区域掩码mask
            return {"img": image, "mask": mask}

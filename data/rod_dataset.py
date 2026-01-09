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
    将一个像素值本身就是类别ID的PIL图像（例如，0代表背景，1代表类别1），
    直接转换为PyTorch的LongTensor。这在分割任务中非常有用，因为损失函数
    （如CrossEntropyLoss）期望的目标掩码（target mask）就是这种格式。
    """

    def __call__(self, pic):
        # 将PIL图像转换为numpy数组，数据类型为无符号8位整数
        img_np = np.array(pic, dtype=np.uint8)
        # 从numpy数组创建Tensor，并转换为LongTensor
        return torch.from_numpy(img_np).long()


class RodDataset(Dataset):
    """
    为燃料棒（ROD）缺陷检测任务定制的数据集类。
    该数据集的核心是为学生-教师网络框架提供数据：
    - 学生网络（StudentNet）接收单通道灰度图。
    - 教师网络（TeacherNet）接收三通道RGB图。
    - 当 use_real_data=False 时（合成模式），通过`cut_paste`方法在正常图像上合成缺陷，生成增强图像和对应的掩码。
    - 当 use_real_data=True 时（真实模式），加载真实图像和对应的真实掩码。
    """

    def __init__(
        self,
        rod_dir,
        resize_shape=[256, 256],
        normalize_mean_l=[0.5],
        normalize_std_l=[0.5],
        normalize_mean_rgb=[0.485, 0.456, 0.406],
        normalize_std_rgb=[0.229, 0.224, 0.225],
        scratch_dir=None,
        dent_dir=None,
        dotted_dir=None,
        rotate_90=False,
        random_rotate=0,
        use_real_data=False,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.use_real_data = use_real_data

        # 如果未指定任何缺陷源目录，自动切换到真实数据模式
        if scratch_dir is None and dent_dir is None and dotted_dir is None:
            self.use_real_data = True

        # --- 为学生网络（灰度图）和教师网络（RGB图）定义不同的预处理管道 ---
        # 学生网络使用单通道灰度图，归一化参数通常设为[0.5], [0.5]
        self.image_preprocessing_l = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean_l, normalize_std_l),
            ]
        )
        # 教师网络使用三通道RGB图，通常使用在ImageNet上预训练的标准归一化参数
        self.image_preprocessing_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean_rgb, normalize_std_rgb),
            ]
        )

        # 统一路径逻辑：rod_dir 应该是包含 images/ 和 labels/ 的父目录
        # 尝试从 rod_dir/images 加载
        self.rod_paths = sorted(glob.glob(os.path.join(rod_dir, "images", "*.png")))
        # 如果为空，尝试直接从 rod_dir 加载 (兼容旧的路径格式或纯背景文件夹)
        if len(self.rod_paths) == 0:
            self.rod_paths = sorted(glob.glob(os.path.join(rod_dir, "*.png")))

        # 数据增强相关参数
        # 如果使用真实数据，强制关闭旋转增强 (根据用户需求：若使用真实图片而非合成图片，则不进行旋转增强)
        if self.use_real_data:
            self.rotate_90 = False
            self.random_rotate = 0
        else:
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate

        if not self.use_real_data:
            # --- 合成模式下的初始化 ---
            # 加载用于合成缺陷的图像和掩码路径
            self.scratch_dir = scratch_dir
            self.scratch_img_paths = sorted(glob.glob(scratch_dir + "/images" + "/*.png")) if scratch_dir else None
            self.scratch_msk_paths = sorted(glob.glob(scratch_dir + "/labels" + "/*.png")) if scratch_dir else None
            self.dent_dir = dent_dir
            self.dent_img_paths = sorted(glob.glob(dent_dir + "/images" + "/*.png")) if dent_dir else None
            self.dent_msk_paths = sorted(glob.glob(dent_dir + "/labels" + "/*.png")) if dent_dir else None
            self.dotted_dir = dotted_dir
            self.dotted_img_paths = sorted(glob.glob(dotted_dir + "/images" + "/*.png")) if dotted_dir else None
            self.dotted_msk_paths = sorted(glob.glob(dotted_dir + "/labels" + "/*.png")) if dotted_dir else None

            # 训练掩码预处理器：cut_paste返回的PIL掩码，其像素值就是类别ID，
            # 且尺寸已对齐，所以只需要直接转换为LongTensor。
            self.mask_preprocessing = PILToLongTensor()
        else:
            # --- 真实模式下的初始化 ---
            # 测试掩码预处理器：从文件加载的PIL掩码，其像素值也是类别ID，
            # 但需要先进行Resize以匹配模型输入尺寸，然后再转换为LongTensor。
            # 注意：Resize操作被移到了__getitem__中，以便和图像进行同步旋转增强(如果启用)
            self.mask_preprocessing = PILToLongTensor()

    def __len__(self):
        return len(self.rod_paths)

    def __getitem__(self, index):
        # --- 1. 加载并预处理图像 ---
        # 统一以灰度模式（'L'）加载，因为所有操作（如cut_paste）都基于灰度图
        image = Image.open(self.rod_paths[index]).convert("L")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if not self.use_real_data:
            # --- 2. 合成模式：加载缺陷、执行数据增强和cut_paste ---
            # 随机加载用于合成的缺陷图像和掩码
            if self.scratch_dir:
                defect_idx = torch.randint(0, len(self.scratch_img_paths), (1,)).item()
                scratch_img = Image.open(self.scratch_img_paths[defect_idx]).convert("L")
                scratch_img = scratch_img.resize(self.resize_shape, Image.BILINEAR)
                scratch_msk = Image.open(self.scratch_msk_paths[defect_idx]).convert("L")
                scratch_msk = scratch_msk.resize(self.resize_shape, Image.NEAREST)
            else:
                scratch_img, scratch_msk = None, None
            if self.dent_dir:
                defect_idx = torch.randint(0, len(self.dent_img_paths), (1,)).item()
                dent_img = Image.open(self.dent_img_paths[defect_idx]).convert("L")
                dent_img = dent_img.resize(self.resize_shape, Image.BILINEAR)
                dent_msk = Image.open(self.dent_msk_paths[defect_idx]).convert("L")
                dent_msk = dent_msk.resize(self.resize_shape, Image.NEAREST)
            else:
                dent_img, dent_msk = None, None
            if self.dotted_dir:
                defect_idx = torch.randint(0, len(self.dotted_img_paths), (1,)).item()
                dotted_img = Image.open(self.dotted_img_paths[defect_idx]).convert("L")
                dotted_img = dotted_img.resize(self.resize_shape, Image.BILINEAR)
                dotted_msk = Image.open(self.dotted_msk_paths[defect_idx]).convert("L")
                dotted_msk = dotted_msk.resize(self.resize_shape, Image.NEAREST)
            else:
                dotted_img, dotted_msk = None, None

            # 对原始无缺陷图像进行旋转增强
            fill_color = 0  # 灰度图旋转时的填充色
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)

            # cut_paste函数接收灰度图，返回合成后的灰度图和对应掩码
            aug_image_l, aug_mask = cut_paste(image, scratch_img, scratch_msk, dent_img, dent_msk, dotted_img, dotted_msk)

            # --- 3. 生成灰度图和RGB图对 ---
            # 从最终的灰度图像（原始图和增强图）创建对应的RGB版本，用于教师网络
            aug_image_rgb = aug_image_l.convert("RGB")
            image_rgb = image.convert("RGB")

            # --- 4. 应用Tensor转换和归一化 ---
            aug_mask = self.mask_preprocessing(aug_mask)
            # 学生网络输入
            aug_image_l = self.image_preprocessing_l(aug_image_l)
            image_l = self.image_preprocessing_l(image)
            # 教师网络输入
            aug_image_rgb = self.image_preprocessing_rgb(aug_image_rgb)
            image_rgb = self.image_preprocessing_rgb(image_rgb)

            # 返回一个包含所有需要的数据的字典
            return {
                "img_origin_l": image_l,          # 原始灰度图 (学生网络)
                "img_origin_rgb": image_rgb,      # 原始RGB图 (教师网络)
                "img_aug_l": aug_image_l,         # 增强灰度图 (学生网络)
                "img_aug_rgb": aug_image_rgb,     # 增强RGB图 (教师网络)
                "mask": aug_mask                  # 增强图对应的分割掩码
            }
        else:
            # --- 2. 真实模式：加载真实掩码并统一返回格式 ---
            # 加载与图像对应的真实掩码
            dir_path, file_name = os.path.split(self.rod_paths[index])
            # 尝试标准结构: rod_dir/labels/file.png
            # 如果 rod_paths 来自 rod_dir/images/file.png，则替换 images 为 labels
            mask_path = os.path.join(dir_path.replace("images", "labels"), file_name)
            
            # 兼容性检查：如果替换后不存在，尝试回退到同级目录查找 labels (针对非标准结构)
            if not os.path.exists(mask_path):
                # 尝试在上级目录找 labels
                mask_path = os.path.join(os.path.dirname(dir_path), "labels", file_name)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
            else:
                # 如果找不到掩码，直接报错
                raise FileNotFoundError(f"找不到对应的掩码文件: {mask_path}，对应的图像为: {self.rod_paths[index]}")

            # 调整掩码尺寸 (使用最近邻插值保持类别ID)
            mask = mask.resize(self.resize_shape, Image.NEAREST)

            # --- 数据增强 (旋转) ---
            # 注意：虽然构造函数中已强制关闭了真实数据的旋转，但保留此代码逻辑以防将来策略变更，
            # 或者如果用户显式希望在真实数据上做增强（需修改构造函数逻辑）。
            # 目前 self.rotate_90 和 self.random_rotate 在 use_real_data=True 时已被置为 False/0。
            if self.rotate_90 or self.random_rotate > 0:
                degree = 0
                if self.rotate_90:
                    degree += np.random.choice(np.array([0, 90, 180, 270]))
                if self.random_rotate > 0:
                    degree += np.random.uniform(-self.random_rotate, self.random_rotate)
                
                if degree != 0:
                    fill_color = 0
                    image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
                    mask = mask.rotate(degree, fillcolor=0, resample=Image.NEAREST)

            mask = self.mask_preprocessing(mask)

            # --- 3. 生成灰度图和RGB图对 ---
            # 在真实模式下，img_aug 和 img_origin 都是加载的真实图像
            image_l_tensor = self.image_preprocessing_l(image)
            image_rgb_tensor = self.image_preprocessing_rgb(image.convert("RGB"))

            # --- 4. 统一返回字典格式 ---
            return {
                "img_origin_l": image_l_tensor,
                "img_origin_rgb": image_rgb_tensor,
                "img_aug_l": image_l_tensor.clone(), # 使用clone避免潜在的引用问题
                "img_aug_rgb": image_rgb_tensor.clone(),
                "mask": mask
            }


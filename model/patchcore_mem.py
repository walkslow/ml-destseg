import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import os
import glob
from typing import Union, List
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MemoryBankSourceDataset(Dataset):
    """
    专门用于构建 PatchCore 记忆库的数据集。
    仅加载原始正常图像，不进行合成缺陷，仅进行基础预处理（Resize, Normalize）。
    """
    def __init__(self, img_dir, resize_shape, mean_l, std_l, mean_rgb, std_rgb, rotate_90=False, random_rotate=0):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        # 兼容 images 子目录结构
        if len(self.img_paths) == 0:
            self.img_paths = sorted(glob.glob(os.path.join(img_dir, "images", "*.png")))
        
        if len(self.img_paths) == 0:
            raise ValueError(f"在目录 {img_dir} 中未找到 PNG 图像")

        self.resize_shape = resize_shape
        self.rotate_90 = rotate_90
        self.random_rotate = random_rotate
        
        self.transform_l = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_l, std_l)
        ])
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_rgb, std_rgb)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 统一以灰度模式加载
        image = Image.open(self.img_paths[idx]).convert("L")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        # 旋转增强 (如果启用)
        fill_color = 0
        if self.rotate_90:
            degree = int(torch.randint(0, 4, (1,)).item() * 90) # 0, 90, 180, 270
            if degree != 0:
                image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
        if self.random_rotate > 0:
            degree = float(torch.empty(1).uniform_(-self.random_rotate, self.random_rotate).item())
            image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)

        # 生成 RGB 版本
        image_rgb = image.convert("RGB")
        
        # 转换和归一化
        img_l = self.transform_l(image)
        img_rgb = self.transform_rgb(image_rgb)
        
        # 返回字典以匹配 train 循环的某些预期 (虽然这里我们主要直接取 tensor)
        # 注意：这里 img_aug 和 img_origin 相同，因为没有合成缺陷
        return {
            "img_origin_l": img_l,
            "img_origin_rgb": img_rgb,
            "img_aug_l": img_l.clone(),
            "img_aug_rgb": img_rgb.clone(),
            # mask 不需要，因为我们不计算损失，只提取特征
        }


class PatchMaker:
    """
    PatchMaker 类
    功能：负责处理特征图，将其转换为 PatchCore 所需的 Patch 特征向量。
    主要工作包括：
    1. 特征图尺寸对齐（上采样）。
    2. 特征图拼接。
    3. 特征展平为 (N, D) 形式的向量。
    """
    def __init__(self, patchsize=3, stride=1):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: List[torch.Tensor], return_spatial_info=False):
        """
        将多层特征图列表转换为 Patch 特征向量。
        
        Args:
            features (List[torch.Tensor]): 来自 TeacherNet 的特征图列表，例如 [x1, x2, x3]。
                                           x1: (B, C1, H1, W1), x2: (B, C2, H2, W2)...
            return_spatial_info (bool): 是否返回空间维度信息。
            
        Returns:
            torch.Tensor: 拼接并展平后的特征向量，形状为 (Total_Patches, Total_Channels)。
        """
        # 以第一层特征图的尺寸作为参考尺寸（通常是分辨率最高的）
        # 注意：PatchCore 原始实现中通常对所有特征图进行 unfold 操作来提取局部 patch，
        # 但在简化版或特定实现中，常直接使用 1x1 patch (即像素点特征) 并通过上采样对齐。
        # 这里我们采用简单的上采样对齐策略，这与 DeSTSeg 中特征对齐的逻辑一致。
        
        # 1. 确定目标分辨率（使用最大的分辨率，通常是第一个特征图的 H, W）
        ref_feature = features[0]
        target_h, target_w = ref_feature.shape[2], ref_feature.shape[3]
        
        processed_features = []
        
        for i, feature in enumerate(features):
            # 将所有特征图上采样到目标分辨率
            # feature: (B, C, H, W)
            if feature.shape[2:] != (target_h, target_w):
                feature = F.interpolate(
                    feature, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            processed_features.append(feature)
            
        # 2. 在通道维度进行拼接
        # concatenated_features: (B, C_total, H, W)
        concatenated_features = torch.cat(processed_features, dim=1)
        
        # 3. 展平为 (B * H * W, C_total)
        # 调整维度顺序为 (B, H, W, C) 然后 reshape
        # 这样每一行代表一个位置的所有层级特征组合
        b, c, h, w = concatenated_features.shape
        reshaped_features = concatenated_features.permute(0, 2, 3, 1).reshape(-1, c)
        
        if return_spatial_info:
            return reshaped_features, (b, h, w)
        
        return reshaped_features


class CoresetSampler:
    """
    CoresetSampler 类
    功能：实现贪婪核心集采样 (Greedy Coreset Sampling)，用于减少记忆库的大小。
    策略：迭代选择与当前已选集合距离最远的点，以最大化覆盖特征空间。
    为了加速，通常使用随机投影降维后再计算距离。
    """
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """
        Args:
            percentage (float): 采样比例 (0 < percentage <= 1)。例如 0.1 表示保留 10% 的特征。
            device (torch.device): 计算设备。
            dimension_to_project_features_to (int): 为了加速距离计算，先将特征投影到的维度。
        """
        self.percentage = percentage
        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _compute_batchwise_differences(
        self, matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两个矩阵之间的成对欧氏距离，支持分批计算以避免 OOM。
        a^2 - 2ab + b^2
        """
        # matrix_a: (M, D) - 可能是巨大的 CPU Tensor
        # matrix_b: (N, D) - 通常较小 (batch 或 单个点)，在 GPU 上
        
        # 确保 matrix_b 在 GPU 上
        matrix_b = matrix_b.to(self.device)
        
        distances_list = []
        batch_size = 2048 # 根据显存大小调整
        
        # 分批处理 matrix_a
        for i in range(0, matrix_a.shape[0], batch_size):
            # 取出一个 batch，移到 GPU
            a_batch = matrix_a[i : i + batch_size].to(self.device)
            
            # 计算 a^2: (Batch, 1)
            a_times_a = a_batch.pow(2).sum(1, keepdim=True)
            # 计算 b^2: (1, N)
            b_times_b = matrix_b.pow(2).sum(1, keepdim=True).t()
            # 计算 2ab: (Batch, N)
            a_times_b = torch.mm(a_batch, matrix_b.t())
            
            # dist = a^2 + b^2 - 2ab
            dist_batch = a_times_a + b_times_b - 2 * a_times_b
            dist_batch = dist_batch.clamp(min=1e-6).sqrt()
            
            distances_list.append(dist_batch)
            
        return torch.cat(distances_list, dim=0)

    def run(self, features: torch.Tensor) -> torch.Tensor:
        """
        执行采样。
        
        Args:
            features (torch.Tensor): 输入特征库，形状 (N, D)。
            
        Returns:
            torch.Tensor: 采样后的特征库，形状 (N * percentage, D)。
        """
        if self.percentage >= 1:
            return features
            
        # 1. 随机投影降维 (Johnson-Lindenstrauss lemma)
        # 如果特征维度很高，计算距离会很慢，投影到低维空间可以近似保持距离关系
        with torch.no_grad():
            if features.shape[1] > self.dimension_to_project_features_to:
                mapper = torch.nn.Linear(
                    features.shape[1], 
                    self.dimension_to_project_features_to, 
                    bias=False
                ).to(self.device)
                
                # 分批投影，避免 OOM
                reduced_features_list = []
                batch_size = 10000
                for i in range(0, features.shape[0], batch_size):
                    batch = features[i : i + batch_size].to(self.device)
                    reduced_batch = mapper(batch)
                    reduced_features_list.append(reduced_batch.cpu()) # 先放回 CPU 拼接
                
                reduced_features = torch.cat(reduced_features_list, dim=0)
            else:
                reduced_features = features
        
        # 尝试将 reduced_features 移动到 GPU 以加速 greedy selection
        # 128 维特征通常较小，例如 100k 数据仅占 ~50MB，1M 数据 ~500MB
        try:
            reduced_features = reduced_features.to(self.device)
        except RuntimeError:
            print("显存不足，无法将所有投影特征移至 GPU，采样速度可能受限。")
            # 可以在这里实现 CPU fallback 或随机降采样
        
        # 2. 贪婪选择
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        
        return features[sample_indices]

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """
        运行近似贪婪核心集选择。
        优化：预计算范数，全 GPU 计算。
        """
        number_of_starting_points = 10
        target_size = int(len(features) * self.percentage)
        
        # 随机选择一些起始点
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()
        
        # 预计算所有点的范数平方: ||x||^2
        # features: (N, D) -> (N, 1)
        # 如果 features 很大，可能在 CPU 上
        features_sq = features.pow(2).sum(dim=1, keepdim=True)
        
        # 初始化 min_distances
        # 确保 min_distances 与 features 在同一设备 (可能是 CPU)
        # 如果 features 在 CPU，我们希望 min_distances 也在 GPU 上以加速 min 操作？
        # 不，如果 features 在 CPU，说明显存放不下，我们必须分批计算距离
        
        if features.is_cuda:
            # === 全 GPU 模式 (快速) ===
            min_distances = torch.full((len(features), 1), float('inf'), device=features.device)
            
            for start_idx in start_points:
                y = features[start_idx : start_idx+1]
                y_sq = features_sq[start_idx : start_idx+1]
                term_xy = torch.mm(features, y.t())
                dist = features_sq + y_sq - 2 * term_xy
                dist = dist.clamp(min=0).sqrt()
                min_distances = torch.minimum(min_distances, dist)
                
            selected_indices = []
            for _ in tqdm.tqdm(range(target_size), desc="构建记忆库采样中..."):
                new_idx = torch.argmax(min_distances).item()
                selected_indices.append(new_idx)
                
                y = features[new_idx : new_idx+1]
                y_sq = features_sq[new_idx : new_idx+1]
                term_xy = torch.mm(features, y.t())
                new_distance_sq = features_sq + y_sq - 2 * term_xy
                new_distance = new_distance_sq.clamp(min=0).sqrt()
                min_distances = torch.minimum(min_distances, new_distance)
                
        else:
            # === CPU-GPU 混合模式 (省显存，稍慢但可行) ===
            # features 在 CPU, features_sq 在 CPU
            # min_distances 放 GPU 以加速 argmin 和 update
            min_distances = torch.full((len(features), 1), float('inf'), device=self.device)
            
            # 定义分批计算距离的辅助函数
            def compute_dist_batch(target_idx):
                y = features[target_idx : target_idx+1].to(self.device) # (1, D)
                y_sq = features_sq[target_idx : target_idx+1].to(self.device)
                
                batch_size = 32768 # 较大 batch 以利用 GPU
                for i in range(0, len(features), batch_size):
                    # 取 batch 送入 GPU
                    batch_feat = features[i : i + batch_size].to(self.device)
                    batch_sq = features_sq[i : i + batch_size].to(self.device)
                    
                    term_xy = torch.mm(batch_feat, y.t())
                    dist = batch_sq + y_sq - 2 * term_xy
                    dist = dist.clamp(min=0).sqrt()
                    
                    # 更新 GPU 上的 min_distances
                    min_distances[i : i + batch_size] = torch.minimum(
                        min_distances[i : i + batch_size], dist
                    )

            # 初始化起始点距离
            for start_idx in start_points:
                compute_dist_batch(start_idx)
                
            selected_indices = []
            for _ in tqdm.tqdm(range(target_size), desc="构建记忆库采样中(混合模式)..."):
                new_idx = torch.argmax(min_distances).item()
                selected_indices.append(new_idx)
                compute_dist_batch(new_idx)
                
        return np.array(selected_indices)


class MemoryBank:
    """
    MemoryBank 类
    功能：管理 PatchCore 的记忆库。
    1. 存储正常样本的特征。
    2. 提供最近邻搜索功能，用于计算异常得分。
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_bank = None # 存储核心特征 (M, D)
        
    def fit(self, features: torch.Tensor, sampling_ratio: float = 0.01):
        """
        构建记忆库。
        
        Args:
            features (torch.Tensor): 所有的训练特征 (N, D)。
            sampling_ratio (float): 采样比例。
        """
        # 针对超大规模特征集的优化：如果特征数量过多，先进行随机下采样
        # 3000万特征太慢了，通常 PatchCore 只需要 10万-50万特征就能很好地工作
        MAX_INITIAL_FEATURES = 500000 
        
        if features.shape[0] > MAX_INITIAL_FEATURES:
            print(f"警告：特征数量 ({features.shape[0]}) 巨大，为保证速度，将随机下采样至 {MAX_INITIAL_FEATURES}...")
            indices = torch.randperm(features.shape[0])[:MAX_INITIAL_FEATURES]
            features = features[indices]
            # 重新计算 sampling_ratio 以保持最终记忆库大小大致不变？
            # 不，通常 sampling_ratio 是针对采样后的集合。
            # 或者我们希望最终保留 N * ratio 个点。
            # 如果原始 N=30M, ratio=0.01 -> target=300k.
            # 现在 N'=500k, ratio=0.01 -> target=5k. 这可能太少了。
            # 我们应该调整 ratio，使得最终数量接近原始目标，或者只是接受较小的记忆库。
            # 这里的策略是：保持 sampling_ratio 不变，因为 500k 的 10% (50k) 通常足够了。
            # 如果用户坚持要 300k 点，那速度必然慢。
            # 让我们稍微调大一点 ratio，确保记忆库不会太小
            # 比如保证至少有 10000 个点，或者 min(50000, N')
            
            # 简单策略：仅截断输入，ratio 不变。这样记忆库变小，但速度快。
            # 对于 30M 数据，通常意味着有很多冗余。
        
        print(f"原始特征数量: {features.shape[0]}, 正在进行核心集采样 (比例: {sampling_ratio})...")
        sampler = CoresetSampler(sampling_ratio, self.device)
        self.memory_bank = sampler.run(features)
        print(f"记忆库构建完成，最终特征数量: {self.memory_bank.shape[0]}")
        
    def save(self, path: str):
        """
        保存记忆库到文件。
        """
        if self.memory_bank is None:
            print("警告：记忆库为空，无法保存。")
            return
        torch.save(self.memory_bank, path)
        print(f"记忆库已保存至 {path}")

    def load(self, path: str):
        """
        从文件加载记忆库。
        """
        if not os.path.exists(path):
            print(f"警告：记忆库文件 {path} 不存在。")
            return
        self.memory_bank = torch.load(path, map_location=self.device)
        print(f"记忆库已从 {path} 加载，特征数量: {self.memory_bank.shape[0]}")

    def predict(self, features: torch.Tensor, n_neighbors: int = 1):
        """
        计算特征到记忆库的最近邻距离。
        
        Args:
            features (torch.Tensor): 测试特征 (N_test, D)。
            n_neighbors (int): 考虑最近的 K 个邻居。
            
        Returns:
            distances (torch.Tensor): 最近邻距离 (N_test, )。
        """
        if self.memory_bank is None:
            raise ValueError("Memory bank 未初始化，请先调用 fit()")
            
        # 确保在同一设备
        features = features.to(self.device)
        self.memory_bank = self.memory_bank.to(self.device)
        
        # 由于显存限制，如果测试特征太多，可能需要分批计算
        # 这里为了简单，先尝试直接计算，如果后续OOM再优化分批
        
        # 计算距离矩阵 (N_test, M_bank)
        # 注意：这里我们使用更高效的方式，避免生成巨大的距离矩阵
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        
        # 使用 PyTorch 的 cdist 计算成对距离，它底层有优化
        # features: (N, D), bank: (M, D) -> dist: (N, M)
        # 如果 N 和 M 很大，cdist 依然会消耗大量显存。
        # 对于实际部署，建议分批处理 query features
        
        batch_size = 1024 # 每次处理 1024 个 query patch
        distances_list = []
        
        with torch.no_grad():
            for i in range(0, features.shape[0], batch_size):
                batch_features = features[i : i + batch_size]
                
                # 计算 batch 到 memory bank 的距离
                # p=2 表示欧氏距离
                dist_batch = torch.cdist(batch_features, self.memory_bank, p=2)
                
                # 获取最近邻距离 (values, indices)
                # topk 如果 n_neighbors=1 就是 min
                # largest=False 表示取最小的距离
                topk_values, _ = torch.topk(dist_batch, k=n_neighbors, dim=1, largest=False)
                
                # 如果 k > 1，通常取平均距离作为分数，这里简单起见，如果 k=1 直接返回
                if n_neighbors == 1:
                    distances_list.append(topk_values.squeeze(1))
                else:
                    distances_list.append(topk_values.mean(dim=1))
        
        return torch.cat(distances_list, dim=0)


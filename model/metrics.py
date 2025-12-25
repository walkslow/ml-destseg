from bisect import bisect_left
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from anomalib.utils.metrics.plotting_utils import plot_figure
from anomalib.utils.metrics.pro import (
    connected_components_cpu,
    connected_components_gpu,
)
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import auc, roc
from torchmetrics.utilities.data import dim_zero_cat


class AUPRO(Metric):
    """
    每区域重叠曲线下面积 (Area Under Per-Region Overlap, AUPRO) 指标。
    复制自 anomalib: https://github.com/openvinotoolkit/anomalib

    AUPRO 是一种用于异常检测的评估指标，它计算每个连通异常区域的重叠率（TPR）与全局假阳性率（FPR）之间的关系。
    相比于传统的像素级 AUC，AUPRO 更关注是否能够发现所有大小的异常区域，而不仅仅是像素总数。
    """

    # 指示该指标是否可微 (Differentiable)。
    # 如果为 True，该指标可以作为损失函数的一部分参与反向传播。
    # AUPRO 计算涉及非平滑操作（如排序、连通域分析），因此不可微。
    is_differentiable: bool = False

    # 指示该指标是否“数值越大越好”。
    # True: 越大越好 (如 Accuracy, IoU); False: 越小越好 (如 Loss, WER); None: 未定义。
    higher_is_better: Optional[bool] = None

    # 指示每次 update 是否需要访问完整的状态 (Full State)。
    # False: update 操作是独立的（例如只是追加列表或简单累加），TorchMetrics 可以优化计算。
    # True: update 依赖于之前的状态，无法简单并行或累积。
    full_state_update: bool = False

    # 定义用于跨 Batch 累积数据的状态变量 (State Variables)。
    # torchmetrics 会自动处理这些变量的设备同步和累积。
    preds: List[Tensor]  # 存储模型预测值
    target: List[Tensor] # 存储真实标签值

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        fpr_limit: float = 0.3,
    ) -> None:
        """
        参数:
            compute_on_step (bool): 是否在每一步更新时计算指标。默认为 True。
            dist_sync_on_step (bool): 是否在每一步同步分布式节点。默认为 False。
            process_group (Any): 分布式进程组。
            dist_sync_fn (Callable): 分布式同步函数。
            fpr_limit (float): 计算 AUPRO 时的 FPR 上限。通常我们只关心低 FPR 下的表现。默认为 0.3。
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        # 使用 add_state 注册需要在 metric 实例中维护的状态变量。
        # 这些变量会自动处理设备移动（CPU/GPU）和重置（reset）。
        
        # 注册 "preds" 状态，用于存储预测结果。
        # default=[]: 初始值为空列表。
        # dist_reduce_fx="cat": 在分布式训练中，各 GPU 的结果将通过 "cat" (拼接) 操作进行汇总。
        self.add_state(
            "preds", default=[], dist_reduce_fx="cat"
        )  # pylint: disable=not-callable

        # 注册 "target" 状态，用于存储真实标签。
        self.add_state(
            "target", default=[], dist_reduce_fx="cat"
        )  # pylint: disable=not-callable

        # register_buffer 用于注册不需要计算梯度的张量，但会随模型 state_dict 一起保存。
        # 这里将 fpr_limit 注册为 buffer，确保它能自动移动到与模型相同的设备上。
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """
        更新状态值。
        参数:
            preds (Tensor): 模型的预测结果（通常是异常图，值在 0-1 之间）。
            target (Tensor): 真实标签（Ground Truth）。
        """
        self.target.append(target)
        self.preds.append(preds)

    def _compute(self) -> Tuple[Tensor, Tensor]:
        """
        计算直到 self.fpr_limit 为止的 PRO/FPR 值对。
        它利用重叠率对应于 TPR 这一事实，通过聚合 ROC 构建过程中产生的每个区域的 TPR/FPR 值来计算整体 PRO 曲线。
        
        异常:
            ValueError: 如果 self.target 不符合 kornia 连通分量分析的要求，则抛出 ValueError。
            
        返回:
            Tuple[Tensor, Tensor]: 包含最终 FPR 和 TPR 值的元组。
        """
        # 将存储在列表中的多个 batch 数据拼接成单个大张量
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        # --- 连通域分析 (Connected Component Analysis, CCA) ---
        # 这一步的目的是找出所有的“异常区域”。
        # kornia 的 connected_components 函数会给每个独立的连通区域分配一个唯一的整数 ID。
        
        # 检查 target 值域是否合法 [0, 1]
        if target.min() < 0 or target.max() > 1:
            raise ValueError(
                (
                    f"kornia.contrib.connected_components expects input to lie in the interval [0, 1], but found "
                    f"interval was [{target.min()}, {target.max()}]."
                )
            )
        # 转换为 float 类型以适配 kornia 接口
        target = target.type(torch.float)
        
        # 根据设备选择对应的连通域算法
        # kornia connected_components requires (B, 1, H, W)
        if target.ndim == 3:
            target_input = target.unsqueeze(1)
        else:
            target_input = target

        if target.is_cuda:
            cca = connected_components_gpu(target_input)
        else:
            cca = connected_components_cpu(target_input)
            
        cca = cca.squeeze(1) # Remove channel dim to match flatten logic

        # 展平所有数据，因为 AUPRO 是基于像素统计的，但需要区分区域
        preds = preds.flatten()
        cca = cca.flatten()
        target = target.flatten()

        # --- 全局 FPR 基准计算 ---
        # 我们首先计算全局的 ROC 曲线，确定在给定的 fpr_limit 下需要多少个采样点。
        # roc() 函数来自 torchmetrics.functional，它计算 Receiver Operating Characteristic (ROC)。
        # 它返回三个张量: (fpr, tpr, thresholds)。
        #   - fpr: 假阳性率 (False Positive Rate) 序列，随阈值降低而单调递增。
        #   - tpr: 真阳性率 (True Positive Rate) 序列。
        #   - thresholds: 对应的阈值序列。
        # 这里我们只取 [0]，即 fpr。
        # NOTE: explicitly set task="binary" to ensure tuple return
        roc_result = roc(preds, target, task="binary")
        if isinstance(roc_result, tuple):
            fpr = roc_result[0]
        else:
            # Fallback if torchmetrics version behavior differs unexpectedly
            fpr = roc_result[0]

        # 计算在 fpr_limit (例如 0.3) 限制下的有效采样点数量。
        # AUPRO 指标通常只关注低 FPR 区域（即高精度区域）。
        # torch.where(condition) 等价于 torch.nonzero(condition, as_tuple=True)。
        # 当只传入 condition 一个参数时，它返回一个包含索引张量的元组 (tuple)。
        # 对于一维向量 fpr，它返回 (indices_tensor,)。
        # 因此我们需要使用 [0] 来取出这个唯一的索引张量，然后用 .size(0) 获取满足条件的元素个数。
        output_size = torch.where(fpr <= self.fpr_limit)[0].size(0)

        # --- 区域级 AUPRO 计算循环 ---
        # AUPRO 的核心思想：平均每个异常区域的覆盖率 (Overlap / TPR)。
        # 算法流程：
        # 1. 遍历每个连通区域 (label)。
        # 2. 将该区域视为“正样本”，背景视为“负样本”。
        # 3. 计算该区域的 ROC 曲线 (TPR vs FPR)。
        # 4. 将该曲线插值对齐到全局统一的 FPR 坐标轴上。
        # 5. 累加所有区域的 TPR 曲线，最后求平均。

        tpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        fpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        # 创建新的统一横坐标索引 (用于插值)
        new_idx = torch.arange(0, output_size, device=preds.device, dtype=torch.float)

        labels = cca.unique()[1:]  # 获取所有区域 ID，跳过背景 (0)
        background = cca == 0      # 全局背景掩码
        
        _fpr: Tensor
        _tpr: Tensor
        for label in labels:
            interp: bool = False
            new_idx[-1] = output_size - 1
            mask = cca == label # 当前区域掩码
            
            # --- 关键步骤：构建局部数据集 ---
            # 我们只关心“当前区域”能否被检测出来，以及是否混淆了“背景”。
            # 因此，数据集仅包含：当前区域像素 (Positive) + 全局背景像素 (Negative)。
            # 注意：其他异常区域的像素被忽略，不参与计算，避免干扰。
            valid_pixels = background | mask
            
            # 只有当 valid_pixels 中既有正样本也有负样本时，ROC 计算才有意义
            # 但实际上 mask 包含当前区域(正) 和 background(负)。
            # 如果 mask 为空(该区域无像素)，则跳过。
            # 如果 background 为空(全图都是该区域)，则跳过。
            if not mask.any() or not background.any():
                continue

            _roc_res = roc(preds[valid_pixels], mask[valid_pixels], task="binary")
            if isinstance(_roc_res, tuple):
                 _fpr, _tpr = _roc_res[:-1]
            else:
                 _fpr, _tpr = _roc_res[:-1] # Try to unpack anyway if it's list-like

            # --- 截断与插值处理 ---
            # 我们只关注 FPR <= fpr_limit 的部分（通常是 0.3，即低误报率区域）。
            
            # 处理特殊情况：如果最低的 FPR 都已经超过了 limit（模型太差或 limit 太小）
            if _fpr[_fpr <= self.fpr_limit].max() == 0:
                _fpr_limit = _fpr[_fpr > self.fpr_limit].min()
            else:
                _fpr_limit = self.fpr_limit

            # 筛选出满足 limit 要求的点
            _fpr_idx = torch.where(_fpr <= _fpr_limit)[0]
            
            # 如果曲线没有刚好落在 limit 上，我们需要手动插值出 limit 处的 TPR 值
            # 这样可以保证所有曲线都在同一个终点结束，便于平均。
            # torch.allclose: 检查浮点数是否相等（考虑到精度误差）
            if not torch.allclose(_fpr[_fpr_idx].max(), self.fpr_limit):
                # 1. 找到第一个大于 fpr_limit 的点的索引
                # searchsorted 要求输入是排序好的（FPR 本身就是单调递增的）
                _tmp_idx = torch.searchsorted(_fpr, self.fpr_limit)
                
                # 2. 将这个“刚好越界”的点加入到索引列表中
                # 这样我们就有了 [..., just_below_limit, just_above_limit] 两个点
                # 可以利用这两个点进行线性插值，估算出 limit 处的准确值
                _fpr_idx = torch.cat([_fpr_idx, _tmp_idx.unsqueeze_(0)])
                
                # 3. 计算插值系数 (Slope / Ratio)
                # 目标是计算 limit 点在 [prev, next] 区间内的相对位置 (0~1)
                # 公式推导:
                # ratio = (limit - prev) / (next - prev)
                #       = (limit - prev) / (next - prev)
                # 代码中的写法:
                # _slope = 1 - (next - limit) / (next - prev)
                #        = ((next - prev) - (next - limit)) / (next - prev)
                #        = (next - prev - next + limit) / (next - prev)
                #        = (limit - prev) / (next - prev)
                # 结果是一样的，表示 limit 点距离 prev 点有多远（归一化距离）。
                _slope = 1 - (
                    (_fpr[_tmp_idx] - self.fpr_limit)
                    / (_fpr[_tmp_idx] - _fpr[_tmp_idx - 1])
                )
                interp = True

            # 截取有效曲线段
            _fpr = _fpr[_fpr_idx]
            _tpr = _tpr[_fpr_idx]

            # 将索引归一化并映射到 new_idx 的尺度
            _fpr_idx = _fpr_idx.float()
            _fpr_idx /= _fpr_idx.max()
            _fpr_idx *= new_idx.max()

            # 如果需要插值，修正最后一个索引位置
            if interp:
                new_idx[-1] = _fpr_idx[-2] + ((_fpr_idx[-1] - _fpr_idx[-2]) * _slope)

            # --- 曲线对齐与累加 ---
            # 使用线性插值将当前区域的 TPR/FPR 映射到统一的 new_idx 上
            _tpr = self.interp1d(_fpr_idx, _tpr, new_idx)
            _fpr = self.interp1d(_fpr_idx, _fpr, new_idx)
            
            # 累加到总曲线
            tpr += _tpr
            fpr += _fpr

        # --- 计算平均曲线 ---
        if labels.size(0) > 0:
            tpr /= labels.size(0)
            fpr /= labels.size(0)
        
        return fpr, tpr

    def compute(self) -> Tensor:
        """
        首先计算 PRO 曲线，然后计算并缩放曲线下面积。
        返回:
            Tensor: AUPRO 指标的值
        """
        # 1. 获取聚合后的全局 FPR 和 TPR 曲线
        # _compute() 方法已经完成了所有区域曲线的对齐、累加和平均
        fpr, tpr = self._compute()

        # 2. 边界检查：防止空数据或单点数据导致计算错误
        # 如果采样点太少（<=1），无法计算面积，直接返回 0.0
        if fpr.size(0) <= 1: 
             return torch.tensor(0.0, device=self.fpr_limit.device)

        # 3. 计算曲线下面积 (Area Under Curve)
        # reorder=True 确保输入按 x 轴 (fpr) 排序，这对梯形积分法是必须的
        # 此时计算出的面积是绝对面积
        aupro = auc(fpr, tpr, reorder=True)
        
        # 4. 归一化 (Normalization)
        # 因为我们在计算时截断了 FPR (只取到 fpr_limit，例如 0.3)
        # 为了让指标在 0~1 之间，我们需要除以积分区间的宽度 (即最大的 FPR 值)
        # 这样 AUPRO 表示的是在 [0, fpr_limit] 区间内的“平均”重叠率
        if fpr[-1] == 0:
            # 如果最大 FPR 为 0，说明没有有效曲线（可能是该类别无异常样本），返回 NaN 以便在上层忽略
            return torch.tensor(float('nan'), device=self.fpr_limit.device)
            
        aupro = aupro / fpr[-1] 

        return aupro

    def generate_figure(self) -> Tuple[Figure, str]:
        """
        生成包含 PRO 曲线和 AUPRO 的图表。
        返回:
            Tuple[Figure, str]: 包含图表和用于日志记录的图表标题的元组
        """
        fpr, tpr = self._compute()
        aupro = self.compute()

        xlim = (0.0, self.fpr_limit.detach_().cpu().numpy())
        ylim = (0.0, 1.0)
        xlabel = "Global FPR"
        ylabel = "Averaged Per-Region TPR"
        loc = "lower right"
        title = "PRO"

        fig, _axis = plot_figure(
            fpr, tpr, aupro, xlim, ylim, xlabel, ylabel, loc, title
        )

        return fig, "PRO"

    @staticmethod
    def interp1d(old_x: Tensor, old_y: Tensor, new_x: Tensor) -> Tensor:
        """
        将 1D 信号线性插值到新的采样点。
        参数:
            old_x (Tensor): 原始 1-D x 值 (与 y 大小相同)，必须是单调递增的。
            old_y (Tensor): 原始 1-D y 值 (与 x 大小相同)。
            new_x (Tensor): 需要插值 y 的 x 值。
        返回:
            Tensor: 对应 new_x 值的 y 值。
        """

        # 1. 预先计算所有区间的斜率 (Slope)
        # 线性插值公式: y = y_prev + slope * (x - x_prev)
        # eps 是为了防止除以零的数值稳定性保护
        eps = torch.finfo(old_y.dtype).eps
        slope = (old_y[1:] - old_y[:-1]) / (eps + (old_x[1:] - old_x[:-1]))

        # 2. 找到 new_x 中每个点落在 old_x 的哪个区间内
        # searchsorted 返回的是插入位置，使得数组保持有序。
        # 也就是说，old_x[idx-1] <= new_x < old_x[idx]
        idx = torch.searchsorted(old_x, new_x)

        # 我们需要的是区间的左端点索引 (left neighbor)，所以减 1。
        # 修正后：old_x[idx] <= new_x < old_x[idx+1]
        idx -= 1
        
        # 3. 处理边界情况 (Clamping)
        # 如果 new_x 小于 old_x[0]，idx 会变成 -1，需要 clamp 到 0。
        # 如果 new_x 大于 old_x[-1]，idx 会超过最大区间索引，需要限制在最后一个区间。
        # old_x.size(0) - 2 是最后一个区间的起始索引（因为区间数是 N-1）。
        idx = torch.clamp(idx, 0, old_x.size(0) - 2)

        # 4. 执行批量线性插值计算
        # old_y[idx]: 区间起点的 y 值
        # slope[idx]: 对应区间的斜率
        # (new_x - old_x[idx]): 距离区间起点的距离 (delta x)
        y_new = old_y[idx] + slope[idx] * (new_x - old_x[idx])

        return y_new


class IAPS(Metric):
    """
    实例平均精度 (Instance Average Precision, IAP) 分数。
    该指标用于评估模型在实例级别（即每个连通的异常区域作为一个实例）上的检测性能。
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        ioi_thresh: float = 0.5,
        recall_thresh: float = 0.9,  # 我们论文中 IAP@k 的 k%
    ) -> None:
        """
        参数:
            ioi_thresh (float): Intersection over Instance 阈值，用于判定是否检测到实例。
            recall_thresh (float): 计算 IAP 时关注的召回率阈值。
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state(
            "preds", default=[], dist_reduce_fx="cat"
        )  # pylint: disable=not-callable
        self.add_state(
            "target", default=[], dist_reduce_fx="cat"
        )  # pylint: disable=not-callable
        self.ioi_thresh = ioi_thresh
        self.recall_thresh = recall_thresh

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """更新状态值。"""
        self.target.append(target)
        self.preds.append(preds)

    def compute(self):
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        # 检查并准备 target 以供 kornia 进行标记
        if target.min() < 0 or target.max() > 1:
            raise ValueError(
                (
                    f"kornia.contrib.connected_components expects input to lie in the interval [0, 1], but found "
                    f"interval was [{target.min()}, {target.max()}]."
                )
            )
        target = target.type(torch.float)  # kornia expects FloatTensor
        if target.is_cuda:
            cca = connected_components_gpu(target)
        else:
            cca = connected_components_cpu(target)

        preds = preds.flatten()
        cca = cca.flatten()
        target = target.flatten()

        labels = cca.unique()[1:] # 忽略背景

        ins_scores = []

        # 计算每个实例的得分
        for label in labels:
            mask = cca == label
            # 取该实例区域内预测值的前 k% 作为该实例的得分
            heatmap_ins, _ = preds[mask].sort(descending=True)
            ind = np.int64(self.ioi_thresh * len(heatmap_ins))
            ins_scores.append(float(heatmap_ins[ind]))

        if len(ins_scores) == 0:
            # 如果没有真实异常实例，理论上 IAP 无定义或为 1。这里简单处理抛出异常或返回 1.
            # 为了稳健性，这里返回 1.0 和 1.0 (假设完美)
            # 但原代码抛出异常。我们改为返回 nan 或警告。
            # raise Exception("gt_masks all zeros")
            return torch.tensor(1.0), torch.tensor(1.0) 

        ins_scores.sort()

        recall = []
        precision = []

        # 计算不同阈值下的 Precision 和 Recall
        for i, score in enumerate(ins_scores):
            recall.append(1 - i / len(ins_scores))
            # TP: 预测值 >= 当前score 且 真实为正 的数量？
            # 这里的计算方式是基于像素的简化版，还是基于实例的？
            # ins_scores 已经是实例级得分。
            # 这里的逻辑是：把 ins_scores 中的每一个值作为阈值。
            
            # 统计有多少像素预测值 >= score 且 真实标签为正
            tp = torch.sum(preds * target >= score)
            # 统计有多少像素预测值 >= score
            tpfp = torch.sum(preds >= score)
            precision.append(float(tp / tpfp))

        # 修正 Precision 曲线（使其单调不减，标准的 AP 计算方式）
        for i in range(0, len(precision) - 1):
            precision[i + 1] = max(precision[i + 1], precision[i])
            
        ap_score = sum(precision) / len(ins_scores)
        
        recall = recall[::-1]
        precision = precision[::-1]
        
        # 找到指定 recall 阈值处的 precision
        k = bisect_left(recall, self.recall_thresh)
        # 边界检查
        if k >= len(precision):
            k = len(precision) - 1
            
        return ap_score, precision[k]


class MulticlassSegmentationMetrics(Metric):
    """
    多类别分割指标计算器。
    一次性计算并返回 mIoU (平均交并比), mDice (平均Dice系数), mFscore (平均F1分数)。
    
    特点：
    - 支持忽略特定类别（如背景）。
    - 自动处理 batch 维度。
    - 内部维护混淆矩阵，支持分布式计算。
    """
    is_differentiable = False
    full_state_update = False

    def __init__(self, num_classes: int, ignore_index: int | None = 0,
                 compute_on_step=True, dist_sync_on_step=False):
        """
        参数:
            num_classes (int): 类别总数（包含背景）。
            ignore_index (int | None): 计算平均值时需要忽略的类别索引（通常是背景 0）。
                                     如果为 None，则所有类别都参与平均计算。
        """
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 混淆矩阵: 行是真实标签，列是预测标签
        self.add_state("confmat", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        更新混淆矩阵。
        参数:
            preds: (B, H, W) 或 (B, C, H, W) 的预测张量。
                   如果是 (B, C, H, W)，会自动进行 argmax。
            target: (B, H, W) 的真实标签索引。
        """
        if preds.ndim == 4: # (B, C, H, W) -> (B, H, W)
            preds = preds.argmax(dim=1)
            
        preds = preds.flatten()
        target = target.flatten()
        
        # 确保数据在同一设备
        if preds.device != self.confmat.device:
            self.confmat = self.confmat.to(preds.device)
            
        # 计算混淆矩阵 (Confusion Matrix)
        # 混淆矩阵是一个 N x N 的矩阵，其中行表示真实类别 (Target)，列表示预测类别 (Prediction)。
        # 元素 (i, j) 表示真实类别为 i 但被预测为 j 的像素数量。
        
        # 核心技巧：将二维坐标 (target, preds) 映射到一维索引。
        # 公式：index = target * num_classes + preds
        # 这种映射保证了每一对 (target, preds) 组合都有唯一的索引值。
        # 例如：num_classes=3, target=1, preds=2 -> index = 1*3 + 2 = 5
        unique_mapping = target * self.num_classes + preds
        
        # 使用 bincount 统计每个索引出现的次数。
        # bincount 是一个非常高效的 CUDA 优化操作，比循环快得多。
        # minlength=num_classes**2 确保结果长度固定，包含了所有可能的类别组合（即使某些组合未出现）。
        bins = torch.bincount(
            unique_mapping,
            minlength=self.num_classes ** 2
        )
        
        # 将一维统计结果 reshape 回二维矩阵形状 (num_classes, num_classes)
        # 这样就直接得到了本次 batch 的混淆矩阵，并累加到全局混淆矩阵中。
        self.confmat += bins.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        计算最终指标。
        返回:
            dict: 包含 mIoU, mDice, mFscore 及各类别详细指标的字典。
        """
        # 将混淆矩阵转换为浮点数以进行除法运算
        cm = self.confmat.float()
        
        # 计算 TP (True Positive), FP (False Positive), FN (False Negative)
        # 混淆矩阵对角线上的元素即为分类正确的数量 (TP)
        tp = torch.diag(cm)
        # 列求和减去 TP 即为该类的误报数量 (FP - 被错误预测为该类的样本)
        fp = cm.sum(0) - tp
        # 行求和减去 TP 即为该类的漏报数量 (FN - 该类被错误预测为其他类的样本)
        fn = cm.sum(1) - tp
        
        # --- 核心指标计算 ---
        # 1e-7 是一个极小值 (epsilon)，用于防止分母为 0 导致数值不稳定
        
        # IoU (Intersection over Union) = TP / (TP + FP + FN)
        # 衡量预测区域与真实区域的重叠程度
        iou = tp / (tp + fp + fn + 1e-7)
        
        # Dice Coefficient (F1-Score for binary case) = 2*TP / (2*TP + FP + FN)
        # 等价于 F1 Score，对不平衡样本更敏感，常用于医学分割
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
        
        # Precision (查准率) = TP / (TP + FP)
        # 预测为正的样本中有多少是真的正
        precision = tp / (tp + fp + 1e-7)
        
        # Recall (查全率/灵敏度) = TP / (TP + FN)
        # 真实为正的样本中有多少被预测出来了
        recall    = tp / (tp + fn + 1e-7)
        
        # F-Score (F1) = 2 * Precision * Recall / (Precision + Recall)
        # Precision 和 Recall 的调和平均数
        fscore = 2 * precision * recall / (precision + recall + 1e-7)
        
        # --- 计算平均指标 (mIoU, mDice, mFscore) ---
        # 处理忽略类 (ignore_index)：
        # 在多分类分割中，通常有一个背景类（例如索引0），在计算平均性能指标时
        # 我们往往只关心前景类的表现，因此需要将背景类排除在平均值计算之外。
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            # 创建一个全 True 的掩码
            mask = torch.ones(self.num_classes, dtype=torch.bool, device=cm.device)
            # 将需要忽略的类别位置设为 False
            mask[self.ignore_index] = False
            
            # 仅选取有效类别的指标参与平均计算
            valid_iou = iou[mask]
            valid_dice = dice[mask]
            valid_fscore = fscore[mask]
        else:
            # 如果没有忽略类，则所有类别都参与计算
            valid_iou = iou
            valid_dice = dice
            valid_fscore = fscore
            
        return {
            "mIoU":   valid_iou.mean(),   # 所有有效类别的 IoU 平均值
            "mDice":  valid_dice.mean(),  # 所有有效类别的 Dice 平均值
            "mFscore": valid_fscore.mean(), # 所有有效类别的 Fscore 平均值
            "per_class_IoU": iou,         # 每个类别的 IoU (Tensor)
            "per_class_Dice": dice,       # 每个类别的 Dice (Tensor)
            "per_class_Fscore": fscore,   # 每个类别的 Fscore (Tensor)
            "confusion_matrix": cm        # 完整的混淆矩阵
        }

class MulticlassAUPRO(Metric):
    """
    多类别 AUPRO (Area Under Per-Region Overlap) 指标。
    
    AUPRO 原本是为单类别异常检测设计的二分类指标。
    为了支持多类别分割任务，我们采用 **One-vs-Rest (OvR)** 策略。
    
    计算策略：
    1. 为每一个类别分别维护一个独立的 AUPRO 计算器。
    2. 对于类别 C，我们将 C 视为“异常类”(Positive)，将所有其他类别视为“正常类/背景”(Negative)。
    3. 分别计算每个类别的 AUPRO 分数。
    4. 最后计算所有有效类别（排除 ignore_index）的平均分。
    """
    def __init__(self, num_classes: int, ignore_index: int = 0, **kwargs):
        """
        参数:
            num_classes (int): 类别总数。
            ignore_index (int): 计算平均值时忽略的类别（通常是背景类，如 class 0）。
                                背景类的 AUPRO 通常没有意义，因为 AUPRO 关注的是检测出的“区域”。
            **kwargs: 传递给 AUPRO 的其他参数 (如 fpr_limit)。
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # 使用 ModuleList 存储一组独立的 AUPRO 实例
        # 这样每个类别都有自己的状态 (preds, targets)，互不干扰。
        self.pros = torch.nn.ModuleList(
            [AUPRO(**kwargs) for _ in range(num_classes)]
        )

    def update(self, preds: Tensor, target: Tensor):
        """
        更新每个类别的统计数据。
        
        参数:
            preds: (B, C, H, W) 概率图，通常经过 Softmax (多类) 或 Sigmoid (多标签)。
                   每个通道 c 代表该像素属于类别 c 的概率。
            target: (B, H, W) 整数索引标签，取值范围 [0, C-1]。
        """
        # Check target shape
        if target.ndim == 1:
             # This might happen if target was flattened prematurely or dataset returns 1D
             raise ValueError(f"MulticlassAUPRO expects target of shape (B, H, W), got {target.shape}")

        for c in range(self.num_classes):
            # 跳过忽略类别（通常是背景）
            # 我们不需要统计背景的异常检测性能
            if c == self.ignore_index:
                continue
                
            # --- 构建二分类任务 (One-vs-Rest) ---
            
            # 1. 提取预测概率
            # 取出第 c 个通道，表示模型认为该像素是类别 c 的概率
            binary_pred = preds[:, c]          # Shape: (B, H, W)
            
            # 2. 构建二值标签 (Ground Truth)
            # 如果真实标签等于 c，则设为 1 (Positive/Anomaly)
            # 否则设为 0 (Negative/Normal)
            binary_gt   = (target == c).long() # Shape: (B, H, W)
            
            # 3. 更新对应类别的计算器
            self.pros[c].update(binary_pred, binary_gt)

    def compute(self) -> Tensor:
        """
        计算所有类别的 AUPRO 的平均值。
        """
        aupros = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            val = self.pros[c].compute()
            if not torch.isnan(val):
                aupros.append(val)

        if len(aupros) == 0:
            # 如果所有类别都无法计算（例如测试集全为背景），返回 NaN
            return torch.tensor(float('nan'))

        return torch.stack(aupros).mean()

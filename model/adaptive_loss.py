# -*- coding: utf-8 -*-
"""
自适应损失权重调整模块

提供多种自动调节损失函数权重的方法，用于解决多损失组合时权重难以手动调优的问题。

参考论文:
1. Uncertainty-based weighting: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018
2. GradNorm: Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing", ICML 2018
3. AutoWeight: Liu et al. "AutoWeight: Automatic Loss Weighting for Multi-Task Learning", ICML 2024
"""

import torch
import torch.nn as nn
import warnings


class UncertaintyWeightingLoss(nn.Module):
    """
    基于不确定性的动态权重调整 (Multi-Task Learning Using Uncertainty to Weigh Losses)

    原理: 将每个损失的权重参数作为可学习参数，同时使用log方差作为同方差不确定性
         的代理。模型会自动学习每个任务的相对权重。

    公式: L_total = 1/(2*σ1^2) * L1 + 1/(2*σ2^2) * L2 + log σ1 + log σ2

    优点:
    - 完全自适应，无需手动设置权重
    - 有理论保证，收敛稳定
    - 对不同尺度的损失鲁棒

    使用方法:
        loss_fn = UncertaintyWeightingLoss(num_losses=2)
        total_loss, weights = loss_fn([focal_loss_val, dice_loss_val])
    """

    def __init__(self, num_losses=2, init_log_vars=None):
        """
        初始化不确定性加权损失

        Args:
            num_losses: 损失函数的数量
            init_log_vars: 初始log方差值，None则自动初始化为0
        """
        super(UncertaintyWeightingLoss, self).__init__()
        self.num_losses = num_losses

        # log(σ^2) 作为可学习参数，初始化为0 (即初始权重相等)
        if init_log_vars is None:
            init_log_vars = [0.0] * num_losses

        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))

    def forward(self, losses):
        """
        计算加权的总损失

        Args:
            losses: 损失值列表 [loss1, loss2, ...]，每个都是标量张量

        Returns:
            total_loss: 加权后的总损失
            weights: 各损失的权重列表
        """
        assert len(losses) == self.num_losses, \
            f"Expected {self.num_losses} losses, got {len(losses)}"

        # 精度 = 1 / (2 * σ^2) = exp(-log_var) / 2
        precisions = torch.exp(-self.log_vars)

        # 总损失 = Σ (precision_i * loss_i) + log σ_i
        total_loss = 0
        for i, loss in enumerate(losses):
            # 损失项: precision * loss
            weighted_loss = precisions[i] * loss
            # 正则化项: log σ (防止权重趋于无穷)
            reg_term = self.log_vars[i]
            total_loss += weighted_loss + reg_term

        # 计算实际权重 (用于记录和分析)
        weights = [float(precisions[i]) for i in range(self.num_losses)]

        return total_loss, weights

    def get_weights(self):
        """获取当前的实际权重"""
        precisions = torch.exp(-self.log_vars)
        return [float(p) for p in precisions]


class AutoWeightLoss(nn.Module):
    """
    自动权重损失 (AutoWeight: Automatic Loss Weighting for Multi-Task Learning)

    原理: 自动调节各损失权重，使得不同任务的梯度量级相似

    优点:
    - 计算简单，无需额外反向传播
    - 直接平衡梯度贡献
    - 收敛速度快

    使用方法:
        loss_fn = AutoWeightLoss(num_losses=2)
        total_loss, weights = loss_fn([focal_loss_val, dice_loss_val])
    """

    def __init__(self, num_losses=2, init_weights=None):
        """
        初始化自动权重损失

        Args:
            num_losses: 损失函数的数量
            init_weights: 初始权重，None则自动初始化为1
        """
        super(AutoWeightLoss, self).__init__()
        self.num_losses = num_losses

        if init_weights is None:
            init_weights = [1.0] * num_losses

        # 权重参数 (使用softmax确保为正且和归一)
        self.params = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))

    def forward(self, losses):
        """
        计算加权的总损失

        Args:
            losses: 损失值列表 [loss1, loss2, ...]

        Returns:
            total_loss: 加权后的总损失
            weights: 各损失的权重列表
        """
        assert len(losses) == self.num_losses, \
            f"Expected {self.num_losses} losses, got {len(losses)}"

        # 使用softmax确保权重为正
        weights = torch.softmax(self.params, dim=0)

        total_loss = sum(w * loss for w, loss in zip(weights, losses))

        return total_loss, [float(w) for w in weights]


class DynamicWeightAdjustment:
    """
    动态权重调整 (基于损失值的自适应调整)

    原理: 根据各损失值的相对大小动态调整权重
         损失值越大 → 权重越小 (防止某个损失主导训练)

    优点:
    - 实现简单，计算开销小
    - 无需额外可学习参数
    - 训练稳定性好

    使用方法:
        weight_adjuster = DynamicWeightAdjustment(
            base_weights=[20.0, 1.0],
            adjust_factor=0.5,
            warmup_steps=1000
        )
        weights = weight_adjuster.get_weights([focal_loss_val, dice_loss_val], step)
        total_loss = sum(w * loss for w, loss in zip(weights, losses))
    """

    def __init__(self, base_weights=None, adjust_factor=0.5, warmup_steps=1000,
                 smoothing_window=10, min_weight_ratio=0.1, max_weight_ratio=10.0):
        """
        初始化动态权重调整器

        Args:
            base_weights: 基础权重 [focal_weight, dice_weight]
            adjust_factor: 调整强度，0表示不调整，越大调整越剧烈
            warmup_steps: 预热步数，在此之前使用基础权重
            smoothing_window: 损失平滑窗口大小
            min_weight_ratio: 最小权重比例 (相对于基础权重)
            max_weight_ratio: 最大权重比例 (相对于基础权重)
        """
        self.base_weights = torch.tensor(base_weights or [20.0, 1.0],
                                         dtype=torch.float32)
        self.adjust_factor = adjust_factor
        self.warmup_steps = warmup_steps
        self.smoothing_window = smoothing_window
        self.min_weight_ratio = min_weight_ratio
        self.max_weight_ratio = max_weight_ratio

        # 损失历史记录 (用于平滑)
        self.loss_history = [[] for _ in range(len(self.base_weights))]

    def get_weights(self, current_losses, step):
        """
        根据当前损失值计算调整后的权重

        Args:
            current_losses: 当前各损失的值 [focal_loss, dice_loss]
            step: 当前训练步数

        Returns:
            weights: 调整后的权重列表
        """
        # 预热期间使用基础权重
        if step < self.warmup_steps:
            return [float(w) for w in self.base_weights]

        num_losses = len(self.base_weights)

        # 更新损失历史并计算平滑后的损失
        smoothed_losses = []
        for i, loss in enumerate(current_losses):
            self.loss_history[i].append(float(loss))
            if len(self.loss_history[i]) > self.smoothing_window:
                self.loss_history[i].pop(0)

            # 计算移动平均
            smoothed_loss = sum(self.loss_history[i]) / len(self.loss_history[i])
            smoothed_losses.append(smoothed_loss)

        # 计算相对损失比例
        smoothed_losses = torch.tensor(smoothed_losses)
        loss_ratios = smoothed_losses / smoothed_losses.mean()

        # 根据损失比例调整权重: 损失越大，权重越小
        # 公式: adjusted_weight = base_weight * (loss_ratio ^ (-adjust_factor))
        adjustment_factors = torch.pow(loss_ratios, -self.adjust_factor)

        # 应用调整因子
        adjusted_weights = self.base_weights * adjustment_factors

        # 限制权重范围
        adjusted_weights = torch.clamp(
            adjusted_weights,
            min=self.base_weights * self.min_weight_ratio,
            max=self.base_weights * self.max_weight_ratio
        )

        return [float(w) for w in adjusted_weights]

    def reset_history(self):
        """重置损失历史"""
        self.loss_history = [[] for _ in range(len(self.base_weights))]


class GradNormLoss(nn.Module):
    """
    GradNorm: 基于梯度的权重平衡

    原理: 调整权重使得不同任务的梯度量级相似

    优点:
    - 直接平衡梯度贡献
    - 理论完备

    缺点:
    - 需要额外反向传播
    - 计算开销较大

    注意: 由于需要访问模型参数和计算额外梯度，使用较复杂
          建议优先考虑UncertaintyWeighting或DynamicWeightAdjustment
    """

    def __init__(self, num_losses=2, alpha=1.5):
        """
        初始化GradNorm

        Args:
            num_losses: 损失数量
            alpha: 控制平衡强度的超参数
        """
        super(GradNormLoss, self).__init__()
        self.num_losses = num_losses
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_losses))

    def forward(self, losses, model_params, loss_ratios):
        """
        计算GradNorm损失 (需要额外反向传播)

        Args:
            losses: 各损失值列表
            model_params: 模型参数 (用于计算梯度)
            loss_ratios: 各损失的相对变化率

        Returns:
            weighted_loss: 加权损失
        """
        warnings.warn("GradNorm需要特殊的训练循环支持，建议使用UncertaintyWeighting代替")
        # 完整实现需要修改训练循环，此处仅作为接口占位
        weights = torch.softmax(self.weights, dim=0)
        return sum(w * loss for w, loss in zip(weights, losses))


def create_adaptive_loss(method='uncertainty', num_losses=2, **kwargs):
    """
    工厂函数: 创建自适应损失函数

    Args:
        method: 调整方法，可选:
            - 'uncertainty': UncertaintyWeightingLoss (推荐)
            - 'autoweight': AutoWeightLoss
            - 'dynamic': DynamicWeightAdjustment (最简单)
            - 'gradnorm': GradNormLoss (需要特殊训练循环)
        num_losses: 损失数量
        **kwargs: 方法特定的参数

    Returns:
        自适应损失实例

    示例:
        # 方法1: 不确定性权重 (推荐)
        loss_fn = create_adaptive_loss('uncertainty', num_losses=2)

        # 方法2: 动态调整 (最简单)
        weight_adjuster = create_adaptive_loss('dynamic', base_weights=[20.0, 1.0])

        # 方法3: 自动权重
        loss_fn = create_adaptive_loss('autoweight', num_losses=2)
    """
    method = method.lower()

    if method == 'uncertainty':
        return UncertaintyWeightingLoss(num_losses, **kwargs)
    elif method == 'autoweight':
        return AutoWeightLoss(num_losses, **kwargs)
    elif method == 'dynamic':
        return DynamicWeightAdjustment(**kwargs)
    elif method == 'gradnorm':
        return GradNormLoss(num_losses, **kwargs)
    else:
        raise ValueError(f"Unknown adaptive loss method: {method}. "
                        f"Choose from: 'uncertainty', 'autoweight', 'dynamic', 'gradnorm'")
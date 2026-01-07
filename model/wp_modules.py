import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Wavelet Pooling (小波池化) ---
def get_wav(in_channels, pool=True):
    """
    使用卷积操作实现小波分解 (Wavelet Decomposition)。
    基于 Haar 小波基。
    
    Args:
        in_channels: 输入通道数
        pool: 如果为True，使用Conv2d（下采样）；如果为False，使用ConvTranspose2d（上采样，未完全实现）。
    
    Returns:
        LL, LH, HL, HH: 四个频带的卷积层
    """
    # Haar 小波基定义
    # 低通算子：本质是求和（均值），当它与图像卷积时，会将相邻的两个像素相加，从而平滑图像，保留低频的轮廓信息
    # 1 / np.sqrt(2) 是一个归一化系数，保证变换前后信号的总能量不变 （即像素值的平方和不变）
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    # 高通算子：本质是求差（差异），当它与图像卷积时，会将相邻的两个像素相减
    # 如果两个像素值接近（平滑区域），结果趋近于0；如果像素值突变（边缘或细节），结果会很大，从而突出图像中的细节和边缘
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    # 构建 2D 滤波器，形状均为 (2, 2)，元素绝对值均为0.5
    # LL（低频）：全图平滑，元素均为0.5，[[0.5, 0.5], [0.5, 0.5]]
    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    # LH（垂直边缘，水平高频）：垂直方向求和，水平方向求差，[[-0.5, 0.5], [-0.5, 0.5]]
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    # HL（水平边缘，垂直高频）：水平方向求和，垂直方向求差，[[-0.5, -0.5], [0.5, 0.5]]
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    # HH（对角高频）：两个方向均求差，[[0.5, -0.5], [-0.5, 0.5]]
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    # .unsqueeze(0) : 增加“通道”维度
    # 卷积核的权重通常是 4D 的： [输出通道, 输入通道, 高, 宽] 。这里的 unsqueeze(0) 是为了先占住 输入通道 这个位置。
    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    # 创建分组卷积层，输入/输出通道对等，每个通道独立进行小波变换
    # kernel_size=2 : 对应 Haar 小波操作的 2x2 窗口。它每次观察相邻的 4 个像素（2x2 区域）来计算均值或差值
    # stride=2 : 步长为 2 意味着卷积核在移动时互不重叠
    # padding=0 : 无填充，特征图的宽度和高度都会缩减为原来的 1/2
    # bias=False：小波变换是一个纯线性的数学加减运算，必须关闭偏置
    # groups=in_channels：深度卷积配置，每个通道独立进行卷积，不共享卷积核
    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    # 冻结权重，因为小波变换是固定的数学运算，不需要训练
    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    # 加载 Haar 滤波器权重
    # 使用LL.weight.data而不是LL.weight，绕过梯度检查，确保了小波变换的权重在整个训练过程中始终保持不变，就像一个固定的物理算子
    # .expand(in_channels, -1, -1, -1)表示第0维维度扩展为in_channels，其他维度保持不变。结果张量形状为(in_channels, 1, 2, 2)
    # 分组卷积中每个卷积层的形状为(out_channels, in_channels / groups, kH, kW)，而这里in_channels=groups，故第1维为1
    # 这样， 输入特征图的每一个通道都会由一个独立的、但权重完全相同的 Haar 算子进行处理
    # expand 只是创建了一个虚拟的视图（View），并不实际分配内存
    # 使用 clone() 可以强制分配一块连续的内存空间，确保权重在模型运行和保存时不会出现引用冲突或内存对齐问题
    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH

class WavePool(nn.Module):
    """
    小波池化模块。
    将输入特征图分解为四个频带：LL (低频), LH (水平高频), HL (垂直高频), HH (对角高频)。
    输出尺寸为输入的 1/2。
    """
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

# --- Helper Modules (辅助模块) ---
class MSCM(nn.Module):
    """
    多尺度通道调制模块 (Multi-Scale Channel Modulation)。
    对应架构图 (c)。
    结合全局分支 (GAP + Linear + Linear) 和局部分支 (Linear + Linear)。
    """
    def __init__(self, d_model=64):
        super(MSCM, self).__init__()
        # 全局分支: GAP -> Linear -> Linear
        # 注意：Linear 作用于最后一个维度 (Channel)
        # WPFormer 源码中未使用 bottleneck 结构，而是保持维度 d_model
        self.global_branch = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        # 局部分支: Linear -> Linear
        self.local_branch = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, N, C]
        # 全局特征建模: 对空间维度 N 求平均 -> [B, 1, C]
        g_pool = torch.mean(x, dim=1, keepdim=True)
        g = self.global_branch(g_pool) # [B, 1, C]
        # 局部特征建模
        l = self.local_branch(x) # [B, N, C]
        # 融合与激活 (利用广播机制 [B, 1, C] + [B, N, C] -> [B, N, C])
        return self.sigmoid(g + l)

class convbnrelu(nn.Module):
    """
    基础卷积块：Conv2d + BatchNorm2d + ReLU
    """
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    """
    深度可分离卷积 (Depthwise Separable Conv)。
    用于在减少参数量的同时提取特征。
    """
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            # 第一步：Depthwise Convolution (深度卷积)
            # 核心参数 g=in_channel，意味着卷积组数等于输入通道数，每个输入通道由一个独立的 3x3 卷积核处理，通道之间互不通信
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            # 第二步：Pointwise Convolution (逐点卷积)
            # 核心参数 k=1 ：使用 1x1 卷积。提取通道特征，负责将通道数从 in_channel 变换到 out_channel 
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
        )

    def forward(self, x):
        return self.conv(x)

# --- D2T Attention (WCA + PCA) ---
class D2T_Attention(nn.Module):
    """
    双塔注意力模块 (Dual-Tower Attention, D2T)，结合了 WCA 和 PCA。
    
    设计理念：
    - Query (Qin): 教师网络特征。
    - Key/Value (Fi): 学生网络特征。
    
    结构对齐架构图:
    - WCA: 频域增强交叉注意力 (图 a)
    - PCA: 原型增强交叉注意力 (图 b)
    - D2T Decoder: (WCA + PCA) -> SA -> Qout
    """
    def __init__(self, d_model, h=8, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        
        # --- WCA (Wavelet-enhanced Cross-Attention) 组件 (图 a) ---
        self.pool = WavePool(d_model)
        self.mscm_wca = MSCM(d_model=d_model)
        self.wca_attn = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.wca_norm = nn.LayerNorm(d_model)
        
        # --- PCA (Prototype-based Cross-Attention) 组件 (图 b) ---
        self.proto_size = 16 # 原型数量 (K)
        self.plu_conv = DSConv3x3(d_model, d_model)
        self.plu_mheads = nn.Linear(d_model, self.proto_size, bias=False)
        self.pca_attn = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.mscm_pca = MSCM(d_model=d_model)
        self.pca_norm = nn.LayerNorm(d_model)
        
        # --- D2T Decoder 组件 (SA) ---
        self.sa_attn = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.sa_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, teacher, student):
        """
        前向传播。
        teacher: Qin [B, C, H, W]
        student: Fi [B, C, H, W]
        """
        B, C, H, W = teacher.shape
        q_in = teacher.flatten(2).transpose(1, 2) # [B, HW, C]
        
        # --- 1. WCA Branch (图 a) ---
        LL, LH, HL, HH = self.pool(student)
        f_fre_h = (HL + LH + HH).flatten(2).transpose(1, 2) # 高频 [B, HW/4, C]
        f_fre_l = LL.flatten(2).transpose(1, 2)             # 低频 [B, HW/4, C]
        
        # 频带调制: MSCM(H+L) * H + L
        wca_wei = self.mscm_wca(f_fre_h + f_fre_l)
        wca_kv = (wca_wei * f_fre_h) + f_fre_l
        
        # Cross Attention
        x_wca, _ = self.wca_attn(query=q_in, key=wca_kv, value=wca_kv)
        x_wca = self.wca_norm(x_wca + q_in) # 局部残差
        
        # --- 2. PCA Branch (图 b) ---
        # PLU: Prototype Learning Unit
        f_i_flat = student.flatten(2).transpose(1, 2) # [B, HW, C]
        f_i_conv = self.plu_conv(student).flatten(2).transpose(1, 2) # [B, HW, C]
        
        # 空间注意力权重 [B, HW, K]
        plu_weights = F.softmax(self.plu_mheads(f_i_conv), dim=1)
        # 聚合原型 [B, K, C]
        protos = plu_weights.transpose(1, 2) @ f_i_flat
        
        # Cross Attention
        # 原因：WPFormer 中 Query 数目(16)与 Protos(16) 相同，可直接相加。
        # 这里 Teacher 是空间特征 (HW=256)，Protos 是全局特征 (K=16)。
        # 必须通过 Cross Attention 将 Protos "对齐/广播" 到空间维度，才能进行后续的融合计算。
        aligned_protos, _ = self.pca_attn(query=q_in, key=protos, value=protos)
        
        # 后置调制 (MSCM)
        # 参考 WPFormer 逻辑: attn = mscw(protos + query), out = query * attn + query
        # 1. 计算调制权重：利用对齐后的 Protos 和 Teacher 特征融合
        pca_wei = self.mscm_pca(aligned_protos + q_in)
        
        # 2. 调制 (Modulation)：
        # WPFormer 中 PCA 分支是“通道调制”逻辑，Protos 仅用于生成门控权重，不直接注入特征值。
        # 因此这里使用权重调制 Teacher (q_in)，而不是 aligned_protos。
        x_pca_modulated = q_in * pca_wei
        
        # 3. 局部残差
        x_pca = self.pca_norm(x_pca_modulated + q_in)
        
        # --- 3. D2T Decoder Fusion ---
        # WCA + PCA
        x_sum = x_wca + x_pca
        
        # SA (Self-Attention)
        x_sa, _ = self.sa_attn(query=x_sum, key=x_sum, value=x_sum)
        x_sa = self.sa_norm(x_sa + x_sum) # 残差
        
        # Final residual with Qin
        x_out = self.final_norm(x_sa + q_in)
        
        # Restore shape
        return x_out.transpose(1, 2).view(B, C, H, W)

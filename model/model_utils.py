from typing import Callable, Optional
# Callable:用于类型提示，表示一个可调用对象（函数、方法、lambda表达式等）。用法：Callable[[参数类型], 返回值类型]
# Optional:用于类型提示，表示一个值可以是指定类型或 None。用法：Optional[类型]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # 3x3卷积层，默认padding=1，stride=1，故输入输出的空间尺寸不变
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation, # 确保输出尺寸不变
        groups=groups, # 控制分组卷积的组数
        bias=False, # 设为 False，因为后续有归一化层
        dilation=dilation, # 控制空洞卷积的扩张率
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    # 1x1卷积层，默认padding=0，stride=1，故输入输出的尺寸不变
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def make_layer(block, inplanes, planes, blocks, stride=1, norm_layer=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    # 参数block在实际应用中为BasicBlock，用于构建ResNet的基本块
    # 当inplanes 等于 planes * block.expansion且stride 为 1 时，残差主路径不改变输入的维度和尺寸，故shortcut无需下采样
    downsample = None
    # 反之，shortcut需要下采样
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            # conv1x1保证了输入输出的通道数一致，这里的stride可能不为1
            # tips:当stride不为1时，残差块会改变输入的尺寸，但实际上在构建StudentNet时，stride都为1，特征图尺寸是通过双线性插值来改变的
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    # 第一个block需要下采样，传入stride和downsample，确保shortcut的输出与残差主路径的输出能够相加
    layers.append(block(inplanes, planes, stride, downsample, norm_layer=norm_layer))
    # 改变输入通道数，后续的block的shortcut无需下采样，默认stride为1
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, norm_layer=norm_layer))

    return nn.Sequential(*layers)


def l2_normalize(input, dim=1, eps=1e-12):
    '''
    L2范数归一化操作,将输入张量在指定维度上进行归一化处理.
    将输入张量的L2范数归一化为1,此时张量的长度信息被去除,只剩下方向信息.
    两个L2归一化张量的点积相对于余弦相似度.
    input: 输入张量,需要进行归一化的数据;
    dim: 归一化的维度,默认为1(通道维度);
    eps: 防止除零的小常数,默认为1e-12;
    '''
    denom = torch.sqrt(torch.sum(input**2, dim=dim, keepdim=True))
    return input / (denom + eps)


class BasicBlock(nn.Module):
    expansion: int = 1 # 默认扩展因子为1，即输出通道数out_planes与基准通道数planes相同

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # 第一个conv3x3传入了stride参数，是因为make_layer的stride参数可能不为1，此时shortcut中的conv1x1即downsample改变了输入的尺寸
        # 故第一个conv3x3接收相同的stride参数，由于conv3x3的padding默认为1，故该conv3x3的输出与shortcut的输出尺寸相同
        # tips:把下面的planes换成planes * expansion更严谨
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # 第二个conv3x3不接收stride参数，使用默认stride为1，不改变输入的尺寸和维度
        # 因为第一个conv3x3和shortcut上的downsample已经将主路径和shortcut的输出尺寸和维度对齐了
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def get_norm_layer(norm: str):
    norm = {
        "BN": nn.BatchNorm2d,
        "LN": nn.LayerNorm,
    }[norm.upper()]
    return norm


def get_act_layer(act: str):
    act = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
    }[act.lower()]
    return act


# 将常用的 Conv+Norm+Act 组合封装成单一模块
class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        conv_kwargs=None,
        norm_layer=None,
        norm_kwargs=None,
        act_layer=None,
        act_kwargs=None,
    ):
        super(ConvNormAct2d, self).__init__()

        conv_kwargs = {}
        if norm_layer: # 当指定 norm_layer 时，卷积层不使用偏置
            conv_kwargs["bias"] = False
        # 为了在步长大于1的情况下，仍然保持输出尺寸与标准"same"填充的一致性，自动计算合适的填充大小
        if padding == "same" and stride > 1:
            # if kernel_size is even, -1 is must
            padding = (kernel_size - 1) // 2

        self.conv = self._build_conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_kwargs,
        )
        self.norm = None
        if norm_layer:
            norm_kwargs = {}
            self.norm = get_norm_layer(norm_layer)(
                num_features=out_channels, **norm_kwargs
            )
        self.act = None
        if act_layer:
            act_kwargs = {}
            self.act = get_act_layer(act_layer)(**act_kwargs)

    def _build_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        conv_kwargs,
    ):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **conv_kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ASPP(nn.Module):
    def __init__(self, input_channels, output_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), # 全局平均池化，输出形状为 [N, C, 1, 1]
                ConvNormAct2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    norm_layer="BN",
                    act_layer="RELU",
                ),
            )
        )
        # 根据扩张率的不同选择不同的卷积核大小和填充方式
        # 创建多个并行的 ConvNormAct2d 模块用于捕获不同尺度的信息
        for atrous_rate in atrous_rates:
            conv_norm_act = ConvNormAct2d # 通过变量赋值可以方便地替换基础模块类型
            modules.append(
                conv_norm_act(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=1 if atrous_rate == 1 else 3,
                    # 设置padding等于扩张率atrous_rate保证了输出的尺寸不变
                    padding=0 if atrous_rate == 1 else atrous_rate,
                    dilation=atrous_rate,
                    norm_layer="BN",
                    act_layer="RELU",
                )
            )

        # 特征提取层，ModuleList中一共有1 + len(atrous_rates)个模块
        self.aspp_feature_extractors = nn.ModuleList(modules)
        # 特征融合层，将ModuleList中所有模块的输出进行融合（需要先在通道维度上concat）
        self.aspp_fusion_layer = ConvNormAct2d(
            (1 + len(atrous_rates)) * output_channels,
            output_channels,
            kernel_size=3,
            norm_layer="BN",
            act_layer="RELU",
        )

    def forward(self, x):
        res = []
        for aspp_feature_extractor in self.aspp_feature_extractors:
            res.append(aspp_feature_extractor(x))
        # res[0]即ModuleList第一个模块的输出先经过了全局平均池化，所以尺寸改变了，需要上采样到与输入x相同的尺寸
        res[0] = F.interpolate(
            input=res[0], size=x.shape[2:], mode="bilinear", align_corners=False
        )  # resize back after global-avg-pooling layer
        # 各个并行分支的输出进行拼接，然后通过特征融合层进行融合降维
        res = torch.cat(res, dim=1)
        res = self.aspp_fusion_layer(res)
        return res

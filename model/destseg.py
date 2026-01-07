import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import ASPP, BasicBlock, l2_normalize, make_layer
from model.wp_modules import D2T_Attention


class TeacherNet(nn.Module):
    """
    教师网络（TeacherNet）
    - 功能：作为特征提取器，为学生网络提供稳定的、高质量的特征表示作为学习目标。
    - 结构：使用在ImageNet上预训练的ResNet-18模型作为编码器。
    - 输入：三通道RGB图像。
    - 输出：三个不同层级的特征图（来自ResNet的layer1, layer2, layer3）。
    - 特点：参数被冻结（`requires_grad=False`），在训练过程中不进行更新，确保其稳定性。
    """

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True, # 加载在ImageNet上预训练的权重
            features_only=True, # 仅提取中间特征，不进行最终分类
            out_indices=[1, 2, 3], # 输出第1、2、3层级的特征图，通道数分别为64、128、256，尺寸分别为原始输入的1/4, 1/8, 1/16
        )
        # 冻结整个教师网络的参数，使其在训练过程中不被更新
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.eval() # 将网络设置为评估模式，关闭dropout、batch normalization等训练时的行为
        x1, x2, x3 = self.encoder(x)
        return (x1, x2, x3)


class StudentNet(nn.Module):
    """
    学生网络（StudentNet）
    - 功能：学习模拟教师网络的特征表示，并用于最终的异常检测。
    - 结构：
        - 编码器（Encoder）：使用一个从零开始训练的ResNet-18，输入通道为1（灰度图）。
        - 解码器（Decoder）：可选的解码器部分（由`ed`标志控制），用于将编码器的高层特征上采样回
          与教师网络输出特征图相同尺寸的层级，但没有使用跳跃连接（skip connections）。
    - 输入：单通道灰度图像。
    - 输出：三个与教师网络输出层级对应的特征图。
    - 特点：参数是可训练的，其目标是使其输出在L2归一化后与教师网络的输出尽可能相似。
    """

    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed  # 控制是否存在解码器（Encoder-Decoder）结构
        if self.ed: # 默认存在解码器结构，学生网络的编-解码器是U-Net结构(但是没有skip connections)
            # BasicBlock是一个残差结构，当输入通道数和输出通道数不相同时，shortcut部分会进行downsample使之对齐
            # 当未指定make_layer的stride参数(默认为1)时，构建的layer是不改变输入的尺寸即宽高的
            # decoder_layer3、decoder_layer2、decoder_layer1的输出通道数分别为256、128、64，与教师网络的输出一一对应
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        # --- 权重初始化 ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming 正态分布初始化卷积核权重，适用于ReLU系列激活函数，有助于缓解梯度消失问题
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 将归一化层的权重(γ)初始化为1，偏置(β)初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # --- 编码器定义 ---
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,  # 不使用预训练权重，从随机初始化开始训练
            features_only=True,  # 仅返回中间特征图
            out_indices=[1, 2, 3, 4],  # 输出所有4个stage的特征图
            in_chans=1,  # 指定输入通道数为1，以适应灰度图
        )

    def forward(self, x):
        # 注意：没有self.eval()，因为StudentNet需要在训练和推理两种模式下工作。
        # 其模式由外部的`model.train()`或`model.eval()`控制。
        x1, x2, x3, x4 = self.encoder(x)

        if not self.ed:
            # 如果没有解码器，直接返回编码器的前三层输出
            return (x1, x2, x3)

        # --- 解码过程 ---
        # 从最深的特征图x4开始，通过解码层和上采样，逐级恢复特征图尺寸。
        b4 = self.decoder_layer4(x4)
        # 使用双线性插值将特征图上采样到目标尺寸（例如x3的尺寸）
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)


class SegmentationNet(nn.Module):
    """
    分割网络（SegmentationNet）
    - 功能：接收由学生和教师网络特征差异计算得出的融合特征，并预测最终的像素级分割掩码。
    - 结构：
        - 一个残差层（`res`）用于初步处理融合特征。
        - 一个分割头（`head`），包含ASPP（空洞空间金字塔池化）模块用于多尺度特征提取，
          后接卷积层，最终输出每个像素属于各个类别的logits。
    - 输入：融合后的多尺度特征张量（通道数为448 = 64+128+256）。
    - 输出：分割logits，形状为 (N, num_classes, H, W)。
    """

    def __init__(self, inplanes=448, num_classes=4):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        # --- 权重初始化 ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # --- 分割头定义 ---
        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),  # 空洞空间金字塔池化模块，用于多尺度特征提取
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),  # 1x1卷积，将通道数调整为类别数，得到分割logits
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        # 直接返回原始logits，损失函数（如CrossEntropyLoss）内部会执行softmax
        return x


class DeSTSeg(nn.Module):
    """
    DeSTSeg 主模型
    - 功能：整合教师网络、学生网络和分割网络，实现端到端的异常分割。
    - 核心思想：通过对比学生网络（在增强/原始灰度图上）和教师网络（在对应RGB图上）的特征差异，
      来识别异常区域。
    - `dest`标志：控制是否使用数据增强（cut_paste）后的图像进行学生-教师特征对比。
    - `ed`标志：传递给StudentNet，控制其是否包含解码器结构。
    """

    def __init__(self, dest=True, ed=True, num_classes=4, use_d2t=False):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_net = StudentNet(ed)
        self.dest = dest  # 控制是否使用数据增强策略的标志位，这是为了和一般的T-S网络（T和S的输入相同）进行区别

        self.use_d2t = use_d2t
        seg_inplanes = 448

        if self.use_d2t:
            # 如果启用 D2T，初始化 D2T_Attention 模块
            # 对应 TeacherNet/StudentNet 的三个输出尺度，通道数分别为 64, 128, 256
            self.d2t_modules = nn.ModuleList([
                D2T_Attention(d_model=64),
                D2T_Attention(d_model=128),
                D2T_Attention(d_model=256)
            ])
            # 启用 D2T 后，SegmentationNet 的输入通道数翻倍 (原始差异特征 + D2T增强特征)
            seg_inplanes *= 2

        self.segmentation_net = SegmentationNet(inplanes=seg_inplanes, num_classes=num_classes)

    def forward(self, img_aug_l, img_aug_rgb, img_origin_l=None, img_origin_rgb=None):
        self.teacher_net.eval()

        # --- 处理推理（inference）时输入不完整的情况 ---
        if img_origin_l is None:
            img_origin_l = img_aug_l.clone()
        if img_origin_rgb is None:
            img_origin_rgb = img_aug_rgb.clone()

        # --- 1. 计算用于分割网络输入的融合特征 ---
        # 教师网络处理增强后的RGB图像
        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug_rgb)
        ]
        # 学生网络处理增强后的灰度图像
        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug_l)
        ]

        # --- 特征融合策略 ---
        # 将教师和学生网络在不同尺度上的特征进行融合，作为分割网络的输入。
        fusion_features = []
        for i, (output_t, output_s) in enumerate(zip(outputs_teacher_aug, outputs_student_aug)):
            # 1. 原始差异特征计算
            # 按元素相乘并取负，作为特征差异的度量。点积越大（越相似），差异值越小。
            diff_feat = -output_t * output_s

            # 2. D2T 结构增强 (如果启用)
            if self.use_d2t:
                # D2T_Attention 输入: Query=Teacher (Actual), Key=Student (Normal)
                # 利用 Wavelet Pooling 和 Prototype Learning 增强特征表示
                d2t_feat = self.d2t_modules[i](teacher=output_t, student=output_s)
                # 将增强特征与原始差异特征拼接
                scale_feat = torch.cat([diff_feat, d2t_feat], dim=1)
            else:
                scale_feat = diff_feat

            # 3. 上采样对齐
            # 上采样到最大特征图的尺寸 (outputs_student_aug[0] 的尺寸)
            upsampled_feat = F.interpolate(
                scale_feat,
                size=outputs_student_aug[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            fusion_features.append(upsampled_feat)

        # 沿通道维度拼接，得到融合特征
        # 未启用 D2T: 64+128+256 = 448 通道
        # 启用 D2T: (64*2)+(128*2)+(256*2) = 896 通道
        output = torch.cat(fusion_features, dim=1)

        # 将融合特征输入分割网络，得到像素级分割结果
        output_segmentation = self.segmentation_net(output)

        # --- 2. 计算用于余弦相似度损失的异常图 ---
        # 根据`dest`标志，选择使用增强图还是原始图的学生网络输出来计算损失
        if self.dest:
            outputs_student = outputs_student_aug
        else:
            outputs_student = [
                l2_normalize(output_s) for output_s in self.student_net(img_origin_l)
            ]
        # 教师网络始终处理原始（未增强）的RGB图像作为基准
        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin_rgb)
        ]

        output_de_st_list = []
        # 逐尺度计算教师和学生网络特征之间的余弦距离，作为该尺度的异常图
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            # a_map = 1 - cos(theta) = 1 - (A·B / ||A||||B||)
            # 由于特征已经L2归一化，||A||=||B||=1，所以 a_map = 1 - A·B
            # 相似度越高，点积越大，a_map值越小（接近0）；差异越大，a_map值越大（可达2）。
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)

        # --- 融合多尺度异常图 ---
        # 将所有尺度的异常图上采样到相同尺寸
        output_de_st = torch.cat(
            [
                F.interpolate(
                    output_de_st_instance,
                    size=outputs_student[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_st_instance in output_de_st_list
            ],
            dim=1,
        )  # 形状: [N, 3, H, W]

        # 沿通道维度逐元素相乘，得到最终的综合异常图。
        # 乘法逻辑：只有在所有尺度上都表现出高异常分数（高余弦距离）的区域，
        # 才被认为是强异常，这有助于抑制噪声和假阳性。
        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True) # 形状: [N, 1, H, W]

        # --- 返回结果 ---
        # output_segmentation: 分割网络的原始logits输出 [N, num_classes, H, W]
        # output_de_st: 融合后的单通道综合异常图 [N, 1, H, W]
        # output_de_st_list: 融合前的多尺度异常图列表，每个元素为 [N, 1, H, W]
        return output_segmentation, output_de_st, output_de_st_list

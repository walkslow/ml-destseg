import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import ASPP, BasicBlock, l2_normalize, make_layer


class TeacherNet(nn.Module):
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
    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed
        if self.ed: # 默认存在解码器结构，学生网络的编-解码器是U-Net结构(但是没有skip connections)
            # BasicBlock是一个残差结构，当输入通道数和输出通道数不相同时，shortcut部分会进行downsample使之对齐
            # 当未指定make_layer的stride参数(默认为1)时，构建的layer是不改变输入的尺寸即宽高的
            # decoder_layer3、decoder_layer2、decoder_layer1的输出通道数分别为256、128、64，与教师网络的输出一一对应
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for m in self.modules(): # 遍历网络中的所有子模块（递归方式）
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming 正态分布初始化卷积核权重，适用于ReLU系列激活函数，有助于缓解梯度消失问题
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 将归一化层的权重(γ)初始化为1，偏置(β)初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False, # 不使用预训练权重，从随机初始化开始训练
            features_only=True, # 仅返回中间特征图，而不是最终分类结果
            out_indices=[1, 2, 3, 4],
        )

    def forward(self, x):
        # 注意：没有self.eval()，因为StudentNet需要在训练和推理两种模式下工作，而TeacherNet只需要在推理模式下工作
        # 通过 model.train() 和 model.eval() 在外部控制StudentNet的工作模式
        x1, x2, x3, x4 = self.encoder(x)
        if not self.ed: # 当没有解码器结构时，直接将编码器的1、2、3层输出作为学生网络的输出
            return (x1, x2, x3)
        x = x4
        # 对x4进行解码，依次还原到x3、x2、x1的尺寸和维度，其中维度在定义解码层时已经保证了输出与教师网络的输出相同
        # 而尺寸通过双线性插值进行上采样实现，x3.size()[2:]即为x3的尺寸，align_corners=False保持边缘区域的平滑过渡
        # 在这一步可以看出该解码器并没有使用编码器的输出，即没有skip connections，是一个普通的编-解码器结构
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)


class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448): # 64+128+256=448
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]), # 空洞空间金字塔池化模块，用于多尺度特征提取
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1), # 1×1卷积层，将通道数降为1用于二分类分割
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x


class DeSTSeg(nn.Module):
    def __init__(self, dest=True, ed=True):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_net = StudentNet(ed)
        self.dest = dest # 控制是否使用数据增强策略的标志位
        self.segmentation_net = SegmentationNet(inplanes=448)

    def forward(self, img_aug, img_origin=None):
        self.teacher_net.eval()

        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        # 提取教师网络对增强图像的特征输出并进行 L2 归一化，即归一化后的3个尺度上的特征图
        # detach()切断梯度传播，节省内存，防止梯度反向传播到教师网络
        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]
        # 提取学生网络对增强图像的特征输出并进行 L2 归一化，即归一化后的3个尺度上的特征图
        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug)
        ]
        output = torch.cat(
            [
                F.interpolate(
                    # L2归一化之后特征输出的点积相对于余弦相似度，点积越大越相似，故取负以表示差异值
                    -output_t * output_s, # 两者按尺度相乘并取负作为差异，拼接后喂入分割网络
                    size=outputs_student_aug[0].size()[2:], # 插值后的尺寸为最大的特征图的尺寸
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1, # 沿channel维度进行拼接，此时output的channel数为64+128+256=448
        )

        # 将融合后的特征输入 segmentation_net 得到分割结果，即单通道的异常分数图
        output_segmentation = self.segmentation_net(output)

        # 根据 dest 标志选择使用增强图像还是原始图像的学生网络输出
        if self.dest:
            outputs_student = outputs_student_aug
        else:
            outputs_student = [
                l2_normalize(output_s) for output_s in self.student_net(img_origin)
            ]
        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]

        output_de_st_list = []
        # 计算教师和学生网络特征的余弦距离，即1-学生特征与教师特征的点积，得到异常分数图
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            # a_map 是单通道的异常分数图，与之前-output_t * output_s的区别：
            # -output_t * output_s直接取负的特征乘积作为差异度量，用于生成分割网络的输入特征，更关注特征间的相对差异关系
            # a_map计算真正的余弦距离（1 - 余弦相似度），余弦相似度范围[−1, 1]，余弦距离范围[0, 2]。异常区域的特征差异更大，得到更高的分数
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)
        # 通过插值和累乘得到最终的差异图
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
        )  # [N, 3, H, W]
        # 为什么使用乘法而不是加法？
        # 每个尺度的a_map可以看作是该尺度下的异常概率，多尺度都判定为异常的区域才应该是真正的异常
        # 乘法对小值更加敏感，任何一个尺度的低异常分数都会显著降低最终结果，有助于减少误检，提高检测精度
        # 加法倾向于累积证据，容易产生过多的假阳性
        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)

        # output_segmentation 是通过 SegmentationNet 网络生成的最终分割结果，是单通道分割图 [N, 1, H, W]，用于提供精细的像素级异常分割掩码
        # output_de_st 是对学生和教师网络的多尺度余弦差异图进行插值和累乘得到的综合异常图，没有经过分割网络，也是单通道异常检测图 [N, 1, H, W]
        # output_de_st_list 是学生和教师网络的多尺度余弦差异图列表，其每个元素都是[N, 1, H, W]。它经过插值和累乘就得到output_de_st
        return output_segmentation, output_de_st, output_de_st_list

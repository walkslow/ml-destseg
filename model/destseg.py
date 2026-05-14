import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import ASPP, BasicBlock, l2_normalize, make_layer
from model.wp_modules import D2T_Attention
from model.patchcore_mem import PatchMaker


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

    def __init__(self, dest=True, ed=True, num_classes=4, use_d2t=False, use_patchcore=False, use_afs=False, afs_ratio=0.5, use_rrs=False, rrs_ratio=0.5):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_net = StudentNet(ed)
        self.dest = dest  # 控制是否使用数据增强策略的标志位，这是为了和一般的T-S网络（T和S的输入相同）进行区别

        self.use_d2t = use_d2t
        self.use_patchcore = use_patchcore
        self.use_afs = use_afs
        self.afs_ratio = afs_ratio
        self.use_rrs = use_rrs
        self.rrs_ratio = rrs_ratio
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

        if self.use_patchcore:
            # 如果启用 PatchCore，增加一个输入通道 (Anomaly Map)
            seg_inplanes += 1
            # 初始化 PatchMaker 用于特征对齐 (参数需与 train.py 中一致)
            self.patch_maker = PatchMaker(patchsize=3, stride=1)

        self.segmentation_net = SegmentationNet(inplanes=seg_inplanes, num_classes=num_classes)

        # --- AFS & RRS 特征选择初始化 ---
        # channel_mask: 用于 AFS (Anomaly Feature Selection) 的通道掩码
        # 这是一个 buffer (不会在反向传播中更新，但会随模型保存/加载)
        # 初始化为全 1，表示默认所有通道都被选中
        # 形状为 [1, C, 1, 1]，方便直接与特征图 (B, C, H, W) 进行广播乘法
        self.register_buffer("channel_mask", torch.ones(1, seg_inplanes, 1, 1))

    def run_afs(self, dataloader, device, ratio=0.5):
        """
        执行 AFS (Anomaly Feature Selection) 算法。
        
        功能：
            计算融合特征图与真实异常掩码之间的余弦相似度，筛选出对异常最敏感的特征通道。
            更新 `self.channel_mask`，将未选中的通道永久置零。
        
        执行时机：
            通常在 Phase 1 (学生网络训练) 结束后，Phase 2 (分割网络训练) 开始前调用一次。
            
        Args:
            dataloader: 包含合成异常数据的 DataLoader。
            device: 计算设备。
            ratio (float): 保留的特征通道比例 (0~1)。
        """
        if not self.use_afs:
            return

        print(f"--- Running AFS (Anomaly Feature Selection) with ratio {ratio} ---")
        self.eval()
        
        num_channels = self.channel_mask.shape[1]
        channel_scores = torch.zeros(num_channels, device=device)
        num_batches = 0
        
        # PatchCore 通道索引 (如果是最后一个通道)
        patchcore_idx = num_channels - 1 if self.use_patchcore else -1

        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(dataloader, desc="AFS Calculating"):
                # 获取数据
                img_aug_l = batch["img_aug_l"].to(device)
                img_aug_rgb = batch["img_aug_rgb"].to(device)
                img_origin_l = batch["img_origin_l"].to(device)
                img_origin_rgb = batch["img_origin_rgb"].to(device)
                mask = batch["mask"].to(device).float() # (B, H, W) -> 需要转为 float 用于计算
                
                # 获取融合特征 (此时尚未应用 AFS/RRS)
                # 我们需要临时 bypass forward 中的特征选择逻辑，或者手动提取特征
                # 这里我们复用 forward 的前半部分逻辑来提取特征
                # 为避免代码重复，我们假设 forward 中应用 mask 是在最后一步
                # 但实际上 forward 会应用 mask。
                # 策略：临时将 channel_mask 设为全 1 (虽然它本来就是全1，除非已经运行过 AFS)
                # 更好的方式：重构 forward 提取特征部分，或者在这里手动执行特征提取步骤
                
                # --- 手动执行特征提取 (复用 forward 逻辑) ---
                # 1. 教师/学生 输出
                teacher_features = self.teacher_net(img_aug_rgb)
                outputs_teacher = [l2_normalize(t.detach()) for t in teacher_features]
                outputs_student = [l2_normalize(s) for s in self.student_net(img_aug_l)]
                
                # 2. PatchCore Map (如果启用)
                patchcore_map_feat = None
                if self.use_patchcore:
                    # --- PatchCore 通道特殊处理说明 ---
                    # 目的：构造一个占位符张量，保持特征图的通道维度与模型定义一致，以便后续进行 cat 操作。
                    # 
                    # 1. 为什么是全0？
                    #    - 在 AFS 阶段，我们主要关注普通特征通道与异常掩码的相似度。
                    #    - 计算真实的 PatchCore 异常图需要调用 memory_bank.predict()，这是一个昂贵的最近邻搜索操作。
                    #    - 此时 memory_bank 可能尚未传入，或者为了节省 AFS 计算时间，我们跳过这一步。
                    #
                    # 2. 为什么这样做没问题？
                    #    - 全0的特征图与任何 Mask 的相似度计算结果都为 0。
                    #    - 理论上这会导致该通道被 AFS 剔除。
                    #    - 但是！我们在 run_afs 函数的后半部分有【PatchCore 保护机制】：
                    #      `channel_scores[patchcore_idx] = max_score + 1.0`
                    #    - 这行代码会强制将 PatchCore 通道的分数设为最高，确保它 100% 被选中保留。
                    #    - 因此，在这里计算它的真实分数是没有意义的，直接用全0占位既安全又高效。
                    target_size = outputs_student[0].size()[2:]
                    patchcore_map_feat = torch.zeros((img_aug_l.shape[0], 1, *target_size), device=device)

                # 3. 特征融合
                fusion_features = []
                for i, (output_t, output_s) in enumerate(zip(outputs_teacher, outputs_student)):
                    diff_feat = -output_t * output_s
                    if self.use_d2t:
                        d2t_feat = self.d2t_modules[i](teacher=output_t, student=output_s)
                        scale_feat = torch.cat([diff_feat, d2t_feat], dim=1)
                    else:
                        scale_feat = diff_feat
                    
                    upsampled_feat = F.interpolate(
                        scale_feat, size=outputs_student[0].size()[2:], mode="bilinear", align_corners=False
                    )
                    fusion_features.append(upsampled_feat)
                
                output = torch.cat(fusion_features, dim=1)
                if self.use_patchcore:
                    output = torch.cat([output, patchcore_map_feat], dim=1)
                
                # output: (B, C, H, W)
                
                # --- 计算相似度分数 ---
                # 将 Mask 下采样到特征图尺寸
                # mask: (B, H, W) -> (B, 1, H, W)
                mask_down = F.interpolate(mask.unsqueeze(1), size=output.shape[2:], mode="nearest")
                
                # 计算 Cosine Similarity
                # 将特征图和 Mask 展平：(B, C, H*W) vs (B, 1, H*W)
                b, c, h, w = output.shape
                feat_flat = output.view(b, c, -1)
                mask_flat = mask_down.view(b, 1, -1)
                
                # 归一化
                feat_norm = F.normalize(feat_flat, p=2, dim=2)
                mask_norm = F.normalize(mask_flat, p=2, dim=2)
                
                # 点积 (B, C)
                # sum(feat * mask, dim=2)
                scores = torch.sum(feat_norm * mask_norm, dim=2)
                
                # 累加 batch 平均分
                channel_scores += scores.mean(dim=0)
                num_batches += 1
        
        # 计算全局平均分
        channel_scores /= num_batches
        
        # --- PatchCore 保护 ---
        if self.use_patchcore:
            # 将 PatchCore 通道的分数设为最大值，确保它一定被选中
            # 或者在筛选后强制置 1。这里选择设为最大值参与 TopK 排序。
            max_score = channel_scores.max()
            channel_scores[patchcore_idx] = max_score + 1.0
            print(f"AFS: PatchCore channel index {patchcore_idx} is protected.")

        # --- 筛选 TopK ---
        num_keep = int(num_channels * ratio)
        # 确保至少保留 1 个
        num_keep = max(1, num_keep)
        
        # 获取 TopK 索引
        _, topk_indices = torch.topk(channel_scores, k=num_keep)
        
        # 更新 channel_mask
        self.channel_mask.zero_() # 先全 0
        self.channel_mask[0, topk_indices, 0, 0] = 1.0 # 选中的置 1
        
        print(f"AFS Completed: Selected {num_keep}/{num_channels} channels.")
        
    def apply_rrs(self, features, ratio=0.5):
        """
        应用 RRS (Reconstruction Residual Selection) 策略。
        
        功能：
            动态计算当前 batch 特征图的响应强度 (GMP + GAP)，筛选出高响应的通道。
            
        执行时机：
            在 forward 函数内部，Phase 2 及推理阶段每次调用。
            
        Args:
            features (torch.Tensor): 输入特征图 (B, C, H, W)。
            ratio (float): 保留比例。
            
        Returns:
            torch.Tensor: 经过 RRS 筛选后的特征图 (未选中通道被置零)。
        """
        # 使用 no_grad 确保 mask 生成过程不参与梯度计算
        with torch.no_grad():
            b, c, h, w = features.shape
            
            # 1. 计算响应分数 Score = GMP + GAP
            # GAP: (B, C)
            gap = F.adaptive_avg_pool2d(features, (1, 1)).view(b, c)
            # GMP: (B, C)
            gmp = F.adaptive_max_pool2d(features, (1, 1)).view(b, c)
            
            scores = gap + gmp # (B, C)
            
            # --- PatchCore 保护 ---
            if self.use_patchcore:
                patchcore_idx = c - 1
                # 设为极大值，确保被 TopK 选中
                # 注意：我们要对每个样本分别处理，这里利用广播机制
                # scores[:, patchcore_idx] = float('inf') # inf 可能会导致 NaN 问题，用最大值+1
                max_val = scores.max()
                scores[:, patchcore_idx] = max_val + 1.0
            
            # 2. 筛选 TopK
            num_keep = int(c * ratio)
            num_keep = max(1, num_keep)
            
            # 获取每个样本的 TopK 索引 (B, K)
            _, topk_indices = torch.topk(scores, k=num_keep, dim=1)
            
            # 3. 生成 Mask (B, C, 1, 1)
            # rrs_mask = torch.zeros_like(features) # (B, C, H, W) -> 显存消耗较大？
            # 优化：生成 (B, C) mask 然后 unsqueeze
            rrs_mask_bc = torch.zeros((b, c), device=features.device)
            # scatter_(dim, index, src)
            # 将 1.0 散布到 topk_indices 指定的位置
            rrs_mask_bc.scatter_(1, topk_indices, 1.0)
            
            rrs_mask = rrs_mask_bc.view(b, c, 1, 1)
            
        # 4. 应用 Mask
        # 注意：features * rrs_mask 会保留 features 的梯度，被 mask 掉的部分梯度变为 0
        return features * rrs_mask

    def forward(self, img_aug_l, img_aug_rgb, img_origin_l=None, img_origin_rgb=None, memory_bank=None):
        self.teacher_net.eval()

        # --- 处理推理（inference）时输入不完整的情况 ---
        if img_origin_l is None:
            img_origin_l = img_aug_l.clone()
        if img_origin_rgb is None:
            img_origin_rgb = img_aug_rgb.clone()

        # --- 1. 计算用于分割网络输入的融合特征 ---
        # 教师网络处理增强后的RGB图像
        # 获取原始特征用于 PatchCore (如果启用)
        teacher_features_aug = self.teacher_net(img_aug_rgb)
        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in teacher_features_aug
        ]
        # 学生网络处理增强后的灰度图像
        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug_l)
        ]

        # --- PatchCore Anomaly Map Calculation ---
        patchcore_map = None
        patchcore_features = None # Phase 1: 返回特征以供收集

        if self.use_patchcore:
            target_size = outputs_student_aug[0].size()[2:] # 目标对齐尺寸
            
            # 统一特征提取逻辑：无论 Phase 1 还是 Phase 2，都需要提取特征
            # features: (B*H*W, D), spatial_info: (B, H, W)
            features, (b, h, w) = self.patch_maker.patchify(teacher_features_aug, return_spatial_info=True)

            if memory_bank is not None:
                 # Phase 2 (or Inference): 记忆库已构建，计算异常图
                 
                 # 计算距离 (Anomaly Score)
                 # distances: (B*H*W, )
                 distances = memory_bank.predict(features)
                 
                 # Reshape 回空间尺寸 (B, 1, H, W)
                 patchcore_map = distances.reshape(b, 1, h, w)
                 
                 # 上采样对齐到最大特征图尺寸
                 patchcore_map = F.interpolate(
                    patchcore_map,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                 )
            else:
                 # Phase 1: 记忆库未构建，返回特征供收集
                 
                 # 生成占位符异常图 (B, 1, H, W)
                 patchcore_map = torch.zeros((b, 1, *target_size), device=img_aug_rgb.device)
                 
                 # 返回 detached 特征以节省显存 (用于构建记忆库)
                 patchcore_features = features.detach()

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
        
        if self.use_patchcore:
             # 拼接 PatchCore Anomaly Map
             output = torch.cat([output, patchcore_map], dim=1)

        # --- 特征选择策略 (AFS & RRS) ---
        # 仅在 Phase 2 (分割网络训练) 及推理阶段生效
        #
        # [生效机制说明]
        # 1. Phase 1 (Student Training):
        #    - AFS: 此时 self.channel_mask 初始化为全 1，output * mask 等于原样输出，不产生影响。
        #    - RRS: 如果开启，虽然会计算并筛选特征，但 Phase 1 的 Loss (loss_de_st) 仅依赖学生/教师网络的中间特征差异，
        #           完全不使用 segmentation_net 的输出。
        #           因此，即使 output 被 RRS 修改，也不会通过 segmentation_net 反向传播影响学生网络的训练。
        #           (仅浪费少量前向计算资源，逻辑上是安全的)
        # 2. Phase 2 (Segmentation Training):
        #    - AFS: 此时 run_afs() 已被调用，self.channel_mask 已更新（部分置 0），特征筛选生效。
        #    - RRS: 此时 loss_seg 直接依赖 segmentation_net 的输出，RRS 的筛选直接影响 Loss 和梯度。
        #
        # 此时 self.channel_mask 可能已经被 AFS 更新过 (如果在 train.py 中调用了 run_afs)
        
        # 1. 应用 AFS Mask (静态)
        if self.use_afs:
            output = output * self.channel_mask
            
        # 2. 应用 RRS Mask (动态)
        if self.use_rrs:
            # 动态应用 RRS (Reconstruction Residual Selection)
            # RRS 策略会计算当前 Batch 中每个样本的特征通道响应 (GMP+GAP)
            # 并保留响应最强的 Top-K 个通道，其余置零。
            
            # 尝试获取 rrs_ratio 参数
            # 由于 RRS 是在 forward 过程中动态调用的，我们需要知道保留比例 ratio
            # 为了保持 forward 接口签名的稳定性（不增加新参数），我们优先尝试从实例属性 self.rrs_ratio 中获取
            # 如果未设置该属性（例如旧代码加载模型），则使用默认值 0.5
            if hasattr(self, 'rrs_ratio'):
                output = self.apply_rrs(output, self.rrs_ratio)
            else:
                # 默认保留 50% 的通道
                # 注意：这只是一个兜底逻辑。在正常流程中，rrs_ratio 会在 __init__ 中被初始化。
                # 如果用户设置了 --rrs_ratio 0.8，那么 self.rrs_ratio 就是 0.8，这里就会执行上面的分支。
                output = self.apply_rrs(output, 0.5)

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
        
        return output_segmentation, output_de_st, output_de_st_list, patchcore_features

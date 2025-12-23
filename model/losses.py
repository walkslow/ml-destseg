import torch
import torch.nn.functional as F


def cosine_similarity_loss(output_de_st_list):
    '''
    计算余弦相似度损失
    :param output_de_st_list: 模型输出的多尺度学生-教师网络特征图余弦距离列表,每个元素的shape为(N, 1, H, W)
    :return: 余弦相似度损失
    '''
    loss = 0
    for instance in output_de_st_list:
        _, _, h, w = instance.shape
        loss += torch.sum(instance) / (h * w)
    return loss


def focal_loss(inputs, targets, gamma=2, reduction='mean'):
    '''
    计算多分类Focal Loss。Focal Loss旨在通过降低易分样本的权重，使模型更专注于难分的样本。
    :param inputs: 模型输出的原始logits, shape为(N, C, H, W), C为类别数
    :param targets: 真实标签(类别索引), shape为(N, H, W)
    :param gamma: 聚焦参数 (focusing parameter)。gamma > 0 会减小易分样本的损失贡献，值越大，效果越强。默认值为2。
    :param reduction: 损失归约方式,支持"mean"或"sum"
    :return: Focal Loss
    '''
    # 确保targets的数据类型为Long, 这是cross_entropy函数的要求
    targets = targets.long()
    # cross_entropy函数期望的targets形状为(N, H, W), 如果输入是(N, 1, H, W)，则移除通道维度
    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    # 1. 计算标准的交叉熵损失，但不进行归约(reduction='none')
    # ce_loss的每个元素是对应像素的交叉熵损失值, ce_loss = -log(p_t)
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')

    # 2. 计算p_t, 即模型预测为真实类别的概率
    # 由于 ce_loss = -log(p_t), 所以 p_t = exp(-ce_loss)
    p_t = torch.exp(-ce_loss)

    # 3. 计算Focal Loss的核心部分
    # (1-p_t)^gamma 是调节因子。对于易分样本(p_t -> 1), 调节因子趋近于0, 损失被抑制。
    # 对于难分样本(p_t -> 0), 调节因子趋近于1, 损失基本不受影响。
    loss = ((1 - p_t) ** gamma) * ce_loss

    # 4. 根据指定的reduction方式对损失进行归约
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def dice_loss(inputs, targets, smooth=1e-5):
    '''
    计算多分类Dice Loss。Dice Loss衡量的是预测与真实标签之间的重叠度，对类别不平衡问题鲁棒。
    :param inputs: 模型输出的原始logits, shape为(N, C, H, W)
    :param targets: 真实标签(类别索引), shape为(N, H, W)
    :param smooth: 平滑因子，防止计算过程中除以零，默认1e-5
    :return: Dice Loss
    '''
    # 1. 将模型的logits输出通过softmax转换为概率分布
    inputs = F.softmax(inputs, dim=1)

    # 2. 准备targets，确保其数据类型和形状符合要求
    targets = targets.long()
    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    # 3. 将类别索引的targets转换为one-hot编码，以便与概率图进行比较
    # 例如，类别索引2会转换为[0, 0, 1, 0]
    num_classes = inputs.shape[1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)

    # 4. 确保概率图和one-hot标签的形状完全一致
    assert inputs.shape == targets_one_hot.shape

    # 5. 计算交集(intersection)
    # 通过逐元素相乘并沿空间维度(H, W)求和，代表了批次中每个图像、每个类别的“正确预测概率总和”，即交集的大小
    intersection = torch.sum(inputs * targets_one_hot, dim=(2, 3))
    
    # 6. 计算并集(union)，在Dice Loss中通常指 |A| + |B|
    # 分别对概率图和one-hot标签沿空间维度求和
    union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

    # 7. 计算Dice系数。公式为: 2 * |A ∩ B| / (|A| + |B|)
    # [:, 1:] 操作是为了忽略背景类别(类别0)，让损失函数更关注前景(缺陷)的分割效果
    dice_coefficient = (2. * intersection[:, 1:] + smooth) / (union[:, 1:] + smooth)

    # 8. Dice Loss是 1 减去 Dice系数的平均值。损失越小，表示重叠度越高。
    return 1 - dice_coefficient.mean()


def l1_loss(inputs, targets, reduction="mean"):
    '''
    L1损失(也称为平均绝对误差,MAE)
    '''
    return F.l1_loss(inputs, targets, reduction=reduction)

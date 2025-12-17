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


def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    '''
    计算Focal Loss,注意目前只适用于二值分割和二分类
    :param inputs: 模型输出的预测值,范围[0,1],shape为(N, 1, H, W)
    :param targets: 真实标签(0或1),shape为(N, 1, H, W)
    :param alpha: 平衡因子(正样本的权重系数),默认-1表示不使用平衡因子
    :param gamma: Focusing参数,控制易分样本的权重衰减程度,默认4
    :param reduction: 损失归约方式,支持"mean"或"sum"
    :return: Focal Loss,shape为(1,)
    '''
    inputs = inputs.float()
    targets = targets.float()
    # 先计算标准交叉熵损失:-[target * log(input) + (1 - target) * log(1 - input)]
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    # 计算模型预测准确的概率p_t
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    # 计算Focal Loss，使得模型更加关注难以分类的样本，减轻大量易分样本对训练的影响
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 调整正负样本对总损失的贡献度，解决正负样本不平衡问题
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def l1_loss(inputs, targets, reduction="mean"):
    '''
    L1损失(也称为平均绝对误差,MAE)
    '''
    return F.l1_loss(inputs, targets, reduction=reduction)

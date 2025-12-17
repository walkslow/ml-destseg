import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from eval import evaluate
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, l1_loss

warnings.filterwarnings("ignore")


# 注意：该train函数只训练一个category，即训练一个类别(这个类别指物体的类别，而不是缺陷的类别)的模型
def train(args, category, rotate_90=False, random_rotate=0):
    # 创建检查点和日志目录
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    # 如果日志目录存在，则递归删除整个目录及其内容
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    # 创建 TensorBoard 可视化记录器，用于记录训练过程中的各种指标和日志。
    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    # 用于优化分割网络的不同组件，使用不同学习率
    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001, # 默认学习率，但会被参数组中设置的学习率覆盖
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    # 用于优化学生网络，使用较高的学习率
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    dataset = MVTecDataset(
        is_train=True,
        # 数据集根目录路径+当前训练的类别名称+/train/good/
        # 为每个类别单独构建训练路径，只使用 "good" 类别的正常样本进行训练（无监督学习）
        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs, # 每次从 dataloader 中取出的数据包含batch_size个样本
        shuffle=True, # 数据在每个 epoch 后会被随机打乱
        num_workers=args.num_workers,
        drop_last=True, # 丢弃最后不足一个batch的数据
    )

    global_step = 0 # 全局训练步数计数器，用于跟踪训练进度

    # 基于 step 而非 epoch 的训练策略，适用于对比学习/自监督学习/大规模预训练
    flag = True # 控制训练循环是否继续的布尔标志

    while flag:
        for _, sample_batched in enumerate(dataloader):
            # 清空两个优化器的梯度缓存
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            # 将数据移至 GPU
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()

            if global_step < args.de_st_steps: # 第一阶段：训练学生网络，冻结分割网络
                model.student_net.train()
                model.segmentation_net.eval()
            else: # 第二阶段：训练分割网络，冻结学生网络
                model.student_net.eval()
                model.segmentation_net.train()

            # 前向传播，获取分割网络和学生-教师网络的输出
            # output_segmentation: 分割网络的输出，范围[0,1],shape为(N, 1, H, W)
            # output_de_st: 学生-教师网络异常检测图，范围[0,1],shape为(N, 1, H, W)
            # output_de_st_list: 学生-教师网络的多尺度异常检测图，每个元素的shape为(N, 1, H, W)
            output_segmentation, output_de_st, output_de_st_list = model(
                img_aug, img_origin
            )

            # 将真实掩码插值到与分割输出相同的尺寸
            # 在语义分割任务中，通常将 ground truth 标签调整到模型输出尺寸
            # 这样可以避免在高分辨率下计算损失带来的噪声和过拟合风险
            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            # 根据不同训练阶段计算相应损失
            # 第一阶段：仅计算学生-教师网络的余弦相似度损失
            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            # 第二阶段：计算分割网络的Focal Loss和L1损失
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)

            # 反向传播与优化
            if global_step < args.de_st_steps:
                total_loss_val = cosine_loss_val
                total_loss_val.backward()
                de_st_optimizer.step() # 根据计算得到的梯度更新模型参数
            else:
                total_loss_val = focal_loss_val + l1_loss_val
                total_loss_val.backward()
                seg_optimizer.step()

            global_step += 1

            # 记录标量值以便可视化训练过程，global_step相当于x坐标
            visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
            visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
            visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
            visualizer.add_scalar("total_loss", total_loss_val, global_step)

            if global_step % args.eval_per_steps == 0:
                evaluate(args, category, model, visualizer, global_step)

            if global_step % args.log_per_steps == 0:
                if global_step < args.de_st_steps:
                    print(
                        f"Training at global step {global_step}, cosine loss: {round(float(cosine_loss_val), 4)}"
                    )
                else:
                    print(
                        f"Training at global step {global_step}, focal loss: {round(float(focal_loss_val), 4)}, l1 loss: {round(float(l1_loss_val), 4)}"
                    )

            # 如果dataloader中的数据全部迭代完了，而此时global_step还没有达到args.steps，
            # 则会重新进入for循环，开启下一个epoch的训练，直到global_step达到args.steps，才会跳出while循环，训练才会停止
            if global_step >= args.steps:
                flag = False
                break

    # 将模型参数字典保存为文件
    torch.save(
        model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 创建命令行参数解析器

    # 定义各种训练相关的命令行参数
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="./datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="./datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_MVTec")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--de_st_steps", type=int, default=1000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=1000)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference，表示取前 T 个最大响应

    parser.add_argument(
        "--custom_training_category", action="store_true", default=False
    )
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    args = parser.parse_args() # 解析命令行输入

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category
        # check
        for category in (
            no_rotation_category + slight_rotation_category + rotation_category
        ):
            assert category in ALL_CATEGORY
    else:
        no_rotation_category = [
            "capsule",
            "metal_nut",
            "pill",
            "toothbrush",
            "transistor",
        ]
        slight_rotation_category = [
            "wood",
            "zipper",
            "cable",
        ]
        rotation_category = [
            "bottle",
            "grid",
            "hazelnut",
            "leather",
            "tile",
            "carpet",
            "screw",
        ]

    # 对每个物体类别单独训练模型
    with torch.cuda.device(args.gpu_id):
        for obj in no_rotation_category:
            print(obj)
            train(args, obj)

        for obj in slight_rotation_category:
            print(obj)
            train(args, obj, rotate_90=False, random_rotate=5)

        for obj in rotation_category:
            print(obj)
            train(args, obj, rotate_90=True, random_rotate=5)

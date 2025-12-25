# -*- coding: utf-8 -*-
# 导入基础工具包
import argparse
import os
import shutil
import warnings
import copy
from datetime import datetime

# 导入PyTorch核心包
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# 导入项目自定义模块
from constant import RESIZE_SHAPE, NORMALIZE_MEAN_L, NORMALIZE_STD_L, NORMALIZE_MEAN_RGB, NORMALIZE_STD_RGB
from data.rod_dataset import RodDataset
from eval import evaluate # 评估函数
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, dice_loss
from visualize import save_metric_plots

# 忽略不必要的警告信息，保持输出整洁
warnings.filterwarnings("ignore")


def train(args):
    """
    主训练函数，负责整个模型的训练流程。
    该函数涵盖了从设备选择、目录创建、模型初始化、优化器和数据加载器设置，
    到执行核心训练循环、计算损失、反向传播和模型保存的全过程。
    :param args: 命令行传入的参数对象
    """
    start_time = datetime.now()
    print(f"--- 训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- 1. 初始化和环境设置 ---
    # 自动检测可用的CUDA设备，如果无可用GPU，则自动切换到CPU。这种设计增强了代码的设备兼容性。
    # 使用f-string动态构建设备字符串，例如 "cuda:0"
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- 使用设备: {device} ---")

    # 确保用于保存模型权重（checkpoint）和训练日志（log）的目录存在，如果不存在则创建。
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # --- 2. 运行命名和日志设置 ---
    # 为了方便实验跟踪和比较，为每次运行生成一个唯一的名称。
    # 名称由一个固定的前缀(run_name_head)、总训练步数(steps)和当前的日期时间戳构成。
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    run_name = f"{args.run_name_head}_{args.steps}_{current_time}"
    log_dir = os.path.join(args.log_path, run_name + "/")
    # 如果该日志目录已存在（例如，短时间内重复运行），则先删除，以确保一个干净的日志环境。
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    # 初始化TensorBoard的SummaryWriter，它会将训练过程中的各项指标（如损失）写入指定的log_dir。
    visualizer = SummaryWriter(log_dir=log_dir)
    print(f"--- TensorBoard 日志目录: {log_dir} ---")

    # --- 3. 模型初始化 ---
    # 实例化DeSTSeg模型。
    # num_classes: 设置为分类任务的类别数，包括背景。
    # dest=True: 启用student-teacher模式，学生网络将使用数据增强后的图像。
    # ed=True: 学生网络将包含一个解码器结构。
    # .to(device): 将模型的所有参数和缓冲区移动到先前选定的设备（GPU或CPU）。
    model = DeSTSeg(num_classes=args.num_classes, dest=True, ed=True).to(device)

    # --- 4. 优化器设置 ---
    # 采用两阶段训练策略，因此需要为学生网络和分割网络分别设置优化器。
    # 为分割网络设置优化器，并采用分层学习率（differentiated learning rates）：
    # ResNet主干部分使用较小的学习率(lr_res)，因为通常加载预训练权重，需要微调。
    # 分割头部分使用较大的学习率(lr_seghead)，因为它是新初始化的，需要更快的学习。
    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,  # 此处的lr是默认值，实际会被上面参数组中的lr覆盖
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    # 为学生网络设置独立的优化器。
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4, # 默认学习率
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    # --- 5. 数据集和数据加载器 ---
    print("--- 初始化训练数据集 ---")
    # 实例化RodDataset，用于加载和预处理数据
    dataset = RodDataset(
        is_train=True, # 明确指定为训练模式
        # 传入各种缺陷数据的路径
        rod_dir=args.rod_dir,
        scratch_dir=args.scratch_dir,
        dent_dir=args.dent_dir,
        dotted_dir=args.dotted_dir,
        # 根据命令行参数动态设置数据增强选项
        rotate_90=args.rotate_90,
        random_rotate=args.random_rotate,
        # 传入图像尺寸和归一化参数
        resize_shape=RESIZE_SHAPE,
        normalize_mean_l=NORMALIZE_MEAN_L,
        normalize_std_l=NORMALIZE_STD_L,
        normalize_mean_rgb=NORMALIZE_MEAN_RGB,
        normalize_std_rgb=NORMALIZE_STD_RGB,
    )

    # 实例化DataLoader，用于高效地批量加载数据
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,          # 每个批次加载的样本数
        shuffle=True,               # 在每个epoch开始时打乱数据顺序，增加模型泛化能力
        num_workers=args.num_workers, # 使用多个子进程并行加载数据，加快数据准备速度
        drop_last=True,             # 如果最后一个批次的样本数不足batch_size，则丢弃该批次
    )

    # --- 6. 训练主循环 ---
    global_step = 0  # 初始化全局步数计数器，用于精确控制训练总长度
    flag = True      # 训练循环的控制标志

    # 最佳模型跟踪
    best_st_loss = float('inf')
    best_seg_metric = float('-inf')
    best_st_state_path = os.path.join(args.checkpoint_path, f"{run_name}_best_st.pckl")
    best_seg_state_path = os.path.join(args.checkpoint_path, f"{run_name}_best_seg.pckl")

    # 使用 "while" 循环和 "global_step" 实现基于步数（step-based）的训练，
    # 而非传统的基于轮次（epoch-based）的训练。这在需要精确控制迭代次数的场景中非常有用，适用于对比学习/自监督学习/大规模预训练
    while flag:
        # 遍历数据加载器，获取每个批次的数据
        for _, sample_batched in enumerate(dataloader):
            # 在每次计算梯度之前，清空优化器的旧梯度
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()

            # 将批次中的所有数据张量移动到目标设备
            img_origin_l = sample_batched["img_origin_l"].to(device)
            img_origin_rgb = sample_batched["img_origin_rgb"].to(device)
            img_aug_l = sample_batched["img_aug_l"].to(device)
            img_aug_rgb = sample_batched["img_aug_rgb"].to(device)
            mask = sample_batched["mask"].to(device)

            # --- 两阶段训练逻辑 ---
            # 阶段一：训练学生网络 (de_st_steps)
            if global_step < args.de_st_steps:
                model.student_net.train()    # 仅学生网络处于训练模式
                model.segmentation_net.eval() # 分割网络处于评估模式，其参数不更新
            # 阶段二：训练分割网络
            else:
                model.student_net.eval()     # 学生网络处于评估模式，冻结其参数
                model.segmentation_net.train() # 仅分割网络处于训练模式

            # --- 前向传播 ---
            # 将数据输入模型，获取三个关键输出
            # output_segmentation: 分割网络的最终输出, shape: (N, C, H, W)
            # output_de_st: 学生-教师网络融合后的单尺度异常图, shape: (N, 1, H, W)
            # output_de_st_list: 学生-教师网络在不同特征层级的多尺度异常图列表
            output_segmentation, output_de_st, output_de_st_list = model(
                img_origin_l=img_origin_l,     # 原始灰度图，用于分割网络
                img_origin_rgb=img_origin_rgb, # 原始RGB图，用于教师网络
                img_aug_l=img_aug_l,           # 增强后的灰度图，用于学生网络
                img_aug_rgb=img_aug_rgb        # 增强后的RGB图，当前模型实现中未使用
            )

            # --- 掩码(mask)尺寸对齐 ---
            # 原始的mask尺寸可能与模型的输出尺寸不一致，需要通过插值进行对齐。
            # F.interpolate 要求输入是4D或5D张量 (N, C, ...)，而mask是3D (N, H, W)。
            # 1. mask.unsqueeze(1): 在通道维度上增加一个维度，变为 (N, 1, H, W)。
            # 2. F.interpolate(...): 使用 'nearest' 最近邻插值，将其空间尺寸调整为与分割输出一致。
            # 3. .squeeze(1): 计算损失时，目标mask应为 (N, H, W)，因此移除通道维度。
            mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=output_segmentation.size()[2:],
                mode="nearest",
            ).squeeze(1).long() # 确保mask的数据类型为long，以匹配损失函数要求

            # --- 损失计算 ---
            # 根据当前所处的训练阶段，计算不同的损失。
            # 阶段一：仅计算学生-教师网络的余弦相似度损失，目标是让学生网络模仿教师网络。
            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            # 阶段二：计算分割网络的损失，由Focal Loss和Dice Loss两部分组成。
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            dice_loss_val = dice_loss(output_segmentation, mask)

            # --- 反向传播与优化 ---
            if global_step < args.de_st_steps:
                total_loss_val = cosine_loss_val
                total_loss_val.backward()  # 计算梯度
                de_st_optimizer.step()     # 更新学生网络的权重
                
                # 跟踪最佳 S-T 模型 (基于 Cosine Loss)
                if cosine_loss_val < best_st_loss:
                    best_st_loss = cosine_loss_val
                    torch.save(model.state_dict(), best_st_state_path)

            else:
                total_loss_val = focal_loss_val + dice_loss_val
                total_loss_val.backward()  # 计算梯度
                seg_optimizer.step()       # 更新分割网络的权重

            global_step += 1 # 更新全局步数

            # --- 阶段切换：加载最佳 S-T 模型 ---
            if global_step == args.de_st_steps:
                if os.path.exists(best_st_state_path):
                    print(f"--- [Phase Switch] Loading best S-T model (Loss: {best_st_loss:.6f}) for Segmentation training ---")
                    # 加载参数
                    model.load_state_dict(torch.load(best_st_state_path))
                    # 重新将模型放到设备上 (以防万一)
                    model.to(device)

            # --- 7. 日志记录和可视化 ---
            # 使用visualizer将各项损失值写入TensorBoard，方便后续分析和可视化。
            # global_step作为x轴，损失值作为y轴。
            visualizer.add_scalar("Loss/Cosine_Loss", cosine_loss_val, global_step)
            visualizer.add_scalar("Loss/Focal_Loss", focal_loss_val, global_step)
            visualizer.add_scalar("Loss/Dice_Loss", dice_loss_val, global_step)
            visualizer.add_scalar("Loss/Total_Loss", total_loss_val, global_step)

            # 定期评估模型性能
            if global_step % args.eval_per_steps == 0 and args.test_dir:
                print(f"--- Running evaluation at step {global_step} ---")
                eval_args = copy.copy(args)
                eval_args.rod_dir = args.test_dir
                try:
                    # evaluate 现在返回指标字典
                    metrics = evaluate(eval_args, model, visualizer, global_step)
                    
                    # 仅在第二阶段 (分割训练) 跟踪最佳分割模型
                    if global_step > args.de_st_steps:
                        # 使用 mIoU 作为主要指标 (AUPRO 可能为 NaN)
                        current_metric = metrics.get("mIoU", 0.0)
                        if current_metric > best_seg_metric:
                            best_seg_metric = current_metric
                            print(f"--- New Best Segmentation Model! mIoU: {best_seg_metric:.4f} ---")
                            torch.save(model.state_dict(), best_seg_state_path)
                            
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                
                # 恢复训练模式
                if global_step < args.de_st_steps:
                    model.student_net.train()
                    model.segmentation_net.eval()
                else:
                    model.student_net.eval()
                    model.segmentation_net.train()

            # 定期在控制台打印训练信息
            if global_step % args.log_per_steps == 0:
                if global_step < args.de_st_steps:
                    print(
                        f"训练步数 {global_step}/{args.steps} | "
                        f"阶段: 学生网络 | "
                        f"余弦损失: {round(float(cosine_loss_val), 4)}"
                    )
                else:
                    print(
                        f"训练步数 {global_step}/{args.steps} | "
                        f"阶段: 分割网络 | "
                        f"Focal损失: {round(float(focal_loss_val), 4)} | "
                        f"Dice损失: {round(float(dice_loss_val), 4)}"
                    )

            # 检查是否达到总训练步数，如果是，则终止训练
            if global_step >= args.steps:
                flag = False
                break

    # --- 8. 保存模型 ---
    # 训练结束后，保存最终的模型。
    # 优先保存训练过程中通过 evaluate 选出的最佳模型 (best_seg_state_path)。
    # 如果没有生成最佳模型（例如未运行评估），则保存当前最后一步的模型。
    
    final_model_path = os.path.join(args.checkpoint_path, run_name + ".pckl")
    
    if os.path.exists(best_seg_state_path):
        print(f"--- 训练完成，正在将最佳模型 (mIoU: {best_seg_metric:.4f}) 移动为最终结果 ---")
        # 复制/移动为最终名称
        shutil.copy(best_seg_state_path, final_model_path)
        # 删除中间文件
        os.remove(best_seg_state_path)
    else:
        print(f"--- 训练完成，未找到最佳模型记录，保存当前模型至: {final_model_path} ---")
        torch.save(model.state_dict(), final_model_path)
        
    # 清理中间过程保存的 S-T 模型
    if os.path.exists(best_st_state_path):
        print(f"--- 清理中间 S-T 模型: {os.path.basename(best_st_state_path)} ---")
        os.remove(best_st_state_path)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"--- 训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"--- 训练总时长: {duration} ---")

    # --- 9. 自动绘制并保存 Loss/Metric 曲线 ---
    print("--- 正在绘制并保存 Loss 和评估指标曲线 ---")
    # 最终保存的目录是 args.vis_path/run_name
    vis_save_dir = os.path.join(args.vis_path, run_name)
    save_metric_plots(log_dir, vis_save_dir)


if __name__ == "__main__":
    # --- 命令行参数定义 ---
    # 使用argparse库来定义和解析命令行参数，这使得脚本的配置更加灵活。
    parser = argparse.ArgumentParser(description='使用DeSTSeg在ROD数据集上进行多分类分割')

    # -- 硬件与并行化参数 --
    parser.add_argument("--gpu_id", type=int, default=0, help="指定使用的GPU编号")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载器使用的工作进程数")

    # -- 数据集路径参数 --
    # default路径使用了 'r' 前缀来表示原始字符串，避免反斜杠 `\` 被错误解析。
    parser.add_argument("--rod_dir", type=str, default=r"D:\Dataset\ForMyThesis\MiniTest\train\good\images", help="完好图像（good）的目录路径")
    parser.add_argument("--scratch_dir", type=str, default=r"D:\Dataset\ForMyThesis\MiniTest\train\scratch", help="划痕缺陷（scratch）的目录路径")
    parser.add_argument("--dent_dir", type=str, default=r"D:\Dataset\ForMyThesis\MiniTest\train\dent", help="凹痕缺陷（dent）的目录路径")
    parser.add_argument("--dotted_dir", type=str, default=None, help="点状缺陷（dotted）的目录路径 (可选)")
    parser.add_argument("--test_dir", type=str, default=r"D:\Dataset\ForMyThesis\MiniTest\eval\images", help="测试集图像目录路径 (用于训练中评估)")
    
    # -- 数据增强参数 --
    parser.add_argument('--rotate_90', action='store_true', help='启用90度旋转数据增强')
    parser.add_argument('--random_rotate', type=int, default=0, help='随机旋转增强的最大角度 (0表示不启用)')

    # -- 模型与日志路径参数 --
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/", help="保存模型检查点的路径")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_ROD", help="运行名称的前缀")
    parser.add_argument("--log_path", type=str, default="./logs/", help="保存TensorBoard日志的路径")
    parser.add_argument("--vis_path", type=str, default="./vis/", help="保存可视化图表的路径")
    parser.add_argument("--num_classes", type=int, default=4, help="类别数量")

    # -- 训练超参数 --
    parser.add_argument("--bs", type=int, default=8, help="训练的批量大小 (Batch Size)")
    parser.add_argument("--lr_de_st", type=float, default=0.4, help="学生网络的学习率")
    parser.add_argument("--lr_res", type=float, default=0.1, help="分割网络中ResNet部分的学习率")
    parser.add_argument("--lr_seghead", type=float, default=0.01, help="分割网络中分割头部分的学习率")
    parser.add_argument("--steps", type=int, default=5000, help="总训练步数")
    parser.add_argument("--de_st_steps", type=int, default=1000, help="第一阶段训练学生网络的步数")
    parser.add_argument("--eval_per_steps", type=int, default=1000, help="每N步评估一次模型")
    parser.add_argument("--log_per_steps", type=int, default=50, help="每N步在控制台记录一次训练信息")
    parser.add_argument("--gamma", type=float, default=2, help="Focal Loss中的gamma参数")

    # 解析命令行传入的参数
    args = parser.parse_args()

    # --- 开始训练 ---
    # 调用主训练函数，并传入解析后的参数。
    # 设备选择逻辑已在train函数内部处理，此处无需额外操作。
    train(args)

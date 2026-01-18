# -*- coding: utf-8 -*-
# 导入基础工具包
import argparse
import os
import shutil
import warnings
import sys
# 忽略所有警告 (包括 torchmetrics 产生的 pkg_resources 弃用警告)
warnings.filterwarnings("ignore")

from datetime import datetime

# 导入PyTorch核心包
import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# 导入项目自定义模块
from constant import RESIZE_SHAPE, NORMALIZE_MEAN_L, NORMALIZE_STD_L, NORMALIZE_MEAN_RGB, NORMALIZE_STD_RGB
from data.rod_dataset import RodDataset
from eval import evaluate # 评估函数
from model.model_utils import setup_seed, seed_worker, get_scheduler, DualLogger
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, dice_loss
from model.adaptive_loss import create_adaptive_loss
from model.patchcore_mem import MemoryBank, MemoryBankSourceDataset # 引入 PatchCore 组件
from visualize import save_metric_plots

def train(args):
    """
    主训练函数，负责整个模型的训练流程。
    该函数涵盖了从设备选择、目录创建、模型初始化、优化器和数据加载器设置，
    到执行核心训练循环、计算损失、反向传播和模型保存的全过程。
    :param args: 命令行传入的参数对象
    """
    # --- 1. 运行命名和日志设置 ---
    # 为了方便实验跟踪和比较，为每次运行生成一个唯一的名称。
    
    # 如果使用了 PatchCore，且 run_name_head 中没有 MemB，则在 run_name_head 前加上前缀
    if args.use_patchcore and "MemB" not in args.run_name_head:
        args.run_name_head = "MemB_" + args.run_name_head

    # 如果使用了 D2T，且 run_name_head 中没有 D2T，则在 run_name_head 前加上前缀
    if args.use_d2t and "D2T" not in args.run_name_head:
        args.run_name_head = "D2T_" + args.run_name_head

    # 处理 rotate_90 逻辑：默认为 True，除非指定了 --no_rotate_90
    args.rotate_90 = not args.no_rotate_90

    # 名称由一个固定的前缀(run_name_head)、总训练步数(steps)和当前的日期时间戳构成。
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    run_name = f"{args.run_name_head}_{args.steps}_{current_time}"
    
    # --- 配置 DualLogger ---
    # 将终端输出同时写入文件
    if not os.path.exists(args.terminal_output_dir):
        os.makedirs(args.terminal_output_dir)
    terminal_log_path = os.path.join(args.terminal_output_dir, run_name + ".txt")
    
    # 保存原始 stdout 以便后续恢复（虽然这里可能不需要恢复，但这是一个好习惯）
    original_stdout = sys.stdout
    sys.stdout = DualLogger(terminal_log_path)
    
    print(f"--- 终端输出将同时保存至: {terminal_log_path} ---")

    print("--- 命令行参数配置 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("----------------------")

    start_time = datetime.now()
    print(f"--- 训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 设置随机种子
    setup_seed(args.seed)

    # --- 2. 初始化和环境设置 ---
    # 处理 GPU 选择
    if args.gpu_id is None:
        args.gpu_id = [-1]
    
    # 检查是否请求使用 CPU
    use_cpu_only = False
    if len(args.gpu_id) == 1 and args.gpu_id[0] == -1:
        use_cpu_only = True
        print("--- 检测到 gpu_id=-1，将强制使用 CPU 模式 ---")
        num_physical_gpus = 0 # 模拟无 GPU 环境
    else:
        # 尝试检测物理 GPU 数量 (不依赖 torch.cuda，避免提前初始化)
        import subprocess
        try:
            # 使用 nvidia-smi 列出所有 GPU，统计行数
            result = subprocess.check_output("nvidia-smi -L", shell=True)
            # 针对 Windows/Linux 换行符差异进行处理，过滤空行
            physical_gpus = [line for line in result.decode('utf-8').strip().split('\n') if line.strip()]
            num_physical_gpus = len(physical_gpus)
            print(f"--- 检测到物理 GPU 数量: {num_physical_gpus} ---")
        except Exception as e:
            # 如果 nvidia-smi 执行失败（如无驱动或未在 PATH 中），则无法验证
            print(f"--- 警告: 无法检测物理 GPU 数量 (nvidia-smi error: {e})，将尝试直接使用 CPU ---")
            num_physical_gpus = 0
            use_cpu_only = True

    # 验证并过滤 GPU ID
    valid_gpu_ids = []
    if use_cpu_only:
        valid_gpu_ids = []
    elif num_physical_gpus > 0:
         for gpu_idx in args.gpu_id:
             if 0 <= gpu_idx < num_physical_gpus:
                 valid_gpu_ids.append(gpu_idx)
             else:
                 print(f"--- 警告: GPU ID {gpu_idx} 超出范围 (0-{num_physical_gpus-1})，将被忽略 ---")
    
    # 设置 CUDA_VISIBLE_DEVICES
    if valid_gpu_ids:
        # 将 int 列表转换为字符串列表用于 join
        gpu_list_str = [str(x) for x in valid_gpu_ids]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list_str)
        print(f"--- 实际使用的 GPU ID: {valid_gpu_ids} ---")
    else:
        # 如果没有有效的 GPU ID，确保 CUDA_VISIBLE_DEVICES 为空，强制 PyTorch 使用 CPU（或看不到 GPU）
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if not use_cpu_only: # 避免重复打印
            print("--- 未指定有效的 GPU，将使用 CPU 训练 ---")

    # 自动检测可用的CUDA设备 (此时受 CUDA_VISIBLE_DEVICES 影响)
    if torch.cuda.is_available() and valid_gpu_ids:
        # 重要：设置 CUDA_VISIBLE_DEVICES 后，逻辑设备 ID 总是从 0 开始
        # 所以这里我们始终使用 cuda:0 作为主设备
        device = torch.device("cuda:0")
        
        # 判断是否启用多卡模式
        # PyTorch 看到的设备数量就是 len(valid_gpu_ids)
        visible_devices = torch.cuda.device_count()
        if visible_devices > 1:
            args.use_multi_gpu = True
            print(f"--- PyTorch 检测到 {visible_devices} 张可见 GPU，启用 DataParallel 模式 ---")
        else:
            args.use_multi_gpu = False
            print("--- PyTorch 检测到单张可见 GPU，启用单卡模式 ---")
    else:
        device = torch.device("cpu")
        args.use_multi_gpu = False
        print("--- 使用 CPU 模式 ---")

    print(f"--- 使用主设备: {device} ---")
    
    # 确保用于保存模型权重（checkpoint）和训练日志（log）的目录存在，如果不存在则创建。
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    log_dir = os.path.join(args.log_dir, run_name + "/")
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
    # use_d2t: 是否使用 WCA 和 PCA 增强结构
    # .to(device): 将模型的所有参数和缓冲区移动到先前选定的设备（GPU或CPU）。
    model = DeSTSeg(
        num_classes=args.num_classes, 
        dest=True, 
        ed=True,
        use_d2t=args.use_d2t,
        use_patchcore=args.use_patchcore
    ).to(device)

    # --- 多卡训练支持 (DataParallel) ---
    if torch.cuda.device_count() > 1 and args.use_multi_gpu:
        print(f"--- 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 模式 ---")
        model = torch.nn.DataParallel(model)
        # 获取原始模型对象，以便访问其子模块 (segmentation_net, student_net)
        # 在 DataParallel 模式下，model.module 才是真正的模型
        real_model = model.module
    else:
        real_model = model

    # --- 自适应损失权重初始化 ---
    adaptive_loss_fn = None
    if args.adaptive_loss_method:
        print(f"--- 启用自适应损失权重调整: {args.adaptive_loss_method} ---")
        if args.adaptive_loss_method == 'dynamic':
            # 动态权重调整 (无需参数，在训练循环中调用)
            adaptive_loss_fn = create_adaptive_loss(
                method='dynamic',
                base_weights=[args.lambda_focal, args.lambda_dice],
                adjust_factor=args.adaptive_adjust_factor,
                warmup_steps=args.adaptive_warmup_steps,
                smoothing_window=args.adaptive_smoothing_window
            )
            print(f"    基础权重: Focal={args.lambda_focal}, Dice={args.lambda_dice}")
            print(f"    调整因子: {args.adaptive_adjust_factor}, 预热步数: {args.adaptive_warmup_steps}")
        elif args.adaptive_loss_method == 'uncertainty':
            # 不确定性权重 (作为可学习模块)
            adaptive_loss_fn = create_adaptive_loss(
                method='uncertainty',
                num_losses=2,
                init_log_vars=[args.adaptive_init_log_var] * 2
            ).to(device)
            print(f"    初始log方差: {args.adaptive_init_log_var}")
            print(f"    将权重参数添加到优化器进行学习")
        elif args.adaptive_loss_method == 'autoweight':
            # 自动权重
            adaptive_loss_fn = create_adaptive_loss(
                method='autoweight',
                num_losses=2,
                init_weights=[args.lambda_focal, args.lambda_dice]
            ).to(device)
            print(f"    初始权重: Focal={args.lambda_focal}, Dice={args.lambda_dice}")
            print(f"    将权重参数添加到优化器进行学习")
        else:
            print(f"--- 警告: 未知的自适应损失方法 {args.adaptive_loss_method}，将使用固定权重 ---")
            adaptive_loss_fn = None
    else:
        print("--- 使用固定损失权重 ---")

    # --- 4. 优化器与学习率调度设置 ---
    # 采用两阶段训练策略，因此需要为学生网络和分割网络分别设置优化器和调度器。

    # 4.1 分割网络优化器
    seg_optimizer = torch.optim.SGD(
        [
            {"params": real_model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": real_model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    # 如果使用 uncertainty 或 autoweight 方法，将权重参数添加到优化器
    if adaptive_loss_fn and args.adaptive_loss_method in ['uncertainty', 'autoweight']:
        seg_optimizer.add_param_group({
            "params": adaptive_loss_fn.parameters(),
            "lr": args.lr_seghead  # 使用与分割头相同的学习率
        })
    
    # 4.2 学生网络优化器
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": real_model.student_net.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    # 4.3 学习率调度器 (Linear Warmup + Cosine Annealing)
    # 计算各阶段的步数
    st_steps = args.de_st_steps
    seg_steps = args.steps - args.de_st_steps
    
    # 创建调度器
    de_st_scheduler = get_scheduler(de_st_optimizer, st_steps)
    seg_scheduler = get_scheduler(seg_optimizer, seg_steps)


    # --- 5. 数据集和数据加载器 ---
    print("--- 初始化训练数据集 ---")
    # 实例化RodDataset，用于加载和预处理数据
    dataset = RodDataset(
        # 传入各种缺陷数据的路径
        rod_dir=args.rod_dir,
        scratch_dir=args.scratch_dir,
        dent_dir=args.dent_dir,
        dotted_dir=args.dotted_dir,
        # 根据命令行参数动态设置数据增强选项
        rotate_90=args.rotate_90,
        random_rotate=args.random_rotate,
        # 新增：控制是否使用真实数据
        use_real_data=args.use_real_train_data,
        # 传入图像尺寸和归一化参数
        resize_shape=RESIZE_SHAPE,
        normalize_mean_l=NORMALIZE_MEAN_L,
        normalize_std_l=NORMALIZE_STD_L,
        normalize_mean_rgb=NORMALIZE_MEAN_RGB,
        normalize_std_rgb=NORMALIZE_STD_RGB,
    )

    # 实例化DataLoader，用于高效地批量加载数据
    g = torch.Generator()
    g.manual_seed(args.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,          # 每个批次加载的样本数
        shuffle=True,               # 在每个epoch开始时打乱数据顺序，增加模型泛化能力
        num_workers=args.num_workers, # 使用多个子进程并行加载数据，加快数据准备速度
        drop_last=True,             # 如果最后一个批次的样本数不足batch_size，则丢弃该批次
        worker_init_fn=seed_worker,
        generator=g,
    )

    # --- 6. 训练主循环 ---
    global_step = 0  # 初始化全局步数计数器，用于精确控制训练总长度
    flag = True      # 训练循环的控制标志

    # --- PatchCore 初始化 ---
    memory_bank = None
    memory_bank_features_list = [] # 用于暂存特征 (放在 CPU)
    num_samples_collected = 0
    total_train_samples = len(dataset)
    memory_bank_finalized = False
    
    if args.use_patchcore:
        print(f"--- 启用 PatchCore: 采样率 {args.patchcore_ratio}, 预计总样本数 {total_train_samples} ---")
        # MemoryBank 负责存储和搜索，注意这里的 device 是用于后续计算距离的主设备
        memory_bank = MemoryBank(device)

        # --- 记忆库预构建逻辑 ---
        if args.use_prebuild_memory_bank and args.memory_bank_source_dir:
            print(f"--- 预构建记忆库: 使用原始正常图像 {args.memory_bank_source_dir} ---")
            
            # 创建专用数据集
            # 旋转增强策略：如果训练启用旋转，记忆库也应该包含旋转后的特征以覆盖特征空间
            mb_dataset = MemoryBankSourceDataset(
                img_dir=args.memory_bank_source_dir,
                resize_shape=RESIZE_SHAPE,
                mean_l=NORMALIZE_MEAN_L,
                std_l=NORMALIZE_STD_L,
                mean_rgb=NORMALIZE_MEAN_RGB,
                std_rgb=NORMALIZE_STD_RGB,
                rotate_90=args.rotate_90,
                random_rotate=args.random_rotate
            )
            
            # 使用较大的 batch_size 加速提取 (仅做推理)
            mb_dataloader = DataLoader(mb_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
            
            model.eval()
            all_features_list = []
            
            print(f"--- 开始提取特征 (数据集大小: {len(mb_dataset)}) ---")
            with torch.no_grad():
                for batch in tqdm.tqdm(mb_dataloader, desc="提取记忆库特征"):
                    img_aug_l = batch["img_aug_l"].to(device)
                    img_aug_rgb = batch["img_aug_rgb"].to(device)
                    
                    # 传入 memory_bank=None 触发 Phase 1 逻辑，提取特征
                    # 注意：如果是 DataParallel，需要确保参数传递正确
                    # DeSTSeg forward: (img_aug_l, img_aug_rgb, img_origin_l, img_origin_rgb, memory_bank)
                    # 我们显式传递，利用 kwargs
                    _, _, _, patchcore_features = model(
                        img_aug_l=img_aug_l, 
                        img_aug_rgb=img_aug_rgb, 
                        memory_bank=None
                    )
                    
                    if patchcore_features is not None:
                        all_features_list.append(patchcore_features.cpu())
            
            if all_features_list:
                all_features = torch.cat(all_features_list, dim=0)
                print(f"--- 特征提取完成，总特征数: {all_features.shape[0]}，开始构建核心集 ---")
                memory_bank.fit(all_features, sampling_ratio=args.patchcore_ratio)
                
                # 保存
                save_path = os.path.join(args.checkpoint_dir, f"{run_name}_memory_bank_prebuilt.pt")
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)
                memory_bank.save(save_path)
                
                memory_bank_finalized = True
                print(f"--- 记忆库预构建完成并已保存至 {save_path}，跳过后续训练中的构建步骤 ---")
                
                # 恢复模型训练模式 (虽然 Phase 1 主要训练 Student，Teacher 始终 eval)
                # 但主循环会处理 model.train() / eval()
            else:
                print("--- 警告: 未提取到任何特征，记忆库预构建失败，将回退到训练中构建 ---")
                memory_bank_finalized = False
        else:
            memory_bank_finalized = False
    else:
        memory_bank_finalized = False

    # --- PatchCore 参数校验与自动修正 ---
    # 仅当 memory_bank 未预构建完成时，才强制要求 Phase 1 遍历整个数据集
    if args.use_patchcore and not memory_bank_finalized:
        import math
        # 计算遍历一次训练集所需的最小步数
        # 实际 batch size = args.bs * args.grad_acc_steps
        actual_bs = args.bs * args.grad_acc_steps
        min_steps_for_one_epoch = math.ceil(total_train_samples / actual_bs)
        
        if args.de_st_steps < min_steps_for_one_epoch:
            print(f"--- 警告: de_st_steps ({args.de_st_steps}) 小于完成一轮 Epoch 所需步数 ({min_steps_for_one_epoch}) ---")
            print(f"--- 自动修正: de_st_steps 调整为 {min_steps_for_one_epoch} ---")
            
            # 调整 Phase 1 步数
            diff = min_steps_for_one_epoch - args.de_st_steps
            args.de_st_steps = min_steps_for_one_epoch
            
            # 同样增加总步数，保持 Phase 2 训练时长不变
            args.steps += diff
            print(f"--- 自动修正: 总训练步数 steps 调整为 {args.steps} ---")
            
            # 更新 Scheduler 需要的步数变量
            st_steps = args.de_st_steps
            seg_steps = args.steps - args.de_st_steps
            
            # 重新创建 Scheduler
            de_st_scheduler = get_scheduler(de_st_optimizer, st_steps)
            seg_scheduler = get_scheduler(seg_optimizer, seg_steps)

    # 预加载验证集 DataLoader (如果指定了 val_rod_dir)
    # 将其放在循环外，避免每次 evaluate 都重新扫描目录和创建 workers
    val_dataloader = None
    if args.val_rod_dir:
        print(f"--- 预加载验证数据集: {args.val_rod_dir} ---")
        val_dataset = RodDataset(
            rod_dir=args.val_rod_dir,
            scratch_dir=args.val_scratch_dir,
            dent_dir=args.val_dent_dir,
            dotted_dir=args.val_dotted_dir,
            use_real_data=args.use_real_val_data,
            # 验证集旋转增强参数与训练集一致
            rotate_90=args.rotate_90,
            random_rotate=args.random_rotate,
            resize_shape=RESIZE_SHAPE,
            normalize_mean_l=NORMALIZE_MEAN_L,
            normalize_std_l=NORMALIZE_STD_L,
            normalize_mean_rgb=NORMALIZE_MEAN_RGB,
            normalize_std_rgb=NORMALIZE_STD_RGB,
        )
        val_g = torch.Generator()
        val_g.manual_seed(args.seed)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.bs, 
            shuffle=False, 
            num_workers=args.num_workers,
            worker_init_fn=seed_worker, 
            generator=val_g
        )

    # 最佳模型跟踪
    best_st_loss = float('inf')
    best_seg_metric = float('-inf')
    best_st_state_path = os.path.join(args.checkpoint_dir, f"{run_name}_best_st.pckl")
    best_seg_state_path = os.path.join(args.checkpoint_dir, f"{run_name}_best_seg.pckl")

    # 使用 "while" 循环和 "global_step" 实现基于步数（step-based）的训练，
    # 而非传统的基于轮次（epoch-based）的训练。这在需要精确控制迭代次数的场景中非常有用，适用于对比学习/自监督学习/大规模预训练
    log_start_time = datetime.now()
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
            # 定义当前训练阶段
            is_student_phase = global_step < args.de_st_steps

            # 阶段一：训练学生网络 (de_st_steps)
            if is_student_phase:
                real_model.student_net.train()    # 仅学生网络处于训练模式
                real_model.segmentation_net.eval() # 分割网络处于评估模式，其参数不更新
            # 阶段二：训练分割网络
            else:
                real_model.student_net.eval()     # 学生网络处于评估模式，冻结其参数
                real_model.segmentation_net.train() # 仅分割网络处于训练模式

            # --- 前向传播 ---
            # 将数据输入模型，获取四个关键输出
            # output_segmentation: 分割网络的最终输出, shape: (N, C, H, W)
            # output_de_st: 学生-教师网络融合后的单尺度异常图, shape: (N, 1, H, W)
            # output_de_st_list: 学生-教师网络在不同特征层级的多尺度异常图列表
            # patchcore_features: 学生-教师网络在教师特征层级的特征图，用于 PatchCore 记忆库构建
            output_segmentation, output_de_st, output_de_st_list, patchcore_features = model(
                img_origin_l=img_origin_l,     # 原始灰度图，用于分割网络
                img_origin_rgb=img_origin_rgb, # 原始RGB图，用于教师网络
                img_aug_l=img_aug_l,           # 增强后的灰度图，用于学生网络
                img_aug_rgb=img_aug_rgb,       # 增强后的RGB图，当前模型实现中未使用
                # 传入 PatchCore 记忆库: 仅在记忆库已构建且不在学生训练阶段时传入，以加速 Phase 1
                memory_bank=memory_bank if (memory_bank_finalized and not is_student_phase) else None
            )

            # --- PatchCore 特征收集 ---
            if args.use_patchcore and not memory_bank_finalized and patchcore_features is not None:
                # 移动到 CPU 并缓存
                memory_bank_features_list.append(patchcore_features.cpu())
                
                # 更新计数
                num_samples_collected += img_origin_rgb.shape[0]
                
                # 检查是否已收集足够的数据
                if num_samples_collected >= total_train_samples:
                    print(f"--- PatchCore: 已收集 {num_samples_collected}/{total_train_samples} 样本，开始构建记忆库... ---")
                    
                    # 合并所有特征
                    all_features = torch.cat(memory_bank_features_list, dim=0)
                    
                    # 构建记忆库 (核心集采样)
                    memory_bank.fit(all_features, sampling_ratio=args.patchcore_ratio)
                    
                    # 保存记忆库
                    bank_save_path = os.path.join(args.checkpoint_dir, f"{run_name}_memory_bank.pt")
                    memory_bank.save(bank_save_path)
                    
                    # 清理缓存
                    memory_bank_features_list = []
                    memory_bank_finalized = True
                    print("--- PatchCore: 记忆库构建完成并保存 ---")

            # --- 损失计算、反向传播与日志记录 ---
            if is_student_phase:
                # --- 阶段一：学生网络训练 ---
                # 仅计算学生-教师网络的余弦相似度损失
                cosine_loss_val = cosine_similarity_loss(output_de_st_list)
                total_loss_val = cosine_loss_val
                
                total_loss_val.backward()
                de_st_optimizer.step()
                de_st_scheduler.step() # 更新学生网络学习率
                
                # 跟踪最佳 S-T 模型 (基于 Cosine Loss)
                if cosine_loss_val < best_st_loss:
                    best_st_loss = cosine_loss_val
                    torch.save(real_model.state_dict(), best_st_state_path)

                # 日志记录：仅记录 Cosine Loss 和 学习率
                visualizer.add_scalar("Loss/Cosine_Loss", cosine_loss_val, global_step + 1)
                visualizer.add_scalar("LR/Student_Net", de_st_optimizer.param_groups[0]["lr"], global_step + 1)

            else:
                # --- 阶段二：分割网络训练 ---
                # 掩码(mask)尺寸对齐 (仅在分割阶段需要)
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=output_segmentation.size()[2:],
                    mode="nearest",
                ).squeeze(1).long()

                # 计算分割网络的损失
                focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
                dice_loss_val = dice_loss(output_segmentation, mask)

                # 使用自适应权重或固定权重计算总损失
                if adaptive_loss_fn:
                    if args.adaptive_loss_method == 'dynamic':
                        # 动态权重调整
                        lambda_focal, lambda_dice = adaptive_loss_fn.get_weights(
                            [focal_loss_val, dice_loss_val],
                            global_step - args.de_st_steps  # 分割阶段的相对步数
                        )
                        total_loss_val = lambda_focal * focal_loss_val + lambda_dice * dice_loss_val
                    else:
                        # uncertainty 或 autoweight 方法
                        total_loss_val, weights = adaptive_loss_fn([focal_loss_val, dice_loss_val])
                        lambda_focal, lambda_dice = weights
                else:
                    # 使用固定权重
                    lambda_focal = args.lambda_focal
                    lambda_dice = args.lambda_dice
                    total_loss_val = lambda_focal * focal_loss_val + lambda_dice * dice_loss_val

                # 梯度累积：损失除以累积步数
                (total_loss_val / args.grad_acc_steps).backward()

                if (global_step + 1) % args.grad_acc_steps == 0:
                    seg_optimizer.step()
                    seg_optimizer.zero_grad()
                    seg_scheduler.step() # 更新分割网络学习率

                # 日志记录：仅记录分割相关损失 和 学习率
                visualizer.add_scalar("Loss/Focal_Loss", focal_loss_val, global_step + 1)
                visualizer.add_scalar("Loss/Dice_Loss", dice_loss_val, global_step + 1)
                visualizer.add_scalar("Loss/Total_Loss", total_loss_val, global_step + 1)

                # 记录自适应权重变化
                if adaptive_loss_fn:
                    visualizer.add_scalar("Adaptive_Weights/Focal_Weight", lambda_focal, global_step + 1)
                    visualizer.add_scalar("Adaptive_Weights/Dice_Weight", lambda_dice, global_step + 1)

                # 分别记录 ResNet (Backbone) 和 Head 的学习率
                visualizer.add_scalar("LR/Seg_ResNet", seg_optimizer.param_groups[0]["lr"], global_step + 1)
                visualizer.add_scalar("LR/Seg_Head", seg_optimizer.param_groups[1]["lr"], global_step + 1)

            global_step += 1 # 更新全局步数

            # 定期在控制台打印训练信息
            if global_step % args.log_per_steps == 0:
                current_time = datetime.now()
                elapsed_time = current_time - log_start_time
                log_start_time = current_time # 重置计时器
                
                if is_student_phase:
                    print(
                        f"训练步数 {global_step}/{args.steps} | "
                        f"阶段: 学生网络 | "
                        f"余弦损失: {round(float(cosine_loss_val), 4)} | "
                        f"耗时: {elapsed_time}"
                    )
                else:
                    # 分割阶段：根据是否使用自适应权重显示不同信息
                    if adaptive_loss_fn:
                        print(
                            f"训练步数 {global_step}/{args.steps} | "
                            f"阶段: 分割网络 | "
                            f"Focal损失: {round(float(focal_loss_val), 4)} | "
                            f"Dice损失: {round(float(dice_loss_val), 4)} | "
                            f"Focal权重: {round(float(lambda_focal), 2)} | "
                            f"Dice权重: {round(float(lambda_dice), 2)} | "
                            f"总损失: {round(float(total_loss_val), 4)} | "
                            f"耗时: {elapsed_time}"
                        )
                    else:
                        print(
                            f"训练步数 {global_step}/{args.steps} | "
                            f"阶段: 分割网络 | "
                            f"Focal损失: {round(float(focal_loss_val), 4)} | "
                            f"Dice损失: {round(float(dice_loss_val), 4)} | "
                            f"总损失: {round(float(total_loss_val), 4)} | "
                            f"耗时: {elapsed_time}"
                        )

            # --- 阶段切换：加载最佳 S-T 模型 ---
            if global_step == args.de_st_steps:
                if os.path.exists(best_st_state_path):
                    print(f"--- [Phase Switch] Loading best S-T model (Loss: {best_st_loss:.6f}) for Segmentation training ---")
                    # 加载参数
                    # 注意：加载参数时需要加载到 real_model (即 model.module)，而不是 DataParallel 包装后的 model
                    # 因为 state_dict 是从 real_model 保存的
                    real_model.load_state_dict(torch.load(best_st_state_path))
                    # 重新将模型放到设备上 (以防万一)
                    real_model.to(device)

            # 定期评估模型性能
            # 修改评估策略：仅在分割训练阶段进行定期评估，并在阶段开始和结束时强制评估
            should_run_eval = False
            if args.val_rod_dir:
                # 1. 分割阶段开始前 (Phase 1 刚结束)
                if global_step == args.de_st_steps:
                    should_run_eval = True
                
                # 2. 分割阶段中的周期性评估 或 训练结束时
                elif global_step > args.de_st_steps:
                    steps_in_phase2 = global_step - args.de_st_steps
                    if steps_in_phase2 % args.eval_per_steps == 0 or global_step == args.steps:
                        should_run_eval = True

            if should_run_eval:
                print(f"--- Running evaluation at step {global_step} ---")
                
                try:
                    # 估算分割阶段的总评估次数
                    seg_phase_steps = args.steps - args.de_st_steps
                    # +1 是为了包含 baseline (step == de_st_steps) 那一次
                    total_evals = seg_phase_steps // args.eval_per_steps + 1
                    # 如果最后一步不能被整除，则最后一步会额外执行一次评估，总数+1
                    if seg_phase_steps % args.eval_per_steps != 0:
                        total_evals += 1

                    # 计算当前是分割阶段的第几次评估
                    # Phase 1 结束时 (Phase 2 开始前) 为第 1 次
                    if global_step == args.de_st_steps:
                        current_eval_idx = 1
                    elif global_step == args.steps:
                        # 如果是最后一步，直接设为最大索引
                        current_eval_idx = total_evals
                    else:
                        # 普通周期性评估
                        # 举例: de_st=500, eval=100. step=600 -> (100)//100 + 1 = 2.
                        current_eval_idx = (global_step - args.de_st_steps) // args.eval_per_steps + 1


                    # Determine if we should visualize this step
                    should_vis = False
                    if args.vis_steps:
                         target_indices = set()
                         for s in args.vis_steps:
                             if s == 0: continue
                             if s > 0:
                                 target_indices.add(s)
                             else:
                                 # e.g. total=5, s=-1 -> 5. s=-2 -> 4.
                                 target_indices.add(total_evals + 1 + s)
                         
                         if current_eval_idx in target_indices:
                             should_vis = True
                    should_vis = should_vis or global_step == args.steps
                    
                    vis_save_dir = None
                    if should_vis:
                        vis_save_dir = os.path.join(args.vis_dir, run_name, "gt_vs_pred")
                        if not os.path.exists(vis_save_dir):
                            os.makedirs(vis_save_dir)

                    # Determine if we should calc aupro
                    # Calculate AUPRO if user requested it OR if it's the final step
                    should_calc_aupro = args.val_calc_aupro or (global_step == args.steps)

                    # evaluate 现在返回指标字典
                    # 注意：直接传入 args 即可，因为我们传入了 dataloader，evaluate 内部不会使用 args 中的路径参数来创建数据集
                    metrics = evaluate(args, model, visualizer, global_step,
                                       vis_gt_pred=should_vis,
                                       vis_save_dir=vis_save_dir,
                                       vis_num_images=args.vis_num_images,
                                       calc_aupro=should_calc_aupro,
                                       dataloader=val_dataloader,
                                       memory_bank=memory_bank)
                    
                    # 仅在第二阶段 (分割训练) 跟踪最佳分割模型
                    # 注意：global_step == args.de_st_steps 时也是评估分割模型(尽管还没开始训练分割头，作为baseline)
                    if global_step >= args.de_st_steps:
                        # 使用 mIoU 作为主要指标 (AUPRO 可能为 NaN)
                        current_metric = metrics.get("mIoU", 0.0)
                        if current_metric > best_seg_metric:
                            best_seg_metric = current_metric
                            print(f"--- New Best Segmentation Model! mIoU: {best_seg_metric:.4f} ---")
                            # 保存模型权重
                            # 注意：如果 model 是 DataParallel，需要保存 model.module.state_dict()
                            # 我们统一使用 real_model.state_dict()
                            torch.save(real_model.state_dict(), best_seg_state_path)
                            
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 恢复训练模式
                # 注意：如果 model 是 DataParallel，需要访问 model.module 来设置子模块的模式
                # 或者直接使用 real_model (它已经指向了 model.module 或 model)
                if global_step < args.de_st_steps:
                    real_model.student_net.train()
                    real_model.segmentation_net.eval()
                else:
                    real_model.student_net.eval()
                    real_model.segmentation_net.train()

            # 检查是否达到总训练步数，如果是，则终止训练
            if global_step >= args.steps:
                flag = False
                break

    # --- 8. 保存模型 ---
    # 训练结束后，保存最终的模型。
    # 优先保存训练过程中通过 evaluate 选出的最佳模型 (best_seg_state_path)。
    # 如果没有生成最佳模型（例如未运行评估），则保存当前最后一步的模型。
    
    final_model_path = os.path.join(args.checkpoint_dir, run_name + ".pckl")
    
    if os.path.exists(best_seg_state_path):
        print(f"--- 训练完成，正在将最佳模型 (mIoU: {best_seg_metric:.4f}) 移动为最终结果 ---")
        # 复制/移动为最终名称
        shutil.copy(best_seg_state_path, final_model_path)
        # 删除中间文件
        os.remove(best_seg_state_path)
    else:
        print(f"--- 训练完成，未找到最佳模型记录，保存当前模型至: {final_model_path} ---")
        # 保存模型权重 (使用 real_model 确保兼容性)
        torch.save(real_model.state_dict(), final_model_path)
        
    # 清理中间过程保存的 S-T 模型
    if os.path.exists(best_st_state_path):
        print(f"--- 清理中间 S-T 模型: {os.path.basename(best_st_state_path)} ---")
        os.remove(best_st_state_path)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"--- 训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"--- 训练总时长: {duration} ---")

    # 确保所有TensorBoard数据写入磁盘
    visualizer.flush()
    visualizer.close()

    # --- 9. 自动绘制并保存 Loss/Metric 曲线 ---
    print("--- 正在绘制并保存 Loss 和评估指标曲线 ---")
    # 最终保存的目录是 args.vis_dir/run_name/metrics
    vis_save_dir = os.path.join(args.vis_dir, run_name, "metrics")
    save_metric_plots(log_dir, vis_save_dir)


if __name__ == "__main__":
    # --- 命令行参数定义 ---
    # 使用argparse库来定义和解析命令行参数，这使得脚本的配置更加灵活。
    parser = argparse.ArgumentParser(description='使用DeSTSeg在ROD数据集上进行多分类分割')

    # -- 硬件与并行化参数 --
    parser.add_argument("--gpu_id", type=int, nargs='+', default=[0], help="指定使用的GPU编号。单卡如 '0'，多卡如 '0 1'，使用CPU如 '-1'")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载器使用的工作进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于复现实验结果")

    # -- 数据集路径参数 (训练) --
    # default路径使用了 'r' 前缀来表示原始字符串，避免反斜杠 `\` 被错误解析
    parser.add_argument("--rod_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\train\good_7500", help="训练集基准图像目录路径 (父目录，包含 images/)")
    parser.add_argument("--scratch_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\train\scratch_200", help="划痕缺陷（scratch）的目录路径")
    parser.add_argument("--dent_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\train\dent_300", help="凹痕缺陷（dent）的目录路径")
    parser.add_argument("--dotted_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\train\dotted_100", help="点状缺陷（dotted）的目录路径 (可选)")
    parser.add_argument("--use_real_train_data", action='store_true', help="训练时是否使用真实缺陷数据 (默认为False，即使用合成)")

    # -- 数据集路径参数 (验证) --
    parser.add_argument("--val_rod_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\eval\good_900", help="验证集图像目录路径 (用于训练中评估)")
    parser.add_argument("--val_scratch_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\eval\scratch_79", help="验证集划痕缺陷目录 (可选)")
    parser.add_argument("--val_dent_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\eval\dent_50", help="验证集凹痕缺陷目录 (可选)")
    parser.add_argument("--val_dotted_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\eval\dotted_19", help="验证集点状缺陷目录 (可选)")
    parser.add_argument("--use_real_val_data", action='store_true', help="验证时是否使用真实缺陷数据 (默认为False，即使用合成)")

    # -- 数据增强参数 --
    # 修改：默认开启旋转增强，提供 --no_rotate_90 参数来关闭
    parser.add_argument('--no_rotate_90', action='store_true', help='禁用90度旋转数据增强')
    parser.add_argument('--random_rotate', type=int, default=10, help='随机旋转增强的最大角度 (0表示不启用)')

    # -- D2T 和 PatchCore 记忆库相关参数 --
    parser.add_argument("--use_d2t", action="store_true", help="是否启用 D2T (WCA + PCA) 模块增强特征表示")
    parser.add_argument("--use_patchcore", action="store_true", help="是否启用 PatchCore 记忆库")
    parser.add_argument("--patchcore_ratio", type=float, default=0.01, help="PatchCore 记忆库采样比例")
    parser.add_argument("--memory_bank_source_dir", type=str, default=r"D:\lh\Datasets\ForMyThesis\RodDefect\train\good_75", help="用于构建PatchCore记忆库的原始正常图像(未复制扩充)目录 (通常不包含合成缺陷)")
    parser.add_argument("--use_prebuild_memory_bank", action='store_true', help="是否在训练前使用原始正常图像预构建记忆库 (避免使用增强后的冗余数据)")

    # -- 模型与训练超参数 --
    parser.add_argument("--num_classes", type=int, default=4, help="类别数量")    
    parser.add_argument("--bs", type=int, default=8, help="训练的批量大小 (Batch Size)")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="梯度累积步数，用于模拟更大的Batch Size。实际BS = bs * grad_acc_steps")
    parser.add_argument("--steps", type=int, default=10000, help="总训练步数")
    parser.add_argument("--de_st_steps", type=int, default=2500, help="第一阶段训练学生网络的步数")
    parser.add_argument("--lr_de_st", type=float, default=0.05, help="学生网络的学习率")
    parser.add_argument("--lr_res", type=float, default=0.0001, help="分割网络中ResNet部分的学习率")
    parser.add_argument("--lr_seghead", type=float, default=0.001, help="分割网络中分割头部分的学习率")
    parser.add_argument("--gamma", type=float, default=2, help="Focal Loss中的gamma参数")
    parser.add_argument("--lambda_focal", type=float, default=20.0, help="Focal Loss 的权重系数")
    parser.add_argument("--lambda_dice", type=float, default=1.0, help="Dice Loss 的权重系数")

    # -- 自适应损失权重参数 --
    parser.add_argument("--adaptive_loss_method", type=str, default=None, choices=['dynamic', 'uncertainty', 'autoweight', None],
                        help="自适应损失权重调整方法: 'dynamic'(动态调整), 'uncertainty'(不确定性权重), 'autoweight'(自动权重), None(固定权重)")
    parser.add_argument("--adaptive_adjust_factor", type=float, default=0.5,
                        help="动态权重调整的强度因子 (0-1)，仅在 dynamic 模式下生效。0表示不调整，越大调整越剧烈")
    parser.add_argument("--adaptive_warmup_steps", type=int, default=1000,
                        help="自适应权重的预热步数，在此期间使用基础权重，仅在 dynamic 模式下生效")
    parser.add_argument("--adaptive_smoothing_window", type=int, default=10,
                        help="损失值的平滑窗口大小，用于减少权重波动，仅在 dynamic 模式下生效")
    parser.add_argument("--adaptive_init_log_var", type=float, default=0.0,
                        help="不确定性权重的初始log方差值，仅在 uncertainty 模式下生效")

    # -- 日志与可视化参数 --
    parser.add_argument("--checkpoint_dir", type=str, default="./saved_model/", help="保存模型检查点的目录路径")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg", help="运行名称的前缀")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="保存TensorBoard日志的目录路径")
    parser.add_argument("--vis_dir", type=str, default="./vis/", help="保存可视化图表的目录路径")
    parser.add_argument("--terminal_output_dir", type=str, default="./terminal_output/", help="保存终端输出日志的目录路径")
    parser.add_argument("--log_per_steps", type=int, default=50, help="每N步在控制台记录一次训练信息")
    parser.add_argument("--eval_per_steps", type=int, default=200, help="在训练分割网络的过程中每N步评估一次模型")
    parser.add_argument("--vis_steps", type=int, nargs='+', default=[10, 20, 30, -2, -1], help="指定需要进行详细可视化（GT vs Pred）的评估步骤序号。1表示第一次，-1表示最后一次。设置为0表示不进行。")
    parser.add_argument("--vis_num_images", type=int, default=16, help="每次详细可视化时保存的图像数量")
    parser.add_argument("--val_calc_aupro", action="store_true", help="是否在验证期间计算 AUPRO 指标 (耗时较长，默认关闭，仅在最后一步计算)")

    # 解析命令行传入的参数
    args = parser.parse_args()

    # --- 开始训练 ---
    # 调用主训练函数，并传入解析后的参数。
    # 设备选择逻辑已在train函数内部处理，此处无需额外操作。
    train(args)


import argparse
import os
import shutil
import warnings
import sys
# 忽略所有警告 (包括 torchmetrics 产生的 pkg_resources 弃用警告)
warnings.filterwarnings("ignore")

from datetime import datetime
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# Constants
from constant import (
    RESIZE_SHAPE, 
    NORMALIZE_MEAN_L, 
    NORMALIZE_STD_L, 
    NORMALIZE_MEAN_RGB, 
    NORMALIZE_STD_RGB
)

# Dataset & Model
from data.rod_dataset import RodDataset
from model.destseg import DeSTSeg
from model.model_utils import setup_seed, seed_worker, DualLogger

# Metrics
from model.metrics import MulticlassSegmentationMetrics, MulticlassAUPRO
from visualize import save_metric_plots
from draw import save_visual_comparison


def evaluate(args, model, visualizer, global_step=0, vis_gt_pred=False, vis_save_dir=None, vis_num_images=4, calc_aupro=True, dataloader=None):
    model.eval()
    
    device = next(model.parameters()).device

    # 初始化多分类分割指标计算器
    # 假设类别 0 是背景，通常在计算 mIoU 时会关心背景，但在某些异常检测设定下可能忽略
    # 这里默认计算所有类别，如果需要忽略背景，设置 ignore_index=0
    seg_metrics = MulticlassSegmentationMetrics(
        num_classes=args.num_classes, 
        ignore_index=None  # 或者 args.ignore_index
    ).to(device)
    
    # 初始化多分类 AUPRO 指标 (One-vs-Rest)
    # AUPRO 通常忽略背景类 (index 0)
    if calc_aupro:
        aupro_metric = MulticlassAUPRO(
            num_classes=args.num_classes, 
            ignore_index=0
        ).to(device)

    # Visualization accumulation
    vis_imgs_accum = []
    vis_gt_accum = []
    vis_pred_accum = []

    with torch.no_grad():
        if dataloader is None:
            print(f"Loading test data from: {args.rod_dir}")
            dataset = RodDataset(
                is_train=False,
                rod_dir=args.rod_dir,
                resize_shape=RESIZE_SHAPE,
                normalize_mean_l=NORMALIZE_MEAN_L,
                normalize_std_l=NORMALIZE_STD_L,
                normalize_mean_rgb=NORMALIZE_MEAN_RGB,
                normalize_std_rgb=NORMALIZE_STD_RGB,
            )
            g = torch.Generator()
            if hasattr(args, 'seed'):
                 g.manual_seed(args.seed)
            else:
                 g.manual_seed(42) # fallback
                 
            dataloader = DataLoader(
                dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                worker_init_fn=seed_worker, generator=g
            )
        else:
            print(f"Using pre-loaded dataloader with {len(dataloader.dataset)} samples")

        for _, sample_batched in enumerate(dataloader):
            # 获取输入数据 (移动到设备)
            img_aug_l = sample_batched["img_aug_l"].to(device)     # 学生网络输入 (灰度)
            img_aug_rgb = sample_batched["img_aug_rgb"].to(device) # 教师网络输入 (RGB)
            img_origin_l = sample_batched["img_origin_l"].to(device)
            img_origin_rgb = sample_batched["img_origin_rgb"].to(device)
            mask = sample_batched["mask"].to(torch.int64).to(device)

            # 模型前向传播
            # 注意：测试模式下 img_aug 和 img_origin 是一样的，但 DeSTSeg 接口需要传入
            output_segmentation, _, _ = model(
                img_aug_l, img_aug_rgb, img_origin_l, img_origin_rgb
            )

            # 将预测结果插值到原始 mask 尺寸 (如果 mask 尺寸与模型输出不同)
            # RodDataset 的 mask 已经在预处理时 resize 到了 RESIZE_SHAPE
            # DeSTSeg 输出的 output_segmentation 也是 RESIZE_SHAPE (因为输入是 RESIZE_SHAPE)
            # 但为了保险起见，或者如果 mask 是原始分辨率，这里做一次插值
            if output_segmentation.shape[-2:] != mask.shape[-2:]:
                output_segmentation = F.interpolate(
                    output_segmentation,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # --- 更新指标 ---
            
            # 1. 分割指标 (mIoU, Dice, etc.)
            # MulticlassSegmentationMetrics 内部会自动对 (B, C, H, W) 进行 argmax
            seg_metrics.update(output_segmentation, mask)
            
            # 2. 异常检测指标 (AUPRO)
            # MulticlassAUPRO 需要概率图，所以先做 Softmax
            if calc_aupro:
                probs = torch.softmax(output_segmentation, dim=1)
                aupro_metric.update(probs, mask)
            
            # --- Advanced Visualization (Accumulation) ---
            if vis_gt_pred and len(vis_imgs_accum) < vis_num_images:
                batch_size = img_origin_l.shape[0]
                remaining = vis_num_images - len(vis_imgs_accum)
                num_to_take = min(batch_size, remaining)
                
                # Append to lists (keep as tensors on CPU to save GPU memory if needed, or keep on device)
                # save_visual_comparison expects list of tensors.
                for i in range(num_to_take):
                    vis_imgs_accum.append(img_origin_l[i].cpu())
                    vis_gt_accum.append(mask[i].cpu())
                    
                    # Prediction
                    pred_mask = torch.argmax(output_segmentation[i], dim=0) # (H, W)
                    vis_pred_accum.append(pred_mask.cpu())

        # --- Generate Visualization if accumulated ---
        if vis_gt_pred and vis_save_dir and len(vis_imgs_accum) > 0:
            print(f"--- Generating Visualization for {len(vis_imgs_accum)} images ---")
            
            # Use global_step for filename as requested
            pred_filename = f"pred_step_{global_step}.png"
            pred_save_path = os.path.join(vis_save_dir, pred_filename)
            save_visual_comparison(vis_imgs_accum, vis_pred_accum, pred_save_path, nrow=int(vis_num_images**0.5) if vis_num_images > 1 else 1)
            
            # Save GT only once if it doesn't exist
            gt_save_path = os.path.join(vis_save_dir, "gt.png")
            if not os.path.exists(gt_save_path):
                 save_visual_comparison(vis_imgs_accum, vis_gt_accum, gt_save_path, nrow=int(vis_num_images**0.5) if vis_num_images > 1 else 1)
            
            print(f"Saved visualization to {vis_save_dir}")

        # --- 计算并打印结果 ---
        results = seg_metrics.compute()
        if calc_aupro:
            aupro_score = aupro_metric.compute()
            aupro_val = aupro_score.item()
        else:
            aupro_val = 0.0
        
        # 提取标量值
        mIoU = results["mIoU"].item()
        mDice = results["mDice"].item()
        mFscore = results["mFscore"].item()
        
        # 记录到 TensorBoard
        if visualizer:
            visualizer.add_scalar("Eval/mIoU", mIoU, global_step)
            visualizer.add_scalar("Eval/mDice", mDice, global_step)
            visualizer.add_scalar("Eval/mFscore", mFscore, global_step)
            if calc_aupro:
                visualizer.add_scalar("Eval/AUPRO", aupro_val, global_step)

        print(f"Eval at step {global_step}")
        print("================================")
        print(f"mIoU:    {mIoU:.4f}")
        print(f"mDice:   {mDice:.4f}")
        print(f"mFscore: {mFscore:.4f}")
        if calc_aupro:
            print(f"AUPRO:   {aupro_val:.4f}")
        else:
            print(f"AUPRO:   Skipped")
        print("--------------------------------")
        print("Per-class IoU:", results["per_class_IoU"].cpu().numpy())
        print("================================")
        
        # 清理状态
        seg_metrics.reset()
        if calc_aupro:
            aupro_metric.reset()

        return {
            "mIoU": mIoU,
            "mDice": mDice,
            "mFscore": mFscore,
            "AUPRO": aupro_val
        }


def test(args):
    # --- 1. 确定 Checkpoint 和 Run Name ---
    # 确定 checkpoint 路径
    ckpt_path = os.path.join(args.checkpoint_dir, args.base_model_name + ".pckl")
    if not os.path.exists(ckpt_path):
        # 尝试不带 .pckl 后缀或其他命名格式，或者直接使用 args.checkpoint_dir 如果它是文件
        if os.path.isfile(args.checkpoint_dir):
            ckpt_path = args.checkpoint_dir
        else:
            # 在 Logger 初始化前报错，只能输出到终端
            print(f"Error: Checkpoint not found at {ckpt_path}")
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
    # 从 checkpoint 文件名生成 run_name
    # 获取文件名（不带路径）
    ckpt_filename = os.path.basename(ckpt_path)
    # 去掉后缀名
    ckpt_name_no_ext = os.path.splitext(ckpt_filename)[0]
    
    run_name = f"Test_{ckpt_name_no_ext}"
    
    # --- 2. 配置 DualLogger ---
    if not os.path.exists(args.terminal_output_dir):
        os.makedirs(args.terminal_output_dir)
    terminal_log_path = os.path.join(args.terminal_output_dir, run_name + ".txt")
    
    original_stdout = sys.stdout
    sys.stdout = DualLogger(terminal_log_path)
    print(f"--- 终端输出将同时保存至: {terminal_log_path} ---")

    setup_seed(args.seed)
    start_time = datetime.now()
    print(f"--- 测试开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 确定计算设备
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    log_dir = os.path.join(args.log_dir, run_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    visualizer = SummaryWriter(log_dir=log_dir)

    # 初始化模型并移动到指定设备
    model = DeSTSeg(dest=True, ed=True, num_classes=args.num_classes).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    # 这里的 map_location 确保权重能加载到正确的设备上（即使用户在 CPU 机器上加载 GPU 训练的权重）
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    # --- Visualization Setup ---
    vis_save_dir = os.path.join(args.vis_dir, run_name, "gt_vs_pred")
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)

    # 传递 vis_gt_pred=True 以及相关参数，以便在测试时生成可视化图像
    evaluate(args, model, visualizer, global_step=0, vis_gt_pred=args.vis_gt_pred, vis_save_dir=vis_save_dir, vis_num_images=args.vis_num_images)

    # 确保所有数据写入磁盘
    visualizer.flush()
    visualizer.close()

    # --- 自动绘制并保存 Loss/Metric 曲线 ---
    print("--- 正在绘制并保存评估指标曲线 ---")
    vis_save_dir = os.path.join(args.vis_dir, run_name, "metrics")
    save_metric_plots(log_dir, vis_save_dir)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"--- 测试结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"--- 测试总时长: {duration} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0, help="指定使用的GPU编号。设置为 -1 表示使用 CPU。")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载工作线程数")
    
    # 路径参数
    parser.add_argument("--rod_dir", type=str, required=True, help="测试图像目录的路径")
    parser.add_argument("--checkpoint_dir", type=str, default="./saved_model/", help="模型检查点保存目录")
    parser.add_argument("--base_model_name", type=str, default="DeSTSeg_Rod_Best", help="模型文件名前缀")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="TensorBoard 日志目录")
    parser.add_argument("--vis_dir", type=str, default="./vis/", help="可视化结果保存目录")
    parser.add_argument("--terminal_output_dir", type=str, default="./terminal_output/", help="终端输出日志保存目录")

    # 模型参数
    parser.add_argument("--bs", type=int, default=16, help="批量大小")
    parser.add_argument("--num_classes", type=int, default=3, help="类别数量（包含背景）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--vis_gt_pred", action="store_true", help="是否可视化 GT 和预测结果")
    parser.add_argument("--vis_num_images", type=int, default=16, help="可视化的图像数量")

    args = parser.parse_args()

    test(args)

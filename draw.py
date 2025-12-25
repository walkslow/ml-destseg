import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torchvision.utils import make_grid

# 定义颜色表 (R, G, B)
# 0: Background (无色/黑色), 1: Red, 2: Green, 3: Blue, 4: Yellow
COLORS = np.array([
    [0, 0, 0],       # Class 0: Background
    [255, 0, 0],     # Class 1: Scratch (Red)
    [0, 255, 0],     # Class 2: Dent (Green)
    [0, 0, 255],     # Class 3: Dotted (Blue)
    [255, 255, 0]    # Class 4: Other (Yellow)
])

def denormalize(img_tensor):
    """
    将 tensor 转换为可视化的 numpy 数组 (H, W, 3)
    img_tensor: (C, H, W)
    """
    img = img_tensor.detach().cpu().numpy()
    
    # CHW -> HWC
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0)) 
        
    # 如果是单通道灰度，转为 3 通道 RGB 以便叠加颜色
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    
    # Min-Max 归一化到 [0, 1]
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
        
    return img

def overlay_mask(img_np, mask_np, alpha=0.5):
    """
    在图像上叠加彩色 mask
    img_np: (H, W, 3) float [0, 1]
    mask_np: (H, W) int class indices
    """
    colored_mask = np.zeros_like(img_np)
    
    # 找出所有非背景像素
    non_background = mask_np > 0
    
    # 填充颜色
    for c in range(1, len(COLORS)):
        # 简单的容错，如果类别超出颜色表范围，循环使用
        color_idx = c if c < len(COLORS) else (c - 1) % (len(COLORS) - 1) + 1
        
        idx = mask_np == c
        colored_mask[idx] = COLORS[color_idx] / 255.0
        
    # 融合: 仅在有 mask 的地方混合
    output = img_np.copy()
    output[non_background] = (1 - alpha) * img_np[non_background] + alpha * colored_mask[non_background]
    
    return output

def save_visual_comparison(images, masks, save_path, nrow=2):
    """
    生成并保存网格图
    images: List of tensors (C, H, W)
    masks: List of tensors (H, W) or (1, H, W)
    save_path: 保存路径 (包含文件名)
    """
    vis_list = []
    
    for img_t, mask_t in zip(images, masks):
        img_np = denormalize(img_t)
        mask_np = mask_t.detach().cpu().numpy().squeeze()
        
        overlay = overlay_mask(img_np, mask_np)
        
        # HWC -> CHW (Tensor)
        vis_tensor = torch.from_numpy(overlay.transpose(2, 0, 1)).float()
        vis_list.append(vis_tensor)
        
    # 拼接成网格
    grid = make_grid(vis_list, nrow=nrow, padding=2)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存
    ndarr = grid.permute(1, 2, 0).numpy()
    # Clip to [0, 1] just in case
    ndarr = np.clip(ndarr, 0, 1)
    
    plt.imsave(save_path, ndarr)

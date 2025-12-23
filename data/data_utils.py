import math

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from PIL import Image
import random

"""The scripts here are copied from DRAEM: https://github.com/VitjanZ/DRAEM"""


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(
    shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: cv2.resize(
        np.repeat(
            np.repeat(
                gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0
            ),
            d[1],
            axis=1,
        ),
        dsize=(shape[1], shape[0]),
    )
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]
    )


rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


def perlin_noise(image, dtd_image, aug_prob=1.0):
    image = np.array(image, dtype=np.float32)
    dtd_image = np.array(dtd_image, dtype=np.float32)
    shape = image.shape[:2]
    min_perlin_scale, max_perlin_scale = 0, 6
    t_x = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    t_y = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    perlin_scalex, perlin_scaley = 2**t_x, 2**t_y

    perlin_noise = rand_perlin_2d_np(shape, (perlin_scalex, perlin_scaley))

    perlin_noise = rot(images=perlin_noise)
    perlin_noise = np.expand_dims(perlin_noise, axis=2)
    threshold = 0.5
    perlin_thr = np.where(
        perlin_noise > threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise),
    )

    img_thr = dtd_image * perlin_thr / 255.0
    image = image / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8
    image_aug = (
        image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
    )
    image_aug = image_aug.astype(np.float32)

    no_anomaly = torch.rand(1).numpy()[0]

    if no_anomaly > aug_prob:
        return image, np.zeros_like(perlin_thr)
    else:
        msk = (perlin_thr).astype(np.float32)
        msk = msk.transpose(2, 0, 1)

        return image_aug, msk


def cut_paste(normal_image, scratch_image, scratch_mask, dent_image, dent_mask, dotted_image, dotted_mask, min_area_ratio=1/8000):
    """
    将不同类型的缺陷（划痕、凹痕、斑点）通过 "cut and paste" 的方式增强到一张正常的图像上。
    该函数遵循以下步骤：
    1. 从每种缺陷的源图像中，随机裁剪一小块缺陷区域。
    2. 在正常的背景图上，随机寻找一个尚未被其他缺陷占用的“干净”位置。
    3. 将裁剪下的缺陷块粘贴到该位置，并同步更新图像和多类别掩码。
    4. 对所有提供的缺陷类型重复此过程。

    Args:
        normal_image (PIL.Image): 正常的背景图像，作为粘贴的目标。
        scratch_image (PIL.Image): 包含划痕缺陷的图像。
        scratch_mask (PIL.Image): 划痕缺陷的掩码，缺陷处像素值为1。
        dent_image (PIL.Image): 包含凹痕缺陷的图像。
        dent_mask (PIL.Image): 凹痕缺陷的掩码，缺陷处像素值为2。
        dotted_image (PIL.Image): 包含斑点缺陷的图像。
        dotted_mask (PIL.Image): 斑点缺陷的掩码，缺陷处像素值为3。
        min_area_ratio (float): 最小缺陷面积比例阈值，默认为1/8000。

    Returns:
        tuple: 包含两个 PIL.Image 对象的元组:
               - aug_image (PIL.Image): 增强后的图像。
               - aug_mask (PIL.Image): 增强后对应的多类别掩码 (0:背景, 1:划痕, 2:凹痕, 3:斑点)。
    """
    # 步骤 1: 初始化
    # 将输入的 PIL Image 转换为 numpy 数组以便于处理。
    # aug_image 用于叠加缺陷图像，aug_mask 用于叠加缺陷掩码。
    aug_image = np.array(normal_image.copy())
    aug_mask = np.zeros_like(aug_image, dtype=np.uint8)

    def process_defect(defect_image, defect_mask, class_id, max_attempts=20):
        """
        处理单个缺陷类型：从随机连通域裁剪、寻找位置、粘贴。
        这是一个内联函数，可以直接修改外部作用域的 aug_image 和 aug_mask。
        """
        nonlocal aug_image, aug_mask
        # 如果缺陷图像或掩码不存在，则直接跳过。
        if defect_image is None or defect_mask is None:
            return

        defect_image_np = np.array(defect_image)
        # defect_mask 必须是 uint8 才能用于 connectedComponentsWithStats
        defect_mask_np = np.array(defect_mask, dtype=np.uint8)

        # 步骤 2: 找到所有缺陷连通域并随机选择一个。
        # 使用 OpenCV 的 connectedComponentsWithStats 来识别所有独立的缺陷区域。
        # 这能处理一个掩码包含多个不相连缺陷块的情况。
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_mask_np, connectivity=8)

        if num_labels <= 1:  # num_labels=1 表示只有背景，没有缺陷。
            return

        # 提取所有缺陷连通域的边界框和面积 (忽略背景标签0)
        defect_bboxes = [stats[i] for i in range(1, num_labels)]
        
        # 计算正常图像的总面积
        img_h, img_w = aug_image.shape
        normal_image_area = img_h * img_w
        
        # 计算最小缺陷面积阈值
        min_defect_area = normal_image_area * min_area_ratio
        
        # 所有连通域都是合适的，但根据面积采用不同的处理策略
        # 随机选择一个缺陷连通域进行处理
        selected_stat = random.choice(defect_bboxes)
        
        x_min, y_min, rect_w, rect_h = selected_stat[cv2.CC_STAT_LEFT], selected_stat[cv2.CC_STAT_TOP], selected_stat[cv2.CC_STAT_WIDTH], selected_stat[cv2.CC_STAT_HEIGHT]
        defect_area = selected_stat[cv2.CC_STAT_AREA]

        # 步骤 3: 根据缺陷面积与正常图像面积的比例动态调整裁剪策略
        # 计算当前缺陷面积与正常图像面积的比例
        area_ratio = defect_area / normal_image_area
        
        # 如果缺陷面积小于阈值，直接裁剪整个连通域
        if area_ratio < min_area_ratio:
            sub_w, sub_h = rect_w, rect_h
            offset_x, offset_y = 0, 0
        else:
            # 对于较大的缺陷，在外接矩形内随机生成一个更小的子矩形
            # 动态调整裁剪比例下限，使得裁剪后的面积至少为min_defect_area
            # 计算满足最小面积要求的最小裁剪比例
            rect_area = rect_w * rect_h
            min_crop_ratio = min_defect_area / rect_area
            
            # 确保裁剪比例在合理范围内
            min_crop_ratio = max(min_crop_ratio, 0.3)  # 最小不低于0.3
            min_crop_ratio = min(min_crop_ratio, 0.9)  # 最大不超过0.9
            
            if rect_h > rect_w:  # 如果高是长边
                sub_h = int(rect_h * random.uniform(min_crop_ratio, 1.0))
                sub_w = rect_w
                offset_y = np.random.randint(0, rect_h - sub_h + 1) if rect_h > sub_h else 0
                offset_x = 0
            else:  # 如果宽是长边或等长
                sub_w = int(rect_w * random.uniform(min_crop_ratio, 1.0))
                sub_h = rect_h
                offset_x = np.random.randint(0, rect_w - sub_w + 1) if rect_w > sub_w else 0
                offset_y = 0

        # 计算子矩形在源图坐标系中的实际起止位置
        sub_rect_y_start = y_min + offset_y
        sub_rect_y_end = sub_rect_y_start + sub_h
        sub_rect_x_start = x_min + offset_x
        sub_rect_x_end = sub_rect_x_start + sub_w

        # 步骤 4: 从源缺陷图像和掩码中，根据子矩形坐标裁剪出一小块。
        cut_defect = defect_image_np[sub_rect_y_start:sub_rect_y_end, sub_rect_x_start:sub_rect_x_end]
        # temp_mask 是裁剪出的小块缺陷掩码，它保留了原始缺陷的精确形状。
        temp_mask = defect_mask_np[sub_rect_y_start:sub_rect_y_end, sub_rect_x_start:sub_rect_x_end]

        # 检查裁剪出的区域是否有效
        if temp_mask.shape[0] == 0 or temp_mask.shape[1] == 0:
            return # 如果裁剪出空区域，则中止此次粘贴

        cut_h, cut_w = temp_mask.shape
        img_h, img_w = aug_image.shape

        # 步骤 5: 寻找一个不重叠的粘贴位置。
        # 为了防止在拥挤的图像上无限循环，设置一个最大尝试次数。
        for _ in range(max_attempts):
            # 随机选择一个粘贴位置的左上角坐标。
            paste_y = np.random.randint(0, img_h - cut_h + 1) if img_h > cut_h else 0
            paste_x = np.random.randint(0, img_w - cut_w + 1) if img_w > cut_w else 0
            
            # 获取 aug_mask 中对应的目标粘贴区域。
            target_region_in_aug_mask = aug_mask[paste_y:paste_y+cut_h, paste_x:paste_x+cut_w]
            
            # 步骤 6: 检查重叠（核心逻辑）。
            # 我们将目标区域的掩码 `target_region_in_aug_mask` 与我们要粘贴的缺陷掩码 `temp_mask` 进行逐元素相乘。
            # `temp_mask > 0` 会生成一个布尔掩码，代表新缺陷的位置。
            # 如果相乘结果的所有元素都为0，说明在 `temp_mask` 为缺陷的区域，`target_region_in_aug_mask` 都是背景(0)，即没有重叠。
            if np.all((target_region_in_aug_mask * (temp_mask > 0)) == 0):
                
                # 步骤 7: 执行粘贴操作。
                # `paste_shape_mask` 是一个布尔掩码，精确定义了要粘贴的缺陷形状。
                paste_shape_mask = temp_mask > 0
                
                # 获取 aug_image 和 aug_mask 中将被修改的区域的引用。
                target_image_region = aug_image[paste_y:paste_y+cut_h, paste_x:paste_x+cut_w]
                target_mask_region = aug_mask[paste_y:paste_y+cut_h, paste_x:paste_x+cut_w]

                # 使用 np.where 根据 `paste_shape_mask` 更新图像和掩码。
                # 在 `paste_shape_mask` 为 True 的地方，使用 `cut_defect` (图像) 和 `class_id` (掩码)。
                # 在 `paste_shape_mask` 为 False 的地方，保持 `target_image_region` 和 `target_mask_region` 不变。
                aug_image[paste_y:paste_y+cut_h, paste_x:paste_x+cut_w] = np.where(
                    paste_shape_mask, cut_defect, target_image_region
                )
                aug_mask[paste_y:paste_y+cut_h, paste_x:paste_x+cut_w] = np.where(
                    paste_shape_mask, class_id, target_mask_region
                )
                
                # 成功粘贴后，立即返回，处理下一个缺陷类型。
                return

    # 步骤 8: 按顺序处理所有类型的缺陷。
    defects = [
        (scratch_image, scratch_mask, 1),  # 类别1: 划痕
        (dent_image, dent_mask, 2),        # 类别2: 凹痕
        (dotted_image, dotted_mask, 3)     # 类别3: 斑点
    ]

    for defect_img, defect_msk, class_id in defects:
        process_defect(defect_img, defect_msk, class_id)

    # 步骤 9: 返回最终增强后的图像和掩码，并转换回 PIL.Image 格式。
    return Image.fromarray(aug_image), Image.fromarray(aug_mask)

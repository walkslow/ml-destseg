# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

本仓库基于 DeSTSeg (CVPR 2023)，已**从原始 MVTec AD 异常检测改造为燃料棒 (ROD) 缺陷的多类分割**。README 中关于 MVTec 的部分已过时——真正的训练入口数据集是 [data/rod_dataset.py](data/rod_dataset.py)，类别为 背景 + scratch / dent / dotted（默认 `--num_classes 4`），缺陷通过 `cut_paste` 在线合成或加载真实标注（`--use_real_train_data`）。

## 常用命令

```bash
# 安装依赖（torch==2.0.0、numpy==1.24.2 等版本被固定）
pip install -r requirements.txt

# 训练（默认数据路径硬编码为 D:\lh\Datasets\ForMyThesis\RodDefect\...）
python train.py --gpu_id 0 --num_workers 16

# 仅 CPU / 多卡（自动启用 DataParallel）
python train.py --gpu_id -1
python train.py --gpu_id 0 1

# 测试单个 checkpoint（默认从 ./saved_model/<base_model_name>.pckl 加载）
python eval.py --gpu_id 0 --rod_dir <test_dir> --base_model_name <ckpt_name>

# 批量实验（每个脚本内部循环 subprocess 调用 train.py）
python run_loss_experiments.py              # 损失权重 / 自适应损失对比
python run_patchcore_experiments.py         # D2T × PatchCore × Loss 的 8 组
python run_feature_selection_experiments.py # AFS/RRS 比例对比

# 训练曲线
tensorboard --logdir=./logs/
```

没有测试套件、没有 lint 配置。验证依赖 TensorBoard 指标 (`mIoU` / `mDice` / `mFscore` / `AUPRO`) 和 `vis/<run_name>/gt_vs_pred/` 下的可视化图。

## 高层架构

### 两阶段训练 ([train.py](train.py))

单一 `while` 循环按 `global_step` 切阶段，**不用 epoch**：

| 阶段 | 步数范围 | 训练对象 | 损失 | 评估 |
|---|---|---|---|---|
| Phase 1 | `0 ~ de_st_steps` | `student_net` | `cosine_similarity_loss(output_de_st_list)` | 不评估 |
| Phase 2 | `de_st_steps ~ steps` | `segmentation_net` | `λ_focal·focal + λ_dice·dice` | 每 `eval_per_steps` 一次 |

阶段切换点 (`global_step == args.de_st_steps`) 会顺序做三件事：
1. 把 Phase 1 中保存的最佳 S-T 权重 (`*_best_st.pckl`) 加载回模型；
2. 如果 `--use_afs`，调用 `real_model.run_afs(dataloader, device, ratio)` **一次性**更新 `channel_mask`；
3. 强制做一次 baseline 评估作为 Phase 2 起点。

最佳 Phase 2 模型按 `mIoU` 选出，训练结束会被复制为 `<run_name>.pckl` 作为最终产物，`*_best_st.pckl` / `*_best_seg.pckl` 中间文件会被删除。

### DeSTSeg 主模型 ([model/destseg.py](model/destseg.py))

`forward(img_aug_l, img_aug_rgb, img_origin_l=None, img_origin_rgb=None, memory_bank=None)` 返回四元组：`(output_segmentation, output_de_st, output_de_st_list, patchcore_features)`。

进入 `segmentation_net` 的融合特征通道数 `seg_inplanes` 由开关组合决定，**改通道顺序前必须同步以下逻辑**：
- 基础：`448` = 64+128+256（三个尺度的 `-teacher · student` 差异特征）
- `--use_d2t`：×2 → `896`（与 `D2T_Attention` 输出在通道维拼接）
- `--use_patchcore`：+1 → 末尾追加 `patchcore_map`

**PatchCore 通道恒定为最后一维**，AFS/RRS 把索引硬编码成 `c-1` 并强制保留（`run_afs` 把分数设为 `max+1.0`、`apply_rrs` 同理）。`channel_mask` 是注册的 buffer（形状 `[1, seg_inplanes, 1, 1]`，初始全 1），随 `state_dict` 保存/加载，因此**老 checkpoint 与新模型的开关组合必须匹配**否则 buffer 形状不一致。

Phase 1 计算 cosine loss 仅依赖 `output_de_st_list`，不接入 `output_segmentation`——所以 RRS 即使在 Phase 1 开启也不会污染学生训练（仅浪费前向算力）。

### PatchCore 记忆库 ([model/patchcore_mem.py](model/patchcore_mem.py))

记忆库构建两种入口，互斥：
1. **预构建** (`--use_prebuild_memory_bank --memory_bank_source_dir <dir>`)：训练前用 `MemoryBankSourceDataset`（仅原图无合成缺陷）跑一遍 `model(memory_bank=None)` 收集特征 → `MemoryBank.fit()` 做 coreset 采样 → 保存为 `<run_name>_memory_bank_prebuilt.pt`。
2. **训练中构建**（默认）：Phase 1 边训练边累积 `patchcore_features`（暂存 CPU），收够 `len(dataset)` 个样本后构建。这要求 `de_st_steps × bs × grad_acc_steps ≥ len(dataset)`，[train.py](train.py) 会自动校正不满足的情况并相应扩大 `steps`。

`memory_bank=None` 是 `forward` 的关键信号：传 `None` → Phase 1 路径，返回 `patchcore_features` 供收集；传非 `None` → Phase 2 路径，调用 `memory_bank.predict()` 得到异常图并拼接到融合特征末尾。

`CoresetSampler` 对 >50 万的输入会先随机下采样防 OOM；特征矩阵超出显存时走 CPU/GPU 混合模式（特征在 CPU、`min_distances` 在 GPU）。

### Run 名称自动前缀

[train.py](train.py) 会按顺序往 `args.run_name_head` 头部插前缀（仅当还没插过）：
1. `Dynamic_<adjust_factor>_` / `Uncertainty_<log_var>_` / `Autoweight_`（自适应损失）
2. `MemB_`（`--use_patchcore`）
3. `D2T_`（`--use_d2t`）

最终格式：`<prefix...><run_name_head>_<steps>_<YYYYMMDDHHMM>`。批量实验脚本依赖这个命名规则区分配置——改动该规则会破坏现有日志/checkpoint 的可追溯性。

### 输出目录约定

每次训练同时写入 4 个目录，由命令行同名参数控制（默认在仓库根下）：
- `saved_model/` — checkpoint (`.pckl`)、记忆库 (`.pt`)
- `logs/<run_name>/` — TensorBoard
- `vis/<run_name>/{gt_vs_pred,metrics}/` — 评估期 GT/Pred 对比图、训练后曲线
- `terminal_output/<run_name>.txt` — `DualLogger` 把 stdout 同步到这里

`.gitignore` 忽略了以上 4 个目录以及 `*experiments.py`、`todo*`、`*.bat`、`*.ps1`——这些都属于本地产物。

## 与代码交互时容易踩的坑

- 数据集路径用 `r"D:\..."` 写死在 argparse default 里，跨机器跑时**必须**显式覆盖。
- `RodDataset` 在 `scratch_dir/dent_dir/dotted_dir` 全为 `None` 时会**自动切到 `use_real_data=True` 并强制关闭旋转增强**——修改这段逻辑时保留 fallback。
- 有效 batch = `bs × grad_acc_steps`，但 `grad_acc_steps` **仅作用于 Phase 2**，Phase 1 不累积梯度。
- 多卡时用 `real_model = model.module` 访问子模块 (`student_net` / `segmentation_net` / `run_afs`)；保存权重统一用 `real_model.state_dict()`，否则 key 会带 `module.` 前缀。
- 全局指令：与用户始终用中文沟通（项目已有的中文注释和文档应保持一致）。

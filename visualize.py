import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tensorboard_scalars(log_dir, tag):
    """从 tensorboard 日志中读取指定 tag 的标量"""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"未找到 event 文件 in {log_dir}")
    
    # 取最新的 event 文件（按修改时间）
    event_file = max(event_files, key=os.path.getmtime)

    # 加载 event 文件
    acc = EventAccumulator(event_file)
    acc.Reload()  # 加载所有数据

    if tag not in acc.Tags()['scalars']:
        raise KeyError(f"Tag '{tag}' 不存在于日志中。可用 tags: {acc.Tags()['scalars']}")

    scalar_events = acc.Scalars(tag)
    steps = [s.step for s in scalar_events]
    values = [s.value for s in scalar_events]
    return steps, values


def plot_tensorboard_losses(
    log_dir: str = "./logs",
    run_name: str = "default_run",
    tags: list = None,
    vis_dir: str = "./vis"
):
    """
    从 TensorBoard 日志中读取指定标量并绘制 loss 曲线图，保存到指定目录。
    兼容 PyTorch 2.0 + tensorboard（无需 SummaryReader）。
    """
    if tags is None:
        tags = ["cosine_loss", "focal_loss", "l1_loss", "total_loss"]

    actual_log_dir = os.path.join(log_dir, run_name)
    save_dir = os.path.join(vis_dir, run_name)
    save_path = os.path.join(save_dir, "all_losses.png")

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(actual_log_dir):
        raise FileNotFoundError(f"日志目录不存在: {actual_log_dir}")

    n_tags = len(tags)
    cols = 2
    rows = (n_tags + 1) // 2

    plt.figure(figsize=(6 * cols, 4 * rows))

    for i, tag in enumerate(tags, 1):
        try:
            steps, values = read_tensorboard_scalars(actual_log_dir, tag)
            plt.subplot(rows, cols, i)
            plt.plot(steps, values, linewidth=1.2)
            plt.title(tag, fontsize=12)
            plt.xlabel("Global Step")
            plt.ylabel("Value")
            plt.grid(True, linestyle="--", alpha=0.7)
        except Exception as e:
            print(f"⚠️ 读取 tag '{tag}' 失败: {e}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def main():
    parser = argparse.ArgumentParser(description="从 TensorBoard 日志绘制并保存 loss 曲线（兼容 PyTorch 2.0）")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="TensorBoard 日志根目录 (默认: ./logs)")
    parser.add_argument("--run_name", type=str, required=True,
                        help="运行名称，用于拼接实际日志路径: log_dir/run_name")
    parser.add_argument("--tags", nargs="+",
                        default=["cosine_loss", "focal_loss", "l1_loss", "total_loss"],
                        help='要绘制的标量标签列表')
    parser.add_argument("--vis_dir", type=str, default="./vis",
                        help="loss 曲线图保存目录 (默认: ./vis)")

    args = parser.parse_args()

    save_path = plot_tensorboard_losses(
        log_dir=args.log_dir,
        run_name=args.run_name,
        tags=args.tags,
        vis_dir=args.vis_dir
    )
    print(f"✅ Loss 曲线已保存至: {save_path}")


if __name__ == "__main__":
    main()
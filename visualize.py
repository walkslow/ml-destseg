import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_all_scalars(log_dir):
    """
    获取 TensorBoard 日志中所有的标量数据
    返回: {tag_name: (steps, values)} 字典
    """
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"未找到 event 文件 in {log_dir}")
    
    # 取最新的 event 文件（按修改时间）
    event_file = max(event_files, key=os.path.getmtime)
    print(f"Reading event file: {event_file}")

    # 加载 event 文件
    # size_guidance 指定加载所有数据，避免采样
    acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
    acc.Reload()

    tags = acc.Tags()['scalars']
    data = {}
    
    for tag in tags:
        scalar_events = acc.Scalars(tag)
        steps = [s.step for s in scalar_events]
        values = [s.value for s in scalar_events]
        data[tag] = (steps, values)
        
    return data


def save_metric_plots(log_dir, save_dir):
    """
    读取 log_dir 下的所有 scalar 指标，并为每个指标单独绘制曲线图，保存到 save_dir。
    图像名称将对应指标名称（斜杠替换为下划线）。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    try:
        data = get_all_scalars(log_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Found {len(data)} metrics. Saving plots to {save_dir}...")

    for tag, (steps, values) in data.items():
        # 清理 tag 名称作为文件名 (e.g. "Loss/Total_Loss" -> "Loss_Total_Loss")
        safe_tag_name = tag.replace("/", "_").replace("\\", "_").replace(" ", "_")
        file_path = os.path.join(save_dir, f"{safe_tag_name}.png")
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=2)
        plt.title(tag)
        plt.xlabel("Global Step")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="从 TensorBoard 日志绘制并保存所有指标曲线")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="TensorBoard 日志目录 (包含 events.out.tfevents...)")
    parser.add_argument("--vis_dir", type=str, default="./vis",
                        help="图像保存根目录")
    parser.add_argument("--run_name", type=str, default=None,
                        help="可选：如果提供，保存目录将为 vis_dir/run_name")

    args = parser.parse_args()

    # 确定保存路径
    save_path = args.vis_dir
    if args.run_name:
        save_path = os.path.join(args.vis_dir, args.run_name)
        
    save_metric_plots(args.log_dir, save_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _viz_utils import configure_matplotlib
from _viz_utils import ensure_output_dir
from _viz_utils import load_pickle
from _viz_utils import route_length_km
from _viz_utils import save_figure
from _viz_utils import to_coord_array
from _viz_utils import to_local_datetime

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成任务 B 行程时间估计可视化图片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT_DIR / "task_B_tte" / "val_input.pkl",
        help="输入 PKL 文件路径",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=ROOT_DIR / "task_B_tte" / "val_gt.pkl",
        help="真实标签 PKL 文件路径",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT_DIR / "outputs" / "tte_visualization",
        help="图片输出目录",
    )
    return parser.parse_args()


def collect_statistics(input_trajectories: list[dict], gt_map: dict) -> dict[str, np.ndarray]:
    travel_times: list[float] = []
    departure_hours: list[int] = []
    point_counts: list[int] = []
    route_lengths: list[float] = []

    for item in input_trajectories:
        traj_id = item["traj_id"]
        gt_item = gt_map[traj_id]
        coords = to_coord_array(item["coords"])

        travel_times.append(float(gt_item["travel_time"]))
        departure_hours.append(to_local_datetime(int(item["departure_timestamp"])).hour)
        point_counts.append(len(coords))
        route_lengths.append(route_length_km(coords))

    return {
        "travel_times": np.asarray(travel_times, dtype=float),
        "departure_hours": np.asarray(departure_hours, dtype=int),
        "point_counts": np.asarray(point_counts, dtype=int),
        "route_lengths": np.asarray(route_lengths, dtype=float),
    }


def plot_travel_time_histogram(travel_times: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(travel_times, bins=40, color="#577590", edgecolor="white", alpha=0.9)
    ax.set_title("行程时间分布")
    ax.set_xlabel("行程时间（秒）")
    ax.set_ylabel("轨迹数量")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_departure_hour_histogram(hours: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(-0.5, 24.5, 1.0)
    ax.hist(hours, bins=bins, color="#90be6d", edgecolor="white", alpha=0.9)
    ax.set_title("出发小时分布（Asia/Shanghai）")
    ax.set_xlabel("小时")
    ax.set_ylabel("轨迹数量")
    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_values, y_values, s=12, alpha=0.35, color="#277da1", edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("行程时间（秒）")
    ax.grid(alpha=0.2)
    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = ensure_output_dir(args.outdir)

    print(f"正在读取输入数据: {args.input}")
    input_trajectories = load_pickle(args.input)
    print(f"正在读取真实数据: {args.gt}")
    gt_trajectories = load_pickle(args.gt)

    if not input_trajectories:
        raise ValueError("输入文件中没有轨迹数据")

    gt_map = {item["traj_id"]: item for item in gt_trajectories}
    stats = collect_statistics(input_trajectories, gt_map)

    plot_travel_time_histogram(stats["travel_times"], output_dir / "travel_time_hist.png")
    plot_departure_hour_histogram(stats["departure_hours"], output_dir / "departure_hour_hist.png")
    plot_scatter(
        stats["point_counts"],
        stats["travel_times"],
        output_dir / "travel_time_vs_point_count.png",
        title="行程时间与轨迹点数关系",
        xlabel="轨迹点数",
    )
    plot_scatter(
        stats["route_lengths"],
        stats["travel_times"],
        output_dir / "travel_time_vs_route_length.png",
        title="行程时间与轨迹长度关系",
        xlabel="轨迹长度（公里）",
    )

    print(f"可视化完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()

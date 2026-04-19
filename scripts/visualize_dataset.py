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
from _viz_utils import sample_items
from _viz_utils import save_figure
from _viz_utils import to_coord_array
from _viz_utils import to_local_datetime
from _viz_utils import valid_coord_array

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成主数据集轨迹探索可视化图片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT_DIR / "data_ds15" / "val.pkl",
        help="输入 PKL 文件路径",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT_DIR / "outputs" / "dataset_visualization",
        help="图片输出目录",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=500,
        help="轨迹抽样上限，用于轨迹折线图",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样种子",
    )
    return parser.parse_args()


def collect_statistics(trajectories: list[dict]) -> dict[str, np.ndarray]:
    point_counts: list[int] = []
    durations_min: list[float] = []
    departure_hours: list[int] = []
    lon_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []

    for item in trajectories:
        coords = to_coord_array(item["coords"])
        timestamps = np.asarray(item["timestamps"], dtype=np.int64).reshape(-1)

        point_counts.append(len(coords))
        if len(timestamps) >= 2:
            durations_min.append(float((timestamps[-1] - timestamps[0]) / 60.0))
            departure_hours.append(to_local_datetime(int(timestamps[0])).hour)
        else:
            durations_min.append(0.0)
            departure_hours.append(0)

        valid_coords = valid_coord_array(coords)
        if len(valid_coords) > 0:
            lon_parts.append(valid_coords[:, 0])
            lat_parts.append(valid_coords[:, 1])

    all_lons = np.concatenate(lon_parts) if lon_parts else np.array([], dtype=float)
    all_lats = np.concatenate(lat_parts) if lat_parts else np.array([], dtype=float)
    return {
        "point_counts": np.asarray(point_counts, dtype=int),
        "durations_min": np.asarray(durations_min, dtype=float),
        "departure_hours": np.asarray(departure_hours, dtype=int),
        "all_lons": all_lons,
        "all_lats": all_lats,
    }


def plot_trajectory_sample(trajectories: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    line_count = 0
    for item in trajectories:
        coords = valid_coord_array(item["coords"])
        if len(coords) < 2:
            continue
        ax.plot(coords[:, 0], coords[:, 1], color="#2a6f97", alpha=0.18, linewidth=0.7)
        line_count += 1

    ax.set_title(f"抽样轨迹折线图（共 {line_count} 条）")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.grid(alpha=0.2)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_point_heatmap(all_lons: np.ndarray, all_lats: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.hexbin(
        all_lons,
        all_lats,
        gridsize=120,
        mincnt=1,
        bins="log",
        cmap="YlOrRd",
    )
    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("点密度（对数）")
    ax.set_title("轨迹点空间热区图")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_histogram(
    values: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    bins,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins, color=color, edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
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


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = ensure_output_dir(args.outdir)

    print(f"正在读取数据: {args.input}")
    trajectories = load_pickle(args.input)
    if not trajectories:
        raise ValueError("输入文件中没有轨迹数据")

    sampled = sample_items(trajectories, args.max_trajs, args.seed)
    stats = collect_statistics(trajectories)

    plot_trajectory_sample(sampled, output_dir / "trajectory_sample.png")
    plot_point_heatmap(stats["all_lons"], stats["all_lats"], output_dir / "point_heatmap.png")
    plot_histogram(
        stats["point_counts"],
        output_dir / "point_count_hist.png",
        title="每条轨迹点数分布",
        xlabel="轨迹点数",
        bins=40,
        color="#577590",
    )
    plot_histogram(
        stats["durations_min"],
        output_dir / "duration_hist.png",
        title="轨迹时长分布",
        xlabel="轨迹时长（分钟）",
        bins=40,
        color="#f3722c",
    )
    plot_departure_hour_histogram(
        stats["departure_hours"],
        output_dir / "departure_hour_hist.png",
    )

    print(f"可视化完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()

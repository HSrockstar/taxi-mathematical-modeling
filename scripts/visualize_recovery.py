from __future__ import annotations

import argparse
import math
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
from _viz_utils import to_bool_array
from _viz_utils import to_coord_array

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成任务 A 轨迹恢复可视化图片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT_DIR / "task_A_recovery" / "val_input_8.pkl",
        help="输入 PKL 文件路径",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=ROOT_DIR / "task_A_recovery" / "val_gt.pkl",
        help="真实轨迹 PKL 文件路径",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT_DIR / "outputs" / "recovery_visualization",
        help="图片输出目录",
    )
    parser.add_argument(
        "--sample-cases",
        type=int,
        default=12,
        help="展示的轨迹样例数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样种子",
    )
    return parser.parse_args()


def collect_statistics(trajectories: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    observed_ratios: list[float] = []
    missing_counts: list[int] = []

    for item in trajectories:
        mask = to_bool_array(item["mask"])
        observed_ratios.append(float(mask.mean() * 100.0))
        missing_counts.append(int((~mask).sum()))

    return np.asarray(observed_ratios, dtype=float), np.asarray(missing_counts, dtype=int)


def plot_recovery_cases(sampled: list[dict], gt_map: dict, output_path: Path) -> None:
    subplot_count = max(1, len(sampled))
    cols = min(4, subplot_count)
    rows = math.ceil(subplot_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.2))
    axes = np.atleast_1d(axes).ravel()

    for index, item in enumerate(sampled):
        ax = axes[index]
        gt_item = gt_map[item["traj_id"]]
        gt_coords = to_coord_array(gt_item["coords"])
        input_coords = to_coord_array(item["coords"])
        mask = to_bool_array(item["mask"])

        observed_coords = input_coords[mask]
        missing_coords = gt_coords[~mask]

        ax.plot(gt_coords[:, 0], gt_coords[:, 1], color="#9c9c9c", linewidth=1.0, alpha=0.9)
        if len(observed_coords) > 0:
            ax.scatter(
                observed_coords[:, 0],
                observed_coords[:, 1],
                color="#277da1",
                s=10,
                label="已知点" if index == 0 else None,
            )
        if len(missing_coords) > 0:
            ax.scatter(
                missing_coords[:, 0],
                missing_coords[:, 1],
                color="#d62828",
                s=10,
                marker="x",
                label="缺失点位置" if index == 0 else None,
            )

        short_id = str(item["traj_id"])[:10]
        ax.set_title(f"轨迹 {short_id}")
        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
        ax.grid(alpha=0.2)

    if sampled:
        axes[0].plot([], [], color="#9c9c9c", linewidth=1.0, label="真实轨迹")
        axes[0].legend(loc="best", fontsize=8)

    for extra_axis in axes[len(sampled) :]:
        extra_axis.axis("off")

    fig.suptitle("轨迹恢复样例对比图", fontsize=14)
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
    sampled = sample_items(input_trajectories, args.sample_cases, args.seed)
    observed_ratios, missing_counts = collect_statistics(input_trajectories)

    plot_recovery_cases(sampled, gt_map, output_dir / "recovery_cases.png")
    plot_histogram(
        observed_ratios,
        output_dir / "observed_ratio_hist.png",
        title="每条轨迹保留率分布",
        xlabel="保留率（%）",
        bins=30,
        color="#43aa8b",
    )
    plot_histogram(
        missing_counts,
        output_dir / "missing_count_hist.png",
        title="每条轨迹待恢复点数量分布",
        xlabel="待恢复点数量",
        bins=30,
        color="#f9844a",
    )

    print(f"可视化完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()

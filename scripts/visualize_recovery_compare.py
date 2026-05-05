"""任务 A：在同一张子图上叠加 真实轨迹 + 已知点 + 三档基线预测，对比可视化。

输出 4 张大图（每张 12 个子图）：
- outputs/recovery_comparison/cases_random_8.png   1/8 难度，随机抽 12 条
- outputs/recovery_comparison/cases_hard_8.png     1/8 难度，挑 linear MAE 最大的 12 条
- outputs/recovery_comparison/cases_random_16.png  1/16 难度，随机抽 12 条
- outputs/recovery_comparison/cases_hard_16.png    1/16 难度，挑 linear MAE 最大的 12 条

每个子图内同时画 4 个图层（中文图例）：
- 灰色折线   真实轨迹（每个时间步连起来，作为参考骨架）
- 蓝色实心圆 已知观测点（mask=True 的位置）
- 绿色 ×    线性插值在缺失点的预测
- 橙色 ▲    三次样条在缺失点的预测
- 红色 +    卡尔曼+RTS 在缺失点的预测
- 灰色空心圆 真实的缺失点位置（用来肉眼对比谁离真值更近）

每个子图标题包含：traj_id、缺失点数、各方法在该轨迹上的 MAE（米）。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _viz_utils import (  # noqa: E402
    configure_matplotlib,
    ensure_output_dir,
    haversine_distance_km,
    load_pickle,
    save_figure,
    to_bool_array,
    to_coord_array,
)

METHOD_STYLES = {
    "linear": {"color": "#43aa8b", "marker": "x", "label": "线性插值预测"},
    "spline": {"color": "#f3722c", "marker": "^", "label": "三次样条预测"},
    "kalman": {"color": "#d62828", "marker": "+", "label": "卡尔曼+RTS 预测"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对比可视化：真实轨迹 + 已知点 + 三档基线预测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--level",
        choices=["8", "16"],
        default=None,
        help="难度档（8 或 16）。不传则两档都跑。",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "recovery_comparison",
        help="图片输出目录",
    )
    p.add_argument("--sample-cases", type=int, default=12, help="每张图的子图数量")
    p.add_argument("--seed", type=int, default=42, help="随机抽样种子")
    return p.parse_args()


def per_traj_mae_m(pred_coords: np.ndarray, gt_coords: np.ndarray, mask: np.ndarray) -> float:
    """单条轨迹在 mask=False 处的 Haversine MAE（米）。"""
    missing = ~mask
    if not missing.any():
        return float("nan")
    a = pred_coords[missing]
    b = gt_coords[missing]
    d_km = haversine_distance_km(a, b)
    return float(d_km.mean() * 1000.0)


def index_by_id(items):
    return {item["traj_id"]: item for item in items}


def select_random(rng: np.random.Generator, ids: list, n: int) -> list:
    if n >= len(ids):
        return list(ids)
    picked = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(picked.tolist())]


def select_hard(per_traj_mae: dict, n: int) -> list:
    """按 linear MAE 从大到小取前 n 条 traj_id。"""
    items = sorted(per_traj_mae.items(), key=lambda kv: -kv[1])
    return [tid for tid, _ in items[:n]]


def plot_panel(
    ax: plt.Axes,
    traj_id,
    timestamps: np.ndarray,
    gt_coords: np.ndarray,
    input_coords: np.ndarray,
    mask: np.ndarray,
    pred_dict: dict,
    show_legend: bool = False,
) -> None:
    """单个子图。pred_dict: {method_name: pred_coords (N,2)}."""
    missing = ~mask

    ax.plot(
        gt_coords[:, 0],
        gt_coords[:, 1],
        color="#9c9c9c",
        linewidth=1.0,
        alpha=0.8,
        label="真实轨迹连线",
        zorder=1,
    )

    ax.scatter(
        gt_coords[missing, 0],
        gt_coords[missing, 1],
        s=18,
        facecolors="none",
        edgecolors="#5a5a5a",
        linewidths=0.7,
        label="真实缺失点位置",
        zorder=2,
    )

    obs = input_coords[mask]
    if len(obs) > 0:
        ax.scatter(
            obs[:, 0],
            obs[:, 1],
            color="#277da1",
            s=22,
            label="已知观测点",
            zorder=4,
        )

    for method, style in METHOD_STYLES.items():
        if method not in pred_dict:
            continue
        pcoords = pred_dict[method]
        ax.scatter(
            pcoords[missing, 0],
            pcoords[missing, 1],
            color=style["color"],
            marker=style["marker"],
            s=28,
            linewidths=1.1,
            label=style["label"],
            alpha=0.9,
            zorder=3,
        )

    short_name = {"linear": "线性", "spline": "样条", "kalman": "卡尔曼"}
    head = f"轨迹 {traj_id} · 缺失 {int(missing.sum())} 点"
    mae_parts = []
    for method in ["linear", "spline", "kalman"]:
        if method in pred_dict:
            mae = per_traj_mae_m(pred_dict[method], gt_coords, mask)
            mae_parts.append(f"{short_name[method]} {mae:.0f}m")
    title_text = head + "\nMAE：" + " | ".join(mae_parts)
    ax.set_title(title_text, fontsize=8.5)

    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)

    if show_legend:
        ax.legend(loc="best", fontsize=7, framealpha=0.85)


def make_figure(
    selected_ids: list,
    title: str,
    inputs_map: dict,
    gts_map: dict,
    preds_maps: dict,
    output_path: Path,
) -> None:
    n = len(selected_ids)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.0))
    axes = np.atleast_1d(axes).ravel()

    for idx, tid in enumerate(selected_ids):
        ax = axes[idx]
        in_item = inputs_map[tid]
        gt_item = gts_map[tid]
        timestamps = np.asarray(in_item["timestamps"], dtype=np.float64)
        gt_coords = to_coord_array(gt_item["coords"])
        input_coords = to_coord_array(in_item["coords"])
        mask = to_bool_array(in_item["mask"])

        pred_dict = {
            method: to_coord_array(pmap[tid]["coords"])
            for method, pmap in preds_maps.items()
            if tid in pmap
        }

        plot_panel(
            ax,
            tid,
            timestamps,
            gt_coords,
            input_coords,
            mask,
            pred_dict,
            show_legend=(idx == 0),
        )

    for extra in axes[n:]:
        extra.axis("off")

    fig.suptitle(title, fontsize=14, y=1.0)
    save_figure(fig, output_path)
    plt.close(fig)


def run_level(level: str, outdir: Path, sample_cases: int, seed: int) -> None:
    print(f"\n=== 处理 1/{level} 难度 ===")

    inputs = load_pickle(ROOT / "task_A_recovery" / f"val_input_{level}.pkl")
    gts = load_pickle(ROOT / "task_A_recovery" / "val_gt.pkl")
    preds = {}
    for method in METHOD_STYLES:
        path = ROOT / "submissions" / "task_A" / f"baseline_{method}_{level}.pkl"
        if not path.exists():
            print(f"  [跳过] 找不到 {path.name}")
            continue
        preds[method] = load_pickle(path)
        print(f"  加载 {path.relative_to(ROOT)}：{len(preds[method])} 条")

    if not preds:
        raise FileNotFoundError("没有任何基线预测文件，先跑 scripts/run_task_a_baselines.py")

    inputs_map = index_by_id(inputs)
    gts_map = index_by_id(gts)
    preds_maps = {m: index_by_id(p) for m, p in preds.items()}

    common_ids = list(inputs_map.keys())
    for pmap in preds_maps.values():
        common_ids = [i for i in common_ids if i in pmap]
    print(f"  对齐后可用轨迹数: {len(common_ids)}")

    rng = np.random.default_rng(seed)
    rand_ids = select_random(rng, common_ids, sample_cases)

    print(f"  正在按 linear MAE 排序挑选困难样例…")
    if "linear" in preds_maps:
        per_mae = {}
        for tid in common_ids:
            per_mae[tid] = per_traj_mae_m(
                to_coord_array(preds_maps["linear"][tid]["coords"]),
                to_coord_array(gts_map[tid]["coords"]),
                to_bool_array(inputs_map[tid]["mask"]),
            )
        hard_ids = select_hard(per_mae, sample_cases)
    else:
        hard_ids = rand_ids

    legend_methods = "、".join(s["label"] for s in METHOD_STYLES.values() if any(True for _ in [s]))
    methods_loaded = "、".join(METHOD_STYLES[m]["label"] for m in preds_maps)

    title_random = f"任务 A · 1/{level} 难度 · 随机 {sample_cases} 条 · 方法对比 ({methods_loaded})"
    title_hard = (
        f"任务 A · 1/{level} 难度 · 困难 {sample_cases} 条 (线性 MAE 最高) · 方法对比 ({methods_loaded})"
    )

    make_figure(
        rand_ids,
        title_random,
        inputs_map,
        gts_map,
        preds_maps,
        outdir / f"cases_random_{level}.png",
    )
    print(f"  写出 {outdir / f'cases_random_{level}.png'}")

    make_figure(
        hard_ids,
        title_hard,
        inputs_map,
        gts_map,
        preds_maps,
        outdir / f"cases_hard_{level}.png",
    )
    print(f"  写出 {outdir / f'cases_hard_{level}.png'}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    outdir = ensure_output_dir(args.outdir)

    levels = [args.level] if args.level else ["8", "16"]
    for level in levels:
        run_level(level, outdir, args.sample_cases, args.seed)

    print(f"\n全部图已写入 {outdir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

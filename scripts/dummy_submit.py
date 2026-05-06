"""阶段1 dummy 提交：用最朴素的方法跑通整条流水线。

任务 A：在每条轨迹的 mask=False 位置，按时间戳做经纬度的线性插值（首尾若仍是 NaN
则用最近的已知点外推为常数）；
任务 B：训练集上预先算好 (路径长度 / 行程时间) 的全局平均速度，验证时
        预测 travel_time = route_length_m(coords) / 平均速度。

跑完会：
1. 把预测写到 submissions/{task_A,task_B}/dummy_*.pkl
2. 调用 src.common.validate_submission_* 做格式校验
3. 与 val_gt 对比，打印 MAE / RMSE / MAPE 基线指标
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Mapping

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import (  # noqa: E402
    load_pkl,
    save_pkl,
    validate_submission_a,
    validate_submission_b,
)
from src.common.geo import route_length_m  # noqa: E402
from src.common.metrics import metrics_a, metrics_b  # noqa: E402


def linear_interp_one(input_item: Mapping) -> dict:
    """最朴素的"按时间戳分别对 lon、lat 线性插值"实现。

    注意：这里没做 ENU 投影，仅作 dummy 闭环用。已知点保持原值。
    """
    timestamps = np.asarray(input_item["timestamps"], dtype=np.float64)
    coords = np.asarray(input_item["coords"], dtype=np.float64).copy()
    mask = np.asarray(input_item["mask"], dtype=bool)

    if mask.sum() == 0:
        coords[~mask] = 0.0
        return {"traj_id": input_item["traj_id"], "coords": coords.astype(np.float32)}

    t_known = timestamps[mask]
    lon_known = coords[mask, 0]
    lat_known = coords[mask, 1]

    missing = ~mask
    coords[missing, 0] = np.interp(timestamps[missing], t_known, lon_known)
    coords[missing, 1] = np.interp(timestamps[missing], t_known, lat_known)
    return {"traj_id": input_item["traj_id"], "coords": coords.astype(np.float32)}


def run_task_a(level: str) -> dict:
    """level 取 '8' 或 '16'。"""
    input_path = ROOT / "task_A_recovery" / f"val_input_{level}.pkl"
    gt_path = ROOT / "task_A_recovery" / "val_gt.pkl"
    output_path = ROOT / "submissions" / "task_A" / f"dummy_{level}.pkl"

    inputs: List[dict] = load_pkl(input_path)
    gts: List[dict] = load_pkl(gt_path)

    preds = [linear_interp_one(item) for item in tqdm(inputs, desc=f"任务A 1/{level}")]

    save_pkl(preds, output_path)
    validate_submission_a(preds, inputs)

    metrics = metrics_a(preds, gts, input_list=inputs)
    metrics["level"] = f"1/{level}"
    metrics["output"] = str(output_path.relative_to(ROOT))
    return metrics


def run_task_b() -> dict:
    """全局平均速度基线。"""
    train_path = ROOT / "data_ds15" / "train.pkl"
    input_path = ROOT / "task_B_tte" / "val_input.pkl"
    gt_path = ROOT / "task_B_tte" / "val_gt.pkl"
    output_path = ROOT / "submissions" / "task_B" / "dummy.pkl"

    print("加载训练集，估计全局平均速度…")
    train_data: List[dict] = load_pkl(train_path)

    total_dist_m = 0.0
    total_time_s = 0.0
    for item in tqdm(train_data, desc="估计平均速度"):
        coords = np.asarray(item["coords"], dtype=np.float64)
        ts = np.asarray(item["timestamps"], dtype=np.float64)
        if len(ts) < 2:
            continue
        dt = float(ts[-1] - ts[0])
        if dt <= 0:
            continue
        dist = route_length_m(coords)
        total_dist_m += dist
        total_time_s += dt

    avg_speed_mps = total_dist_m / max(total_time_s, 1e-6)
    print(f"全局平均速度: {avg_speed_mps:.3f} m/s ({avg_speed_mps * 3.6:.2f} km/h)")

    inputs: List[dict] = load_pkl(input_path)
    gts: List[dict] = load_pkl(gt_path)

    preds = []
    for item in tqdm(inputs, desc="任务B"):
        dist = route_length_m(item["coords"])
        tt = dist / avg_speed_mps if avg_speed_mps > 0 else 0.0
        preds.append({"traj_id": item["traj_id"], "travel_time": float(tt)})

    save_pkl(preds, output_path)
    validate_submission_b(preds, inputs)

    metrics = metrics_b(preds, gts)
    metrics["avg_speed_mps"] = avg_speed_mps
    metrics["output"] = str(output_path.relative_to(ROOT))
    return metrics


def main() -> None:
    print("=" * 70)
    print("阶段1 dummy 提交：闭环验证")
    print("=" * 70)

    summary: list[tuple[str, dict]] = []

    for level in ("8", "16"):
        t0 = time.time()
        m = run_task_a(level)
        m["elapsed_s"] = round(time.time() - t0, 2)
        summary.append((f"任务A 1/{level} 线性插值", m))

    t0 = time.time()
    m = run_task_b()
    m["elapsed_s"] = round(time.time() - t0, 2)
    summary.append(("任务B 全局速度基线", m))

    print()
    print("=" * 70)
    print("dummy 基线指标汇总")
    print("=" * 70)
    for name, m in summary:
        print(f"\n[{name}]")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

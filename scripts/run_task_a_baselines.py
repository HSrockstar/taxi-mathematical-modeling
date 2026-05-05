"""阶段2 任务 A 三档基线对比：linear / spline / kalman × 1/8 / 1/16。

输出 6 个 pkl 到 submissions/task_A/baseline_{method}_{level}.pkl，
并打印对比表（也保存到 outputs/task_a_baseline_metrics.csv）。
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import load_pkl, save_pkl, validate_submission_a  # noqa: E402
from src.common.metrics import metrics_a  # noqa: E402
from src.task_a.baseline_interp import METHOD_REGISTRY  # noqa: E402


METHODS = ["linear", "spline", "kalman"]
LEVELS = ["8", "16"]


def run_one(method: str, level: str, gts: list, output_dir: Path) -> dict:
    input_path = ROOT / "task_A_recovery" / f"val_input_{level}.pkl"
    inputs = load_pkl(input_path)

    fn = METHOD_REGISTRY[method]
    t0 = time.time()
    preds = [fn(item) for item in tqdm(inputs, desc=f"{method:>6} 1/{level}", leave=False)]
    elapsed = time.time() - t0

    out_path = output_dir / f"baseline_{method}_{level}.pkl"
    save_pkl(preds, out_path)
    validate_submission_a(preds, inputs)

    m = metrics_a(preds, gts, input_list=inputs)
    return {
        "method": method,
        "level": f"1/{level}",
        "MAE_m": round(m["MAE"], 3),
        "RMSE_m": round(m["RMSE"], 3),
        "n_points": m["n_points"],
        "n_traj": m["n_traj"],
        "elapsed_s": round(elapsed, 2),
        "output": str(out_path.relative_to(ROOT)),
    }


def main() -> None:
    output_dir = ROOT / "submissions" / "task_A"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载 val_gt …")
    gts = load_pkl(ROOT / "task_A_recovery" / "val_gt.pkl")

    rows = []
    for level in LEVELS:
        for method in METHODS:
            row = run_one(method, level, gts, output_dir)
            rows.append(row)
            print(
                f"[{row['method']:>6}  {row['level']}]  "
                f"MAE={row['MAE_m']:>9.3f}m  RMSE={row['RMSE_m']:>9.3f}m  "
                f"耗时={row['elapsed_s']:>6.2f}s"
            )

    print()
    print("=" * 78)
    print("任务 A 基线对比表")
    print("=" * 78)
    header = f"{'method':>8} | {'level':>6} | {'MAE (m)':>10} | {'RMSE (m)':>10} | {'time (s)':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:>8} | {row['level']:>6} | "
            f"{row['MAE_m']:>10.3f} | {row['RMSE_m']:>10.3f} | {row['elapsed_s']:>9.2f}"
        )

    csv_path = ROOT / "outputs" / "task_a_baseline_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n指标已写入 {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

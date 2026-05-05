"""验收脚本：把目前所有 9 份 pkl 都过一遍格式校验，并复核指标。

任何一项失败会立刻 raise，全部通过会打印汇总表。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import load_pkl, validate_submission_a, validate_submission_b  # noqa: E402
from src.common.metrics import metrics_a, metrics_b  # noqa: E402


def check_task_a():
    input_8 = load_pkl(ROOT / "task_A_recovery" / "val_input_8.pkl")
    input_16 = load_pkl(ROOT / "task_A_recovery" / "val_input_16.pkl")
    gt = load_pkl(ROOT / "task_A_recovery" / "val_gt.pkl")

    rows = []
    plan = [
        ("8", input_8, "dummy_8.pkl"),
        ("8", input_8, "baseline_linear_8.pkl"),
        ("8", input_8, "baseline_spline_8.pkl"),
        ("8", input_8, "baseline_kalman_8.pkl"),
        ("16", input_16, "dummy_16.pkl"),
        ("16", input_16, "baseline_linear_16.pkl"),
        ("16", input_16, "baseline_spline_16.pkl"),
        ("16", input_16, "baseline_kalman_16.pkl"),
    ]
    for level, inputs, fname in plan:
        path = ROOT / "submissions" / "task_A" / fname
        preds = load_pkl(path)
        validate_submission_a(preds, inputs)
        validate_submission_a(preds, gt)
        m = metrics_a(preds, gt, input_list=inputs)
        rows.append((level, fname, m["MAE"], m["RMSE"], m["n_points"]))
        print(f"  [PASS] task_A/{fname}  rows={len(preds)}")

    print()
    print("Task A   level | submission                  |   MAE(m) |  RMSE(m) | n_points")
    print("-" * 80)
    for level, fname, mae, rmse, n in rows:
        print(f"          1/{level:<3} | {fname:<27} | {mae:>8.3f} | {rmse:>8.3f} | {n}")


def check_task_b():
    inputs = load_pkl(ROOT / "task_B_tte" / "val_input.pkl")
    gt = load_pkl(ROOT / "task_B_tte" / "val_gt.pkl")
    path = ROOT / "submissions" / "task_B" / "dummy.pkl"
    preds = load_pkl(path)
    validate_submission_b(preds, inputs)
    validate_submission_b(preds, gt)
    m = metrics_b(preds, gt)
    print()
    print(f"  [PASS] task_B/dummy.pkl  rows={len(preds)}")
    print()
    print(f"Task B   MAE={m['MAE']:.2f}s  RMSE={m['RMSE']:.2f}s  MAPE={m['MAPE']:.2f}%  n={m['n_traj']}")


def main():
    print("=" * 80)
    print("交付物格式校验 + 指标复核")
    print("=" * 80)
    check_task_a()
    check_task_b()
    print()
    print("=" * 80)
    print("全部 9 个 pkl 通过 validate_submission_* 格式校验。")
    print("=" * 80)


if __name__ == "__main__":
    main()

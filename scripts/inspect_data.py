"""阶段0 数据自检脚本：加载 5 个关键 pkl，把字段、形状、第一条样本结构打印出来。

用法（项目根目录下）:
    python scripts/inspect_data.py
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

TARGET_FILES = [
    ROOT / "data_ds15" / "val.pkl",
    ROOT / "task_A_recovery" / "val_input_8.pkl",
    ROOT / "task_A_recovery" / "val_input_16.pkl",
    ROOT / "task_A_recovery" / "val_gt.pkl",
    ROOT / "task_B_tte" / "val_input.pkl",
    ROOT / "task_B_tte" / "val_gt.pkl",
]


def describe_value(value: Any) -> str:
    """对单个值生成易读的 shape/dtype 说明。"""
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if isinstance(value, list):
        head = type(value[0]).__name__ if value else "empty"
        return f"list len={len(value)} (first item type: {head})"
    if isinstance(value, (int, float, str)):
        return f"{type(value).__name__}={value!r}"
    return type(value).__name__


def inspect_file(path: Path) -> None:
    print("=" * 70)
    print(f"文件: {path.relative_to(ROOT)}")
    if not path.exists():
        print("  [警告] 文件不存在，跳过")
        return

    with path.open("rb") as f:
        data = pickle.load(f)

    print(f"  顶层类型: {type(data).__name__}")
    if isinstance(data, list):
        print(f"  样本数: {len(data)}")
        if not data:
            return
        sample = data[0]
        print(f"  第一条样本类型: {type(sample).__name__}")
        if isinstance(sample, dict):
            print(f"  字段名: {list(sample.keys())}")
            print(f"  各字段说明:")
            for key, value in sample.items():
                print(f"    - {key}: {describe_value(value)}")

            coords = sample.get("coords")
            if isinstance(coords, np.ndarray) and coords.size > 0:
                nan_mask = np.isnan(coords).any(axis=1) if coords.ndim == 2 else np.isnan(coords)
                n_total = len(coords)
                n_nan = int(nan_mask.sum())
                print(
                    f"  coords 缺失统计: 总点数 {n_total}, NaN 点数 {n_nan} "
                    f"({n_nan / max(n_total, 1):.1%})"
                )
            mask = sample.get("mask")
            if mask is not None:
                mask_arr = np.asarray(mask, dtype=bool)
                print(
                    f"  mask 统计: True(已知) {mask_arr.sum()}, False(待预测) "
                    f"{(~mask_arr).sum()}"
                )
            timestamps = sample.get("timestamps")
            if timestamps is not None and len(timestamps) >= 2:
                ts = np.asarray(timestamps, dtype=np.int64)
                diffs = np.diff(ts)
                print(
                    f"  timestamps: 起 {ts[0]}, 止 {ts[-1]}, "
                    f"步长 min={diffs.min()}s mean={diffs.mean():.1f}s max={diffs.max()}s"
                )
    print()


def main() -> None:
    for path in TARGET_FILES:
        inspect_file(path)


if __name__ == "__main__":
    main()

"""任务 A 推理入口（基线版）。

用法（项目根目录）:
    python -m src.task_a.predict --method kalman \
        --input task_A_recovery/val_input_8.pkl \
        --output submissions/task_A/baseline_kalman_8.pkl
可选 --gt 提供 val_gt.pkl 路径以打印评测指标。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from ..common.io import load_pkl, save_pkl, validate_submission_a
from ..common.metrics import metrics_a
from .baseline_interp import METHOD_REGISTRY


def predict(
    input_path: str | Path,
    output_path: str | Path,
    method: str = "linear",
    show_progress: bool = True,
) -> list[dict]:
    if method not in METHOD_REGISTRY:
        raise ValueError(f"未知 method: {method}. 可选: {list(METHOD_REGISTRY)}")
    fn = METHOD_REGISTRY[method]

    inputs = load_pkl(input_path)
    iterator = inputs
    if show_progress:
        iterator = tqdm(inputs, desc=f"任务A {method}")

    preds = [fn(item) for item in iterator]
    save_pkl(preds, output_path)
    validate_submission_a(preds, inputs)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHOD_REGISTRY), default="linear")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", default=None, help="若提供则附带评测")
    args = ap.parse_args()

    t0 = time.time()
    preds = predict(args.input, args.output, method=args.method)
    elapsed = time.time() - t0
    print(f"已写出 {args.output}（{len(preds)} 条），耗时 {elapsed:.2f}s")

    if args.gt:
        gts = load_pkl(args.gt)
        inputs = load_pkl(args.input)
        m = metrics_a(preds, gts, input_list=inputs)
        print(f"评测: MAE={m['MAE']:.3f}m  RMSE={m['RMSE']:.3f}m  "
              f"n_points={m['n_points']}  n_traj={m['n_traj']}")


if __name__ == "__main__":
    main()

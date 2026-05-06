"""评测指标。

任务 A：仅在 val_input 的 mask=False 处累计 Haversine 距离的 MAE / RMSE。
任务 B：travel_time 的 MAE / RMSE / MAPE。
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .geo import haversine_m
from .io import index_by_traj_id


def metrics_a(
    pred_list: Iterable[Mapping],
    gt_list: Iterable[Mapping],
    input_list: Iterable[Mapping] | None = None,
) -> dict:
    """任务 A 指标。

    Parameters
    ----------
    pred_list : 提交结果（每条含 traj_id, coords）。
    gt_list   : 真值（每条含 traj_id, coords）。
    input_list: 可选，val_input_*（每条含 mask）。若提供，仅在 mask=False 处累计；
                若不提供，则在 gt 中所有点累计（仅供调试，不是官方评测口径）。
    """
    pred_map = index_by_traj_id(pred_list)
    gt_map = index_by_traj_id(gt_list)
    input_map = index_by_traj_id(input_list) if input_list is not None else None

    diffs: list[np.ndarray] = []
    for traj_id, gt_item in gt_map.items():
        if traj_id not in pred_map:
            raise ValueError(f"预测结果缺少 traj_id={traj_id}")
        pred = np.asarray(pred_map[traj_id]["coords"], dtype=np.float64)
        gt = np.asarray(gt_item["coords"], dtype=np.float64)
        if pred.shape != gt.shape:
            raise ValueError(
                f"traj_id={traj_id} pred shape {pred.shape} 与 gt {gt.shape} 不匹配"
            )

        if input_map is not None:
            mask = np.asarray(input_map[traj_id]["mask"], dtype=bool)
            select = ~mask
        else:
            select = np.ones(len(gt), dtype=bool)
        if not select.any():
            continue
        d = haversine_m(pred[select, 0], pred[select, 1], gt[select, 0], gt[select, 1])
        diffs.append(np.asarray(d, dtype=np.float64).reshape(-1))

    if not diffs:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n_points": 0, "n_traj": 0}

    all_d = np.concatenate(diffs)
    return {
        "MAE": float(np.mean(all_d)),
        "RMSE": float(np.sqrt(np.mean(all_d ** 2))),
        "n_points": int(len(all_d)),
        "n_traj": len(diffs),
    }


def metrics_b(
    pred_list: Iterable[Mapping],
    gt_list: Iterable[Mapping],
    eps_seconds: float = 1.0,
) -> dict:
    """任务 B 指标。MAPE 的分母用 max(true, eps) 防止除零。"""
    pred_map = index_by_traj_id(pred_list)
    gt_map = index_by_traj_id(gt_list)

    preds, trues = [], []
    for traj_id, gt_item in gt_map.items():
        if traj_id not in pred_map:
            raise ValueError(f"预测结果缺少 traj_id={traj_id}")
        preds.append(float(pred_map[traj_id]["travel_time"]))
        trues.append(float(gt_item["travel_time"]))

    preds_a = np.asarray(preds, dtype=np.float64)
    trues_a = np.asarray(trues, dtype=np.float64)
    diffs = preds_a - trues_a
    abs_diffs = np.abs(diffs)
    return {
        "MAE": float(abs_diffs.mean()),
        "RMSE": float(np.sqrt(np.mean(diffs ** 2))),
        "MAPE": float(np.mean(abs_diffs / np.maximum(trues_a, eps_seconds)) * 100.0),
        "n_traj": int(len(preds_a)),
    }

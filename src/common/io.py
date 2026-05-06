"""pkl 文件读写 + 提交格式校验。"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import numpy as np


def load_pkl(path: str | Path) -> Any:
    """读取 pickle 文件。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj: Any, path: str | Path) -> None:
    """写入 pickle 文件，自动创建父目录。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def _coerce_coords(coords: Any) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"coords 形状必须是 (N, 2)，实际 {arr.shape}")
    return arr


def validate_submission_a(
    pred_list: Iterable[Mapping[str, Any]],
    ref_list: Iterable[Mapping[str, Any]],
) -> None:
    """校验任务 A 提交：traj_id 完整、长度对齐、coords 无 NaN。

    ref_list 既可以是 val_input_*（含 mask）也可以是 val_gt（含完整 coords），
    两者都暴露 traj_id 与 coords 长度。
    """
    pred_list = list(pred_list)
    ref_list = list(ref_list)
    ref_map = {item["traj_id"]: len(item["coords"]) for item in ref_list}

    seen_ids: set = set()
    for item in pred_list:
        if "traj_id" not in item or "coords" not in item:
            raise ValueError("任务 A 每条预测必须包含 traj_id 和 coords")
        traj_id = item["traj_id"]
        if traj_id not in ref_map:
            raise ValueError(f"未知 traj_id: {traj_id}")
        if traj_id in seen_ids:
            raise ValueError(f"重复 traj_id: {traj_id}")
        seen_ids.add(traj_id)

        coords = _coerce_coords(item["coords"])
        if len(coords) != ref_map[traj_id]:
            raise ValueError(
                f"traj_id={traj_id} 的 coords 长度 {len(coords)} != 预期 {ref_map[traj_id]}"
            )
        if np.isnan(coords).any():
            raise ValueError(f"traj_id={traj_id} 的 coords 仍含 NaN")

    missing = set(ref_map.keys()) - seen_ids
    if missing:
        raise ValueError(f"缺失 {len(missing)} 条预测，例如 traj_id={next(iter(missing))}")


def validate_submission_b(
    pred_list: Iterable[Mapping[str, Any]],
    ref_list: Iterable[Mapping[str, Any]],
) -> None:
    """校验任务 B 提交：traj_id 完整、travel_time 是有限实数。"""
    pred_list = list(pred_list)
    ref_ids = {item["traj_id"] for item in ref_list}

    seen: set = set()
    for item in pred_list:
        if "traj_id" not in item or "travel_time" not in item:
            raise ValueError("任务 B 每条预测必须包含 traj_id 和 travel_time")
        traj_id = item["traj_id"]
        if traj_id not in ref_ids:
            raise ValueError(f"未知 traj_id: {traj_id}")
        if traj_id in seen:
            raise ValueError(f"重复 traj_id: {traj_id}")
        seen.add(traj_id)

        tt = item["travel_time"]
        if not isinstance(tt, (int, float, np.floating, np.integer)):
            raise ValueError(f"traj_id={traj_id} 的 travel_time 类型非法: {type(tt)}")
        if not np.isfinite(float(tt)):
            raise ValueError(f"traj_id={traj_id} 的 travel_time 非有限值: {tt}")
        if float(tt) < 0:
            raise ValueError(f"traj_id={traj_id} 的 travel_time 为负: {tt}")

    missing = ref_ids - seen
    if missing:
        raise ValueError(f"缺失 {len(missing)} 条预测，例如 traj_id={next(iter(missing))}")


def index_by_traj_id(items: Iterable[Mapping[str, Any]]) -> dict:
    """按 traj_id 建立索引，方便对齐 pred/gt。"""
    out: dict = {}
    for item in items:
        out[item["traj_id"]] = item
    return out

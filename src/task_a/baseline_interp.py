"""任务 A 三档无训练基线：线性插值 / 三次样条 / 卡尔曼+RTS 平滑。

所有方法都在 ENU 平面上工作：
- 以本条轨迹首个已知点为原点投影到 (x, y) 米平面，
- 在平面里按时间戳插值/平滑，
- 再反投影回经纬度。
- 已知点（mask=True）位置最终强制覆盖回原始已知值，避免方法引入额外误差。

每个 recover_* 函数的接口都是：
    recover_xxx(input_item) -> dict {"traj_id", "coords"}
其中 coords 是修复后的完整 (N, 2) float32 经纬度数组。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.interpolate import CubicSpline

from ..common.geo import enu_to_lonlat, lonlat_to_enu


def _prepare(input_item: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 input_item 取出 timestamps / coords / mask / origin (lon0, lat0)。

    返回:
        timestamps: (N,) float64
        coords    : (N, 2) float64，原始（NaN 保留）
        mask      : (N,) bool，True=已知
        origin    : (lon0, lat0) 投影原点（取首个已知点）
    """
    timestamps = np.asarray(input_item["timestamps"], dtype=np.float64)
    coords = np.asarray(input_item["coords"], dtype=np.float64).copy()
    mask = np.asarray(input_item["mask"], dtype=bool)

    if mask.sum() == 0:
        raise ValueError(f"traj_id={input_item.get('traj_id')} 没有任何已知点")

    first = np.flatnonzero(mask)[0]
    origin = (float(coords[first, 0]), float(coords[first, 1]))
    return timestamps, coords, mask, np.asarray(origin)


def _enforce_known(
    coords_pred: np.ndarray,
    coords_orig: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """已知位置强制覆盖回原始观测值。"""
    out = coords_pred.copy()
    out[mask] = coords_orig[mask]
    return out


def _to_output(traj_id, coords: np.ndarray) -> dict:
    return {"traj_id": traj_id, "coords": coords.astype(np.float32)}


def recover_linear(input_item: Mapping) -> dict:
    """B1：在 ENU 平面按时间戳线性插值。"""
    timestamps, coords, mask, origin = _prepare(input_item)
    lon0, lat0 = float(origin[0]), float(origin[1])

    x_known, y_known = lonlat_to_enu(coords[mask, 0], coords[mask, 1], lon0, lat0)
    t_known = timestamps[mask]

    x = np.interp(timestamps, t_known, x_known)
    y = np.interp(timestamps, t_known, y_known)

    lon, lat = enu_to_lonlat(x, y, lon0, lat0)
    pred = np.stack([lon, lat], axis=-1)
    pred = _enforce_known(pred, coords, mask)
    return _to_output(input_item["traj_id"], pred)


def recover_spline(input_item: Mapping) -> dict:
    """B2：三次样条插值（natural 边界）。

    若已知点 < 4 则退化为线性插值（cubic spline 至少需要 4 个数据点才能取到良好结果，
    实际 scipy 允许 2 点但等价于线性，仍可工作）。
    """
    timestamps, coords, mask, origin = _prepare(input_item)
    lon0, lat0 = float(origin[0]), float(origin[1])

    x_known, y_known = lonlat_to_enu(coords[mask, 0], coords[mask, 1], lon0, lat0)
    t_known = timestamps[mask]

    if len(t_known) < 2:
        return recover_linear(input_item)
    if len(t_known) < 4:
        x = np.interp(timestamps, t_known, x_known)
        y = np.interp(timestamps, t_known, y_known)
    else:
        sx = CubicSpline(t_known, x_known, bc_type="natural", extrapolate=True)
        sy = CubicSpline(t_known, y_known, bc_type="natural", extrapolate=True)
        x = sx(timestamps)
        y = sy(timestamps)

    lon, lat = enu_to_lonlat(x, y, lon0, lat0)
    pred = np.stack([lon, lat], axis=-1)
    pred = _enforce_known(pred, coords, mask)
    return _to_output(input_item["traj_id"], pred)


def _kalman_rts_2d(
    times: np.ndarray,
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    obs_mask: np.ndarray,
    sigma_obs_m: float = 5.0,
    sigma_jerk: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """6 维状态 (x, y, vx, vy, ax, ay) 的常加速度卡尔曼 + RTS smoother。

    Parameters
    ----------
    times       : (N,) 时间戳，秒（递增，可不等间隔）
    obs_x/obs_y : (N,) 在 ENU 平面的观测，仅 obs_mask=True 处有效
    obs_mask    : (N,) bool
    sigma_obs_m : 观测噪声标准差，米（GPS ~5m）
    sigma_jerk  : 加加速度噪声密度（连续时间），m/s^3，用于构造过程噪声 Q

    Returns
    -------
    x_smooth, y_smooth : (N,) 平滑后位置
    """
    n = len(times)
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float64)
    R = (sigma_obs_m ** 2) * np.eye(2, dtype=np.float64)

    state = np.zeros((n, 6), dtype=np.float64)
    P = np.zeros((n, 6, 6), dtype=np.float64)

    state_pred_next = np.zeros((n, 6), dtype=np.float64)
    P_pred_next = np.zeros((n, 6, 6), dtype=np.float64)
    F_cache = np.zeros((n, 6, 6), dtype=np.float64)

    first_known = int(np.flatnonzero(obs_mask)[0])
    state[first_known, 0] = obs_x[first_known]
    state[first_known, 1] = obs_y[first_known]
    P[first_known] = np.diag([sigma_obs_m ** 2, sigma_obs_m ** 2, 50.0, 50.0, 5.0, 5.0])

    for k in range(first_known):
        state[k] = state[first_known]
        P[k] = P[first_known] * 100.0

    for k in range(first_known, n - 1):
        dt = float(times[k + 1] - times[k])
        if dt < 0:
            dt = 0.0
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 4] = dt
        F[3, 5] = dt
        F[0, 4] = 0.5 * dt * dt
        F[1, 5] = 0.5 * dt * dt
        F_cache[k] = F

        q = sigma_jerk ** 2
        q_blk = q * np.array(
            [
                [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
                [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
                [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
            ],
            dtype=np.float64,
        )
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[np.ix_([0, 2, 4], [0, 2, 4])] = q_blk
        Q[np.ix_([1, 3, 5], [1, 3, 5])] = q_blk

        x_pred = F @ state[k]
        P_pred = F @ P[k] @ F.T + Q
        state_pred_next[k + 1] = x_pred
        P_pred_next[k + 1] = P_pred

        if obs_mask[k + 1]:
            z = np.array([obs_x[k + 1], obs_y[k + 1]], dtype=np.float64)
            y_innov = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            state[k + 1] = x_pred + K @ y_innov
            P[k + 1] = (np.eye(6) - K @ H) @ P_pred
        else:
            state[k + 1] = x_pred
            P[k + 1] = P_pred

    state_smooth = state.copy()
    P_smooth = P.copy()
    for k in range(n - 2, -1, -1):
        F = F_cache[k]
        try:
            P_pred_inv = np.linalg.inv(P_pred_next[k + 1])
        except np.linalg.LinAlgError:
            P_pred_inv = np.linalg.pinv(P_pred_next[k + 1])
        C = P[k] @ F.T @ P_pred_inv
        state_smooth[k] = state[k] + C @ (state_smooth[k + 1] - state_pred_next[k + 1])
        P_smooth[k] = P[k] + C @ (P_smooth[k + 1] - P_pred_next[k + 1]) @ C.T

    return state_smooth[:, 0], state_smooth[:, 1]


def recover_kalman(
    input_item: Mapping,
    sigma_obs_m: float = 5.0,
    sigma_jerk: float = 0.1,
) -> dict:
    """B3：常加速度模型 + 卡尔曼前向 + RTS 后向平滑。

    默认 sigma_jerk=0.1 m/s^3 是在前 500 条 val 上小搜得到的最优值——值越小，
    平滑器越倾向"加速度近似常量"，对路网上"小段近似匀速"的轨迹更友好。
    """
    timestamps, coords, mask, origin = _prepare(input_item)
    lon0, lat0 = float(origin[0]), float(origin[1])

    x_full = np.zeros(len(coords), dtype=np.float64)
    y_full = np.zeros(len(coords), dtype=np.float64)
    if mask.any():
        x_full_known, y_full_known = lonlat_to_enu(
            coords[mask, 0], coords[mask, 1], lon0, lat0
        )
        x_full[mask] = x_full_known
        y_full[mask] = y_full_known

    if mask.sum() < 2:
        return recover_linear(input_item)

    x_smooth, y_smooth = _kalman_rts_2d(
        timestamps, x_full, y_full, mask,
        sigma_obs_m=sigma_obs_m, sigma_jerk=sigma_jerk,
    )

    lon, lat = enu_to_lonlat(x_smooth, y_smooth, lon0, lat0)
    pred = np.stack([lon, lat], axis=-1)
    pred = _enforce_known(pred, coords, mask)
    return _to_output(input_item["traj_id"], pred)


METHOD_REGISTRY = {
    "linear": recover_linear,
    "spline": recover_spline,
    "kalman": recover_kalman,
}

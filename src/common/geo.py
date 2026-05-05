"""地理工具：本地 ENU 投影 与 Haversine 距离。

ENU 投影解决"经度 1 度的米数随纬度变化"的问题：
以每条轨迹首个有效点为原点，把 (lon, lat) 转成 (x_east, y_north) 米平面，
所有插值/平滑/神经网络都在该平面内进行，最后再换回经纬度。
"""
from __future__ import annotations

import numpy as np


EARTH_R = 6371008.8


def lonlat_to_enu(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
    lon0: float,
    lat0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """以 (lon0, lat0) 为原点的本地东-北平面投影（小区域近似）。"""
    lat0_rad = np.radians(lat0)
    x = np.radians(np.asarray(lon, dtype=np.float64) - lon0) * np.cos(lat0_rad) * EARTH_R
    y = np.radians(np.asarray(lat, dtype=np.float64) - lat0) * EARTH_R
    return x, y


def enu_to_lonlat(
    x: np.ndarray | float,
    y: np.ndarray | float,
    lon0: float,
    lat0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ENU 平面坐标反投影回经纬度。"""
    lat0_rad = np.radians(lat0)
    lon = lon0 + np.degrees(np.asarray(x, dtype=np.float64) / (EARTH_R * np.cos(lat0_rad)))
    lat = lat0 + np.degrees(np.asarray(y, dtype=np.float64) / EARTH_R)
    return lon, lat


def haversine_m(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """两个点（标量或同形数组）的球面距离，单位米。"""
    lon1_r = np.radians(np.asarray(lon1, dtype=np.float64))
    lat1_r = np.radians(np.asarray(lat1, dtype=np.float64))
    lon2_r = np.radians(np.asarray(lon2, dtype=np.float64))
    lat2_r = np.radians(np.asarray(lat2, dtype=np.float64))

    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(a))


def route_length_m(coords: np.ndarray) -> float:
    """轨迹总长度（米），自动跳过 NaN 段。"""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        return 0.0
    valid = ~np.isnan(coords).any(axis=1)
    pts = coords[valid]
    if len(pts) < 2:
        return 0.0
    return float(haversine_m(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1]).sum())

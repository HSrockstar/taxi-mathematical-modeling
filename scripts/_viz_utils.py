from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle
import random
from typing import Sequence

import numpy as np

SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def ensure_output_dir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def sample_items(items: Sequence, limit: int, seed: int):
    if limit <= 0 or limit >= len(items):
        return list(items)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), limit))
    return [items[index] for index in indices]


def to_local_datetime(timestamp: int) -> datetime:
    utc_time = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    return utc_time.astimezone(SHANGHAI_TZ)


def to_coord_array(coords) -> np.ndarray:
    array = np.asarray(coords, dtype=float)
    return array.reshape(-1, 2)


def to_bool_array(mask) -> np.ndarray:
    return np.asarray(mask, dtype=bool).reshape(-1)


def valid_coord_array(coords) -> np.ndarray:
    coord_array = to_coord_array(coords)
    valid_mask = ~np.isnan(coord_array).any(axis=1)
    return coord_array[valid_mask]


def haversine_distance_km(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    if points_a.shape != points_b.shape:
        raise ValueError("输入坐标形状不一致")

    lon_a = np.radians(points_a[:, 0])
    lat_a = np.radians(points_a[:, 1])
    lon_b = np.radians(points_b[:, 0])
    lat_b = np.radians(points_b[:, 1])

    delta_lon = lon_b - lon_a
    delta_lat = lat_b - lat_a

    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    arc = 2.0 * np.arcsin(np.sqrt(haversine))
    earth_radius_km = 6371.0088
    return earth_radius_km * arc


def route_length_km(coords) -> float:
    coord_array = valid_coord_array(coords)
    if len(coord_array) < 2:
        return 0.0
    return float(haversine_distance_km(coord_array[:-1], coord_array[1:]).sum())


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


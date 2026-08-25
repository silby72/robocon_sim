"""2D LiDAR model (default 10 Hz, STL-19P-like).

Casts ``n_beams`` rays over the field of view via the vectorised DDA. Optional
motion distortion (default ON): each beam is fired at a slightly different time
across the scan, so the beam's pose is interpolated over the scan window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Sensor, SensorParams
from ..env.occupancy_grid import OccupancyGrid
from ..env.raycast import raycast


@dataclass
class LidarParams(SensorParams):
    rate_hz: float = 10.0
    angle_min_rad: float = -np.pi
    angle_max_rad: float = np.pi
    n_beams: int = 450                 # 0.8 deg resolution
    range_min_m: float = 0.12
    range_max_m: float = 12.0
    range_noise_std_m: float = 0.02
    range_noise_per_m: float = 0.0     # additional sigma proportional to range
    enable_motion_distortion: bool = True
    mount_x_m: float = 0.0             # offset from base_link
    mount_y_m: float = 0.0
    mount_yaw_rad: float = 0.0


@dataclass
class LidarScan:
    t: float
    angles: np.ndarray
    ranges: np.ndarray


class Lidar(Sensor):
    def __init__(self, params: LidarParams, grid: OccupancyGrid,
                 rng: np.random.Generator) -> None:
        super().__init__(params, rng)
        self.lp = params
        self.grid = grid
        self.angles = np.linspace(params.angle_min_rad, params.angle_max_rad,
                                  params.n_beams, endpoint=False)

    def _sensor_pose(self, base_pose: np.ndarray) -> np.ndarray:
        x, y, th = base_pose
        c, s = np.cos(th), np.sin(th)
        sx = x + c * self.lp.mount_x_m - s * self.lp.mount_y_m
        sy = y + s * self.lp.mount_x_m + c * self.lp.mount_y_m
        return np.array([sx, sy, th + self.lp.mount_yaw_rad])

    def _measure(self, t: float, truth: dict) -> LidarScan:
        """``truth`` provides ``pose`` and optionally ``pose_start``/``twist``
        for motion distortion (pose at scan start, body twist)."""
        p = self.lp
        base_pose = np.asarray(truth["pose"], dtype=float)

        if p.enable_motion_distortion and "twist" in truth:
            twist = np.asarray(truth["twist"], dtype=float)  # world vx, vy, wz
            scan_time = self._period
            frac = np.linspace(0.0, scan_time, p.n_beams, endpoint=False)
            ranges = np.empty(p.n_beams)
            # interpolate pose per beam then cast individually-angled single rays
            for i in range(p.n_beams):
                dp = twist * frac[i]
                pose_i = base_pose + dp
                spose = self._sensor_pose(pose_i)
                r = raycast(self.grid, spose[0], spose[1],
                            np.array([spose[2] + self.angles[i]]), p.range_max_m)
                ranges[i] = r[0]
        else:
            spose = self._sensor_pose(base_pose)
            ranges = raycast(self.grid, spose[0], spose[1],
                             spose[2] + self.angles, p.range_max_m)

        # noise (distance-proportional component optional)
        finite = np.isfinite(ranges)
        sigma = p.range_noise_std_m + p.range_noise_per_m * np.where(finite, ranges, 0.0)
        noise = sigma * self.rng.standard_normal(p.n_beams)
        ranges = np.where(finite, ranges + noise, ranges)

        # apply min/max clipping to valid-range semantics
        ranges = np.where(ranges < p.range_min_m, np.inf, ranges)
        ranges = np.where(ranges > p.range_max_m, np.inf, ranges)
        return LidarScan(t=t, angles=self.angles.copy(), ranges=ranges)

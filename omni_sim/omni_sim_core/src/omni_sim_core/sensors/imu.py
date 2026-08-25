"""IMU model (default 200 Hz).

Per channel (angular rate, acceleration):

    measured = (1 + s) * true + b(t) + n(t)

with white noise ``n``, a random-walk bias ``db = w_b`` and a constant scale
factor ``s``. An optional misalignment matrix (identity by default) couples axes.

Parameters may be given directly (sigma_noise, sigma_bias_walk) or via Allan
parameters (angle/velocity random walk, bias instability) for later use with
values identified from real logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import Sensor, SensorParams


@dataclass
class ImuParams(SensorParams):
    rate_hz: float = 200.0
    # gyro
    gyro_noise_std: float = 1.0e-3         # rad/s (white)
    gyro_bias_walk_std: float = 1.0e-5     # rad/s per sqrt(s) integrated
    gyro_scale_factor: float = 0.0
    # accel
    accel_noise_std: float = 1.0e-2        # m/s^2
    accel_bias_walk_std: float = 1.0e-4
    accel_scale_factor: float = 0.0
    misalignment: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    # Optional Allan-variance style specification (overrides std if > 0)
    gyro_arw: float = 0.0                  # angle random walk (rad/sqrt(s))
    gyro_bias_instability: float = 0.0     # rad/s


@dataclass
class ImuReading:
    t: float
    angular_velocity_z: float
    linear_acceleration: np.ndarray  # [ax, ay] body frame


class Imu(Sensor):
    def __init__(self, params: ImuParams, rng: np.random.Generator) -> None:
        super().__init__(params, rng)
        self.ip = params
        self._gyro_bias = 0.0
        self._accel_bias = np.zeros(2)
        self._M = np.asarray(params.misalignment, dtype=float)

    def reset(self) -> None:
        super().reset()
        self._gyro_bias = 0.0
        self._accel_bias = np.zeros(2)

    def _gyro_white_std(self) -> float:
        if self.ip.gyro_arw > 0.0:
            return self.ip.gyro_arw * np.sqrt(self.ip.rate_hz)
        return self.ip.gyro_noise_std

    def _measure(self, t: float, truth: dict) -> ImuReading:
        p = self.ip
        dt = self._period
        # bias random walk
        self._gyro_bias += p.gyro_bias_walk_std * np.sqrt(dt) * self.rng.standard_normal()
        self._accel_bias += p.accel_bias_walk_std * np.sqrt(dt) * self.rng.standard_normal(2)

        wz_true = truth["angular_velocity_z"]
        acc_true = np.asarray(truth["linear_acceleration"], dtype=float)

        wz = ((1 + p.gyro_scale_factor) * wz_true + self._gyro_bias
              + self._gyro_white_std() * self.rng.standard_normal())

        acc3 = np.array([acc_true[0], acc_true[1], 0.0])
        acc3 = self._M @ acc3
        acc = ((1 + p.accel_scale_factor) * acc3[:2] + self._accel_bias
               + p.accel_noise_std * self.rng.standard_normal(2))
        return ImuReading(t=t, angular_velocity_z=float(wz),
                          linear_acceleration=self._quantize(acc))

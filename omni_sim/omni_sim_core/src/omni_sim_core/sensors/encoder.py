"""Wheel encoder model (default 1 kHz).

Returns quantised *angle* (counts), never velocity directly, so the quantization
noise naturally shows up when the consumer differentiates for speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Sensor, SensorParams


@dataclass
class EncoderParams(SensorParams):
    rate_hz: float = 1000.0
    counts_per_rev: int = 8192   # CPR


@dataclass
class EncoderReading:
    t: float
    angle_rad: np.ndarray   # quantised shaft angle per wheel
    counts: np.ndarray


class Encoder(Sensor):
    def __init__(self, params: EncoderParams, rng: np.random.Generator) -> None:
        super().__init__(params, rng)
        self.ep = params
        self._rad_per_count = 2 * np.pi / params.counts_per_rev

    def _measure(self, t: float, truth: dict) -> EncoderReading:
        theta = np.asarray(truth["wheel_angle_rad"], dtype=float)
        counts = np.round(theta / self._rad_per_count)
        return EncoderReading(t=t, angle_rad=counts * self._rad_per_count,
                              counts=counts.astype(np.int64))

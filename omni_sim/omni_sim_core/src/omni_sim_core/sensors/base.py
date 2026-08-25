"""Common sensor base: rate, latency FIFO, jitter, dropout, quantization.

A sensor is polled every simulation step with ``maybe_sample(t, truth)``. It
fires asynchronously at its own rate, buffers each reading for ``latency``
seconds, and only releases readings whose release time has passed. All
randomness uses an explicitly injected ``numpy.random.Generator``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any

import numpy as np


@dataclass
class SensorParams:
    rate_hz: float = 100.0
    latency_s: float = 0.0
    jitter_std_s: float = 0.0
    drop_rate: float = 0.0
    quantize_resolution: float = 0.0   # 0 disables; else round to this step
    quantize_bits: int = 0             # 0 disables


class Sensor:
    def __init__(self, params: SensorParams, rng: np.random.Generator) -> None:
        self.p = params
        self.rng = rng
        self._period = 1.0 / params.rate_hz
        self.reset()

    def reset(self) -> None:
        self._next_fire = 0.0
        self._buffer: deque[tuple[float, Any]] = deque()
        self._last_output: Any = None

    # -- helpers subclasses use ------------------------------------------
    def _quantize(self, value: np.ndarray) -> np.ndarray:
        p = self.p
        if p.quantize_resolution > 0.0:
            value = np.round(value / p.quantize_resolution) * p.quantize_resolution
        if p.quantize_bits > 0:
            span = np.max(np.abs(value)) if np.ndim(value) else abs(value)
            if span > 0:
                step = span / (2 ** (p.quantize_bits - 1))
                value = np.round(value / step) * step
        return value

    def _measure(self, t: float, truth: Any) -> Any:
        """Override: produce a noisy measurement from ground truth."""
        raise NotImplementedError

    # -- main poll --------------------------------------------------------
    def maybe_sample(self, t: float, truth: Any) -> Any | None:
        """Advance the sensor. Returns a freshly *released* reading or None."""
        released = None

        # Fire (create) new readings that are due.
        while t + 1e-12 >= self._next_fire:
            fire_t = self._next_fire
            self._next_fire += self._period
            if self.p.drop_rate > 0.0 and self.rng.random() < self.p.drop_rate:
                continue
            jitter = (self.rng.normal(0.0, self.p.jitter_std_s)
                      if self.p.jitter_std_s > 0.0 else 0.0)
            meas = self._measure(fire_t, truth)
            release_t = fire_t + max(0.0, jitter) + self.p.latency_s
            self._buffer.append((release_t, meas))

        # Release readings whose latency has elapsed.
        while self._buffer and self._buffer[0][0] <= t + 1e-12:
            _, meas = self._buffer.popleft()
            released = meas
            self._last_output = meas

        return released

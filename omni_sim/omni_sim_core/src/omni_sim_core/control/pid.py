"""PID controller: derivative-on-measurement, LPF on the derivative term,
and anti-windup by both integral clamping and back-calculation.

Runs under zero-order hold at a fixed control period ``dt``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDParams:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -np.inf
    output_max: float = np.inf
    derivative_lpf_tau: float = 0.0     # 0 disables the filter
    anti_windup_kb: float = 0.0         # back-calculation gain (0 => clamp only)


class PID:
    def __init__(self, params: PIDParams, dt: float) -> None:
        self.p = params
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_meas = None
        self._deriv_filt = 0.0

    def update(self, setpoint: float, measurement: float) -> float:
        p = self.p
        error = setpoint - measurement

        # Derivative on measurement (avoids setpoint-step derivative kick).
        if self._prev_meas is None:
            raw_deriv = 0.0
        else:
            raw_deriv = -(measurement - self._prev_meas) / self.dt
        self._prev_meas = measurement

        if p.derivative_lpf_tau > 0.0:
            alpha = self.dt / (p.derivative_lpf_tau + self.dt)
            self._deriv_filt += alpha * (raw_deriv - self._deriv_filt)
            deriv = self._deriv_filt
        else:
            deriv = raw_deriv

        integral_candidate = self._integral + p.ki * error * self.dt
        unsat = p.kp * error + integral_candidate + p.kd * deriv
        out = float(np.clip(unsat, p.output_min, p.output_max))

        # Anti-windup: clamp the integral, optionally back-calculate.
        if p.anti_windup_kb > 0.0:
            self._integral = integral_candidate + p.anti_windup_kb * (out - unsat) * self.dt
        elif out == unsat:
            self._integral = integral_candidate
        # else: saturated with clamp-only -> freeze integral (no accumulation)

        return out

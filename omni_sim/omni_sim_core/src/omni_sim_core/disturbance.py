"""Independent disturbance injection ports.

Disturbances are addressable by ``target`` so several may act at once:

    motor_torque   : N*m at a motor axis          (DOB's main target)
    body_wrench    : N / N*m on the body           (collisions, slopes)
    control_input  : N*m added to the command      (actuator faults)

Every source draws from an explicitly-passed ``numpy.random.Generator`` so runs
are bit-reproducible for a given seed (no global RNG state).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

TARGETS = ("motor_torque", "body_wrench", "control_input")


@dataclass
class DisturbanceSpec:
    target: str = "motor_torque"
    waveform: str = "step"
    amplitude: float = 0.0
    start_s: float = 0.0
    stop_s: float = float("inf")
    # index into the target vector (which motor / which wrench component)
    index: int = 0
    # waveform-specific
    frequency_hz: float = 1.0        # sine
    freq_start_hz: float = 0.1       # chirp
    freq_end_hz: float = 10.0        # chirp
    ramp_rate: float = 1.0           # ramp (per second)
    sigma: float = 1.0               # white / colored noise std
    color_alpha: float = 0.9         # colored noise AR(1) coefficient
    csv_path: str | None = None      # replay

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"unknown disturbance target: {self.target}")


class DisturbanceSource:
    """A single time-varying scalar disturbance on one target/index."""

    def __init__(self, spec: DisturbanceSpec, duration_s: float,
                 rng: np.random.Generator) -> None:
        self.spec = spec
        self.rng = rng
        self._colored_state = 0.0
        self._csv: np.ndarray | None = None
        self._csv_t: np.ndarray | None = None
        if spec.waveform == "csv":
            self._load_csv(spec.csv_path)

    def _load_csv(self, path: str | None) -> None:
        if path is None:
            raise ValueError("csv waveform requires csv_path")
        data = np.loadtxt(Path(path), delimiter=",", ndmin=2)
        self._csv_t = data[:, 0]
        self._csv = data[:, 1]

    def value(self, t: float, dt: float) -> float:
        s = self.spec
        if t < s.start_s or t > s.stop_s:
            # colored noise still needs to keep evolving? keep simple: gate output
            return 0.0
        tau = t - s.start_s
        w = s.waveform
        if w == "step":
            return s.amplitude
        if w == "ramp":
            return s.amplitude + s.ramp_rate * tau
        if w == "sine":
            return s.amplitude * np.sin(2 * np.pi * s.frequency_hz * tau)
        if w == "chirp":
            duration = max(s.stop_s - s.start_s, 1e-9)
            if not np.isfinite(duration):
                duration = 1.0
            k = (s.freq_end_hz - s.freq_start_hz) / duration
            phase = 2 * np.pi * (s.freq_start_hz * tau + 0.5 * k * tau * tau)
            return s.amplitude * np.sin(phase)
        if w == "white":
            return s.amplitude + s.sigma * self.rng.standard_normal()
        if w == "colored":
            self._colored_state = (s.color_alpha * self._colored_state
                                   + s.sigma * self.rng.standard_normal())
            return s.amplitude + self._colored_state
        if w == "csv":
            return float(np.interp(t, self._csv_t, self._csv))
        raise ValueError(f"unknown waveform: {w}")


class DisturbanceManager:
    """Aggregates sources and returns summed disturbance vectors per target."""

    def __init__(self, specs: list[DisturbanceSpec], duration_s: float,
                 rng: np.random.Generator,
                 n_motors: int = 1, n_wrench: int = 3, n_control: int = 1) -> None:
        self.sources = [DisturbanceSource(s, duration_s, rng) for s in specs]
        self.sizes = {"motor_torque": n_motors,
                      "body_wrench": n_wrench,
                      "control_input": n_control}

    def vector(self, target: str, t: float, dt: float) -> np.ndarray:
        out = np.zeros(self.sizes[target])
        for src in self.sources:
            if src.spec.target == target:
                out[src.spec.index] += src.value(t, dt)
        return out

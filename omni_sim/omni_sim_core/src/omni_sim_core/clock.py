"""Multi-rate time management.

Three independent, fully configurable periods drive the simulation:

    dt_sim   : plant integration step        (default 0.1 ms, 10 kHz)
    dt_motor : current/velocity loop + DOB    (default 1 ms,   1 kHz)
    dt_nav   : trajectory / path following     (default 20 ms,  50 Hz)

Controllers run under zero-order hold: between control ticks their command is
held constant. ``dt_sim`` must divide the other periods (checked at startup).

Integration and control are separate: the plant is integrated at ``dt_sim`` and
the motor-loop controller (PID + DOB) updates only at ``dt_motor``. The control
period can be set directly as ``control_rate_hz`` (a friendlier knob than a raw
period); when given it overrides ``dt_motor = 1 / control_rate_hz``. This lets an
experiment ask "how fast must this loop run?" independently of the integrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _is_integer_multiple(period: float, base: float, tol: float = 1e-9) -> bool:
    ratio = period / base
    return abs(ratio - round(ratio)) <= tol * max(1.0, ratio)


@dataclass
class ClockConfig:
    dt_sim: float = 1.0e-4
    dt_motor: float = 1.0e-3
    dt_nav: float = 2.0e-2
    # Optional: set the motor control rate directly (Hz). Overrides dt_motor.
    # Default None keeps the current 1 kHz (dt_motor = 1 ms) behaviour.
    control_rate_hz: Optional[float] = None

    def __post_init__(self) -> None:
        if self.control_rate_hz is not None:
            if self.control_rate_hz <= 0:
                raise ValueError("control_rate_hz must be > 0")
            self.dt_motor = 1.0 / self.control_rate_hz

    def validate(self) -> None:
        if self.dt_sim <= 0 or self.dt_motor <= 0 or self.dt_nav <= 0:
            raise ValueError("all periods must be positive")
        if not _is_integer_multiple(self.dt_motor, self.dt_sim):
            raise ValueError(
                f"dt_motor ({self.dt_motor}) must be an integer multiple of "
                f"dt_sim ({self.dt_sim})"
            )
        if not _is_integer_multiple(self.dt_nav, self.dt_sim):
            raise ValueError(
                f"dt_nav ({self.dt_nav}) must be an integer multiple of "
                f"dt_sim ({self.dt_sim})"
            )


class MultiRateClock:
    """Advances simulation time by ``dt_sim`` and reports which loops fire.

    Firing is decided by integer step counters so it is exact regardless of
    floating point accumulation. The clock fires a control loop on the step
    whose time is an integer multiple of that loop's period.
    """

    def __init__(self, config: ClockConfig | None = None) -> None:
        self.config = config or ClockConfig()
        self.config.validate()
        self._motor_stride = int(round(self.config.dt_motor / self.config.dt_sim))
        self._nav_stride = int(round(self.config.dt_nav / self.config.dt_sim))
        self.reset()

    def reset(self) -> None:
        self.step_index = 0

    @property
    def t(self) -> float:
        return self.step_index * self.config.dt_sim

    def advance(self) -> None:
        self.step_index += 1

    def motor_fires(self) -> bool:
        return self.step_index % self._motor_stride == 0

    def nav_fires(self) -> bool:
        return self.step_index % self._nav_stride == 0

    def num_steps(self, duration_s: float) -> int:
        return int(round(duration_s / self.config.dt_sim))

"""Fixed-step RK4 integrator.

We intentionally do *not* use ``scipy.integrate.solve_ivp``: variable step
sizing cannot be synchronised with the fixed control periods and hurts bit-for-bit
reproducibility. A plain fixed-step RK4 keeps the whole pipeline deterministic.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

DerivFn = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(f: DerivFn, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """One classical Runge-Kutta 4 step. ``f(t, y) -> dy/dt``.

    The command held by ``f`` is constant across the step (zero-order hold),
    so the four stages differ only through the state, as intended.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

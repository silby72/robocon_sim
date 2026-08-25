"""Disturbance observer (DOB).

Ideal estimate:

    d_hat = Q(s) [ P_n^{-1}(s) omega - tau_cmd ]

``P_n^{-1}(s) = J_n s + B_n`` contains a differentiator and is improper on its
own, so we never realise it directly. Instead we implement two proper transfer
functions, discretised with Tustin (bilinear):

    A(s) = Q(s) P_n^{-1}(s) = (J_n s + B_n) / D_Q(s)   applied to omega
    B(s) = Q(s)             = 1 / D_Q(s)               applied to tau_cmd
    d_hat = A(omega) - B(tau_cmd)

``Q`` is a low-pass of selectable order (1st order, or 2nd-order Butterworth)
with cutoff time-constant ``tau_q`` (changeable at runtime for robustness study).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete


class _IIR:
    """Single-input single-output IIR filter, one sample per ``step``.

    Direct Form II transposed. ``b``, ``a`` are the discrete numerator and
    denominator (a[0] normalised to 1).
    """

    def __init__(self, b: np.ndarray, a: np.ndarray) -> None:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        b = b / a[0]
        a = a / a[0]
        self.b = b
        self.a = a
        self.z = np.zeros(max(len(a), len(b)) - 1)

    def reset(self) -> None:
        self.z[:] = 0.0

    def step(self, x: float) -> float:
        b, a, z = self.b, self.a, self.z
        y = b[0] * x + (z[0] if z.size else 0.0)
        for i in range(1, len(b)):
            zi = b[i] * x - a[i] * y
            if i < len(z):
                zi += z[i]
            z[i - 1] = zi
        # if denominator shorter than numerator, remaining handled above
        return y


def _discretize(num: list[float], den: list[float], dt: float) -> _IIR:
    bd, ad, _ = cont2discrete((num, den), dt, method="bilinear")
    return _IIR(np.ravel(bd), np.ravel(ad))


@dataclass
class DOBParams:
    enabled: bool = True
    tau_q: float = 5.0e-3
    order: int = 1  # 1 or 2 (Butterworth)


class DisturbanceObserver:
    def __init__(self, params: DOBParams, nominal, dt: float) -> None:
        self.p = params
        self.dt = dt
        self.Jn = nominal.Jn
        self.Bn = nominal.Bn
        self._build()

    def _q_denominator(self) -> list[float]:
        tq = self.p.tau_q
        if self.p.order == 1:
            return [tq, 1.0]
        if self.p.order == 2:
            return [tq * tq, np.sqrt(2.0) * tq, 1.0]
        raise ValueError("DOB order must be 1 or 2")

    def _build(self) -> None:
        dq = self._q_denominator()
        # A(s) = (Jn s + Bn) / D_Q(s)
        self._filt_omega = _discretize([self.Jn, self.Bn], dq, self.dt)
        # B(s) = 1 / D_Q(s)
        self._filt_tau = _discretize([1.0], dq, self.dt)
        self._d_hat = 0.0

    def reset(self) -> None:
        self._filt_omega.reset()
        self._filt_tau.reset()
        self._d_hat = 0.0

    def set_tau_q(self, tau_q: float) -> None:
        """Change the cutoff at runtime (re-builds and resets the filters)."""
        self.p.tau_q = tau_q
        self._build()

    @property
    def d_hat(self) -> float:
        return self._d_hat

    def update(self, omega_meas: float, tau_cmd: float) -> float:
        if not self.p.enabled:
            self._d_hat = 0.0
            return 0.0
        self._d_hat = self._filt_omega.step(omega_meas) - self._filt_tau.step(tau_cmd)
        return self._d_hat

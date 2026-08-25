"""Motor layer -- the TRUE plant that the DOB has to compensate.

Per-motor mechanical dynamics (SI units throughout):

    J_m w_dot + B_m w + tau_fric(w) + tau_cog(theta) = tau_em

where ``tau_em = Kt * i`` is the electromagnetic torque. The electrical
subsystem is algebraic by default (``i = tau_cmd / Kt`` -> ``tau_em = tau_cmd``),
and a first-order current lag can be switched on:

    L di/dt + R i = u - Ke w ,   u chosen so that i -> i_cmd = tau_cmd / Kt

Friction is Coulomb + Stribeck with a ``tanh`` smoothing of the sign near zero
speed. Cogging is an optional sinusoid in shaft angle. Torque and speed
saturation are always active (needed for anti-windup experiments).

The state vector for ``N`` motors is laid out as::

    [ theta_0..theta_{N-1}, omega_0..omega_{N-1}, (i_0..i_{N-1}) ]

the current block being present only when electrical dynamics are enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MotorParams:
    """True-plant motor parameters (one shared spec for all motors)."""

    n_motors: int = 1

    inertia_kgm2: float = 5.0e-4          # J_m
    damping_nms: float = 1.0e-4           # B_m (viscous)
    torque_constant_nm_a: float = 0.5     # Kt
    resistance_ohm: float = 1.0           # R
    inductance_h: float = 1.0e-3          # L
    back_emf_v_s: float = 0.5             # Ke
    enable_electrical_dynamics: bool = False

    # Friction (Coulomb + Stribeck)
    friction_coulomb_nm: float = 0.0      # tau_c
    friction_static_nm: float = 0.0       # tau_s
    stribeck_speed_rad_s: float = 0.1     # omega_s
    friction_smooth_eps: float = 1.0e-2   # tanh smoothing width

    # Cogging (off by default)
    enable_cogging: bool = False
    cogging_amplitude_nm: float = 0.0     # A_cog
    cogging_pole_pairs: int = 8           # N_p

    # Saturation (never disabled)
    torque_max_nm: float = 5.0
    omega_max_rad_s: float = 500.0

    def __post_init__(self) -> None:
        if self.n_motors < 1:
            raise ValueError("n_motors must be >= 1")


class MotorArray:
    """Vectorised bank of ``n_motors`` identical-parameter motors."""

    def __init__(self, params: MotorParams) -> None:
        self.p = params
        self.n = params.n_motors
        self.reset()

    # -- state layout -----------------------------------------------------
    @property
    def has_current_state(self) -> bool:
        return self.p.enable_electrical_dynamics

    @property
    def state_size(self) -> int:
        return self.n * (3 if self.has_current_state else 2)

    def reset(self, theta: np.ndarray | None = None,
              omega: np.ndarray | None = None) -> None:
        self.state = np.zeros(self.state_size)
        if theta is not None:
            self.state[: self.n] = theta
        if omega is not None:
            self.state[self.n: 2 * self.n] = omega

    def theta(self, state: np.ndarray | None = None) -> np.ndarray:
        s = self.state if state is None else state
        return s[: self.n]

    def omega(self, state: np.ndarray | None = None) -> np.ndarray:
        s = self.state if state is None else state
        return s[self.n: 2 * self.n]

    def current(self, state: np.ndarray | None = None) -> np.ndarray:
        if not self.has_current_state:
            raise AttributeError("electrical dynamics disabled: no current state")
        s = self.state if state is None else state
        return s[2 * self.n: 3 * self.n]

    # -- physics ----------------------------------------------------------
    def friction_torque(self, omega: np.ndarray) -> np.ndarray:
        p = self.p
        smooth_sign = np.tanh(omega / p.friction_smooth_eps)
        coulomb = p.friction_coulomb_nm * smooth_sign
        stribeck = (p.friction_static_nm - p.friction_coulomb_nm) * np.exp(
            -((omega / p.stribeck_speed_rad_s) ** 2)
        ) * smooth_sign
        viscous = p.damping_nms * omega
        return coulomb + stribeck + viscous

    def cogging_torque(self, theta: np.ndarray) -> np.ndarray:
        if not self.p.enable_cogging:
            return np.zeros_like(theta)
        return self.p.cogging_amplitude_nm * np.sin(self.p.cogging_pole_pairs * theta)

    def _electromagnetic_torque(self, tau_cmd: np.ndarray,
                                state: np.ndarray) -> np.ndarray:
        tau_cmd = np.clip(tau_cmd, -self.p.torque_max_nm, self.p.torque_max_nm)
        if not self.has_current_state:
            return tau_cmd
        return self.p.torque_constant_nm_a * self.current(state)

    def deriv(self, t: float, state: np.ndarray, tau_cmd: np.ndarray,
              tau_ext: np.ndarray | None = None) -> np.ndarray:
        """State derivative under a held torque command and external torque.

        ``tau_ext`` is a disturbance injected at the motor axis (N*m).
        """
        p = self.p
        theta = self.theta(state)
        omega = self.omega(state)
        tau_cmd = np.clip(tau_cmd, -p.torque_max_nm, p.torque_max_nm)
        if tau_ext is None:
            tau_ext = np.zeros(self.n)

        tau_em = self._electromagnetic_torque(tau_cmd, state)
        omega_dot = (
            tau_em
            - self.friction_torque(omega)
            - self.cogging_torque(theta)
            + tau_ext
        ) / p.inertia_kgm2

        # Soft speed saturation: stop accelerating past omega_max.
        over = np.abs(omega) >= p.omega_max_rad_s
        omega_dot = np.where(over & (np.sign(omega) == np.sign(omega_dot)),
                             0.0, omega_dot)

        dtheta = omega
        if not self.has_current_state:
            return np.concatenate([dtheta, omega_dot])

        i = self.current(state)
        i_cmd = tau_cmd / p.torque_constant_nm_a
        u = p.resistance_ohm * i_cmd + p.back_emf_v_s * omega  # steady-state feed
        di = (u - p.resistance_ohm * i - p.back_emf_v_s * omega) / p.inductance_h
        return np.concatenate([dtheta, omega_dot, di])

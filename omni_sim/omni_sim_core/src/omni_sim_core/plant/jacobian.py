"""Jacobian layer -- static map between wheel space and body space.

Four-wheel omni. Wheel ``i`` has mount angle ``alpha_i``, sits at distance ``L``
from the body centre, wheel radius ``r``:

    theta_dot_w = (1/r) J v_body ,

    J = [[-sin a_i, cos a_i, L]] stacked over the four wheels (4 x 3).

Forward (wheels -> body):  v_body   = r * pinv(J) theta_dot_w
Force map (wheels -> body): F_body  = J^T f_wheel
Torque allocation:          tau     = r * pinv(J^T) F_body

This layer is *static* in this spec (no wheel dynamics/slip). The
wheel-velocity <-> body-velocity conversion is factored into methods so a
``WheelDynamics`` block can be inserted later without touching callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _default_alphas() -> list[float]:
    return [np.deg2rad(a) for a in (45.0, 135.0, 225.0, 315.0)]


@dataclass
class JacobianParams:
    mount_angles_rad: list[float] = field(default_factory=_default_alphas)
    body_radius_m: float = 0.2      # L
    wheel_radius_m: float = 0.05    # r


class JacobianLayer:
    def __init__(self, params: JacobianParams | None = None) -> None:
        self.p = params or JacobianParams()
        alphas = np.asarray(self.p.mount_angles_rad, dtype=float)
        L = self.p.body_radius_m
        # J maps body velocity -> (r * wheel angular velocity)
        self.J = np.stack([-np.sin(alphas), np.cos(alphas),
                           np.full_like(alphas, L)], axis=1)  # (n_wheels, 3)
        self.J_pinv = np.linalg.pinv(self.J)
        self.Jt_pinv = np.linalg.pinv(self.J.T)

    @property
    def n_wheels(self) -> int:
        return self.J.shape[0]

    # -- kinematics (the interface that future WheelDynamics plugs into) ---
    def wheel_velocity_from_body(self, v_body: np.ndarray) -> np.ndarray:
        """Body velocity [vx, vy, wz] -> wheel angular velocities [rad/s]."""
        return (self.J @ v_body) / self.p.wheel_radius_m

    def body_velocity_from_wheel(self, wheel_omega: np.ndarray) -> np.ndarray:
        """Wheel angular velocities [rad/s] -> body velocity [vx, vy, wz]."""
        return self.p.wheel_radius_m * (self.J_pinv @ wheel_omega)

    # -- statics ----------------------------------------------------------
    def body_force_from_wheel(self, f_wheel: np.ndarray) -> np.ndarray:
        """Wheel tangential forces [N] -> body wrench [Fx, Fy, Mz]."""
        return self.J.T @ f_wheel

    def wheel_torque_from_body_force(self, f_body: np.ndarray) -> np.ndarray:
        """Body wrench [Fx, Fy, Mz] -> wheel torques [N*m] (min-norm alloc)."""
        return self.p.wheel_radius_m * (self.Jt_pinv @ f_body)

    def wheel_force_from_torque(self, wheel_torque: np.ndarray) -> np.ndarray:
        """Wheel torque [N*m] -> tangential contact force [N]."""
        return wheel_torque / self.p.wheel_radius_m

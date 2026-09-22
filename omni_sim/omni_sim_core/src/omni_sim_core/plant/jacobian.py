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
    """Either the idealised symmetric layout, or a real one.

    The symmetric fields describe four identical wheels on a circle: one mount
    angle each, one shared ``L``, one shared ``r``, no gearbox. That is a fine
    default and a poor description of an actual robot, whose wheels sit at
    measured positions with their own radii and gear ratios -- so
    ``drive_matrix`` can carry the real thing instead, built from
    ``config/robot/chassis.yaml`` by ``mechanism.jacobian.build_drive_jacobian``.

    ``drive_matrix`` maps a body twist to **motor-shaft** angular velocity
    (gear ratio included), which is the quantity the encoders actually measure;
    ``contact_gain`` is the matching per-wheel ``n_i / r_i`` that turns motor
    torque into tangential contact force. Give both or neither.
    """

    mount_angles_rad: list[float] = field(default_factory=_default_alphas)
    body_radius_m: float = 0.2      # L
    wheel_radius_m: float = 0.05    # r
    drive_matrix: Sequence[Sequence[float]] | None = None   # (n_wheels, 3)
    contact_gain: Sequence[float] | None = None             # n_i / r_i


class JacobianLayer:
    """Static wheel <-> body map.

    Two matrices, because two different things are being mapped:

    ``J``   body twist -> motor-shaft angular velocity  (kinematics, geared)
    ``A``   body twist -> tangential speed at the contact point  (statics)

    They differ by ``contact_gain`` per wheel. Keeping both explicit is what
    lets a gearbox exist: the wrench a motor torque produces is ``J.T @ tau``
    either way, but the *contact force* in between is ``(n/r) tau``, and
    collapsing the two loses the gear ratio.
    """

    def __init__(self, params: JacobianParams | None = None) -> None:
        self.p = params or JacobianParams()
        if self.p.drive_matrix is not None:
            self.J = np.asarray(self.p.drive_matrix, dtype=float)
            if self.p.contact_gain is None:
                raise ValueError("drive_matrix needs a matching contact_gain "
                                 "(n_i / r_i per wheel)")
            self._gain = np.asarray(self.p.contact_gain, dtype=float)
            if self._gain.shape != (self.J.shape[0],):
                raise ValueError(
                    f"contact_gain has {self._gain.shape} entries for "
                    f"{self.J.shape[0]} wheels")
        else:
            alphas = np.asarray(self.p.mount_angles_rad, dtype=float)
            L = self.p.body_radius_m
            # rows map body velocity -> contact-point tangential speed
            A = np.stack([-np.sin(alphas), np.cos(alphas),
                          np.full_like(alphas, L)], axis=1)   # (n_wheels, 3)
            self._gain = np.full(len(alphas), 1.0 / self.p.wheel_radius_m)
            self.J = A * self._gain[:, None]
        self.A = self.J / self._gain[:, None]
        self.J_pinv = np.linalg.pinv(self.J)
        self.At_pinv = np.linalg.pinv(self.A.T)

    @property
    def n_wheels(self) -> int:
        return self.J.shape[0]

    # -- kinematics (the interface that future WheelDynamics plugs into) ---
    def wheel_velocity_from_body(self, v_body: np.ndarray) -> np.ndarray:
        """Body velocity [vx, vy, wz] -> motor-shaft angular velocities [rad/s]."""
        return self.J @ v_body

    def body_velocity_from_wheel(self, wheel_omega: np.ndarray) -> np.ndarray:
        """Motor-shaft angular velocities [rad/s] -> body velocity [vx, vy, wz]."""
        return self.J_pinv @ wheel_omega

    # -- statics ----------------------------------------------------------
    def body_force_from_wheel(self, f_wheel: np.ndarray) -> np.ndarray:
        """Wheel tangential forces [N] -> body wrench [Fx, Fy, Mz]."""
        return self.A.T @ f_wheel

    def wheel_torque_from_body_force(self, f_body: np.ndarray) -> np.ndarray:
        """Body wrench [Fx, Fy, Mz] -> motor torques [N*m] (min-norm alloc)."""
        return (self.At_pinv @ f_body) / self._gain

    def wheel_force_from_torque(self, wheel_torque: np.ndarray) -> np.ndarray:
        """Motor torque [N*m] -> tangential contact force [N]."""
        return wheel_torque * self._gain

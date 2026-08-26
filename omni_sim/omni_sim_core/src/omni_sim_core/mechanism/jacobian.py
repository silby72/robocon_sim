"""Drive / odometry jacobian from a wheel layout.

Input : a validated ``Chassis``.
Output: the drive jacobian ``J`` (body twist ``[vx, vy, wz]`` -> per-wheel
motor-shaft angular velocity), its pseudo-inverse (odometry: wheel speeds ->
body twist), and the singular-configuration guard.

Per-wheel (drive axis ``theta_i``, contact point ``(x_i, y_i)``, radius ``r_i``,
gear ratio ``n_i``, motor:wheel):

    omega_i = ( cos(theta_i) vx + sin(theta_i) vy
                + (x_i sin(theta_i) - y_i cos(theta_i)) wz ) / r_i * n_i

Derivation: the wheel's ground-contact point moves at
``v + wz x r_i`` = ``[vx - wz*y_i, vy + wz*x_i]``; its roll speed is the
projection onto the drive axis, divided by ``r_i`` (wheel rad/s) and multiplied
by ``n_i`` (motor shaft rad/s, which is what the encoder sees). ``reverse``
flips the row sign.

Coordinates are the *geometric centre* (base_link). A centre-of-mass offset is
NOT folded into the kinematics here -- it belongs in the mass matrix as the
translation/rotation coupling terms (see ``build_mass_matrix``).
"""
from __future__ import annotations

import logging

import numpy as np

from .schema import Chassis

logger = logging.getLogger(__name__)

# Above this 2-norm condition number the wheel set cannot robustly span the
# 3-DOF body twist (e.g. two wheels, or all drive axes parallel).
CONDITION_NUMBER_LIMIT = 1.0e6


class SingularConfigError(ValueError):
    """Raised when the drive jacobian is too ill-conditioned to invert."""


def build_drive_jacobian(chassis: Chassis) -> np.ndarray:
    """Return the ``(n_wheels, 3)`` drive jacobian for a chassis."""
    rows = []
    for w in chassis.drive_wheels:
        th = w.drive_axis_rad
        x, y = w.position_m
        c, s = np.cos(th), np.sin(th)
        gain = (w.gear_ratio / w.radius_m) * (-1.0 if w.reverse else 1.0)
        rows.append(gain * np.array([c, s, x * s - y * c]))
    return np.asarray(rows, dtype=float)


class DriveJacobian:
    """Drive jacobian with its inverse and a condition-number guard."""

    def __init__(self, chassis: Chassis) -> None:
        self.chassis = chassis
        self.J = build_drive_jacobian(chassis)
        self.condition_number = float(np.linalg.cond(self.J))
        logger.info("drive jacobian: %d wheels, condition number %.3g",
                    self.J.shape[0], self.condition_number)
        if not np.isfinite(self.condition_number) or \
                self.condition_number > CONDITION_NUMBER_LIMIT:
            raise SingularConfigError(
                f"drive jacobian condition number {self.condition_number:.3g} "
                f"exceeds {CONDITION_NUMBER_LIMIT:.0e}: the wheels do not span "
                "all 3 DOF (too few wheels, or parallel drive axes)")
        self.J_pinv = np.linalg.pinv(self.J)

    @property
    def n_wheels(self) -> int:
        return self.J.shape[0]

    def wheel_speeds_from_body(self, body_twist: np.ndarray) -> np.ndarray:
        """Body twist [vx, vy, wz] -> motor-shaft angular velocities [rad/s]."""
        return self.J @ np.asarray(body_twist, dtype=float)

    def body_from_wheel_speeds(self, wheel_omega: np.ndarray) -> np.ndarray:
        """Odometry: motor-shaft angular velocities -> body twist (min-norm)."""
        return self.J_pinv @ np.asarray(wheel_omega, dtype=float)


def build_mass_matrix(chassis: Chassis) -> np.ndarray:
    """3x3 body-frame mass matrix about the geometric centre.

    With the centre of mass at ``(cx, cy)`` relative to base_link, mass ``m`` and
    inertia about the CoM ``Izz``, the mass matrix picks up off-diagonal
    translation/rotation coupling::

        M = [[ m,      0,     -m*cy          ],
             [ 0,      m,      m*cx          ],
             [-m*cy,   m*cx,   Izz + m*(cx^2+cy^2) ]]

    The off-diagonal terms vanish only when the CoM sits on the geometric centre.
    """
    com = chassis.center_of_mass
    m = com.mass_kg
    cx, cy = com.position_m
    izz_center = com.inertia_zz_kgm2 + m * (cx * cx + cy * cy)  # parallel axis
    return np.array([
        [m,        0.0,      -m * cy],
        [0.0,      m,         m * cx],
        [-m * cy,  m * cx,    izz_center],
    ])

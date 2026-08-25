"""Body layer -- 3-DOF rigid body in the world frame.

    M vx_dot = Fx_world - Dx vx_world
    M vy_dot = Fy_world - Dy vy_world
    Iz wz_dot = Mz        - Dtheta wz

Wrenches arrive in the *body* frame (from the jacobian layer) and are rotated
into the world frame each step. State is [x, y, theta, vx, vy, wz] with the
linear velocities expressed in the world frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BodyParams:
    mass_kg: float = 10.0            # M
    inertia_z_kgm2: float = 0.5      # Iz
    drag_x_ns_m: float = 5.0         # Dx
    drag_y_ns_m: float = 5.0         # Dy
    drag_theta_nms: float = 0.5      # Dtheta


class RigidBody:
    STATE_SIZE = 6  # x, y, theta, vx, vy, wz

    def __init__(self, params: BodyParams | None = None) -> None:
        self.p = params or BodyParams()
        self.reset()

    def reset(self, pose: np.ndarray | None = None,
              twist: np.ndarray | None = None) -> None:
        self.state = np.zeros(self.STATE_SIZE)
        if pose is not None:
            self.state[:3] = pose
        if twist is not None:
            self.state[3:] = twist

    @property
    def pose(self) -> np.ndarray:
        return self.state[:3].copy()

    @property
    def twist_world(self) -> np.ndarray:
        return self.state[3:].copy()

    def twist_body(self, state: np.ndarray | None = None) -> np.ndarray:
        """Velocity expressed in the body frame [vx_b, vy_b, wz]."""
        s = self.state if state is None else state
        theta = s[2]
        c, sn = np.cos(theta), np.sin(theta)
        vx_w, vy_w, wz = s[3], s[4], s[5]
        return np.array([c * vx_w + sn * vy_w, -sn * vx_w + c * vy_w, wz])

    def deriv(self, t: float, state: np.ndarray,
              wrench_body: np.ndarray) -> np.ndarray:
        """State derivative given a body-frame wrench [Fx_b, Fy_b, Mz]."""
        theta = state[2]
        vx_w, vy_w, wz = state[3], state[4], state[5]
        c, sn = np.cos(theta), np.sin(theta)

        Fx_w = c * wrench_body[0] - sn * wrench_body[1]
        Fy_w = sn * wrench_body[0] + c * wrench_body[1]
        Mz = wrench_body[2]

        ax = (Fx_w - self.p.drag_x_ns_m * vx_w) / self.p.mass_kg
        ay = (Fy_w - self.p.drag_y_ns_m * vy_w) / self.p.mass_kg
        alpha = (Mz - self.p.drag_theta_nms * wz) / self.p.inertia_z_kgm2

        return np.array([vx_w, vy_w, wz, ax, ay, alpha])

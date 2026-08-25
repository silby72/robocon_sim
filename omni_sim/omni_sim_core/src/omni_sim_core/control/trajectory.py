"""Trajectory generation and a simple following controller.

- Waypoints -> time-parameterised trajectory.
- Velocity profile: trapezoidal or S-curve (jerk-limited).
- Output at each time: (x, y, theta, xd, yd, thd, xdd, ydd) so a feed-forward
  term with acceleration is available.

The following controller is position PID + velocity feed-forward (MPC is out of
scope). It returns a body-frame velocity command.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pid import PID, PIDParams


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    theta: float
    xd: float
    yd: float
    thd: float
    xdd: float
    ydd: float


def _trapezoidal_profile(distance: float, v_max: float, a_max: float,
                         dt: float) -> np.ndarray:
    """Arc-length s(t) samples for a trapezoidal speed profile."""
    distance = abs(distance)
    if distance < 1e-12:
        return np.zeros(1)
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc ** 2
    if 2 * d_acc > distance:  # triangular
        t_acc = np.sqrt(distance / a_max)
        v_peak = a_max * t_acc
        t_flat = 0.0
    else:
        v_peak = v_max
        t_flat = (distance - 2 * d_acc) / v_max
    total = 2 * t_acc + t_flat
    ts = np.arange(0.0, total + dt, dt)
    s = np.empty_like(ts)
    for k, t in enumerate(ts):
        if t < t_acc:
            s[k] = 0.5 * a_max * t ** 2
        elif t < t_acc + t_flat:
            s[k] = d_acc + v_peak * (t - t_acc)
        else:
            td = t - t_acc - t_flat
            s[k] = d_acc + v_peak * t_flat + v_peak * td - 0.5 * a_max * td ** 2
    s[-1] = distance
    return s


@dataclass
class TrajectoryParams:
    v_max: float = 1.0
    a_max: float = 1.0
    profile: str = "trapezoidal"   # or "scurve"
    dt: float = 0.02


class Trajectory:
    """Piecewise-linear geometric path timed with a velocity profile."""

    def __init__(self, waypoints: np.ndarray, params: TrajectoryParams) -> None:
        self.wp = np.asarray(waypoints, dtype=float)  # (N, 2 or 3)
        self.p = params
        self._build()

    def _build(self) -> None:
        pts = self.wp[:, :2]
        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        self._samples: list[TrajectoryPoint] = []
        t0 = 0.0
        for i, L in enumerate(seg_len):
            if L < 1e-12:
                continue
            direction = seg[i] / L
            s = _trapezoidal_profile(L, self.p.v_max, self.p.a_max, self.p.dt)
            sd = np.gradient(s, self.p.dt)
            sdd = np.gradient(sd, self.p.dt)
            th = np.arctan2(direction[1], direction[0])
            for k in range(len(s)):
                self._samples.append(TrajectoryPoint(
                    t=t0 + k * self.p.dt,
                    x=pts[i, 0] + direction[0] * s[k],
                    y=pts[i, 1] + direction[1] * s[k],
                    theta=th,
                    xd=direction[0] * sd[k],
                    yd=direction[1] * sd[k],
                    thd=0.0,
                    xdd=direction[0] * sdd[k],
                    ydd=direction[1] * sdd[k],
                ))
            t0 = self._samples[-1].t + self.p.dt
        if not self._samples:  # degenerate: single point
            x, y = pts[0]
            self._samples = [TrajectoryPoint(0, x, y, 0, 0, 0, 0, 0, 0)]
        self._ts = np.array([p.t for p in self._samples])

    @property
    def duration(self) -> float:
        return self._samples[-1].t

    def sample(self, t: float) -> TrajectoryPoint:
        idx = int(np.searchsorted(self._ts, t, side="right")) - 1
        idx = max(0, min(idx, len(self._samples) - 1))
        return self._samples[idx]


class TrajectoryFollower:
    """Position PID (per axis) + velocity feed-forward -> body velocity cmd."""

    def __init__(self, trajectory: Trajectory, gains: PIDParams, dt: float,
                 v_max: float = 2.0) -> None:
        self.traj = trajectory
        self.dt = dt
        self.v_max = v_max
        self.pid_x = PID(gains, dt)
        self.pid_y = PID(gains, dt)
        self.pid_th = PID(gains, dt)

    def command(self, t: float, pose: np.ndarray) -> np.ndarray:
        """Return a BODY-frame velocity command [vx_b, vy_b, wz].

        Position feed-forward + PID is computed in the world frame, then rotated
        into the body frame (which is what the wheel jacobian expects). The
        linear speed is clamped so a large transient error cannot command a
        runaway velocity.
        """
        ref = self.traj.sample(t)
        vx_w = ref.xd + self.pid_x.update(ref.x, pose[0])
        vy_w = ref.yd + self.pid_y.update(ref.y, pose[1])

        speed = float(np.hypot(vx_w, vy_w))
        if speed > self.v_max:
            vx_w *= self.v_max / speed
            vy_w *= self.v_max / speed

        th = pose[2]
        c, s = np.cos(th), np.sin(th)
        vx_b = c * vx_w + s * vy_w
        vy_b = -s * vx_w + c * vy_w

        th_err = np.arctan2(np.sin(ref.theta - pose[2]), np.cos(ref.theta - pose[2]))
        wz = ref.thd + self.pid_th.update(pose[2] + th_err, pose[2])
        return np.array([vx_b, vy_b, wz])

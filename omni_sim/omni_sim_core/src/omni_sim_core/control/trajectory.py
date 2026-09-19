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


@dataclass
class ProfileLimits:
    """Limits for :class:`SampledTrajectory` (SI units)."""
    v_max: float = 1.2
    a_max: float = 1.5        # tangential accel/decel [m/s^2]
    a_lat_max: float = 2.0    # lateral (cornering) accel [m/s^2]


class SampledTrajectory:
    """Time-parameterised motion along a densely sampled geometric path.

    :class:`Trajectory` times each polyline leg with its own trapezoid, so it
    comes to a **full stop at every waypoint** and its heading jumps by the
    turn angle at each vertex. This class instead takes an already-smooth,
    arc-length-sampled path (see ``planning.smooth.smooth_path``) and fits one
    continuous speed profile over the whole thing, stopping only at the ends.

    The profile is the standard forward-backward pass over arc length:

        v_limit = min(v_max, sqrt(a_lat_max / curvature))     per sample
        forward   v[i]   <= sqrt(v[i-1]^2 + 2 a_max ds)       can we speed up?
        backward  v[i]   <= sqrt(v[i+1]^2 + 2 a_max ds)       must we slow down?

    which is the continuous form of what ``planning.corner`` can only do at
    vertices. Heading is **not** taken from the path tangent: the robot is
    holonomic (§2.6), so ``theta`` comes from an independent
    ``OrientationProfile`` -- or stays fixed if none is given. Tying heading to
    the tangent is what makes a chassis spin on the spot at every corner.

    Exposes ``sample(t)`` / ``duration`` so it drops into
    :class:`TrajectoryFollower` wherever :class:`Trajectory` fits.
    """

    def __init__(self, points: np.ndarray, s: np.ndarray, curvature: np.ndarray,
                 limits: ProfileLimits | None = None, *,
                 orientation=None, theta: float = 0.0) -> None:
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.s = np.asarray(s, dtype=float).ravel()
        self.curvature = np.asarray(curvature, dtype=float).ravel()
        self.limits = limits or ProfileLimits()
        self._orientation = orientation
        self._theta_fixed = float(theta)
        self._build()

    def _build(self) -> None:
        lim = self.limits
        n = len(self.points)
        if n < 2:
            self._t = np.zeros(max(n, 1))
            self._v = np.zeros(max(n, 1))
            self._theta = np.full(max(n, 1), self._theta_fixed)
            return

        ds = np.diff(self.s)
        ds = np.maximum(ds, 1e-9)
        v = np.minimum(lim.v_max,
                       np.sqrt(lim.a_lat_max / np.maximum(self.curvature, 1e-9)))
        v[0] = 0.0
        v[-1] = 0.0
        for i in range(1, n):              # forward: how fast can we arrive?
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * lim.a_max * ds[i - 1]))
        for i in range(n - 2, -1, -1):     # backward: can we still stop in time?
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * lim.a_max * ds[i]))

        dt = 2.0 * ds / np.maximum(v[:-1] + v[1:], 1e-6)
        self._t = np.concatenate([[0.0], np.cumsum(dt)])
        self._v = v

        if self._orientation is not None:
            s_norm = self.s / max(self.s[-1], 1e-9)
            self._theta = np.array([self._orientation.sample(u) for u in s_norm])
        else:
            self._theta = np.full(n, self._theta_fixed)

    @property
    def duration(self) -> float:
        return float(self._t[-1])

    def sample(self, t: float) -> TrajectoryPoint:
        t = float(np.clip(t, 0.0, self.duration))
        i = int(np.searchsorted(self._t, t, side="right")) - 1
        i = max(0, min(i, len(self._t) - 2)) if len(self._t) > 1 else 0
        if len(self._t) < 2:
            x, y = self.points[0]
            return TrajectoryPoint(t, x, y, self._theta[0], 0, 0, 0, 0, 0)

        span = max(self._t[i + 1] - self._t[i], 1e-9)
        a = (t - self._t[i]) / span
        p = self.points[i] * (1 - a) + self.points[i + 1] * a
        speed = self._v[i] * (1 - a) + self._v[i + 1] * a
        heading = self._theta[i] * (1 - a) + self._theta[i + 1] * a

        d = self.points[i + 1] - self.points[i]
        n = float(np.hypot(d[0], d[1]))
        tangent = d / n if n > 1e-9 else np.array([1.0, 0.0])
        accel = (self._v[i + 1] - self._v[i]) / span
        thd = (self._theta[i + 1] - self._theta[i]) / span
        return TrajectoryPoint(
            t=t, x=float(p[0]), y=float(p[1]), theta=float(heading),
            xd=float(tangent[0] * speed), yd=float(tangent[1] * speed),
            thd=float(thd),
            xdd=float(tangent[0] * accel), ydd=float(tangent[1] * accel),
        )


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

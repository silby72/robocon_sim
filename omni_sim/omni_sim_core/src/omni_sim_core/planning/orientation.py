"""Body-heading profile, generated independently of the path (§2.6).

The robot is holonomic, so heading is decoupled from the translation path. Three
modes (spec §2.7):

    fixed     -- one constant heading everywhere
    waypoint  -- a heading given at each path vertex, interpolated by arc length
    target    -- always face a fixed world point (e.g. the basket)

Every mode ``unwrap``s the result: without it, heading jumps by 2*pi at the
+-pi branch cut and every downstream rate limit / interpolation blows up
(P1 acceptance #12).

Time synchronisation with translation is a one-way dependency (§2.10): the
translation trajectory sets T_trans; the heading is fit into it; if the angular
rate/accel limits cannot meet T_trans, T is extended and translation is rebuilt.
``min_rotation_time`` provides the T_rot half of that loop; because T only ever
grows, the loop converges.
"""
from __future__ import annotations

import numpy as np

from .types import OrientationProfile, Path


def _arc_length_s(points: np.ndarray) -> np.ndarray:
    """Normalised cumulative arc length in [0, 1] (0 for a degenerate path)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.zeros(len(pts))
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(pts))
    return s / total


def build_orientation_profile(path: Path, mode: str = "fixed", *,
                              theta: float | None = None,
                              thetas=None,
                              target: tuple[float, float] | None = None
                              ) -> OrientationProfile:
    """Build an :class:`OrientationProfile` for ``path`` in the chosen mode."""
    pts = path.points
    s = _arc_length_s(pts)

    if mode == "fixed":
        if theta is None:
            raise ValueError("mode='fixed' requires theta")
        th = np.full(len(pts), float(theta))

    elif mode == "waypoint":
        if thetas is None:
            raise ValueError("mode='waypoint' requires thetas (one per vertex)")
        th = np.asarray(thetas, dtype=float)
        if th.shape[0] != len(pts):
            raise ValueError(f"thetas has {th.shape[0]} entries, path has {len(pts)}")

    elif mode == "target":
        if target is None:
            raise ValueError("mode='target' requires a target point")
        tx, ty = target
        th = np.arctan2(ty - pts[:, 1], tx - pts[:, 0])

    else:
        raise ValueError(f"unknown orientation mode {mode!r}")

    th = np.unwrap(th)  # §2.7: continuous heading, no +-pi jumps
    return OrientationProfile(s=s, theta=th)


def min_rotation_time(theta_profile: OrientationProfile, omega_max: float,
                      alpha_max: float) -> float:
    """Minimum time to sweep the profile's total heading change (§2.10).

    Trapezoidal angular profile: accelerate at ``alpha_max`` to ``omega_max``,
    cruise, decelerate. Returns 0 for no net rotation.
    """
    th = theta_profile.theta
    total = float(abs(th[-1] - th[0])) if len(th) else 0.0
    if total < 1e-12:
        return 0.0
    t_acc = omega_max / alpha_max
    d_acc = 0.5 * alpha_max * t_acc ** 2
    if 2 * d_acc >= total:                # triangular (never reaches omega_max)
        return 2.0 * np.sqrt(total / alpha_max)
    t_flat = (total - 2 * d_acc) / omega_max
    return 2 * t_acc + t_flat

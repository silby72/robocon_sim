"""C2-continuous smoothing of a planned polyline -- corner.py's deferred half.

``corner.py`` caps the speed at each polyline vertex *instead of* making the
path curvature-continuous ("Rather than curvature continuity (B-splines,
deferred)"). This module is that deferred half: fit a cubic B-spline to the
polyline, sample it densely, and report tangent and curvature per sample, so a
time parameterisation can honour a lateral-acceleration limit continuously
along the whole path rather than only at the vertices.

Geometry only -- no time, no velocity (invariant 8 / §2.8). The dense samples
are a piecewise-linear approximation of the C2 curve: fine enough that its
derivative is effectively continuous at the rate the controller runs at.

**Why a B-spline and not an interpolating spline.** A B-spline *approximates*
its control points: the curve stays inside the convex hull of the control
polygon, so smoothing can only cut corners *inward*, never bulge outward.
A cubic interpolating spline would pass through every vertex exactly but
overshoots on tight turns -- outward, into the obstacle the planner had just
cleared, which is precisely the wrong failure mode next to a wall. Corner
cutting is still a deviation from the cleared polyline, so callers should
check :func:`max_deviation` against the clearance they planned with.

The knot vector is *clamped*, so the first and last control points are
interpolated exactly: a plan still starts at the robot and ends at the goal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Path

_EPS = 1e-12


@dataclass(frozen=True)
class SmoothPath:
    """A C2 curve sampled at (near-)uniform arc length.

    ``points[i]`` is at arc length ``s[i]`` with unit ``tangent[i]`` and
    unsigned ``curvature[i]`` [1/m]; a straight stretch has curvature 0.
    """

    points: np.ndarray      # (M, 2) [m]
    s: np.ndarray           # (M,) cumulative arc length [m]
    tangent: np.ndarray     # (M, 2) unit
    curvature: np.ndarray   # (M,) [1/m], unsigned

    @property
    def length(self) -> float:
        return float(self.s[-1]) if len(self.s) else 0.0

    def __len__(self) -> int:
        return len(self.points)


def _clamped_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """Clamped uniform knot vector: endpoints interpolated, interior uniform."""
    n_interior = n_ctrl - degree - 1
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1] if n_interior > 0 else []
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])


def _polyline_sample(pts: np.ndarray, ds: float) -> SmoothPath:
    """Fallback for paths too short to spline: resample the polyline itself."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s_src = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_src[-1])
    if total < _EPS:
        return SmoothPath(points=pts[:1].copy(), s=np.zeros(1),
                          tangent=np.array([[1.0, 0.0]]), curvature=np.zeros(1))
    n_out = max(2, int(round(total / ds)) + 1)
    s_out = np.linspace(0.0, total, n_out)
    p = np.column_stack([np.interp(s_out, s_src, pts[:, 0]),
                         np.interp(s_out, s_src, pts[:, 1])])
    d = np.gradient(p, s_out, axis=0)
    norm = np.maximum(np.hypot(d[:, 0], d[:, 1]), _EPS)
    return SmoothPath(points=p, s=s_out, tangent=d / norm[:, None],
                      curvature=np.zeros(n_out))


def _resample_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Even points every ``spacing`` along the polyline, ends kept exactly."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s_src = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_src[-1])
    if total < _EPS:
        return pts[:1].copy()
    n = max(2, int(np.ceil(total / max(spacing, _EPS))) + 1)
    s_out = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(s_out, s_src, pts[:, 0]),
                            np.interp(s_out, s_src, pts[:, 1])])


def smooth_path(path: Path | np.ndarray, *, ds: float = 0.02,
                degree: int = 3, control_ds: float = 0.30) -> SmoothPath:
    """Fit a clamped B-spline to ``path`` and sample it every ``ds`` metres.

    The polyline is first resampled to control points ``control_ds`` apart.
    That spacing is the knob on corner cutting: a cubic B-spline rounds a
    corner over about two control intervals, so the deviation from the
    original polyline scales with ``control_ds``. Feeding the raw planner
    vertices in directly (metres apart, sometimes with a near-duplicate pair
    where two plan legs were stitched) rounds corners by half a metre --
    more than the clearance the plan was built with. Check the result with
    :func:`max_deviation` and shrink ``control_ds`` if it is too large.

    ``degree`` drops automatically when there are too few control points to
    support it (a cubic needs 4); below quadratic the result is the polyline
    itself, resampled, with zero curvature.
    """
    from scipy.interpolate import BSpline

    raw = np.asarray(path.points if isinstance(path, Path) else path, dtype=float)
    raw = raw.reshape(-1, 2)
    if len(raw) < 2:
        return _polyline_sample(raw if len(raw) else np.zeros((1, 2)), ds)

    pts = _resample_polyline(raw, control_ds)
    deg = min(int(degree), len(pts) - 1)
    if deg < 2:
        return _polyline_sample(raw, ds)

    spl = BSpline(_clamped_knots(len(pts), deg), pts, deg)
    d1_spl, d2_spl = spl.derivative(1), spl.derivative(2)

    # Dense pass in the spline parameter to measure arc length, then resample
    # uniformly in arc length -- the parameter is not proportional to distance,
    # and every downstream speed/acceleration limit is per metre, not per u.
    rough_len = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    n_dense = max(256, int(4 * rough_len / max(ds, _EPS)))
    u_dense = np.linspace(0.0, 1.0, n_dense)
    p_dense = spl(u_dense)
    seg = np.hypot(*np.diff(p_dense, axis=0).T)
    s_dense = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_dense[-1])
    if total < _EPS:
        return _polyline_sample(pts, ds)

    n_out = max(2, int(round(total / ds)) + 1)
    s_out = np.linspace(0.0, total, n_out)
    u_out = np.interp(s_out, s_dense, u_dense)

    p = spl(u_out)
    d1 = d1_spl(u_out)
    d2 = d2_spl(u_out)
    speed_u = np.maximum(np.hypot(d1[:, 0], d1[:, 1]), _EPS)  # |dp/du|
    tangent = d1 / speed_u[:, None]
    # planar curvature: |x' y'' - y' x''| / |p'|^3   (parameter-independent)
    cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    curvature = cross / speed_u ** 3
    return SmoothPath(points=p, s=s_out, tangent=tangent, curvature=curvature)


def max_deviation(path: Path | np.ndarray, smooth: SmoothPath) -> float:
    """Largest distance from any smoothed sample to the original polyline [m].

    This is how far corner cutting ate into the clearance the planner worked
    with: if it exceeds ``r_circ``-based margin, the smoothed curve is no
    longer guaranteed by that plan and the caller should re-check it.
    """
    pts = np.asarray(path.points if isinstance(path, Path) else path, dtype=float)
    pts = pts.reshape(-1, 2)
    if len(pts) < 2 or len(smooth) == 0:
        return 0.0
    a = pts[:-1]                      # (S,2) segment starts
    ab = pts[1:] - a                  # (S,2) segment vectors
    ab_len2 = np.maximum((ab ** 2).sum(axis=1), _EPS)
    worst = 0.0
    for q in smooth.points:
        t = np.clip(((q - a) * ab).sum(axis=1) / ab_len2, 0.0, 1.0)
        closest = a + t[:, None] * ab
        worst = max(worst, float(np.hypot(*(q - closest).T).min()))
    return worst


def speed_limit_from_curvature(curvature: np.ndarray, a_lat_max: float,
                               v_max: float) -> np.ndarray:
    """Per-sample speed cap from a lateral-acceleration budget: v <= sqrt(a/k).

    The continuous version of ``corner.corner_speed_limits``, which can only
    evaluate this at polyline vertices because a polyline has no curvature
    anywhere else (it is zero on the legs and undefined at the corners).
    """
    k = np.maximum(np.asarray(curvature, dtype=float), _EPS)
    return np.minimum(v_max, np.sqrt(a_lat_max / k))

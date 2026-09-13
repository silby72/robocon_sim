"""Corner speed limits for a polyline path (DECISIONS.md §2.9).

The smoothed path is still a polyline -- it has corners. Rather than curvature
continuity (B-splines, deferred), we cap the speed at each corner so a trapezoid
profile does not fly through it:

    v_corner <= sqrt(a_max * R)
    R = delta * cos(phi/2) / (1 - cos(phi/2))

``phi`` is the deflection (turn) angle at the vertex (0 = straight, pi = U-turn)
and ``delta`` is the allowed lateral deviation from the corner. A straight vertex
gives R = inf (no limit); a U-turn gives R = 0 (full stop).

The tangent offset ``R * tan(phi/2)`` is where the inscribed arc meets each leg.
Two sharp corners a short segment apart will have arcs that eat into each other;
we clamp R so the offset never exceeds half the shorter adjacent segment
(P1 acceptance #11). This is unavoidable on consecutive tight corners.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _turn_angles(points: np.ndarray) -> np.ndarray:
    """Deflection angle phi at each vertex; endpoints get 0 (no corner)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    phi = np.zeros(n)
    for i in range(1, n - 1):
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        na, nb = np.hypot(*a), np.hypot(*b)
        if na < _EPS or nb < _EPS:
            continue
        cos_t = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        phi[i] = np.arccos(cos_t)
    return phi


def corner_radii(points: np.ndarray, delta: float = 0.05
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, phi) per vertex, with R clamped so arcs never overlap (§2.9).

    Endpoints and straight vertices get ``R = inf`` (no corner constraint).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    phi = _turn_angles(pts)
    R = np.full(n, np.inf)
    if n < 3:
        return R, phi

    seg_len = np.hypot(*np.diff(pts, axis=0).T)  # length n-1: seg i = pts[i]->pts[i+1]
    for i in range(1, n - 1):
        p = phi[i]
        if p < 1e-6:
            continue  # effectively straight -> no limit
        c = np.cos(p / 2.0)
        Ri = delta * c / (1.0 - c)
        # clamp: tangent offset R*tan(phi/2) <= half the shorter adjacent segment
        half_min_seg = 0.5 * min(seg_len[i - 1], seg_len[i])
        tan_half = np.tan(p / 2.0)
        max_R = half_min_seg / tan_half if tan_half > _EPS else np.inf
        R[i] = min(Ri, max_R)
    return R, phi


def corner_speed_limits(points: np.ndarray, a_max: float,
                        delta: float = 0.05) -> np.ndarray:
    """Per-vertex speed cap ``sqrt(a_max * R)`` [m/s]; inf where unconstrained."""
    R, _ = corner_radii(points, delta)
    return np.sqrt(a_max * R)


def tangent_offsets(points: np.ndarray, delta: float = 0.05) -> np.ndarray:
    """Tangent offset ``R * tan(phi/2)`` per vertex (0 at endpoints/straights).

    After clamping this is <= half the shorter adjacent segment everywhere, which
    is exactly what "arcs do not eat into each other" means (acceptance #11).
    """
    pts = np.asarray(points, dtype=float)
    R, phi = corner_radii(pts, delta)
    off = np.zeros(len(pts))
    interior = np.isfinite(R) & (phi > 1e-6)
    off[interior] = R[interior] * np.tan(phi[interior] / 2.0)
    return off

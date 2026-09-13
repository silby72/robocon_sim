"""Analytic minimum distance between convex shapes (依頼 §4.2).

Collision judging is done against the *true* geometry, not the occupancy grid --
so discretisation error never leaks into the verdict, and the planner is not
grading itself with its own inflation radius. Everything here is exact.

All obstacle primitives (box, segment, cylinder) and the robot footprint are
convex, so:

    * polygon vs polygon : SAT for overlap (=> 0), else min edge-pair distance
    * polygon vs circle  : point-to-polygon distance of the centre, minus r

Distances are >= 0; 0 means touching or overlapping.
"""
from __future__ import annotations

import numpy as np


def _point_segment_distance(p, a, b) -> float:
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-15:
        return float(np.hypot(*(p - a)))
    t = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)
    proj = a + t * ab
    return float(np.hypot(*(p - proj)))


def _segment_segment_distance(p1, p2, p3, p4) -> float:
    """Shortest distance between segments p1p2 and p3p4 (0 if they cross)."""
    p1, p2, p3, p4 = map(np.asarray, (p1, p2, p3, p4))
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    if a < 1e-15 and e < 1e-15:
        return float(np.hypot(*(p1 - p3)))
    if a < 1e-15:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-15:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-15 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    c1 = p1 + d1 * s
    c2 = p3 + d2 * t
    return float(np.hypot(*(c1 - c2)))


def _point_in_convex(p, poly) -> bool:
    """True if point p is inside convex polygon poly (CCW or CW)."""
    p = np.asarray(p)
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _sat_overlap(A, B) -> bool:
    """Separating-axis test for two convex polygons (touching counts as overlap)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            L = np.hypot(*axis)
            if L < 1e-15:
                continue
            axis = axis / L
            pa = A @ axis
            pb = B @ axis
            if pa.max() < pb.min() - 1e-12 or pb.max() < pa.min() - 1e-12:
                return False   # a separating axis exists -> no overlap
    return True


def polygon_polygon_distance(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if _sat_overlap(A, B):
        return 0.0
    best = np.inf
    na, nb = len(A), len(B)
    for i in range(na):
        for j in range(nb):
            d = _segment_segment_distance(A[i], A[(i + 1) % na],
                                          B[j], B[(j + 1) % nb])
            if d < best:
                best = d
    return float(best)


def point_to_polygon_distance(p, poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float)
    if _point_in_convex(p, poly):
        return 0.0
    n = len(poly)
    return min(_point_segment_distance(p, poly[i], poly[(i + 1) % n])
               for i in range(n))


def polygon_circle_distance(poly: np.ndarray, center, r: float) -> float:
    return float(max(point_to_polygon_distance(center, poly) - r, 0.0))

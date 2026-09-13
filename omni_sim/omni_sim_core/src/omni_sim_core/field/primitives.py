"""2D geometric primitives with a height interval (依頼 §3.1).

The field is never modelled in 3D. Each primitive is a 2D shape (box / cylinder /
segment) annotated with the height band ``[z_lo, z_hi]`` it occupies. A single
definition feeds two consumers:

    * the slicer rasterises it into occupancy grids (planning)
    * the evaluator asks it for its exact 2D shape and min-distance (collision)

``shape_2d`` is the single source of truth for both -- rasterisation and distance
are derived from it, so they can never disagree about where an obstacle is.

Units are SI (metres) here; the spec loader converts from mm on the way in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion

# logical cell states (kept small; mapped to PGM bytes / OccupancyGrid elsewhere)
FREE = np.uint8(0)
OCC = np.uint8(1)
UNKNOWN = np.uint8(2)


@dataclass(frozen=True)
class Primitive:
    """A 2D shape occupying ``[z_lo, z_hi]`` in height. Base for box/cylinder/segment."""

    name: str
    z_lo: float
    z_hi: float

    # -- height queries ---------------------------------------------------
    def contains_z(self, z: float) -> bool:
        """Does the plane at height ``z`` pass through this primitive?"""
        return self.z_lo <= z <= self.z_hi

    def overlaps_band(self, lo: float, hi: float) -> bool:
        """Does the height band ``[lo, hi]`` intersect this primitive?"""
        return self.z_hi > lo and self.z_lo < hi

    # -- geometry (subclasses implement) ----------------------------------
    def shape_2d(self):
        raise NotImplementedError

    def filled_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def distance_to_point(self, p) -> float:
        """Unsigned distance from point ``p`` to the shape (0 if inside)."""
        raise NotImplementedError

    def surface_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Outer 1-cell ring of the filled shape (what a LiDAR sees, §3.3).

        ``filled XOR erosion(filled)`` is the outermost occupied layer. For a
        shape already 1 cell thin, erosion empties it and the surface is the
        shape itself (thin walls survive -- P2 acceptance #6).
        """
        filled = self.filled_mask(X, Y)
        if not filled.any():
            return filled
        eroded = binary_erosion(filled)
        return filled & ~eroded


@dataclass(frozen=True)
class Box(Primitive):
    """Axis-aligned rectangle: centre (cx, cy), full size (w, h)."""

    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0

    def shape_2d(self):
        x0, x1 = self.cx - self.w / 2, self.cx + self.w / 2
        y0, y1 = self.cy - self.h / 2, self.cy + self.h / 2
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return ("polygon", corners)

    def filled_mask(self, X, Y):
        x0, x1 = self.cx - self.w / 2, self.cx + self.w / 2
        y0, y1 = self.cy - self.h / 2, self.cy + self.h / 2
        return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)

    def distance_to_point(self, p) -> float:
        px, py = p
        dx = max(abs(px - self.cx) - self.w / 2, 0.0)
        dy = max(abs(py - self.cy) - self.h / 2, 0.0)
        return float(np.hypot(dx, dy))


@dataclass(frozen=True)
class Cylinder(Primitive):
    """Circle: centre (cx, cy), diameter d."""

    cx: float = 0.0
    cy: float = 0.0
    d: float = 0.0

    @property
    def r(self) -> float:
        return self.d / 2.0

    def shape_2d(self):
        return ("circle", np.array([self.cx, self.cy]), self.r)

    def filled_mask(self, X, Y):
        return (X - self.cx) ** 2 + (Y - self.cy) ** 2 <= self.r ** 2

    def distance_to_point(self, p) -> float:
        px, py = p
        return float(max(np.hypot(px - self.cx, py - self.cy) - self.r, 0.0))


@dataclass(frozen=True)
class Segment(Primitive):
    """A wall of thickness ``t`` centred on the line (x0,y0)->(x1,y1).

    Represented as a (possibly rotated) rectangle so a diagonal divider works.
    """

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    t: float = 0.05

    def _rect_corners(self):
        p0 = np.array([self.x0, self.y0])
        p1 = np.array([self.x1, self.y1])
        d = p1 - p0
        L = np.hypot(*d)
        if L < 1e-12:
            n = np.array([0.0, 0.0])
            u = np.array([0.0, 0.0])
        else:
            u = d / L
            n = np.array([-u[1], u[0]])
        h = self.t / 2.0
        return np.array([p0 + n * h, p1 + n * h, p1 - n * h, p0 - n * h])

    def shape_2d(self):
        return ("polygon", self._rect_corners())

    def filled_mask(self, X, Y):
        return _point_in_polygon(X, Y, self._rect_corners())

    def distance_to_point(self, p) -> float:
        # distance to the centre segment, minus half thickness (clamped at 0)
        px, py = p
        a = np.array([self.x0, self.y0])
        b = np.array([self.x1, self.y1])
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else np.clip(((np.array([px, py]) - a) @ ab) / L2, 0, 1)
        proj = a + t * ab
        return float(max(np.hypot(px - proj[0], py - proj[1]) - self.t / 2.0, 0.0))


def _point_in_polygon(X: np.ndarray, Y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon over grids X, Y (same shape)."""
    inside = np.zeros(X.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > Y) != (yj > Y))
        # avoid divide-by-zero on horizontal edges
        denom = np.where(np.abs(yj - yi) < 1e-15, 1e-15, yj - yi)
        x_cross = (xj - xi) * (Y - yi) / denom + xi
        inside ^= cond & (X < x_cross)
        j = i
    return inside

"""Robot footprint -- the REAL rectangle, orientation included (依頼 §4.2).

Planning is done with the circumscribed circle (orientation-independent, §2.5),
but judging is done with the true rectangle. "Plan with the circle, judge with
the rectangle" is the correct asymmetry: the circle is conservative, so if the
real footprint collides, the bug is on the planning side, not here.

Because judging uses the rotated rectangle, min clearance genuinely depends on
theta (P3 acceptance #3), which is the whole point of decoupling heading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..field.primitives import Primitive
from .geometry_dist import polygon_circle_distance, polygon_polygon_distance


@dataclass(frozen=True)
class RobotFootprint:
    length: float          # [m] extent along body +x
    width: float           # [m] extent along body +y

    @property
    def r_circ(self) -> float:
        """Circumscribed radius -- the value planning should use for r_circ."""
        return 0.5 * float(np.hypot(self.length, self.width))

    def corners(self, x: float, y: float, theta: float) -> np.ndarray:
        """World-frame corners of the rectangle at pose (x, y, theta)."""
        hl, hw = self.length / 2.0, self.width / 2.0
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (local @ R.T) + np.array([x, y])

    def distance_to(self, prim: Primitive, x: float, y: float,
                    theta: float) -> float:
        """Exact min distance from the footprint at (x,y,theta) to a primitive."""
        poly = self.corners(x, y, theta)
        shape = prim.shape_2d()
        if shape[0] == "circle":
            _, center, r = shape
            return polygon_circle_distance(poly, center, r)
        return polygon_polygon_distance(poly, shape[1])

    def min_distance(self, prims, x: float, y: float, theta: float):
        """Return (min_distance, nearest_primitive_name) over ``prims``."""
        best_d, best_name = np.inf, None
        for p in prims:
            d = self.distance_to(p, x, y, theta)
            if d < best_d:
                best_d, best_name = d, p.name
        return float(best_d), best_name

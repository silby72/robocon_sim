"""Line-of-sight shortcut (DECISIONS.md §2.3-2.4, spec §2.4).

Any-angle is done the *decoupled* way: run A* first, then greedily shortcut the
grid-staircase polyline. Keeping search and smoothing separate lets each be
tested in isolation (§2.3); Theta* fuses them and is deferred.

The line-of-sight test reuses the existing DDA in ``env/raycast`` -- we do NOT
write a second raycaster. It is cast against the *inflated* (lethal) grid, so a
shortcut can never cut inside the safety margin (P1 acceptance #9).

Endpoint handling is the classic footgun: whether a wall-hugging segment counts
as blocked depends on whether the endpoint cells are included. We include BOTH
endpoints (the conservative choice) and pin it with a test. ``raycast`` returns 0
when the *start* cell is occupied and, with ``max_range = d``, any lethal cell
the segment enters (including the goal cell, whose entry distance <= d) yields a
finite range -- so both ends are covered by construction.
"""
from __future__ import annotations

import numpy as np

from ..env.raycast import raycast
from .cost_field import CostField
from .types import Path


def line_of_sight(cost_field: CostField, p0, p1) -> bool:
    """True if the straight segment ``p0 -> p1`` crosses no lethal cell.

    Both endpoint cells are included (conservative). Uses the shared DDA.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = float(np.hypot(*(p1 - p0)))
    lethal_grid = cost_field.as_lethal_grid()
    if d < 1e-9:
        return not cost_field.is_lethal_world(p0[0], p0[1])
    angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    # cast just to the goal point; any lethal cell entered within d -> finite.
    rng = raycast(lethal_grid, p0[0], p0[1], np.array([angle]), max_range=d)[0]
    return not np.isfinite(rng)


def shortcut_path(cost_field: CostField, path: Path) -> Path:
    """Greedy any-angle shortcut. Result length <= input length (§ acceptance #10).

    From each kept vertex, jump to the farthest still-visible vertex. The output
    is a subset of the input vertices, so by the triangle inequality it is no
    longer than the input and (line-of-sight against the lethal grid) collision
    free.
    """
    pts = path.points
    n = len(pts)
    if n <= 2:
        return Path(pts.copy())

    kept = [0]
    i = 0
    while i < n - 1:
        # farthest j > i visible from i
        j_best = i + 1
        for j in range(n - 1, i, -1):
            if line_of_sight(cost_field, pts[i], pts[j]):
                j_best = j
                break
        kept.append(j_best)
        i = j_best
    return Path(pts[kept])

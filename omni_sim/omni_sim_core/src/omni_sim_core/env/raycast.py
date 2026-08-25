"""Vectorised grid raycasting (DDA / Amanatides-Woo).

All beams advance together as numpy arrays. A pure-Python per-beam loop cannot
sustain 450 beams at 10 Hz, so the outer loop is over DDA *steps* (bounded by
range/resolution) and every operation inside is vectorised across beams.

Returns ranges in metres; a beam that reaches ``max_range`` without hitting an
occupied cell returns ``inf`` (matching the ``LaserScan`` convention).
"""
from __future__ import annotations

import numpy as np

from .occupancy_grid import OCCUPIED, OccupancyGrid


def raycast(grid: OccupancyGrid, x: float, y: float, angles_world: np.ndarray,
            max_range: float) -> np.ndarray:
    res = grid.meta.resolution
    ox, oy, _ = grid.meta.origin
    angles = np.asarray(angles_world, dtype=float)
    n = angles.size

    # robustness: a non-finite origin (e.g. a diverged pose) has no valid cast
    if not (np.isfinite(x) and np.isfinite(y)):
        return np.full(n, np.inf)

    dx = np.cos(angles)
    dy = np.sin(angles)

    # single origin, one cell shared by all beams -> broadcast to per-beam arrays
    col = np.full(n, int(np.floor((x - ox) / res)), dtype=np.int64)
    row = np.full(n, int(np.floor((y - oy) / res)), dtype=np.int64)

    step_x = np.where(dx >= 0, 1, -1)
    step_y = np.where(dy >= 0, 1, -1)

    with np.errstate(divide="ignore"):
        t_delta_x = np.where(dx != 0, res / np.abs(dx), np.inf)
        t_delta_y = np.where(dy != 0, res / np.abs(dy), np.inf)

        x_next = ox + np.where(dx >= 0, col + 1, col) * res
        y_next = oy + np.where(dy >= 0, row + 1, row) * res
        t_max_x = np.where(dx != 0, (x_next - x) / dx, np.inf)
        t_max_y = np.where(dy != 0, (y_next - y) / dy, np.inf)

    ranges = np.full(n, np.inf)
    active = np.ones(n, dtype=bool)

    # If the starting cell is already occupied, range is 0.
    start_occ = np.array([grid.is_occupied(int(r), int(c))
                          for r, c in zip(row, col)])
    ranges[start_occ] = 0.0
    active &= ~start_occ

    max_steps = int(2 * max_range / res) + 2
    for _ in range(max_steps):
        if not active.any():
            break
        step_in_x = t_max_x <= t_max_y
        t_cross = np.where(step_in_x, t_max_x, t_max_y)

        # advance cell indices
        col = np.where(active & step_in_x, col + step_x, col)
        row = np.where(active & ~step_in_x, row + step_y, row)
        t_max_x = np.where(active & step_in_x, t_max_x + t_delta_x, t_max_x)
        t_max_y = np.where(active & ~step_in_x, t_max_y + t_delta_y, t_max_y)

        # beams that exceeded range -> no hit
        beyond = active & (t_cross > max_range)
        active &= ~beyond

        in_bounds = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
        oob = active & ~in_bounds
        active &= ~oob  # left the map without hitting -> inf

        # occupancy lookup only for in-bounds active beams
        check = active & in_bounds
        if check.any():
            idx = np.where(check)[0]
            occ = grid.grid[row[idx], col[idx]] == OCCUPIED
            hit_idx = idx[occ]
            ranges[hit_idx] = t_cross[hit_idx]
            active[hit_idx] = False

    return ranges

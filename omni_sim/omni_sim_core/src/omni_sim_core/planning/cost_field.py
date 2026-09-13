"""Cost field derived from an occupancy grid (DECISIONS.md §2.4).

One EDT, computed once, drives *both* collision (hard inflation) and preference
(soft cost). The two are kept as separate concepts on purpose -- hard guarantees
safety, soft merely nudges the path off the wall. Merging them is how you end up
either scraping walls or finding no path at all.

    cost(u -> v) = ||v - u|| * (1 + c(v))          # multiplicative, §2.4

Multiplicative (not additive) so the weight ``w`` does not have to be re-tuned
when the map size changes.

The subtle trap (§2.4): an occupancy grid stores an obstacle's *outline*; its
interior is "unknown" == free. Left alone, the EDT gives interior cells a
positive distance-to-obstacle, so the hard inflation silently switches off inside
the obstacle and the planner routes through its middle. The interior must be
sealed to occupied so its EDT distance is 0 (P1 acceptance #2).

The spec suggested ``binary_fill_holes`` for this, but that fills *any* enclosed
background -- including a bordered arena's entire drivable interior (the wall
ring encloses it), which is catastrophic. There is no purely topological way to
tell an obstacle interior from a room or a wall-split half. So we seal a free
component only when it is (a) enclosed (does not touch the grid border) AND
(b) smaller than ``seal_area_frac`` of the map. Obstacle interiors are small and
enclosed -> sealed; drivable areas and split halves are large -> kept free.
This was raised with the user and is the chosen semantics (see PlanConfig).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from ..env.occupancy_grid import OCCUPIED, OccupancyGrid
from .types import PlanConfig

# sentinel for a lethal cell in the per-cell cost array
LETHAL = np.inf


class CostField:
    """Derives lethal mask + soft cost from an occupancy grid, once."""

    def __init__(self, grid: OccupancyGrid, cfg: PlanConfig) -> None:
        self.grid = grid
        self.cfg = cfg
        self.resolution = grid.meta.resolution

        occupied = grid.grid == OCCUPIED
        # §2.4 trap: seal small enclosed free pockets (obstacle interiors) BEFORE
        # the EDT, but never seal large / border-touching free space (arenas).
        occupied = self._seal_enclosed_pockets(occupied, cfg.seal_area_frac)
        self.occupied = np.ascontiguousarray(occupied)

        free = ~self.occupied
        # distance (in metres) from each cell to the nearest occupied cell.
        # occupied cells get 0 (they are the zeros of ``free``).
        self.dist = distance_transform_edt(free) * self.resolution

        r_circ = cfg.r_circ
        d_cutoff = cfg.d_cutoff
        if d_cutoff is None:
            # exp(-k*(d-r_circ)) has decayed to exp(-3) ~ 5% at this distance.
            d_cutoff = r_circ + 3.0 / cfg.soft_k
        self.d_cutoff = d_cutoff

        # hard: occupied OR within a circumscribed radius of an obstacle.
        self.lethal = self.occupied | (self.dist < r_circ)

        # soft: r_circ <= d < d_cutoff, decaying away from the wall.
        soft = (self.dist >= r_circ) & (self.dist < d_cutoff)
        c = np.zeros_like(self.dist)
        c[soft] = cfg.soft_weight * np.exp(-cfg.soft_k * (self.dist[soft] - r_circ))
        self.soft_cost = c

        self.height, self.width = self.lethal.shape

    # -- construction helpers ---------------------------------------------
    @staticmethod
    def _seal_enclosed_pockets(occupied: np.ndarray, area_frac: float) -> np.ndarray:
        """Mark as occupied any free component that is enclosed AND small.

        Enclosed == does not touch the grid border. Small == area below
        ``area_frac`` of the whole map. Both conditions must hold, so obstacle
        interiors get sealed while a bordered arena's interior (large) and a
        wall-split half (border-touching) stay free.
        """
        if area_frac <= 0.0:
            return occupied.copy()
        free = ~occupied
        labels, n = label(free)  # 4-connectivity is fine (conservative) here
        if n == 0:
            return occupied.copy()
        border = np.unique(np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        border_set = set(int(b) for b in border)
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        threshold = area_frac * free.size
        sealed = occupied.copy()
        for lab in range(1, n + 1):
            if lab in border_set:
                continue                       # open to the world -> drivable
            if sizes[lab] < threshold:
                sealed |= labels == lab        # enclosed + small -> obstacle interior
        return sealed

    # -- queries ----------------------------------------------------------
    def is_lethal_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.height and 0 <= col < self.width):
            return True
        return bool(self.lethal[row, col])

    def is_lethal_world(self, x: float, y: float) -> bool:
        row, col = self.grid.world_to_grid(x, y)
        return self.is_lethal_cell(row, col)

    def as_lethal_grid(self) -> OccupancyGrid:
        """The lethal mask as an OccupancyGrid, so ``env/raycast`` can walk it.

        Used by the line-of-sight shortcut (§2.4) -- it must test the *inflated*
        obstacles, not the raw ones, or a shortcut would clip the safety margin.
        """
        g = np.where(self.lethal, OCCUPIED, np.uint8(0)).astype(np.uint8)
        return OccupancyGrid(g, self.grid.meta)

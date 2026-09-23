"""Likelihood field for 2D LiDAR, built from the *localization* raster.

``field/slicer.py`` already produces two rasters per layer for exactly this
reason, and getting them the wrong way round fails quietly:

    nav  filled -- a wall's interior is occupied, because a planner must not
         route through it.
    loc  surface only, 1 cell thick -- because this is what a LiDAR can see.
         Score against the filled raster and a pose buried inside a wall reads
         as a perfect match: every beam ends on "occupied".

So this consumes ``grid(layer, "loc")``. There are 1828 occupied cells in the
2027 ground layer against 92324 in the nav raster; the difference is the point.

The model is the standard likelihood field: precompute the distance from every
cell to the nearest surface, then score a beam by how far its endpoint lands
from one. That is a lookup per beam instead of a raycast per beam, which is
what makes a few hundred particles affordable at LiDAR rate. It ignores which
surface was hit and cannot model occlusion, both of which are fine here and
neither of which a beam model would get right either at this map resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..env.occupancy_grid import OCCUPIED, OccupancyGrid


@dataclass(frozen=True)
class LikelihoodFieldParams:
    sigma_hit_m: float = 0.08      # how forgiving a beam endpoint is
    z_hit: float = 0.85            # weight of the "hit a real surface" term
    z_rand: float = 0.15           # flat term: clutter, unmapped objects
    max_dist_m: float = 2.0        # distances are clipped here (and so is cost)
    beam_stride: int = 6           # use every Nth beam
    max_beam_range_m: float = 12.0


class LikelihoodField:
    """A scorer for one map. Swap it when the robot changes level."""

    def __init__(self, grid: OccupancyGrid,
                 params: LikelihoodFieldParams | None = None) -> None:
        self.grid = grid
        self.p = params or LikelihoodFieldParams()
        res = grid.meta.resolution
        occupied = grid.grid == OCCUPIED
        if not occupied.any():
            raise ValueError("the localization raster has no surfaces at all; "
                             "this looks like a nav raster or an empty map")
        dist = distance_transform_edt(~occupied) * res
        np.clip(dist, 0.0, self.p.max_dist_m, out=dist)
        # Precomputed log-probability per cell. Doing the exponential here
        # rather than per beam per particle is most of the speed.
        hit = np.exp(-0.5 * (dist / self.p.sigma_hit_m) ** 2)
        prob = self.p.z_hit * hit + self.p.z_rand / self.p.max_beam_range_m
        self._logp = np.log(prob)
        self._floor = float(self._logp.min())   # for endpoints off the map

    @property
    def n_surface_cells(self) -> int:
        return int((self.grid.grid == OCCUPIED).sum())

    def log_likelihood(self, poses: np.ndarray, angles: np.ndarray,
                       ranges: np.ndarray, mount=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Score every pose in ``poses`` (Nx3) against one scan, vectorised.

        Beams that returned no hit (``inf``) carry no information about where
        the robot is -- they say only "nothing within range along this ray" --
        so they are dropped rather than scored as a maximally bad endpoint,
        which would punish poses in open space for being in open space.
        """
        poses = np.atleast_2d(np.asarray(poses, dtype=float))
        angles = np.asarray(angles, dtype=float)
        ranges = np.asarray(ranges, dtype=float)

        s = self.p.beam_stride
        a, r = angles[::s], ranges[::s]
        good = np.isfinite(r) & (r < self.p.max_beam_range_m)
        a, r = a[good], r[good]
        if a.size == 0:
            return np.zeros(len(poses))

        mx, my, myaw = mount
        th = poses[:, 2][:, None]                       # (N, 1)
        # sensor origin in world, per pose
        sx = poses[:, 0][:, None] + np.cos(th) * mx - np.sin(th) * my
        sy = poses[:, 1][:, None] + np.sin(th) * mx + np.cos(th) * my
        beam = th + myaw + a[None, :]                   # (N, B)
        ex = sx + r[None, :] * np.cos(beam)
        ey = sy + r[None, :] * np.sin(beam)

        res = self.grid.meta.resolution
        ox, oy, _ = self.grid.meta.origin
        cols = np.floor((ex - ox) / res).astype(np.intp)
        rows = np.floor((ey - oy) / res).astype(np.intp)
        inb = ((rows >= 0) & (rows < self.grid.height)
               & (cols >= 0) & (cols < self.grid.width))
        out = np.full(rows.shape, self._floor)
        out[inb] = self._logp[rows[inb], cols[inb]]
        return out.sum(axis=1)

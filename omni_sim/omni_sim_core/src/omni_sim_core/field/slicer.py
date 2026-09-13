"""Slice height-annotated primitives into occupancy grids (依頼 §3.3).

Two purposes, built *the opposite way* -- getting them backwards fails silently:

    localization ``<name>``  : cross-section at floor + lidar_height.
                               SURFACE only (1 cell), interior + exterior UNKNOWN.
                               Filling it would let the likelihood field score a
                               pose buried inside a wall as a perfect match.

    nav ``<name>_nav``       : the height band [floor+band_lo, floor+band_hi].
                               FILLED. Outside the layer's drivable extent is
                               LETHAL (a drop). Leaving it as a hollow outline
                               would turn a wall interior into a fake pocket the
                               planner's escape logic burrows into.

The band selection is what makes layering work without any 3D: a primitive is an
obstacle *for a layer* only if it protrudes into that layer's robot band. The
layer's own platform sits below the band, so it is never stamped as an obstacle.
"""
from __future__ import annotations

import numpy as np

from .primitives import FREE, OCC, UNKNOWN, Primitive
from .spec import FieldSpec, LayerSpec


def grid_axes(spec: FieldSpec) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Cell-centre coordinate grids X, Y (row 0 = bottom, +y up), and shape."""
    res = spec.resolution
    ox, oy = spec.origin
    cols = int(round(spec.size[0] / res))
    rows = int(round(spec.size[1] / res))
    xs = ox + (np.arange(cols) + 0.5) * res
    ys = oy + (np.arange(rows) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)   # X[r,c], Y[r,c]; Y increases with r
    return X, Y, rows, cols


def slice_localization(spec: FieldSpec, prims: list[Primitive],
                       layer: LayerSpec) -> np.ndarray:
    """Trinary raster: UNKNOWN everywhere, OCC on the surfaces the LiDAR sees."""
    X, Y, rows, cols = grid_axes(spec)
    z = layer.floor_z + spec.lidar_height
    raster = np.full((rows, cols), UNKNOWN, dtype=np.uint8)
    for p in prims:
        if p.contains_z(z):
            raster[p.surface_mask(X, Y)] = OCC
    return raster


def slice_nav(spec: FieldSpec, prims: list[Primitive],
              layer: LayerSpec) -> np.ndarray:
    """Trinary raster: FREE on the drivable surface, OCC on obstacles + drops."""
    X, Y, rows, cols = grid_axes(spec)
    lo = layer.floor_z + spec.robot_band[0]
    hi = layer.floor_z + spec.robot_band[1]

    raster = np.full((rows, cols), FREE, dtype=np.uint8)
    if layer.extent is not None:
        x0, y0, x1, y1 = layer.extent
        off_slab = ~((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
        raster[off_slab] = OCC          # off the platform is a fall -> lethal
    for hx0, hy0, hx1, hy1 in layer.holes:
        hole = (X >= hx0) & (X <= hx1) & (Y >= hy0) & (Y <= hy1)
        raster[hole] = OCC              # a hole punched in the slab is a drop too
    for p in prims:
        if p.overlaps_band(lo, hi):
            raster[p.filled_mask(X, Y)] = OCC
    return raster

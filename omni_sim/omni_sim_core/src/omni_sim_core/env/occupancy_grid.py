"""Occupancy grid, ROS ``map_server`` compatible (PGM + YAML).

The PGM is read with Pillow/numpy (no OpenCV dependency). Occupancy is stored as
a uint8 grid: 100 = occupied, 0 = free (unknown is treated as free here).
Grid row 0 corresponds to the *bottom* of the image, matching the ROS map
convention where ``origin`` is the world coordinate of the lower-left pixel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

OCCUPIED = np.uint8(100)
FREE = np.uint8(0)


@dataclass
class MapMetadata:
    resolution: float
    origin: tuple[float, float, float]
    occupied_thresh: float = 0.65
    free_thresh: float = 0.196
    negate: int = 0


class OccupancyGrid:
    def __init__(self, grid: np.ndarray, meta: MapMetadata) -> None:
        # grid[row, col], row index increases with world +y
        self.grid = np.ascontiguousarray(grid, dtype=np.uint8)
        self.meta = meta
        self.height, self.width = self.grid.shape

    # -- construction -----------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "OccupancyGrid":
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        meta = MapMetadata(
            resolution=float(cfg["resolution"]),
            origin=tuple(cfg["origin"]),
            occupied_thresh=float(cfg.get("occupied_thresh", 0.65)),
            free_thresh=float(cfg.get("free_thresh", 0.196)),
            negate=int(cfg.get("negate", 0)),
        )
        img_path = (yaml_path.parent / cfg["image"]).resolve()
        img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float64)
        # ROS: p_occ = (255 - x) / 255 (or x/255 if negate). Image row 0 is top,
        # world +y is up, so flip vertically.
        img = np.flipud(img)
        if meta.negate:
            p_occ = img / 255.0
        else:
            p_occ = (255.0 - img) / 255.0
        grid = np.where(p_occ >= meta.occupied_thresh, OCCUPIED, FREE).astype(np.uint8)
        return cls(grid, meta)

    # -- coordinate transforms -------------------------------------------
    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        ox, oy, _ = self.meta.origin
        col = int((x - ox) / self.meta.resolution)
        row = int((y - oy) / self.meta.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        ox, oy, _ = self.meta.origin
        x = ox + (col + 0.5) * self.meta.resolution
        y = oy + (row + 0.5) * self.meta.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_occupied(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return bool(self.grid[row, col] == OCCUPIED)


def generate_rect_field(width_m: float, height_m: float, resolution: float,
                        wall_thickness_cells: int = 1,
                        origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
                        ) -> OccupancyGrid:
    """Empty rectangular arena bordered by walls -- works with no map file."""
    cols = int(round(width_m / resolution))
    rows = int(round(height_m / resolution))
    grid = np.full((rows, cols), FREE, dtype=np.uint8)
    t = wall_thickness_cells
    grid[:t, :] = OCCUPIED
    grid[-t:, :] = OCCUPIED
    grid[:, :t] = OCCUPIED
    grid[:, -t:] = OCCUPIED
    meta = MapMetadata(resolution=resolution, origin=origin)
    return OccupancyGrid(grid, meta)

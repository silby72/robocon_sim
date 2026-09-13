"""LayeredField -- one spec, many occupancy grids + the true geometry (§3.4).

Not a 3D physics model: just ``layer -> raster`` dictionaries plus the list of
SI primitives. ``grid(layer, purpose)`` hands the planner an
:class:`OccupancyGrid`; ``geometry(layer)`` hands the evaluator the exact shapes
that can collide at that layer (P3 checks against these, never the grid, so it is
not grading its own discretisation -- §3.1).

Rasters are trinary (free / occupied / unknown) so localization can keep the
"unknown" distinction that RViz shows and P2 acceptance #2 checks. An
``OccupancyGrid`` is binary, so unknown folds into free there -- which is exactly
right: only the surface cells should read as obstacles for the likelihood field.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..env.occupancy_grid import FREE as OG_FREE
from ..env.occupancy_grid import OCCUPIED as OG_OCC
from ..env.occupancy_grid import MapMetadata, OccupancyGrid
from .primitives import OCC, Primitive
from .slicer import slice_localization, slice_nav
from .spec import FieldSpec

# PGM byte values: black = occupied, mid-grey = unknown, white = free.
# Chosen so env/occupancy_grid.from_yaml reads occupied as OCCUPIED and both
# unknown and free as FREE (unknown 205 -> p_occ 0.196 < occupied_thresh).
_PGM = {0: 254, 1: 0, 2: 205}   # {FREE: white, OCC: black, UNKNOWN: grey}


class LayeredField:
    """Occupancy grids sliced from a :class:`FieldSpec`, per layer & purpose."""

    def __init__(self, spec: FieldSpec, unknowns: dict[str, float] | None = None):
        self.spec = spec
        self.unknowns = dict(unknowns or {})
        self.primitives = spec.build_primitives(self.unknowns)
        self._raster_cache: dict[tuple[str, str], np.ndarray] = {}

    @classmethod
    def from_yaml(cls, path, unknowns=None) -> "LayeredField":
        return cls(FieldSpec.from_yaml(path), unknowns)

    # -- rasters ----------------------------------------------------------
    def raster(self, layer: str, purpose: str = "nav") -> np.ndarray:
        """Trinary raster (row 0 = bottom). purpose in {'nav', 'loc'}."""
        key = (layer, purpose)
        if key not in self._raster_cache:
            ls = self.spec.layers[layer]
            if purpose == "nav":
                r = slice_nav(self.spec, self.primitives, ls)
            elif purpose in ("loc", "localization"):
                r = slice_localization(self.spec, self.primitives, ls)
            else:
                raise ValueError(f"purpose must be 'nav' or 'loc', got {purpose!r}")
            self._raster_cache[key] = r
        return self._raster_cache[key]

    def _meta(self) -> MapMetadata:
        ox, oy = self.spec.origin
        return MapMetadata(resolution=self.spec.resolution, origin=(ox, oy, 0.0))

    def grid(self, layer: str, purpose: str = "nav") -> OccupancyGrid:
        """OccupancyGrid for planning/raycast (unknown folds into free)."""
        r = self.raster(layer, purpose)
        g = np.where(r == OCC, OG_OCC, OG_FREE).astype(np.uint8)
        return OccupancyGrid(g, self._meta())

    # -- true geometry (for evaluation) -----------------------------------
    def geometry(self, layer: str) -> list[Primitive]:
        """Primitives that can collide with the robot at ``layer`` (§3.4)."""
        ls = self.spec.layers[layer]
        lo = ls.floor_z + self.spec.robot_band[0]
        hi = ls.floor_z + self.spec.robot_band[1]
        return [p for p in self.primitives if p.overlaps_band(lo, hi)]

    # -- output (map_server compatible) -----------------------------------
    def write(self, out_dir: str | Path, layers=None, purposes=("nav", "loc"),
              prefix: str = "field") -> list[Path]:
        """Write PGM + YAML per (layer, purpose); returns the YAML paths.

        Round-trips through env/occupancy_grid.from_yaml (P2 acceptance #5).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        layers = layers or list(self.spec.layers)
        ox, oy = self.spec.origin
        written = []
        for layer in layers:
            for purpose in purposes:
                r = self.raster(layer, purpose)
                pgm_vals = np.vectorize(_PGM.get)(r).astype(np.uint8)
                img = np.flipud(pgm_vals)   # PGM row 0 = north (top)
                suffix = "" if purpose == "nav" else f"_{purpose}"
                stem = out_dir / f"{prefix}_{layer}{suffix}"
                _write_pgm(stem.with_suffix(".pgm"), img)
                stem.with_suffix(".yaml").write_text(
                    f"image: {stem.name}.pgm\n"
                    f"resolution: {self.spec.resolution}\n"
                    f"origin: [{ox}, {oy}, 0.0]\n"
                    "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
                written.append(stem.with_suffix(".yaml"))
        return written


def _write_pgm(path: Path, img: np.ndarray) -> None:
    rows, cols = img.shape
    with open(path, "wb") as f:
        f.write(b"P5\n")
        f.write(f"# {path.name} -- generated by LayeredField\n".encode())
        f.write(f"{cols} {rows}\n255\n".encode())
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

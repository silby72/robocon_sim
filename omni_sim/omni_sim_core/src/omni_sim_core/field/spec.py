"""Load a field spec (rulebook dimensions) into SI primitives (依頼 §3.1-3.2).

The spec is authored in mm with provenance comments; we only ever *read* it, so
the comments are preserved trivially (no round-trip needed -- invariant 4). Sweep
runs override the ``unknowns`` in memory rather than rewriting the file.

A ``z`` bound may be a number (mm) or a short expression over the unknowns, e.g.
``z: [600, "600 + l1_barrier_h"]`` -- so an obstacle's height can be a swept
parameter (this is what P2 acceptance #4 exercises: barrier top crossing the
LiDAR plane).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .primitives import Box, Cylinder, Primitive, Segment


@dataclass(frozen=True)
class LayerSpec:
    name: str
    floor_z: float                       # [m]
    extent: tuple[float, float, float, float] | None  # (x0,y0,x1,y1) [m] or None
    holes: tuple[tuple[float, float, float, float], ...] = ()  # drops in the slab


@dataclass
class FieldSpec:
    size: tuple[float, float]            # (W, H) [m]
    resolution: float                    # [m/cell]
    origin: tuple[float, float]          # world coord of lower-left pixel [m]
    layers: dict[str, LayerSpec]
    lidar_height: float                  # [m] above each layer floor
    robot_band: tuple[float, float]      # (lo, hi) [m] above floor
    unknown_defaults: dict[str, float]   # name -> default value [mm] (raw)
    _raw_geometry: list[dict] = field(default_factory=list)
    _to_m: float = 1.0                   # unit scale (mm->m = 1e-3)

    # -- loading ----------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "FieldSpec":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        units = cfg.get("units", "mm")
        to_m = {"mm": 1e-3, "m": 1.0}[units]

        size = tuple(float(v) * to_m for v in cfg["field"]["size"])
        res = float(cfg["resolution"]) * to_m
        origin = tuple(float(v) * to_m for v in cfg["field"].get("origin", [0, 0]))

        layers = {}
        for name, ld in cfg["layers"].items():
            ext = ld.get("extent")
            if ext is not None:
                (x0, y0), (x1, y1) = ext
                ext = (x0 * to_m, y0 * to_m, x1 * to_m, y1 * to_m)
            holes = tuple(tuple(v * to_m for v in h) for h in ld.get("holes", []))
            layers[name] = LayerSpec(name=name,
                                     floor_z=float(ld["floor_z"]) * to_m,
                                     extent=ext, holes=holes)

        lidar_h = float(cfg["lidar_height"]) * to_m
        band = tuple(float(v) * to_m for v in cfg["robot_band"])
        unknown_defaults = {k: float(v["default"])
                            for k, v in cfg.get("unknowns", {}).items()}

        return cls(size=size, resolution=res, origin=origin, layers=layers,
                   lidar_height=lidar_h, robot_band=band,
                   unknown_defaults=unknown_defaults,
                   _raw_geometry=list(cfg.get("geometry", [])), _to_m=to_m)

    # -- primitive construction ------------------------------------------
    def _resolve_z(self, value, unknowns_raw: dict[str, float]) -> float:
        """Resolve a z bound (raw units) to metres, evaluating unknown exprs."""
        if isinstance(value, (int, float)):
            raw = float(value)
        else:  # string expression over unknown names, in raw (mm) units
            raw = float(eval(value, {"__builtins__": {}}, dict(unknowns_raw)))
        return raw * self._to_m

    def _wall_thickness(self, g: dict) -> float:
        """Wall thickness in metres, floored at one cell so sub-cell walls still
        render 1 cell thick and never vanish (P2 acceptance #6)."""
        raw = g.get("thickness", self.resolution / self._to_m)  # in raw units
        return max(raw * self._to_m, self.resolution)

    def build_primitives(self, unknowns: dict[str, float] | None = None
                         ) -> list[Primitive]:
        """Instantiate SI primitives. ``unknowns`` overrides defaults (raw units)."""
        raw = dict(self.unknown_defaults)
        if unknowns:
            raw.update(unknowns)
        m = self._to_m
        prims: list[Primitive] = []
        for g in self._raw_geometry:
            name = g["name"]
            typ = g["type"]
            z_lo = self._resolve_z(g["z"][0], raw)
            z_hi = self._resolve_z(g["z"][1], raw)
            if typ == "box":
                cx, cy = (v * m for v in g["center"])
                w, h = (v * m for v in g["size"])
                prims.append(Box(name=name, z_lo=z_lo, z_hi=z_hi,
                                 cx=cx, cy=cy, w=w, h=h))
            elif typ == "cylinder":
                cx, cy = (v * m for v in g["center"])
                prims.append(Cylinder(name=name, z_lo=z_lo, z_hi=z_hi,
                                      cx=cx, cy=cy, d=g["diameter"] * m))
            elif typ == "segment":
                x0, y0, x1, y1 = (v * m for v in g["seg"])
                t = self._wall_thickness(g)
                prims.append(Segment(name=name, z_lo=z_lo, z_hi=z_hi,
                                     x0=x0, y0=y0, x1=x1, y1=y1, t=t))
            elif typ == "outline":
                # a rectangular wall (perimeter barrier) -> 4 segments
                cx, cy = (v * m for v in g["center"])
                w, h = (v * m for v in g["size"])
                t = self._wall_thickness(g)
                prims.extend(_outline_segments(name, z_lo, z_hi, cx, cy, w, h, t))
            else:
                raise ValueError(f"unknown geometry type {typ!r} for {name!r}")
        return prims


def _outline_segments(name, z_lo, z_hi, cx, cy, w, h, t) -> list[Segment]:
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    edges = [("_s", x0, y0, x1, y0), ("_n", x0, y1, x1, y1),
             ("_w", x0, y0, x0, y1), ("_e", x1, y0, x1, y1)]
    return [Segment(name=name + suf, z_lo=z_lo, z_hi=z_hi,
                    x0=a, y0=b, x1=c, y1=d, t=t) for suf, a, b, c, d in edges]

"""Ramps as actual inclines, not flat rectangles that teleport height.

``ConnectorSpec`` gives a ramp a *footprint* -- the region carved drivable on
both layers it joins -- which was enough to make a multi-level plan exist, but
not enough to make climbing one cost anything. The robot crossed a 600 mm rise
with no gravity opposing it, so a drivetrain that could not physically get up
the ramp looked identical to one that could.

This adds the missing half: the surface height across a connector, and the
horizontal force gravity applies to a body standing on it.

The model stays 2D. ``(x, y)`` are horizontal projections (the occupancy grid
is a top-down map), so the along-slope gravity ``m g sin(a)`` projects onto the
horizontal plane as ``m g sin(a) cos(a)``; that is what is returned. Two things
are deliberately *not* modelled, because nothing else in the simulator models
them either: traction (the normal load drops by ``cos(a)``, and there is no
friction or slip model to spend it on -- see plant/jacobian.py), and the
``cos(a)`` foreshortening of the drive force itself, which is a 14 % effect at
30 degrees and shrinks fast at believable ramp angles.

Pitch is reported so a viewer can show it, and so a tipping check has
something to read later; the rigid body itself is still planar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .spec import ConnectorSpec, FieldSpec

G = 9.80665


@dataclass(frozen=True)
class Slope:
    """One connector, with the incline worked out."""

    name: str
    rect: tuple[float, float, float, float]
    axis: int            # 0 = climbs along x, 1 = along y
    sign: float          # +1 if height rises with the axis, -1 if it falls
    z_lo: float          # [m] floor height at the bottom
    z_hi: float          # [m] at the top
    links: tuple[str, ...]

    @property
    def rise(self) -> float:
        return self.z_hi - self.z_lo

    @property
    def run(self) -> float:
        return self.rect[self.axis + 2] - self.rect[self.axis]

    @property
    def angle_rad(self) -> float:
        return math.atan2(self.rise, self.run)

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad)

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def height_at(self, x: float, y: float) -> float:
        """Surface height [m] -- linear from the bottom edge to the top one."""
        v = (x, y)[self.axis]
        lo, hi = self.rect[self.axis], self.rect[self.axis + 2]
        u = (v - lo) / max(hi - lo, 1e-9)
        if self.sign < 0:
            u = 1.0 - u
        return self.z_lo + float(np.clip(u, 0.0, 1.0)) * self.rise

    def uphill(self) -> tuple[float, float]:
        """World-frame unit vector pointing up the slope (horizontal)."""
        return (self.sign, 0.0) if self.axis == 0 else (0.0, self.sign)


class SlopeField:
    """The connectors of a field spec, as inclines."""

    def __init__(self, slopes: tuple[Slope, ...]) -> None:
        self.slopes = slopes

    @classmethod
    def from_spec(cls, spec: FieldSpec) -> "SlopeField":
        out = []
        for c in spec.connectors:
            out.append(cls._slope(spec, c))
        return cls(tuple(out))

    @staticmethod
    def _slope(spec: FieldSpec, c: ConnectorSpec) -> Slope:
        axis, sign = c.climb_axis(spec.layers)
        zs = sorted(spec.layers[n].floor_z for n in c.links)
        return Slope(name=c.name, rect=c.rect, axis=axis, sign=sign,
                     z_lo=zs[0], z_hi=zs[-1], links=tuple(c.links))

    def at(self, x: float, y: float) -> Slope | None:
        for s in self.slopes:
            if s.contains(x, y):
                return s
        return None

    def gravity_wrench_body(self, x: float, y: float, theta: float,
                            mass_kg: float) -> np.ndarray:
        """Body-frame ``[Fx, Fy, Mz]`` gravity applies here. Zero off a ramp.

        Points *downhill*: a robot that stops climbing rolls back, which is
        the whole point of modelling this.
        """
        s = self.at(x, y)
        if s is None or abs(s.rise) < 1e-9:
            return np.zeros(3)
        a = s.angle_rad
        mag = mass_kg * G * math.sin(a) * math.cos(a)
        ux, uy = s.uphill()
        fx_w, fy_w = -mag * ux, -mag * uy          # downhill
        c, sn = math.cos(theta), math.sin(theta)
        return np.array([c * fx_w + sn * fy_w, -sn * fx_w + c * fy_w, 0.0])

    def describe(self) -> list[str]:
        return [f"{s.name}: {s.rise * 1000:.0f} mm rise over "
                f"{s.run * 1000:.0f} mm run = {s.angle_deg:.1f} deg"
                for s in self.slopes]


def max_climb_angle_rad(drive_force_n: float, mass_kg: float) -> float:
    """Steepest slope a given horizontal drive force can hold, ignoring drag.

    Solves ``F = m g sin(a) cos(a)``, i.e. ``sin(2a) = 2F/(m g)``. Note the
    right-hand side peaks at ``a = 45 deg``: past that a *horizontal* push
    gains nothing, which is the honest limit of a planar model rather than a
    real robot's tipping or traction limit -- both of which bite first.
    """
    ratio = 2.0 * drive_force_n / (mass_kg * G)
    if ratio >= 1.0:
        return math.pi / 4          # planar model saturates here
    return 0.5 * math.asin(ratio)

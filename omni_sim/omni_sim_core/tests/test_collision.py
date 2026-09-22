"""The body must be stopped by what it is drawn against.

Two defects these pin down, both of which let the chassis visibly sink into
geometry on screen:

* ``resolve_level`` used to commit to "the highest level free at the centre",
  which meant driving into the 600 mm L1 slab wall anywhere along its 6 m face
  re-labelled the robot as being *on* L1 the moment its centre passed the slab
  edge -- and being on the slab is legal, so the wall stopped nothing.
* ``is_blocked`` used to test four corner points. A wall thinner than the body,
  or a wall corner poking into the middle of an edge, passed between them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omni_sim_core.evaluation.footprint import RobotFootprint
from omni_sim_core.ui.leveled_field_2027 import LeveledField

MAPS = Path(__file__).resolve().parents[2] / "maps"
YAMLS = [MAPS / f"field_2027_{lvl}.yaml" for lvl in ("ground", "l1", "l2")]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in YAMLS),
                                reason="generated maps/ not present")


@pytest.fixture
def lf():
    return LeveledField(*(str(p) for p in YAMLS), team="red")


@pytest.fixture
def fp():
    return RobotFootprint(length=0.9, width=0.9)   # config/robot/chassis.yaml


def _occupied_under(lf, levels, poly):
    """Cells the polygon covers that are occupied on *every* level in
    ``levels`` -- an independent oracle for what is genuinely solid there.

    Even-odd point-in-polygon and a plain loop, deliberately *not* the convex
    cross-product test ``is_blocked`` uses, so agreement between the two means
    something. Intersecting across levels (rather than, say, taking the
    smallest per-level count) is the whole point mid-gate: the ground raster
    and the L1 raster each call most of the other's floor a wall, so only what
    both refuse is actually solid.
    """
    g = lf.grids[levels[0]]
    res, (ox, oy, _) = g.meta.resolution, g.meta.origin
    c0 = int(np.floor((poly[:, 0].min() - ox) / res))
    c1 = int(np.floor((poly[:, 0].max() - ox) / res)) + 2
    r0 = int(np.floor((poly[:, 1].min() - oy) / res))
    r1 = int(np.floor((poly[:, 1].max() - oy) / res)) + 2
    n = 0
    for row in range(r0, r1):
        for col in range(c0, c1):
            x = ox + (col + 0.5) * res
            y = oy + (row + 0.5) * res
            inside, j = False, len(poly) - 1
            for i in range(len(poly)):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if (yi > y) != (yj > y):
                    if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                        inside = not inside
                j = i
            if not inside:
                continue
            solid = all(not lf.grids[lv].in_bounds(row, col)
                        or lf.grids[lv].is_occupied(row, col) for lv in levels)
            if solid:
                n += 1
    return n


def test_the_level_cannot_change_away_from_a_gate(lf):
    """Driving east into the L1 slab's west wall at y = 7.5 -- 4.5 m north of
    the Ramp -- must leave the robot on the ground, whatever its centre is
    standing over."""
    level = "ground"
    for x in np.arange(2.30, 3.01, 0.02):
        level, _ = lf.resolve_level(level, float(x), 7.5)
    assert level == "ground", (
        "the robot climbed onto L1 by driving into the slab wall")


def test_the_level_does_change_through_the_ramp(lf):
    """...and the Ramp still works, or the fix above would just be a wall.

    The route is: north up the slope (no gate -- only the ground grid clears
    the slope), then east off the landing onto the slab, which is the move the
    gate covers."""
    level = "ground"
    for y in np.arange(3.2, 6.31, 0.02):            # up the slope
        level, _ = lf.resolve_level(level, 2.0, float(y))
    assert level == "ground", "the slope itself should not promote to L1"
    for x in np.arange(2.0, 3.81, 0.02):            # off the landing, onto L1
        level, _ = lf.resolve_level(level, float(x), 6.2)
    assert level == "l1", "the Ramp no longer gets the robot onto L1"


def test_no_accepted_pose_overlaps_a_wall(lf, fp):
    """Off a gate, a pose the checker accepts covers no occupied cell at all.

    Swept across the L1 slab's west wall and around the Mustika Pillar, at
    every 15 deg of heading -- the pillar is the case corner sampling was
    worst at, being narrower than the body.
    """
    # Step sizes are a deliberate compromise: the oracle below is a plain
    # Python loop over cells, so a fine sweep costs minutes. These catch the
    # defect this pins down (the pillar case fails at any resolution) while
    # keeping the file a few seconds.
    thetas = np.deg2rad(np.arange(0, 90, 22.5))
    cases = [("ground", np.arange(0.80, 3.20, 0.06), np.array([7.5])),
             ("ground", np.arange(4.30, 6.70, 0.10), np.arange(6.80, 9.20, 0.10))]
    checked = 0
    for level, xs, ys in cases:
        for th in thetas:
            for x in xs:
                for y in ys:
                    lf._gate_pending = None
                    _, levels = lf.resolve_level(level, float(x), float(y))
                    if len(levels) > 1:
                        continue                  # mid-gate: see the next test
                    poly = fp.corners(float(x), float(y), float(th))
                    if lf.is_blocked(levels, poly):
                        continue
                    checked += 1
                    assert _occupied_under(lf, levels, poly) == 0, (
                        f"accepted ({x:.2f}, {y:.2f}) th={th:.2f} on {levels[0]} "
                        "but the body covers occupied cells")
    assert checked > 50, "the sweep accepted almost nothing -- check the setup"


def test_a_thin_wall_between_the_corners_is_not_missed(lf, fp):
    """The L1 perimeter barrier is 50 mm thick -- an eighteenth of the body.

    Placed so the barrier runs through the middle of the footprint with no
    corner inside it, corner sampling saw nothing. (The slab edge alone would
    also refuse this pose; the point is that the *barrier cells* are counted.)
    """
    poly = fp.corners(2.50, 7.50, 0.0)            # centred on the west barrier
    assert _occupied_under(lf, ("l1",), poly) > 0, "test pose misses the barrier"
    assert lf.is_blocked(("l1",), poly)


def test_the_gate_tolerance_is_at_most_one_cell(lf, fp):
    """Mid-crossing, a one-cell tolerance bridges the seam between two layer
    rasters. That is the only place it is justified and the only place it is
    granted -- and it stays one cell, not one body."""
    res = lf.grids["ground"].meta.resolution
    worst = 0
    for th in np.deg2rad(np.arange(0, 90, 22.5)):
        for x in np.arange(1.90, 3.40, 0.06):
            for y in np.arange(5.50, 6.60, 0.10):
                lf._gate_pending = None
                _, levels = lf.resolve_level("l1", float(x), float(y))
                if len(levels) < 2:
                    continue
                poly = fp.corners(float(x), float(y), float(th))
                if lf.is_blocked(levels, poly):
                    continue
                worst = max(worst, _occupied_under(lf, levels, poly))
    # one cell of skin around the body's outline, not a body-sized overlap
    perimeter_cells = int(np.ceil(4 * 0.7 / res))
    assert worst <= perimeter_cells, (
        f"mid-gate tolerance let {worst} solid cells under the body "
        f"(a one-cell skin would be about {perimeter_cells})")

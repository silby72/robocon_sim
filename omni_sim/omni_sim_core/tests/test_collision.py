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


def _covered_cells(lf, level, poly):
    """Cells the polygon covers that are occupied -- by an independent method.

    Even-odd point-in-polygon and a plain loop, deliberately *not* the convex
    cross-product test ``is_blocked`` uses, so agreement between the two means
    something.
    """
    g = lf.grids[level]
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
            if not g.in_bounds(row, col) or g.is_occupied(row, col):
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
    """...and the Ramp still works, or the fix above would just be a wall."""
    level = "ground"
    for x in np.arange(1.2, 3.61, 0.02):
        level, _ = lf.resolve_level(level, float(x), 4.75)   # through the Ramp
    assert level == "l1", "the Ramp no longer gets the robot onto L1"


def test_no_accepted_pose_overlaps_a_wall(lf, fp):
    """Off a gate, a pose the checker accepts covers no occupied cell at all.

    Swept across the L1 slab's west wall and around the Mustika Pillar, at
    every 15 deg of heading -- the pillar is the case corner sampling was
    worst at, being narrower than the body.
    """
    thetas = np.deg2rad(np.arange(0, 90, 15))
    cases = [("ground", np.arange(1.70, 3.20, 0.03), np.array([7.5])),
             ("ground", np.arange(4.70, 6.30, 0.05), np.arange(7.20, 8.80, 0.05))]
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
                    assert _covered_cells(lf, levels[0], poly) == 0, (
                        f"accepted ({x:.2f}, {y:.2f}) th={th:.2f} on {levels[0]} "
                        "but the body covers occupied cells")
    assert checked > 100, "the sweep accepted almost nothing -- check the setup"


def test_a_thin_wall_between_the_corners_is_not_missed(lf, fp):
    """The L1 perimeter barrier is 50 mm thick -- an eighteenth of the body.

    Placed so the barrier runs through the middle of the footprint with no
    corner inside it, corner sampling saw nothing. (The slab edge alone would
    also refuse this pose; the point is that the *barrier cells* are counted.)
    """
    poly = fp.corners(2.50, 7.50, 0.0)            # centred on the west barrier
    assert _covered_cells(lf, "l1", poly) > 0, "test pose misses the barrier"
    assert lf.is_blocked(("l1",), poly)


def test_the_gate_tolerance_is_at_most_one_cell(lf, fp):
    """Mid-crossing, a one-cell tolerance bridges the seam between two layer
    rasters. That is the only place it is justified and the only place it is
    granted -- and it stays one cell, not one body."""
    res = lf.grids["ground"].meta.resolution
    worst = 0
    for th in np.deg2rad(np.arange(0, 90, 15)):
        for x in np.arange(1.90, 3.40, 0.03):
            for y in np.arange(2.80, 3.40, 0.05):
                lf._gate_pending = None
                _, levels = lf.resolve_level("l1", float(x), float(y))
                if len(levels) < 2:
                    continue
                poly = fp.corners(float(x), float(y), float(th))
                if lf.is_blocked(levels, poly):
                    continue
                # cells occupied on *every* level in play, i.e. genuinely solid
                shared = min(_covered_cells(lf, lv, poly) for lv in levels)
                worst = max(worst, shared)
    # one cell of skin around the body's outline, not a body-sized overlap
    perimeter_cells = int(np.ceil(4 * 0.9 / res))
    assert worst <= perimeter_cells, (
        f"mid-gate tolerance let {worst} solid cells under the body "
        f"(a one-cell skin would be about {perimeter_cells})")

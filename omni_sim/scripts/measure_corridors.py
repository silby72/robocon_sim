#!/usr/bin/env python3
"""Measure what a given chassis can actually reach on each field layer.

The transition waypoints in ``ui/leveled_field_2027.py`` are pinned to the
free bands this prints. They are not tunable by feel: a waypoint that drifts
into inflation makes A* fail with "goal shifted out of inflation", and one
that drifts the other way makes the stitched leg longer than the gate that is
supposed to cover it.

    python scripts/measure_corridors.py                 # the 0.9 m chassis
    python scripts/measure_corridors.py --size 0.7      # what would fit
    python scripts/measure_corridors.py --sweep         # largest chassis that
                                                        # can still reach L2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.field.layered_field import LayeredField      # noqa: E402
from omni_sim_core.planning.cost_field import CostField         # noqa: E402
from omni_sim_core.planning.types import PlanConfig             # noqa: E402

SPEC = ROOT / "config" / "field" / "robocon2027.yaml"
RAMP_EXIT_M = (2.4, 4.75)      # where the red ramp lands on L1
STAIRS_X_M = (4.7, 6.3)        # the Stairs gate's x-span


def _components(mask: np.ndarray):
    from scipy.ndimage import label
    return label(mask)


def reachable_on_l1(cf: CostField, res: float):
    """Free cells on L1 connected to the ramp exit (the only way up)."""
    lab, _ = _components(~cf.lethal)
    cid = lab[int(RAMP_EXIT_M[1] / res), int(RAMP_EXIT_M[0] / res)]
    if cid == 0:
        return None
    return lab == cid


def free_bands(cf: CostField, x: float, res: float):
    """Contiguous free y-intervals in the column at ``x`` [m]."""
    col = ~cf.lethal[:, int(x / res)]
    bands, start = [], None
    for i, v in enumerate(col):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start * res, i * res))
            start = None
    if start is not None:
        bands.append((start * res, len(col) * res))
    return bands


def analyse(size_m: float, verbose: bool = True) -> bool:
    """Print the corridors for a square chassis; return True if L2 is reachable."""
    r_circ = size_m * math.sqrt(2) / 2
    lf = LayeredField.from_yaml(SPEC)
    cfg = PlanConfig(r_circ=r_circ)
    grids = {l: lf.grid(l) for l in ("ground", "l1", "l2")}
    cfs = {l: CostField(grids[l], cfg) for l in grids}
    res = grids["ground"].meta.resolution

    reach = reachable_on_l1(cfs["l1"], res)
    if reach is None:
        if verbose:
            print(f"{size_m:.2f} m chassis (r_circ {r_circ:.3f}): "
                  "the ramp exit itself is lethal -- L1 unreachable")
        return False

    ys, xs = np.nonzero(reach)
    at_stairs = (xs * res >= STAIRS_X_M[0]) & (xs * res <= STAIRS_X_M[1])
    ok = bool(at_stairs.any())

    if verbose:
        print(f"=== square chassis {size_m:.3f} m  ->  r_circ {r_circ:.3f} m ===")
        print(f"ground, ramp approach (y=4.75): free x "
              f"{_fmt(free_bands_x(cfs['ground'], 4.75, res))}")
        print(f"L1 reachable from the ramp: x {xs.min()*res:.2f}..{xs.max()*res:.2f}, "
              f"y {ys.min()*res:.2f}..{ys.max()*res:.2f}  ({reach.sum()} cells)")
        ring = free_bands(cfs["l1"], 3.2, res)
        print(f"L1 west ring at x=3.20: {_fmt(ring)}")
        if ok:
            sx = xs[at_stairs] * res
            sy = ys[at_stairs] * res
            print(f"L1 under the Stairs gate: x {sx.min():.2f}..{sx.max():.2f}, "
                  f"y {sy.min():.2f}..{sy.max():.2f}  <- L1-side waypoint goes here")
        else:
            print(f"L1 under the Stairs gate: NONE "
                  f"(reach stops at x={xs.max()*res:.2f}) -- L2 unreachable")
        print(f"L2 entry band at x=4.80: {_fmt(free_bands(cfs['l2'], 4.8, res))}")
    return ok


def free_bands_x(cf: CostField, y: float, res: float):
    """Contiguous free x-intervals in the row at ``y`` [m]."""
    row = ~cf.lethal[int(y / res), :]
    bands, start = [], None
    for i, v in enumerate(row):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start * res, i * res))
            start = None
    if start is not None:
        bands.append((start * res, len(row) * res))
    return bands


def _fmt(bands):
    return ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in bands) or "(none)"


def sweep():
    print("largest square chassis that can still reach L2:")
    best = None
    for size in np.arange(0.40, 1.21, 0.01):
        if analyse(float(size), verbose=False):
            best = float(size)
    print(f"  {best:.2f} m" if best else "  none in 0.40..1.20 m")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=0.9, help="square side [m]")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        sweep()
    else:
        analyse(a.size)

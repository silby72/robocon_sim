#!/usr/bin/env python3
"""依頼 §5.3② — sensitivity of the L1 localization grid to l1_barrier_h.

The barrier height is a rulebook unknown. As it rises past the L1 LiDAR plane
(600 + 185 = 785 mm) the perimeter starts reflecting beams, so the localization
grid gains a strong outer feature. We sweep the height and measure the beam
hit-rate from L1 poses, then report the threshold: turns "unknown" into
"unknown, but the threshold is 0.19 m".

    python experiments/sweep_unknown_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.field import LayeredField                    # noqa: E402
from omni_sim_core.env.raycast import raycast                   # noqa: E402
from omni_sim_core.sweep import run_sweep, find_boundary        # noqa: E402

SPEC = ROOT / "config" / "field" / "robocon2027.yaml"
# Poses just inside each L1 edge, each casting a fan of beams OUTWARD (toward the
# perimeter). The L2 slab and other features are inward, so only the perimeter
# barrier can return these beams -- the signal is not swamped. Without the
# barrier they escape over the edge (inf); with it they hit at short range.
EDGE_PROBES = [
    ((2.8, 5.5), np.pi),          # near west edge, beams facing west
    ((8.2, 5.5), 0.0),            # near east edge, beams facing east
    ((5.5, 2.8), -np.pi / 2),     # near south edge, beams facing south
    ((3.5, 8.2), np.pi / 2),      # near north edge (off the pocket), facing north
]
FAN = np.radians(40.0)
N_BEAMS = 41
MAX_RANGE = 3.0


def run_one(params, rng):
    h_mm = params["l1_barrier_h"] * 1000.0
    field = LayeredField.from_yaml(SPEC, unknowns={"l1_barrier_h": h_mm})
    grid = field.grid("l1", "loc")
    hits = total = 0
    for (x, y), heading in EDGE_PROBES:
        angles = heading + np.linspace(-FAN, FAN, N_BEAMS)
        r = raycast(grid, x, y, angles, max_range=MAX_RANGE)
        hits += int(np.isfinite(r).sum())
        total += N_BEAMS
    return {"hit_rate": hits / total}


def main() -> None:
    heights = np.round(np.arange(0.05, 0.401, 0.01), 3)   # 0.05..0.40 m
    conds = [{"l1_barrier_h": float(h)} for h in heights]
    res = run_sweep(conds, run_one, seed=42, parallel=4)
    out = res.to_csv(ROOT / "results" / "barrier_sensitivity.csv")
    print(res.report())
    print(f"csv -> {out}")

    rows = res.ok()
    lo = np.mean([r["hit_rate"] for r in rows[:3]])
    hi = np.mean([r["hit_rate"] for r in rows[-3:]])
    thr = 0.5 * (lo + hi)                                 # midpoint hit-rate
    b = find_boundary(rows, x="l1_barrier_h",
                      predicate=lambda r: r["hit_rate"] <= thr)
    print(f"\nBOUNDARY: L1 perimeter becomes visible (hit-rate steps up) at "
          f"l1_barrier_h ~ {b:.3f} m (expected ~0.185 m = LiDAR plane)"
          if b else "no step found")


if __name__ == "__main__":
    main()

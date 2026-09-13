#!/usr/bin/env python3
"""依頼 §5.3① — max robot width through the L1 corridor pinch (1.2 m).

Outputs the CSV and, more importantly, the *boundary*: the largest footprint
width that still clears the pinch with >= 0.15 m margin. That single number is
the requirement the mechanism team can build to.

    python experiments/sweep_footprint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.sweep import SweepSpec, run_sweep, find_boundary   # noqa: E402
from scenarios import corridor_scene, plan_and_eval                   # noqa: E402

PINCH = 1.2   # [m] the corridor narrows to this over the L2 step


def run_one(params, rng):
    grid, geo, start, goal = corridor_scene(gap_width=PINCH)
    w = params["footprint_width"]
    return plan_and_eval(grid, geo, start, goal, footprint_len=w, footprint_wid=w,
                         planner=params.get("planner", "astar"))


def _refine(width: float) -> bool:
    """Predicate at an arbitrary width, for bisection sharpening the boundary."""
    grid, geo, start, goal = corridor_scene(gap_width=PINCH)
    m = plan_and_eval(grid, geo, start, goal, width, width, planner="astar")
    return bool(m["success"]) and m["min_clearance"] >= 0.15


def main() -> None:
    spec = SweepSpec.from_yaml(ROOT / "config" / "sweeps" / "footprint_boundary.yaml")
    res = run_sweep(spec.conditions(), run_one, seed=spec.seed,
                    parallel=spec.parallel)
    out = res.to_csv(ROOT / "results" / "footprint_boundary.csv")
    print(res.report())
    print(f"csv -> {out}")

    ok = [r for r in res.ok() if r.get("planner") == "astar"]
    b = find_boundary(ok, x="footprint_width",
                      predicate=lambda r: r["success"] and r["min_clearance"] >= 0.15,
                      refine=_refine, tol=2e-3)
    print(f"\nBOUNDARY (astar): max footprint width through a {PINCH} m pinch "
          f"with >=0.15 m margin = {b:.3f} m" if b else "no boundary found")


if __name__ == "__main__":
    main()

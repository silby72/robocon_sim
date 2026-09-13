"""Reusable sweep scenarios: build a scene, plan, judge -> metrics (依頼 §5).

A "scene" is a nav OccupancyGrid (for planning) plus the exact obstacle geometry
(for judging) -- the same one-definition-two-consumers split as the field layer,
so the evaluator never grades the planner's own discretisation.

These live outside the core so the sweep engine stays generic. They are top-level
functions, hence picklable / fork-safe for ``multiprocessing``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.env.occupancy_grid import (FREE, OCCUPIED, MapMetadata,   # noqa: E402
                                              OccupancyGrid)
from omni_sim_core.field.primitives import Box                               # noqa: E402
from omni_sim_core.planning import (CostField, GridPlanner, PlanConfig,       # noqa: E402
                                    shortcut_path)
from omni_sim_core.evaluation import (RobotFootprint, evaluate_run,           # noqa: E402
                                      path_to_trajectory)


# --------------------------------------------------------------------------- #
# scene construction
# --------------------------------------------------------------------------- #
def _blank(size, res, origin=(0.0, 0.0)):
    cols = int(round(size[0] / res))
    rows = int(round(size[1] / res))
    grid = np.full((rows, cols), FREE, dtype=np.uint8)
    meta = MapMetadata(resolution=res, origin=(origin[0], origin[1], 0.0))
    return OccupancyGrid(grid, meta)


def _stamp_boxes(grid: OccupancyGrid, boxes: list[Box]) -> None:
    res = grid.meta.resolution
    ox, oy, _ = grid.meta.origin
    rows, cols = grid.grid.shape
    xs = ox + (np.arange(cols) + 0.5) * res
    ys = oy + (np.arange(rows) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    for b in boxes:
        grid.grid[b.filled_mask(X, Y)] = OCCUPIED


def corridor_scene(gap_width: float, res: float = 0.05):
    """A straight corridor that narrows in the middle to ``gap_width`` (依頼 §5.3①).

    Outer corridor 1.5 m wide, a central pinch of ``gap_width``. Robot traverses
    south->north. Returns (grid, geometry, start, goal).
    """
    size = (4.0, 6.0)
    grid = _blank(size, res)
    half = 0.75                       # outer corridor half-width (1.5 m)
    cx = 2.0
    boxes = [
        Box("wall_w", 0, 1, cx=cx - half - 0.5, cy=3.0, w=1.0, h=6.0),
        Box("wall_e", 0, 1, cx=cx + half + 0.5, cy=3.0, w=1.0, h=6.0),
        # the pinch: two blocks that squeeze the corridor to gap_width over y in [2.7,3.3]
        Box("pinch_w", 0, 1, cx=cx - gap_width / 2 - 0.5, cy=3.0, w=1.0, h=0.6),
        Box("pinch_e", 0, 1, cx=cx + gap_width / 2 + 0.5, cy=3.0, w=1.0, h=0.6),
    ]
    _stamp_boxes(grid, boxes)
    return grid, boxes, (cx, 0.4), (cx, 5.6)


def gap_scene(opening_width: float, res: float = 0.05):
    """Two blocks with an ``opening_width`` gap, approached diagonally (依頼 §5.3③)."""
    size = (5.0, 5.0)
    grid = _blank(size, res)
    cx = 2.5
    boxes = [
        Box("block_w", 0, 1, cx=cx - opening_width / 2 - 0.75, cy=2.5, w=1.5, h=1.5),
        Box("block_e", 0, 1, cx=cx + opening_width / 2 + 0.75, cy=2.5, w=1.5, h=1.5),
    ]
    _stamp_boxes(grid, boxes)
    return grid, boxes, (0.6, 0.6), (4.4, 4.4)   # diagonal approach


# --------------------------------------------------------------------------- #
# plan + judge -> metrics
# --------------------------------------------------------------------------- #
def plan_and_eval(grid, geometry, start, goal, footprint_len, footprint_wid,
                  planner="astar", soft_weight=2.0, soft_k=4.0):
    """Plan with the circumscribed circle, judge with the real footprint.

    Returns the §5.1 metric dict: success, min_clearance, path_length,
    expanded_nodes, plan_time_s.
    """
    fp = RobotFootprint(length=footprint_len, width=footprint_wid)
    cfg = PlanConfig(r_circ=fp.r_circ, soft_weight=soft_weight, soft_k=soft_k)
    cf = CostField(grid, cfg)
    res = GridPlanner(cf).plan(start, goal)
    if not res.success:
        return {"success": 0, "min_clearance": float("nan"),
                "path_length": float("nan"), "expanded_nodes": res.expanded_nodes,
                "plan_time_s": res.elapsed_s}
    path = res.path
    if planner == "astar_shortcut":
        path = shortcut_path(cf, path)
    traj = path_to_trajectory(path, spacing=grid.meta.resolution, heading="tangent")
    ev = evaluate_run(fp, geometry, traj, goal_xy=goal)
    return {
        "success": int(ev.checks["no_collision"]),
        "min_clearance": ev.min_clearance,
        "path_length": path.length,
        "expanded_nodes": res.expanded_nodes,
        "plan_time_s": res.elapsed_s,
    }

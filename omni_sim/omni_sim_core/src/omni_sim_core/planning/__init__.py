"""Path-planning layer (rclpy-free, GUI-free, deterministic).

Design decisions live in ``DECISIONS.md`` §2. The public surface is a *pure*
function ``plan(start, goal) -> PlanResult`` over a static occupancy grid, plus
post-processing (line-of-sight shortcut, corner speed limits) and an orientation
profile that is generated *independently* of the path (holonomic robot, so theta
is decoupled from the translation — §2.6).

Invariants honoured here:
    * no ``rclpy`` / ``PySide6`` import (core stays ROS/GUI independent)
    * every random draw takes an explicit ``numpy.random.Generator`` (none here —
      planning is deterministic by construction)
    * ``Path`` carries geometry only, never time (§2.8 / invariant 8)
"""
from __future__ import annotations

from .types import Path, OrientationProfile, PlanResult, PlanConfig
from .cost_field import CostField
from .grid_planner import GridPlanner, plan
from .shortcut import line_of_sight, shortcut_path
from .corner import corner_radii, corner_speed_limits
from .orientation import build_orientation_profile

__all__ = [
    "Path",
    "OrientationProfile",
    "PlanResult",
    "PlanConfig",
    "CostField",
    "GridPlanner",
    "plan",
    "line_of_sight",
    "shortcut_path",
    "corner_radii",
    "corner_speed_limits",
    "build_orientation_profile",
]

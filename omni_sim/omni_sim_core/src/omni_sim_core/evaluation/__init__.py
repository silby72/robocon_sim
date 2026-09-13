"""Evaluation harness (依頼 §4): judge a drive against the TRUE geometry.

Collision is analytic (footprint rectangle vs exact primitives), never grid-based
-- so the planner is not scored with its own discretisation or inflation radius.
rclpy/GUI free, deterministic.
"""
from __future__ import annotations

from .footprint import RobotFootprint
from .evaluator import EvalResult, EvalThresholds, evaluate_run, path_to_trajectory
from .report import write_report

__all__ = [
    "RobotFootprint",
    "EvalResult",
    "EvalThresholds",
    "evaluate_run",
    "path_to_trajectory",
    "write_report",
]

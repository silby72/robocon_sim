"""Sweep engine (依頼 §5): continuous parameter exploration, deterministic.

Declares a parameter space, runs each condition headless in parallel with an
independent per-condition RNG (result invariant to process count), writes tidy
CSV, and extracts a *boundary* rather than a scatter of pass/fail points.
rclpy/GUI free.
"""
from __future__ import annotations

from .spec import SweepSpec
from .engine import run_sweep
from .results import SweepResults
from .boundary import find_boundary, bisect_boundary

__all__ = [
    "SweepSpec",
    "run_sweep",
    "SweepResults",
    "find_boundary",
    "bisect_boundary",
]

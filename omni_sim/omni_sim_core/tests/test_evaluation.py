"""P3 evaluation-harness acceptance + unit tests (依頼 §4).

Numbered ``acceptance N:`` comments map to the 4 P3 criteria. Headless,
deterministic, judged against exact geometry.
"""
from __future__ import annotations

import numpy as np
import pytest

from omni_sim_core.field.primitives import Box, Cylinder
from omni_sim_core.evaluation import (EvalThresholds, RobotFootprint,
                                      evaluate_run, write_report)


def cyl(name, cx, cy, d):
    return Cylinder(name=name, z_lo=0.0, z_hi=1.0, cx=cx, cy=cy, d=d)


# --------------------------------------------------------------------------- #
def test_acceptance1_straight_path_min_distance_matches_analytic():
    """acceptance 1: on a known straight path, min clearance == analytic value."""
    fp = RobotFootprint(length=0.4, width=0.4)
    obstacle = cyl("c", 2.0, 0.0, 1.0)     # radius 0.5 at (2,0)
    # robot sits at origin, axis-aligned: right edge at x=0.2, so
    # clearance = (2 - 0.2) - 0.5 = 1.3 m
    d, name = fp.min_distance([obstacle], 0.0, 0.0, 0.0)
    assert name == "c"
    assert d == pytest.approx(1.3, abs=1e-9)

    # a straight drive along x from -1..1: closest approach is at x = something,
    # min over the path equals the analytic closest distance.
    xs = np.linspace(-1.0, 1.0, 21)
    traj = np.column_stack([xs, xs, np.zeros_like(xs), np.zeros_like(xs)])
    res = evaluate_run(fp, [obstacle], traj, goal_xy=(1.0, 0.0))
    # closest at x=1: right edge 1.2, distance 2-1.2-0.5 = 0.3
    assert res.min_clearance == pytest.approx(0.3, abs=1e-9)


def test_acceptance2_driving_into_wall_detects_collision():
    """acceptance 2: a drive that rams the obstacle registers a collision."""
    fp = RobotFootprint(length=0.4, width=0.4)
    obstacle = cyl("c", 0.3, 0.0, 1.0)     # radius 0.5 -> robot overlaps near x=0
    xs = np.linspace(-2.0, 0.0, 21)
    traj = np.column_stack([xs, xs, np.zeros_like(xs), np.zeros_like(xs)])
    res = evaluate_run(fp, [obstacle], traj, goal_xy=(0.0, 0.0))
    assert res.collisions >= 1
    assert res.checks["no_collision"] is False
    assert res.min_clearance == 0.0


def test_acceptance3_rotating_footprint_changes_min_distance():
    """acceptance 3: rotating the footprint changes clearance correctly.

    A square rotated 45 deg pokes a corner toward the obstacle, so clearance
    drops from (edge) to (circumscribed radius)."""
    fp = RobotFootprint(length=0.4, width=0.4)
    obstacle = cyl("c", 2.0, 0.0, 1.0)
    d0, _ = fp.min_distance([obstacle], 0.0, 0.0, 0.0)          # edge faces it
    d45, _ = fp.min_distance([obstacle], 0.0, 0.0, np.pi / 4)   # corner faces it
    assert d0 == pytest.approx(1.3, abs=1e-9)
    assert d45 == pytest.approx(2.0 - fp.r_circ - 0.5, abs=1e-9)
    assert d45 < d0


def test_acceptance4_same_input_identical_result():
    """acceptance 4: identical inputs -> identical verdict (no RNG)."""
    fp = RobotFootprint(length=0.5, width=0.4)
    obs = [cyl("a", 1.0, 1.0, 0.6), Box("b", 0.0, 1.0, cx=3.0, cy=0.0,
                                        w=1.0, h=1.0)]
    xs = np.linspace(0.0, 3.0, 31)
    traj = np.column_stack([xs, xs, 0.3 * np.sin(xs), 0.1 * xs])
    r1 = evaluate_run(fp, obs, traj, goal_xy=(3.0, 0.0), goal_theta=0.3)
    r2 = evaluate_run(fp, obs, traj, goal_xy=(3.0, 0.0), goal_theta=0.3)
    assert r1.min_clearance == r2.min_clearance
    assert r1.collisions == r2.collisions
    assert r1.reach_error == r2.reach_error
    assert r1.heading_error_deg == r2.heading_error_deg
    assert r1.checks == r2.checks


# --------------------------------------------------------------------------- #
def test_box_and_segment_distance_symmetry():
    fp = RobotFootprint(length=0.4, width=0.4)
    wall = Box("w", 0.0, 1.0, cx=0.0, cy=2.0, w=4.0, h=0.2)  # horizontal wall
    # footprint top edge at y=0.2, wall bottom edge at y=1.9 -> gap 1.7
    d, _ = fp.min_distance([wall], 0.0, 0.0, 0.0)
    assert d == pytest.approx(1.7, abs=1e-9)


def test_reach_and_heading_error():
    fp = RobotFootprint(0.4, 0.4)
    traj = np.array([[0.0, 0.0, 0.0, 0.0],
                     [1.0, 0.9, 0.1, np.radians(3.0)]])
    res = evaluate_run(fp, [], traj, goal_xy=(1.0, 0.0),
                       goal_theta=0.0, thresholds=EvalThresholds())
    assert res.reach_error == pytest.approx(np.hypot(0.1, 0.1), abs=1e-9)
    assert res.heading_error_deg == pytest.approx(3.0, abs=1e-9)
    assert res.checks["reach"] and res.checks["heading"]


def test_write_report_creates_files(tmp_path):
    from omni_sim_core.planning.types import Path as PlanPath
    fp = RobotFootprint(0.4, 0.4)
    obstacle = cyl("c", 2.0, 0.0, 1.0)
    xs = np.linspace(0.0, 1.0, 6)
    traj = np.column_stack([xs, xs, np.zeros_like(xs), np.zeros_like(xs)])
    res = evaluate_run(fp, [obstacle], traj, goal_xy=(1.0, 0.0))
    planned = [PlanPath(np.column_stack([xs, np.zeros_like(xs)]))]
    out = write_report(res, tmp_path, traj, planned, timestamp="fixed")
    assert (out / "summary.txt").exists()
    assert (out / "trajectory.csv").exists()
    assert (out / "plan.csv").exists()
    # route.png only if matplotlib is installed; do not require it

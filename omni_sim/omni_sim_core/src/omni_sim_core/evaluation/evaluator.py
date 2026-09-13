"""Per-run evaluation of a drive against the true field geometry (依頼 §4).

Judged, per run (依頼 §4.1):

    1. every segment planned successfully
    2. collision count == 0
    3. min distance to any obstacle >= 0.15 m
    4. reach error (true pose vs goal) <= 0.25 m
    5. final heading error <= 5 deg

Collision is checked against ``field.geometry()`` (exact shapes), never the grid
(§4.2). Nothing here draws a random number, so a run is reproducible from its
inputs alone (P3 acceptance #4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..planning.types import PlanResult
from .footprint import RobotFootprint


@dataclass(frozen=True)
class EvalThresholds:
    clearance_min: float = 0.15     # [m]
    reach_max: float = 0.25         # [m]
    heading_max_deg: float = 5.0    # [deg]


@dataclass
class EvalResult:
    success: bool
    plan_success: bool
    collisions: int
    min_clearance: float
    reach_error: float
    heading_error_deg: float
    # per-check verdicts (§4.1 items 1..5)
    checks: dict[str, bool] = field(default_factory=dict)
    # time series for trajectory.csv
    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    clearances: np.ndarray = field(default_factory=lambda: np.empty(0))
    nearest: list[str] = field(default_factory=list)
    # planning info for plan.csv
    plan_expanded: list[int] = field(default_factory=list)
    plan_times: list[float] = field(default_factory=list)


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def path_to_trajectory(path, spacing: float, heading=0.0) -> np.ndarray:
    """Resample a geometric Path into a pose log (N,4) for evaluation.

    Not a dynamic simulation -- it stamps the planned geometry at a fixed spacing
    so the footprint-clearance metric can be swept cheaply. ``heading`` is a
    constant angle, or 'tangent' to face along the path. Time column is a running
    index (planning is time-free -- invariant 8; the "t" here is just an ordinal).
    """
    pts = np.asarray(path.points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        th = 0.0 if heading == "tangent" else float(heading)
        return np.array([[0.0, pts[0, 0], pts[0, 1], th]]) if len(pts) else np.empty((0, 4))
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        L = float(np.hypot(*seg))
        if L < 1e-12:
            continue
        n = max(int(np.ceil(L / spacing)), 1)
        tang = np.arctan2(seg[1], seg[0])
        for k in range(n):
            p = a + seg * (k / n)
            th = tang if heading == "tangent" else float(heading)
            out.append([len(out), p[0], p[1], th])
    th_end = np.arctan2(pts[-1, 1] - pts[-2, 1], pts[-1, 0] - pts[-2, 0]) \
        if heading == "tangent" else float(heading)
    out.append([len(out), pts[-1, 0], pts[-1, 1], th_end])
    return np.asarray(out, dtype=float)


def evaluate_run(footprint: RobotFootprint, geometry, trajectory: np.ndarray,
                 goal_xy, plans: list[PlanResult] | None = None,
                 goal_theta: float | None = None,
                 thresholds: EvalThresholds | None = None) -> EvalResult:
    """Evaluate one drive.

    ``trajectory`` is the *true* pose log, shape (N, 4) = [t, x, y, theta].
    ``geometry`` is the exact primitives that can collide (``field.geometry``).
    ``plans`` are the per-segment PlanResults (for the plan-success check + CSV).
    """
    th = thresholds or EvalThresholds()
    traj = np.asarray(trajectory, dtype=float).reshape(-1, 4)
    n = len(traj)

    # per-pose clearance against the true geometry
    clearances = np.empty(n)
    nearest = []
    for i, (_, x, y, theta) in enumerate(traj):
        d, name = footprint.min_distance(geometry, x, y, theta)
        clearances[i] = d
        nearest.append(name or "")

    # collisions: count episodes where the footprint overlaps an obstacle
    collided = clearances <= 1e-9
    collisions = int(np.sum(collided[1:] & ~collided[:-1])) + int(collided[:1].sum())
    min_clearance = float(clearances.min()) if n else np.inf

    # reach + heading error at the final true pose
    final = traj[-1] if n else np.array([0, *goal_xy, 0.0])
    reach_error = float(np.hypot(final[1] - goal_xy[0], final[2] - goal_xy[1]))
    heading_error_deg = (abs(np.degrees(_wrap(final[3] - goal_theta)))
                         if goal_theta is not None else 0.0)

    plan_success = bool(plans is None or all(p.success for p in plans))

    checks = {
        "plan_success": plan_success,
        "no_collision": collisions == 0,
        "clearance": min_clearance >= th.clearance_min,
        "reach": reach_error <= th.reach_max,
        "heading": (goal_theta is None) or (heading_error_deg <= th.heading_max_deg),
    }

    return EvalResult(
        success=all(checks.values()),
        plan_success=plan_success,
        collisions=collisions,
        min_clearance=min_clearance,
        reach_error=reach_error,
        heading_error_deg=heading_error_deg,
        checks=checks,
        times=traj[:, 0].copy(),
        clearances=clearances,
        nearest=nearest,
        plan_expanded=[p.expanded_nodes for p in (plans or [])],
        plan_times=[p.elapsed_s for p in (plans or [])],
    )

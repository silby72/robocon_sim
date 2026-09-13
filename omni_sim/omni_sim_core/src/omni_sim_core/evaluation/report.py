"""Write an evaluation run to ``results/<timestamp>/`` (依頼 §4.3).

    summary.txt      per-check verdict table
    trajectory.csv   true trajectory, per-step min clearance + nearest obstacle
    plan.csv         planned path, expanded nodes, plan time
    route.png        true path (solid) vs planned path (dotted) -- viz optional

matplotlib is an optional extra (viz); route.png is skipped, not errored, if it
is missing -- the harness stays headless-clean.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from ..planning.types import Path as PlanPath
from .evaluator import EvalResult


def write_report(result: EvalResult, out_root: str | Path,
                 true_trajectory: np.ndarray,
                 planned_paths: list[PlanPath] | None = None,
                 timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_root) / ts
    out.mkdir(parents=True, exist_ok=True)
    traj = np.asarray(true_trajectory, dtype=float).reshape(-1, 4)

    _summary(result, out / "summary.txt")
    _trajectory_csv(result, traj, out / "trajectory.csv")
    _plan_csv(result, planned_paths, out / "plan.csv")
    _route_png(traj, planned_paths, out / "route.png")
    return out


def _summary(r: EvalResult, path: Path) -> None:
    lines = ["evaluation summary", "=" * 40,
             f"OVERALL: {'PASS' if r.success else 'FAIL'}", ""]
    labels = {"plan_success": "1. all segments planned",
              "no_collision": "2. collision count == 0",
              "clearance": "3. min clearance >= threshold",
              "reach": "4. reach error <= threshold",
              "heading": "5. heading error <= threshold"}
    for key, label in labels.items():
        ok = r.checks.get(key, True)
        lines.append(f"  [{'x' if ok else ' '}] {label}")
    lines += ["",
              f"collisions       : {r.collisions}",
              f"min clearance    : {r.min_clearance:.4f} m",
              f"reach error      : {r.reach_error:.4f} m",
              f"heading error    : {r.heading_error_deg:.3f} deg",
              f"expanded nodes   : {r.plan_expanded}",
              f"plan time (s)    : {[round(t, 4) for t in r.plan_times]}"]
    path.write_text("\n".join(lines) + "\n")


def _trajectory_csv(r: EvalResult, traj: np.ndarray, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "theta", "min_clearance", "nearest_obstacle"])
        for i in range(len(traj)):
            t, x, y, th = traj[i]
            clr = r.clearances[i] if i < len(r.clearances) else ""
            near = r.nearest[i] if i < len(r.nearest) else ""
            w.writerow([t, x, y, th, clr, near])


def _plan_csv(r: EvalResult, paths, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "point_index", "x", "y",
                    "expanded_nodes", "plan_time_s"])
        for seg, pp in enumerate(paths or []):
            exp = r.plan_expanded[seg] if seg < len(r.plan_expanded) else ""
            pt = r.plan_times[seg] if seg < len(r.plan_times) else ""
            for k, (x, y) in enumerate(pp.points):
                w.writerow([seg, k, x, y, exp if k == 0 else "",
                            pt if k == 0 else ""])


def _route_png(traj: np.ndarray, paths, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    if paths:
        for pp in paths:
            ax.plot(pp.points[:, 0], pp.points[:, 1], ":", color="tab:orange",
                    lw=1.5, label="planned")
    if len(traj):
        ax.plot(traj[:, 1], traj[:, 2], "-", color="tab:blue", lw=1.8,
                label="true")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    ax.set_title("route: true (solid) vs planned (dotted)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

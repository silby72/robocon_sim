"""Real-time matplotlib view of the robot following a path on the field map.

Windowed (needs a display):
    python3 experiments/live_view.py

Headless -> GIF (no display needed):
    python3 experiments/live_view.py --save results/live_view.gif --duration 8

The robot tracks a rectangular loop that dodges the field obstacle; the LiDAR
point cloud, travelled path and tracking error update live.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.clock import ClockConfig                       # noqa: E402
from omni_sim_core.simulator import RobotSim, RobotConfig         # noqa: E402
from omni_sim_core.control.pid import PIDParams                   # noqa: E402
from omni_sim_core.control.trajectory import (                    # noqa: E402
    Trajectory, TrajectoryParams, TrajectoryFollower)
from omni_sim_core.sensors.lidar import Lidar, LidarParams        # noqa: E402
from omni_sim_core.env.occupancy_grid import (                    # noqa: E402
    OccupancyGrid, generate_rect_field)
from omni_sim_core.ui.viewer import RealtimeViewer, ViewerConfig  # noqa: E402


def build(map_yaml: str | None, duration: float):
    clk = ClockConfig(dt_sim=1e-3, dt_motor=1e-3, dt_nav=2e-2)
    sim = RobotSim(RobotConfig(clock=clk), seed=0)

    if map_yaml:
        grid = OccupancyGrid.from_yaml(map_yaml)
    else:
        grid = generate_rect_field(6.0, 4.0, 0.05)

    # rectangular loop dodging the obstacle (~x in [3.0,3.3], y in [1.5,2.5])
    waypoints = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 3.0],
                          [1.0, 3.0], [1.0, 1.0]])
    traj = Trajectory(waypoints, TrajectoryParams(v_max=1.2, a_max=1.5, dt=0.02))
    follower = TrajectoryFollower(traj, PIDParams(kp=2.0, ki=0.0, kd=0.0),
                                  dt=clk.dt_nav, v_max=1.5)

    sim.reset(pose=np.array([1.0, 1.0, 0.0]))

    lidar = Lidar(LidarParams(n_beams=240, enable_motion_distortion=False),
                  grid, np.random.default_rng(0))

    viewer = RealtimeViewer(sim, follower, lidar,
                            ViewerConfig(duration_s=duration, interval_ms=50))
    return viewer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "maps" / "field.yaml"))
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--save", default=None, help="write a GIF instead of a window")
    args = ap.parse_args(argv)

    map_yaml = args.map if Path(args.map).exists() else None
    viewer = build(map_yaml, args.duration)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        viewer.save(str(out))
        print(f"[live_view] wrote {out}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()

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
from omni_sim_core.ui.leveled_field_2027 import LeveledField      # noqa: E402


# RED team, ground start -> up the Ramp onto L1 -> around inside L1's own
# (red) half, generously clear of the L2 platform's footprint -> back down
# the Ramp -> then a deliberate run at the blue half, which rules 3.2
# (operating zones) and 6.2.2 (opponent's territory) say must stop dead at
# the centre divider (x=5.5) instead of crossing. Only the Ramp rectangle
# ever changes level ground<->L1 -- matching how the real field is built.
# (L1<->L2 via the Stairs gate uses the same mechanism -- see
# leveled_field_2027.py -- just not exercised by this particular demo path,
# since a 0.9 m-wide square footprint needs very generous clearance from
# L2's edges and the Central Pillar to cross reliably.) Override with
# --waypoints for a different map (note: territory/gates assume red).
DEFAULT_WAYPOINTS = [
    [1.0, 1.2], [1.0, 3.2], [7.0, 3.2],
]


def build(map_yaml: str | None, duration: float, waypoints=None,
          chassis_yaml: str | None = None, team: str = "red"):
    clk = ClockConfig(dt_sim=1e-3, dt_motor=1e-3, dt_nav=2e-2)
    sim = RobotSim(RobotConfig(clock=clk), seed=0)

    leveled_field = None
    if map_yaml:
        grid = OccupancyGrid.from_yaml(map_yaml)
        map_dir = Path(map_yaml).parent
        l1_yaml, l2_yaml = map_dir / "field_2027_l1.yaml", map_dir / "field_2027_l2.yaml"
        if Path(map_yaml).stem == "field_2027_ground" and l1_yaml.exists() and l2_yaml.exists():
            leveled_field = LeveledField(map_yaml, str(l1_yaml), str(l2_yaml), team=team)
    else:
        grid = generate_rect_field(6.0, 4.0, 0.05)

    waypoints = np.asarray(waypoints if waypoints is not None
                           else DEFAULT_WAYPOINTS, dtype=float)
    traj = Trajectory(waypoints, TrajectoryParams(v_max=1.2, a_max=1.5, dt=0.02))
    follower = TrajectoryFollower(traj, PIDParams(kp=2.0, ki=0.0, kd=0.0),
                                  dt=clk.dt_nav, v_max=1.5)

    sim.reset(pose=np.array([waypoints[0][0], waypoints[0][1], 0.0]))

    lidar = Lidar(LidarParams(n_beams=240, enable_motion_distortion=False),
                  grid, np.random.default_rng(0))

    viewer = RealtimeViewer(sim, follower, lidar,
                            ViewerConfig(duration_s=duration, interval_ms=50,
                                        chassis_yaml=chassis_yaml),
                            leveled_field=leveled_field)
    return viewer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "maps" / "field_2027_ground.yaml"))
    ap.add_argument("--chassis", default=str(ROOT / "config" / "robot" / "chassis.yaml"),
                    help="config/robot/chassis.yaml; drives the drawn footprint/wheels")
    ap.add_argument("--team", choices=["red", "blue"], default="red",
                    help="which side's territory/ramp/start zone to use")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--save", default=None, help="write a GIF instead of a window")
    ap.add_argument("--waypoints", default=None,
                    help="loop to track, as 'x,y x,y ...' in metres. The first "
                         "point is also the start pose. Default suits "
                         "maps/field.yaml; a different map needs its own.")
    args = ap.parse_args(argv)

    wp = None
    if args.waypoints:
        wp = [[float(v) for v in pt.split(",")] for pt in args.waypoints.split()]

    map_yaml = args.map if Path(args.map).exists() else None
    chassis_yaml = args.chassis if Path(args.chassis).exists() else None
    viewer = build(map_yaml, args.duration, wp, chassis_yaml, args.team)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        viewer.save(str(out))
        print(f"[live_view] wrote {out}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()

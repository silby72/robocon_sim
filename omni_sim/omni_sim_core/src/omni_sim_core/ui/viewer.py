"""Real-time matplotlib viewer (spec 10.3): robot pose, LiDAR point cloud,
reference trajectory + travelled path, and the tracking-error trace.

matplotlib only; seaborn is used for styling *if available* (optional). Works in
two modes:

- windowed  : ``RealtimeViewer(...).show()``  -> live FuncAnimation window
- headless  : ``RealtimeViewer(...).save("out.gif")``  -> writes a GIF (Agg)

Colours echo the serial2can console dashboard (dark panel, cyan accent, green
good / red bad) so the whole toolchain looks of a piece.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# palette matched to ui/console.py (approx 256-colour equivalents)
_BG = "#1c1c1c"
_PANEL = "#262626"
_MUTED = "#8a8a8a"
_TITLE = "#00afd7"      # cyan 45
_ACCENT = "#5fd7ff"     # 81
_GOOD = "#00d75f"       # 48
_WARN = "#ffaf00"       # 214
_BAD = "#ff5f5f"        # 203
_TEXT = "#d0d0d0"       # 252


def _apply_style() -> None:
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid", context="talk")
    except Exception:
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")


@dataclass
class ViewerConfig:
    duration_s: float = 20.0
    trail_len: int = 400          # points kept in the travelled-path trail
    lidar_stride: int = 2         # plot every Nth beam (thin the cloud)
    interval_ms: int = 50         # animation frame period (wall clock)


class RealtimeViewer:
    """Drives a RobotSim + trajectory follower and animates the result.

    Parameters
    ----------
    sim : RobotSim
    follower : TrajectoryFollower  (provides body-velocity commands)
    lidar : Lidar                  (its grid is used for the map background)
    config : ViewerConfig
    """

    def __init__(self, sim, follower, lidar, config: ViewerConfig | None = None):
        self.sim = sim
        self.follower = follower
        self.lidar = lidar
        self.grid = lidar.grid
        self.cfg = config or ViewerConfig()

        self._path_x: list[float] = []
        self._path_y: list[float] = []
        self._err_t: list[float] = []
        self._err: list[float] = []
        self._steps_per_frame = max(
            1, int(round(self.cfg.interval_ms * 1e-3 / sim.clock.config.dt_sim)))
        self._torque = np.zeros(sim.jac.n_wheels)

    # -- one animation frame: advance the sim, refresh artists ------------
    def _extent(self):
        g = self.grid
        ox, oy, _ = g.meta.origin
        return [ox, ox + g.width * g.meta.resolution,
                oy, oy + g.height * g.meta.resolution]

    def _build_figure(self):
        import matplotlib.pyplot as plt
        _apply_style()
        fig = plt.figure(figsize=(12, 6.5))
        fig.patch.set_facecolor(_BG)
        gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[3, 1])
        ax_map = fig.add_subplot(gs[:, 0])
        ax_err = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[1, 1])
        for ax in (ax_map, ax_err, ax_info):
            ax.set_facecolor(_PANEL)

        # occupancy background
        ax_map.imshow(self.grid.grid, origin="lower", extent=self._extent(),
                      cmap="Greys", vmin=0, vmax=100, alpha=0.9, zorder=0)
        ax_map.set_title("omni_sim  ·  live view", color=_TITLE)
        ax_map.set_xlabel("x [m]", color=_TEXT)
        ax_map.set_ylabel("y [m]", color=_TEXT)
        ax_map.set_aspect("equal")

        # reference trajectory (static)
        ref_pts = np.array([[p.x, p.y] for p in self.follower.traj._samples])
        ax_map.plot(ref_pts[:, 0], ref_pts[:, 1], "--", color=_MUTED, lw=1.5,
                    label="reference", zorder=1)

        # dynamic artists
        (self._trail,) = ax_map.plot([], [], "-", color=_ACCENT, lw=2.0,
                                     label="path", zorder=2)
        self._scan = ax_map.scatter([], [], s=6, c=_WARN, label="LiDAR", zorder=3)
        (self._robot,) = ax_map.plot([], [], "o", color=_GOOD, ms=12, zorder=4)
        self._heading = ax_map.annotate("", xytext=(0, 0), xy=(0, 0),
                                        arrowprops=dict(arrowstyle="->", color=_GOOD,
                                                        lw=2), zorder=5)
        ax_map.legend(loc="upper right", framealpha=0.3, fontsize=9)

        ax_err.set_title("tracking error", color=_ACCENT)
        ax_err.set_ylabel("|e| [m]", color=_TEXT)
        ax_err.set_xlabel("t [s]", color=_TEXT)
        (self._err_line,) = ax_err.plot([], [], "-", color=_BAD, lw=1.8)
        self._ax_err = ax_err

        ax_info.axis("off")
        self._info = ax_info.text(0.02, 0.95, "", va="top", ha="left",
                                  family="monospace", fontsize=11, color=_TEXT,
                                  transform=ax_info.transAxes)

        self.fig = fig
        self.ax_map = ax_map
        return fig

    def _lidar_world_points(self):
        p = self.lidar._sensor_pose(self.sim.body.pose)
        scan = self.lidar._measure(self.sim.clock.t,
                                   {"pose": self.sim.body.pose})
        r = scan.ranges[::self.cfg.lidar_stride]
        a = scan.angles[::self.cfg.lidar_stride] + p[2]
        finite = np.isfinite(r)
        xs = p[0] + r[finite] * np.cos(a[finite])
        ys = p[1] + r[finite] * np.sin(a[finite])
        return np.column_stack([xs, ys]) if xs.size else np.empty((0, 2))

    def _advance(self):
        # Recompute the follower command at the nav rate (ZOH between ticks),
        # not once per animation frame -- coarse control makes the loop unstable.
        torque = self._torque
        for _ in range(self._steps_per_frame):
            if self.sim.clock.nav_fires():
                cmd_vel = self.follower.command(self.sim.clock.t, self.sim.body.pose)
                torque = self.sim._wheel_torque_from_cmd_vel(cmd_vel, kp=1.5)
            self.sim.step(torque)
            self.sim.update_odometry()
        self._torque = torque

    def _update(self, _frame):
        self._advance()
        pose = self.sim.body.pose
        ref = self.follower.traj.sample(self.sim.clock.t)

        self._path_x.append(pose[0]); self._path_y.append(pose[1])
        if len(self._path_x) > self.cfg.trail_len:
            self._path_x = self._path_x[-self.cfg.trail_len:]
            self._path_y = self._path_y[-self.cfg.trail_len:]
        self._trail.set_data(self._path_x, self._path_y)

        pts = self._lidar_world_points()
        self._scan.set_offsets(pts if pts.size else np.empty((0, 2)))

        self._robot.set_data([pose[0]], [pose[1]])
        hx, hy = 0.35 * np.cos(pose[2]), 0.35 * np.sin(pose[2])
        self._heading.set_position((pose[0], pose[1]))
        self._heading.xy = (pose[0] + hx, pose[1] + hy)

        err = float(np.hypot(ref.x - pose[0], ref.y - pose[1]))
        self._err_t.append(self.sim.clock.t); self._err.append(err)
        self._err_line.set_data(self._err_t, self._err)
        self._ax_err.relim(); self._ax_err.autoscale_view()

        drift = float(np.linalg.norm(pose[:2] - self.sim.odom_pose[:2]))
        self._info.set_text(
            f"t      {self.sim.clock.t:6.2f} s\n"
            f"pose   {pose[0]:5.2f}, {pose[1]:5.2f}\n"
            f"theta  {pose[2]:+5.2f} rad\n"
            f"track  {err:5.3f} m\n"
            f"odom   {drift:5.3f} m")
        return (self._trail, self._scan, self._robot, self._err_line, self._info)

    def _n_frames(self) -> int:
        frame_dt = self._steps_per_frame * self.sim.clock.config.dt_sim
        return max(1, int(round(self.cfg.duration_s / frame_dt)))

    def _animation(self):
        from matplotlib.animation import FuncAnimation
        self._build_figure()
        return FuncAnimation(self.fig, self._update, frames=self._n_frames(),
                             interval=self.cfg.interval_ms, blit=False,
                             repeat=False)

    def show(self):
        import matplotlib.pyplot as plt
        self._anim = self._animation()
        plt.tight_layout()
        plt.show()

    def save(self, path: str, fps: int = 20):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.animation import PillowWriter
        anim = self._animation()
        anim.save(path, writer=PillowWriter(fps=fps))
        return path

"""ROS 2 (Jazzy) node wrapping omni_sim_core.

This is the *only* file allowed to import rclpy. It drives ``RobotSim`` and the
sensor models, publishes /clock in simulation time, the sensor topics, odom and
ground-truth odom, and exposes reset/pause/set_pose services.

Real-time behaviour is controlled by ``real_time_factor``:
    > 0 : advance sim so that it tracks wall-clock * factor
    = 0 : advance a fixed batch each wall tick (as fast as possible)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float64MultiArray, Header, String
from std_srvs.srv import SetBool, Trigger

from omni_sim_core.clock import ClockConfig
from omni_sim_core.simulator import RobotSim, RobotConfig
from omni_sim_core.sensors.imu import Imu as ImuModel, ImuParams
from omni_sim_core.sensors.lidar import Lidar as LidarModel, LidarParams
from omni_sim_core.sensors.encoder import Encoder as EncoderModel, EncoderParams
from omni_sim_core.env.occupancy_grid import OccupancyGrid, generate_rect_field
from omni_sim_core.planning import (GridPlanner, CostField, PlanConfig,
                                    shortcut_path, smooth_path, max_deviation)
from omni_sim_core.control.pid import PIDParams
from omni_sim_core.control.trajectory import (SampledTrajectory, ProfileLimits,
                                              TrajectoryFollower)
from omni_sim_core.evaluation.footprint import RobotFootprint
from omni_sim_core.ui.leveled_field_2027 import LeveledField, LEVEL_ORDER

from .tf_broadcaster import RobotTf, yaw_to_quat


def _same_point(a, b, tol: float = 1e-6) -> bool:
    return abs(float(a[0]) - float(b[0])) < tol and abs(float(a[1]) - float(b[1])) < tol


def _sim_time_msg(t: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(t)
    msg.nanosec = int(round((t - int(t)) * 1e9))
    return msg


class SimNode(Node):
    def __init__(self) -> None:
        super().__init__("omni_sim")

        self.declare_parameter("dt_sim", 1.0e-4)
        self.declare_parameter("dt_motor", 1.0e-3)
        self.declare_parameter("dt_nav", 2.0e-2)
        self.declare_parameter("real_time_factor", 1.0)
        self.declare_parameter("wall_tick_s", 0.01)
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("seed", 0)

        dt_sim = self.get_parameter("dt_sim").value
        dt_motor = self.get_parameter("dt_motor").value
        dt_nav = self.get_parameter("dt_nav").value
        self.rtf = float(self.get_parameter("real_time_factor").value)
        self.wall_tick = float(self.get_parameter("wall_tick_s").value)
        seed = int(self.get_parameter("seed").value)

        clk = ClockConfig(dt_sim=dt_sim, dt_motor=dt_motor, dt_nav=dt_nav)
        self.sim = RobotSim(RobotConfig(clock=clk), seed=seed)

        self.declare_parameter("team", "red")
        team = str(self.get_parameter("team").value)

        map_yaml = self.get_parameter("map_yaml").value
        self._leveled = None
        if map_yaml:
            self.grid = OccupancyGrid.from_yaml(map_yaml)
            # Auto-detect the real ABU Robocon 2027 field (same heuristic as
            # experiments/live_view.py) and, if its sibling L1/L2 grids are
            # there too, plan across all three levels instead of just the
            # single ground-layer grid -- see LeveledField for why a goal
            # anywhere inside the L1/L2 footprint used to fail outright
            # (that whole area reads as lethal on the ground grid alone).
            map_dir = Path(map_yaml).parent
            l1_yaml, l2_yaml = map_dir / "field_2027_l1.yaml", map_dir / "field_2027_l2.yaml"
            if (Path(map_yaml).stem == "field_2027_ground"
                    and l1_yaml.exists() and l2_yaml.exists()):
                self._leveled = LeveledField(map_yaml, str(l1_yaml), str(l2_yaml), team=team)
        else:
            self.grid = generate_rect_field(6.0, 4.0, 0.05)
            self.get_logger().info("no map_yaml given; using a 6x4 m rect field")
        # place robot roughly in the middle
        self.sim.reset(pose=np.array([1.0, 1.0, 0.0]))

        # Default False: sensors/lidar.py's motion-distortion path calls
        # raycast() once per beam instead of vectorised over all of them
        # (see ui/viewer.py's LeveledField work for the full writeup), which
        # turns each scan into a multi-second stall and makes the node look
        # hung. Fixing the vectorization is tracked separately; until then,
        # every launch path (ros2 run, sim.launch.py, sim_with_dashboard.
        # launch.py) needs this off by default rather than relying on each
        # one to remember to pass it -- forgetting it once already cost a
        # long debugging detour that looked exactly like a hang.
        self.declare_parameter("lidar_motion_distortion", False)
        lidar_distortion = bool(self.get_parameter("lidar_motion_distortion").value)

        rng = np.random.default_rng(seed)
        self.imu = ImuModel(ImuParams(), rng)
        self.lidar = LidarModel(
            LidarParams(enable_motion_distortion=lidar_distortion), self.grid, rng)
        self.encoder = EncoderModel(EncoderParams(), rng)

        # --- pub/sub ---
        clock_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST)
        self.pub_clock = self.create_publisher(Clock, "/clock", clock_qos)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", 10)
        self.pub_imu = self.create_publisher(Imu, "/imu/data_raw", 20)
        self.pub_joint = self.create_publisher(JointState, "/joint_states", 20)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 20)
        self.pub_gt = self.create_publisher(Odometry, "/ground_truth/odom", 20)
        self.pub_plan = self.create_publisher(PathMsg, "/plan", 1)
        # The C2-smoothed curve actually followed, as distinct from /plan's
        # A* polyline: corner cutting moves one off the other by up to the
        # circumscribed-vs-inscribed margin, and that difference is exactly
        # what a viewer needs to show to make the clearance argument visible.
        self.pub_traj = self.create_publisher(PathMsg, "/trajectory", 1)

        # "Plan with the circle, judge with the rectangle" (evaluation/
        # footprint.py): A* inflates by the circumscribed radius, which is
        # orientation-independent and conservative, while collision judging
        # below uses the real rotated rectangle. r_circ therefore comes from
        # the chassis, not a guess -- a 0.9 m square's circumscribed radius is
        # 0.64 m, so the old hard-coded 0.45 was under-inflating every plan.
        self.declare_parameter("chassis_yaml", "")
        chassis_yaml = self._resolve_repo_file(
            str(self.get_parameter("chassis_yaml").value), "config/robot/chassis.yaml")
        self._footprint = self._load_footprint(chassis_yaml)
        default_r = self._footprint.r_circ if self._footprint is not None else 0.45
        self.declare_parameter("planner_r_circ", float(default_r))
        plan_cfg = PlanConfig(r_circ=float(self.get_parameter("planner_r_circ").value))
        if self._leveled is not None:
            # Plan on the territory-masked grids, judge with the raw ones: the
            # opponent's half is a rule, not an obstacle, so it exists only in
            # is_blocked() unless it is stamped in here too. See
            # LeveledField.territory_masked_grid for what that cost.
            self._planners = {
                lvl: GridPlanner(CostField(self._leveled.territory_masked_grid(lvl),
                                           plan_cfg))
                for lvl in self._leveled.grids}
        else:
            self._planners = {"ground": GridPlanner(CostField(self.grid, plan_cfg))}

        # trajectory-following params: same defaults experiments/live_view.py
        # uses, which is where this PID + trapezoidal-profile combination was
        # actually tuned against this robot's dynamics.
        self.declare_parameter("traj_v_max", 1.2)
        self.declare_parameter("traj_a_max", 1.5)
        self.declare_parameter("traj_a_lat_max", 2.0)
        self.declare_parameter("traj_kp", 2.0)
        self.declare_parameter("traj_control_ds", 0.30)
        # How far the timed reference may run ahead of the robot before its
        # clock is held. Too small and normal tracking error stalls the run;
        # too large and a genuinely stuck robot is dragged at full command for
        # seconds before anyone notices.
        self.declare_parameter("traj_lookahead", 0.5)
        self.declare_parameter("traj_goal_tol", 0.05)
        self.declare_parameter("traj_stall_timeout", 3.0)
        self.declare_parameter("traj_settle_timeout", 2.0)
        self._traj_v_max = float(self.get_parameter("traj_v_max").value)
        self._traj_a_max = float(self.get_parameter("traj_a_max").value)
        self._traj_a_lat_max = float(self.get_parameter("traj_a_lat_max").value)
        self._traj_kp = float(self.get_parameter("traj_kp").value)
        self._traj_control_ds = float(self.get_parameter("traj_control_ds").value)
        self._traj_lookahead = float(self.get_parameter("traj_lookahead").value)
        self._traj_goal_tol = float(self.get_parameter("traj_goal_tol").value)
        self._traj_stall_timeout = float(self.get_parameter("traj_stall_timeout").value)
        self._traj_settle_timeout = float(self.get_parameter("traj_settle_timeout").value)
        self._follower: TrajectoryFollower | None = None
        self._traj_t = 0.0          # governed position along the trajectory
        self._traj_stalled = 0.0    # sim seconds the reference has been held
        self._traj_settle = 0.0     # sim seconds spent past the profile's end
        self._cur_level = "ground"      # only changes through a ramp/stairs gate
        self._collision_steps = 0
        self._collision_logged = False
        self._blocked = False
        # How far the robot may travel between collision checks -- also the
        # bound on how far it can enter anything before being pushed back out.
        self.declare_parameter("collision_check_ds", 0.005)
        self._collision_check_ds = float(
            self.get_parameter("collision_check_ds").value)
        self._safe_pose = self.sim.body.pose.copy()     # last pose known clear
        self._checked_pose = self.sim.body.pose.copy()  # last pose checked

        # --- what the browser needs to draw the same robot this node judges --
        # The dashboard used to draw a 9 px dot on a hand-copied sketch of the
        # field. Publishing the scene means the page renders the *configured*
        # chassis (config/robot/chassis.yaml -- the file the GUI's chassis page
        # edits) against the *real* field spec, so "it looks clear but the sim
        # says blocked" stops being possible: both sides are now the same
        # geometry. See omni_sim_core/ui/web_scene.py.
        self.declare_parameter("field_yaml", "")
        field_yaml = self._resolve_repo_file(
            str(self.get_parameter("field_yaml").value),
            "config/field/robocon2027.yaml")
        self.pub_scene = self.create_publisher(String, "/sim/scene", 1)
        self.pub_state = self.create_publisher(String, "/sim/state", 10)
        self._scene_json = self._build_scene(field_yaml, chassis_yaml, team)
        # Re-published on a timer rather than latched: rosbridge bridges a
        # browser subscription with its own QoS, and a transient-local
        # publisher silently fails to reach a volatile subscriber -- a page
        # opened after the node started would render an empty field forever.
        self.create_timer(2.0, self._publish_scene)
        self._last_state_pub = -1.0

        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal_pose, 10)
        self.create_subscription(Float64MultiArray, "/sim/wheel_torque_cmd",
                                 self._on_wheel_torque, 10)
        self.create_subscription(Float64MultiArray, "/sim/disturbance",
                                 self._on_disturbance, 10)

        self.create_service(Trigger, "/sim/reset", self._srv_reset)
        self.create_service(SetBool, "/sim/pause", self._srv_pause)

        self.tf = RobotTf(self)

        self._cmd_vel = np.zeros(3)
        self._wheel_torque_cmd = None
        self._ext_wrench = np.zeros(3)
        self._paused = False
        self._prev_twist_world = np.zeros(3)
        self._pub_counts = {"scan": 0, "imu": 0, "joint": 0, "odom": 0}
        self._steps_per_tick = max(1, int(round(self.rtf * self.wall_tick / dt_sim))) \
            if self.rtf > 0 else int(round(self.wall_tick / dt_sim))

        # optional serial2can-style live console dashboard (off by default so it
        # does not compete with RViz / log output)
        self.declare_parameter("dashboard", False)
        self._dash = None
        if bool(self.get_parameter("dashboard").value):
            from omni_sim_core.ui import ConsoleDashboard
            self._dash = ConsoleDashboard("omni_sim  ·  ROS 2 node")
            self._dash.start()

        self.create_timer(self.wall_tick, self._on_tick)
        self.get_logger().info(
            f"omni_sim started: rtf={self.rtf}, {self._steps_per_tick} steps/tick")

    def _resolve_repo_file(self, given: str, rel: str) -> str:
        """``given`` if set, else ``rel`` under the repo the map came from.

        ``map_yaml`` is passed as ``$PWD/maps/field_2027_ground.yaml``, so its
        grandparent is the repo root and the chassis/field specs sit at known
        paths under it. Without this, forgetting ``chassis_yaml:=`` silently
        disabled collision judging altogether -- the robot drove through walls
        and the only clue was the absence of a log line.
        """
        if given:
            return given
        map_yaml = str(self.get_parameter("map_yaml").value)
        if not map_yaml:
            return ""
        cand = Path(map_yaml).resolve().parent.parent / rel
        if cand.exists():
            self.get_logger().info(f"using {cand} (not given explicitly)")
            return str(cand)
        return ""

    def _build_scene(self, field_yaml: str, chassis_yaml: str, team: str) -> str:
        try:
            from omni_sim_core.ui.web_scene import scene_json
            js = scene_json(field_yaml=field_yaml or None,
                            chassis_yaml=chassis_yaml or None, team=team)
        except Exception as exc:
            self.get_logger().warn(f"could not build /sim/scene: {exc}")
            return ""
        self.get_logger().info(f"/sim/scene ready ({len(js)} bytes)")
        return js

    def _publish_scene(self) -> None:
        if self._scene_json:
            self.pub_scene.publish(String(data=self._scene_json))

    def _publish_state(self, t: float) -> None:
        """Level / collision / following state, ~10 Hz, as compact JSON.

        Separate from the scene on purpose: the scene is static and large, this
        is small and changes every tick. A front-end that dims the walls not in
        play needs ``level`` from here and ``walls[].blocks`` from there.
        """
        # Throttled on *sim* time, which /sim/reset rewinds to 0. A plain
        # "has 0.1 s elapsed" test then stays false until the sim has re-run
        # every second it had already run, so the topic goes silent after a
        # reset -- for minutes, with no error anywhere. Treat time moving
        # backwards as a reason to publish, not a reason to wait.
        elapsed = t - self._last_state_pub
        if 0.0 <= elapsed < 0.1:
            return
        self._last_state_pub = t
        pose = self.sim.body.pose
        self.pub_state.publish(String(data=json.dumps({
            "t": round(float(t), 3),
            "level": self._cur_level,
            "blocked": bool(self._blocked),
            "collision_steps": int(self._collision_steps),
            "following": self._follower is not None,
            "paused": bool(self._paused),
            "pose": [round(float(pose[0]), 4), round(float(pose[1]), 4),
                     round(float(pose[2]), 4)],
            "cmd_vel": [round(float(v), 3) for v in self._cmd_vel],
        }, separators=(",", ":"))))

    def _load_footprint(self, chassis_yaml: str) -> RobotFootprint | None:
        """Real chassis rectangle from ``config/robot/chassis.yaml``.

        Imported lazily (ruamel lives behind ``mechanism/``) and optional: with
        no chassis file the node keeps its previous behaviour of no collision
        judging at all.
        """
        if not chassis_yaml or not Path(chassis_yaml).exists():
            return None
        try:
            from omni_sim_core.mechanism.yaml_rt import rt_load
            from omni_sim_core.mechanism.schema import Chassis
            chassis = Chassis.from_doc(rt_load(chassis_yaml))
        except Exception as exc:  # malformed file, missing ruamel, ...
            self.get_logger().warn(f"could not load chassis {chassis_yaml}: {exc}")
            return None
        size = float(chassis.footprint.size_m)
        self.get_logger().info(
            f"chassis footprint {size:.3f} m square (r_circ {0.5 * size * 2 ** 0.5:.3f} m)")
        return RobotFootprint(length=size, width=size)

    def destroy_node(self) -> bool:
        if self._dash is not None:
            self._dash.stop()
        return super().destroy_node()

    # -- callbacks --------------------------------------------------------
    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self._wheel_torque_cmd = None  # cmd_vel takes over
        self._follower = None  # manual input cancels trajectory following

    def _on_wheel_torque(self, msg: Float64MultiArray) -> None:
        self._wheel_torque_cmd = np.asarray(msg.data, dtype=float)
        self._follower = None  # manual input cancels trajectory following

    def _on_disturbance(self, msg: Float64MultiArray) -> None:
        d = np.asarray(msg.data, dtype=float)
        self._ext_wrench = d[:3] if d.size >= 3 else np.pad(d, (0, 3 - d.size))

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        """A* from the current ground-truth pose to ``msg``, published as a
        ``nav_msgs/Path`` on ``/plan`` (geometry only -- see planning/types.py,
        Path carries no time/velocity, matching this node's own true/nominal
        separation discipline of not mixing planning with control).

        With a ``LeveledField`` map, start and goal can be on different
        levels: the single-level A* only ever sees its own grid, where
        everywhere off its own footprint (i.e. every other level) reads as
        lethal, so it can't reach a goal on L1/L2 by itself. Instead this
        plans one A* segment per level crossed and stitches them together at
        each gate's waypoint pair (see LeveledField.transition_waypoint_pair).
        """
        start = (float(self.sim.body.pose[0]), float(self.sim.body.pose[1]))
        goal = (float(msg.pose.position.x), float(msg.pose.position.y))

        if self._leveled is None:
            start_level = goal_level = "ground"
        else:
            start_level = self._leveled.level_of(*start)
            goal_level = self._leveled.level_of(*goal)
        i, j = LEVEL_ORDER.index(start_level), LEVEL_ORDER.index(goal_level)
        levels_path = list(LEVEL_ORDER[i:j + 1] if i <= j else LEVEL_ORDER[j:i + 1][::-1])

        waypoints = [start]
        for k in range(len(levels_path) - 1):
            p_from, p_to = self._leveled.transition_waypoint_pair(
                levels_path[k], levels_path[k + 1])
            waypoints.append(p_from)
            waypoints.append(p_to)
        waypoints.append(goal)

        all_points: list[tuple[float, float]] = []
        all_warnings: list[str] = []
        for k, level in enumerate(levels_path):
            p0, p1 = waypoints[2 * k], waypoints[2 * k + 1]
            planner = self._planners[level]
            result = planner.plan(p0, p1)
            if not result.success:
                all_warnings.append(f"[{level}] {p0}->{p1}: {'; '.join(result.warnings)}")
                self.get_logger().warn(
                    f"plan failed on leg {k + 1}/{len(levels_path)} ({level}): "
                    f"{'; '.join(result.warnings)}")
                self.pub_plan.publish(PathMsg(header=self._stamped_header("map")))
                self.pub_traj.publish(PathMsg(header=self._stamped_header("map")))
                return
            seg = shortcut_path(planner.cf, result.path).points
            if all_points and len(seg) and _same_point(seg[0], all_points[-1]):
                # Drop only a genuinely duplicated point. The legs are NOT
                # contiguous across a gate: leg k ends on the *from* side of
                # the ramp/stairs and leg k+1 starts on the *to* side, a gap
                # apart. Dropping seg[0] unconditionally deleted the waypoint
                # that puts the robot inside the gate rectangle, so the path
                # went straight from one side of the gate to the *next* free
                # vertex beyond it -- a diagonal that leaves the gate. The
                # robot then crossed where no level is in play, was judged
                # against a single level, and jammed mid-transition.
                seg = seg[1:]
            all_points.extend((float(x), float(y)) for x, y in seg)
            all_warnings.extend(result.warnings)

        self._publish_path(self.pub_plan, all_points)
        if len(levels_path) > 1:
            self.get_logger().info(
                f"plan ok: {' -> '.join(levels_path)} ({len(all_points)} points)")
        if all_warnings:
            self.get_logger().info(f"plan warnings: {'; '.join(all_warnings)}")

        self._start_following(all_points)

    def _start_following(self, points: list[tuple[float, float]]) -> None:
        """Smooth the planned polyline into a C2 curve, time it, and follow it.

        Three things this deliberately does *not* do the naive way:

        * the polyline is B-spline smoothed (``planning.smooth``) rather than
          followed corner-to-corner, so the chassis doesn't decelerate to a
          full stop at every planner vertex;
        * the speed profile is one continuous forward-backward pass with a
          curvature-derived lateral limit, not a per-leg trapezoid;
        * heading is held **fixed** at whatever the robot is facing now. The
          robot is holonomic, so tying heading to the path tangent buys
          nothing and costs a spin-in-place at every corner.

        Corner cutting moves the followed curve off the cleared polyline, so
        the deviation is checked against the margin the plan was inflated
        with; too large and this falls back to following the polyline itself.
        A stray /cmd_vel or /sim/wheel_torque_cmd cancels following entirely
        (see _on_cmd_vel / _on_wheel_torque).
        """
        if len(points) < 2:
            self._follower = None
            return
        poly = np.array(points, dtype=float)
        sm = smooth_path(poly, ds=0.02, control_ds=self._traj_control_ds)
        dev = max_deviation(poly, sm)
        # How far the curve may leave the cleared polyline is NOT some fraction
        # of r_circ. A* only guarantees the centreline is r_circ from anything
        # lethal, so every millimetre of corner cutting spends that buffer
        # directly. What is genuinely spare is the "plan with the circle, judge
        # with the rectangle" asymmetry: the plan reserved the circumscribed
        # radius while the body only ever occupies the inscribed one, so
        # r_circ - r_insc is real slack and nothing beyond it is. For the 0.9 m
        # square that is 0.636 - 0.450 = 0.186 m -- less than half of r_circ,
        # which would have licensed clipping a wall in the 0.228 m-wide L1 ring.
        if self._footprint is not None:
            budget = self._footprint.r_circ - min(self._footprint.length,
                                                  self._footprint.width) / 2.0
        else:
            budget = 0.45 * (np.sqrt(2) - 1.0)
        if dev > budget:
            self.get_logger().warn(
                f"smoothing deviated {dev:.2f} m (> the {budget:.2f} m the "
                "circumscribed-vs-inscribed margin can absorb); following the "
                "unsmoothed polyline instead")
            sm = smooth_path(poly, ds=0.02, control_ds=0.05)

        limits = ProfileLimits(v_max=self._traj_v_max, a_max=self._traj_a_max,
                               a_lat_max=self._traj_a_lat_max)
        traj = SampledTrajectory(sm.points, sm.s, sm.curvature, limits,
                                 theta=float(self.sim.body.pose[2]))
        dt_nav = self.sim.clock.config.dt_nav
        self._follower = TrajectoryFollower(
            traj, PIDParams(kp=self._traj_kp, ki=0.0, kd=0.0),
            dt=dt_nav, v_max=self._traj_v_max)
        self._traj_t = 0.0
        self._traj_stalled = 0.0
        self._traj_settle = 0.0
        self._wheel_torque_cmd = None
        self._publish_path(self.pub_traj, sm.points)
        self.get_logger().info(
            f"following planned path: {traj.duration:.1f} s, {sm.length:.2f} m, "
            f"smoothing deviation {dev:.3f} m")

    def _stamped_header(self, frame_id: str) -> Header:
        return Header(stamp=_sim_time_msg(self.sim.clock.t), frame_id=frame_id)

    def _publish_path(self, pub, points) -> None:
        out = PathMsg()
        out.header = self._stamped_header("map")
        for x, y in points:
            pose = PoseStamped()
            pose.header = out.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            out.poses.append(pose)
        pub.publish(out)

    def _srv_reset(self, req, resp):
        self.sim.reset(pose=np.array([1.0, 1.0, 0.0]))
        self.imu.reset(); self.lidar.reset(); self.encoder.reset()
        self._follower = None
        self._cur_level = "ground"
        self._safe_pose = self.sim.body.pose.copy()
        self._checked_pose = self.sim.body.pose.copy()
        resp.success = True
        resp.message = "sim reset"
        return resp

    def _srv_pause(self, req, resp):
        self._paused = req.data
        resp.success = True
        resp.message = "paused" if req.data else "resumed"
        return resp

    # -- main loop --------------------------------------------------------
    def _advance_following(self) -> None:
        """Recompute ``_cmd_vel`` from the active trajectory follower, once
        per nav tick -- not every dt_sim step, which would run the follower's
        internal PID far faster than the gains (tuned for dt_nav) expect.
        Stops itself a moment after the profile's timed duration elapses,
        holding the final PID-converged pose and freeing /cmd_vel for manual
        use again without waiting for an explicit cancel."""
        if self._follower is None or not self.sim.clock.nav_fires():
            return
        traj = self._follower.traj
        pose = self.sim.body.pose
        dt_nav = self.sim.clock.config.dt_nav

        # The trajectory's own clock is *governed*, not just elapsed sim time.
        # A time-indexed reference assumes the robot keeps up; anything that
        # delays it -- a collision resolving to zero velocity, a slow gate
        # crossing -- makes the reference run away, and the old
        # "t > duration + 0.5 -> done" test then declared success wherever the
        # robot happened to be. That is the "it stops part-way" symptom: no
        # error, no warning, just a stopped robot metres short of the goal.
        # Holding the reference whenever the robot lags turns a delay into a
        # delay instead of an abandonment.
        ref = traj.sample(self._traj_t)
        lag = float(np.hypot(ref.x - pose[0], ref.y - pose[1]))
        if lag < self._traj_lookahead:
            self._traj_t += dt_nav
            self._traj_stalled = 0.0
        else:
            self._traj_stalled += dt_nav

        goal = traj.points[-1]
        d_goal = float(np.hypot(goal[0] - pose[0], goal[1] - pose[1]))

        if self._traj_stalled > self._traj_stall_timeout:
            self._stop_following(
                f"giving up {d_goal:.2f} m short: held at ({pose[0]:.2f}, "
                f"{pose[1]:.2f}) on {self._cur_level} for "
                f"{self._traj_stalled:.1f} s -- the reference is {lag:.2f} m "
                "ahead and the robot is not closing it", warn=True)
            return

        if self._traj_t >= traj.duration:
            # The profile ends at zero speed, so the last few centimetres are
            # the PID settling. Bound that too, and report the miss rather
            # than sitting on it.
            self._traj_settle += dt_nav
            if d_goal < self._traj_goal_tol:
                self._stop_following(f"reached the goal ({d_goal:.3f} m error)")
                return
            if self._traj_settle > self._traj_settle_timeout:
                self._stop_following(
                    f"stopped {d_goal:.2f} m from the goal after "
                    f"{self._traj_settle:.1f} s of settling", warn=True)
                return

        self._cmd_vel = self._follower.command(self._traj_t, pose)

    def _stop_following(self, why: str, warn: bool = False) -> None:
        self._follower = None
        self._cmd_vel = np.zeros(3)
        log = self.get_logger().warn if warn else self.get_logger().info
        log(f"following ended: {why}")

    def _resolve_collision(self) -> None:
        """Undo a motion that put the real (rotated) footprint into a wall, the
        opponent's half, or off the edge of the level it's on.

        Same rules and the same ``LeveledField`` the matplotlib viewer uses,
        so both front-ends agree about what the robot can physically do --
        until this was here, /cmd_vel drove straight through L1 and across the
        centre divider. The level is *persistent state* that only changes
        inside a ramp/stairs gate; see LeveledField.resolve_level for why a
        0.9 m body needs both levels valid while it straddles a gate.

        Run on *distance moved*, not every integration step. The check now
        samples the whole footprint outline rather than four corners, which
        costs ~60 us; at dt_sim = 1e-4 that would be 0.6 s of CPU per simulated
        second, and the robot only moves 0.12 mm per step anyway. Checking
        every ``collision_check_ds`` of travel (or of corner travel, for a pure
        rotation) keeps the cost negligible and bounds how far the body can
        enter anything to that same distance -- 5 mm by default, against the
        443 mm the previous corner-sampling allowed.
        """
        if self._leveled is None or self._footprint is None:
            self._blocked = False
            return
        pose = self.sim.body.pose
        moved = float(np.hypot(pose[0] - self._checked_pose[0],
                               pose[1] - self._checked_pose[1]))
        swept = self._footprint.r_circ * abs(
            float(np.arctan2(np.sin(pose[2] - self._checked_pose[2]),
                             np.cos(pose[2] - self._checked_pose[2]))))
        if max(moved, swept) < self._collision_check_ds:
            return
        self._checked_pose = pose.copy()

        new_level, collision_levels = self._leveled.resolve_level(
            self._cur_level, pose[0], pose[1])
        corners = self._footprint.corners(pose[0], pose[1], pose[2])
        if self._leveled.is_blocked(collision_levels, corners):
            # Restore the heading too, not just the position: a rotation in
            # place can drive a corner into a wall all by itself, and undoing
            # only the translation leaves it there for good.
            self.sim.body.state[:3] = self._safe_pose
            self.sim.body.state[3:6] = 0.0
            self._checked_pose = self._safe_pose.copy()
            self._collision_steps += 1
            self._blocked = True
            if not self._collision_logged:
                self.get_logger().warn(
                    f"blocked at ({pose[0]:.2f}, {pose[1]:.2f}) on {self._cur_level}"
                    " -- wall, level edge, or opponent territory")
                self._collision_logged = True
        else:
            if self._cur_level != new_level:
                self.get_logger().info(f"level {self._cur_level} -> {new_level}")
                # LiDAR sees whichever level the robot is standing on. Swapping
                # the grid on the one sensor (rather than keeping one Lidar per
                # level) keeps a single firing schedule: an idle Lidar's
                # maybe_sample() would fire a catch-up burst when reactivated.
                self.lidar.grid = self._leveled.grids[new_level]
            self._cur_level = new_level
            self._collision_logged = False
            self._blocked = False
            self._safe_pose = pose.copy()

    def _wheel_torque(self) -> np.ndarray:
        if self._wheel_torque_cmd is not None:
            return self._wheel_torque_cmd
        return self.sim._wheel_torque_from_cmd_vel(self._cmd_vel, kp=0.2)

    def _on_tick(self) -> None:
        if self._paused:
            return
        dt_sim = self.sim.clock.config.dt_sim
        for _ in range(self._steps_per_tick):
            self._advance_following()
            torque = self._wheel_torque()
            self.sim.step(torque)
            self._resolve_collision()
            self.sim.update_odometry(
                wheel_angle=self.encoder._rad_per_count * np.round(
                    self.sim.wheel_angle_true / self.encoder._rad_per_count))
            t = self.sim.clock.t
            self._poll_and_publish(t, dt_sim)
        # publish clock once per tick with the latest sim time
        self.pub_clock.publish(Clock(clock=_sim_time_msg(self.sim.clock.t)))
        if self._dash is not None:
            self._render_dashboard()

    def _render_dashboard(self) -> None:
        if not self._dash.should_render():
            return
        from omni_sim_core.ui.console import Section, C, health_color

        pose = self.sim.body.pose
        twist = self.sim.body.twist_body()
        odom = self.sim.odom_pose
        drift = float(np.linalg.norm(pose[:2] - odom[:2]))
        wheel = self.jac_wheel_speeds()
        dc = health_color(drift, warn=0.05, bad=0.2)
        mode = "TORQUE" if self._wheel_torque_cmd is not None else "CMD_VEL"
        state = f"{C.warn}PAUSED{C.reset}" if self._paused else f"{C.good}RUN{C.reset}"

        sim_sec = Section("SIMULATION", [
            f"sim_time {self.sim.clock.t:8.3f} s   rtf {self.rtf:.2f}   "
            f"{self._steps_per_tick} steps/tick   {state}",
            f"input    {mode:<8}  cmd_vel [{self._cmd_vel[0]:+.2f} "
            f"{self._cmd_vel[1]:+.2f} {self._cmd_vel[2]:+.2f}]",
        ])
        pose_sec = Section("ROBOT (ground truth)", [
            f"pose   x {pose[0]:7.3f}  y {pose[1]:7.3f}  th {pose[2]:+7.3f} rad",
            f"twist  vx {twist[0]:+6.3f} vy {twist[1]:+6.3f} wz {twist[2]:+6.3f} (body)",
            f"wheels " + " ".join(f"{w:+6.2f}" for w in wheel) + " rad/s",
        ])
        odom_sec = Section("ODOMETRY", [
            f"odom   x {odom[0]:7.3f}  y {odom[1]:7.3f}  th {odom[2]:+7.3f} rad",
            f"drift  {dc}{drift:7.4f} m{C.reset}  (map->odom correction)",
        ])
        sens_sec = Section("SENSORS published", [
            f"scan {self._pub_counts['scan']:<7} imu {self._pub_counts['imu']:<7} "
            f"joint {self._pub_counts['joint']:<7} odom {self._pub_counts['odom']}",
        ])
        self._dash.render("running", [sim_sec, pose_sec, odom_sec, sens_sec])

    def jac_wheel_speeds(self) -> np.ndarray:
        return self.sim.jac.wheel_velocity_from_body(self.sim.body.twist_body())

    def _poll_and_publish(self, t: float, dt_sim: float) -> None:
        stamp = _sim_time_msg(t)

        # ground-truth twists / accel for sensor truth
        twist_world = self.sim.body.twist_world
        accel_world = (twist_world[:2] - self._prev_twist_world[:2]) / dt_sim
        self._prev_twist_world = twist_world.copy()
        th = self.sim.body.pose[2]
        c, s = math.cos(th), math.sin(th)
        accel_body = np.array([c * accel_world[0] + s * accel_world[1],
                               -s * accel_world[0] + c * accel_world[1]])

        imu_truth = {"angular_velocity_z": float(twist_world[2]),
                     "linear_acceleration": accel_body}
        reading = self.imu.maybe_sample(t, imu_truth)
        if reading is not None:
            self._publish_imu(stamp, reading)

        enc = self.encoder.maybe_sample(
            t, {"wheel_angle_rad": self.sim.wheel_angle_true})
        if enc is not None:
            self._publish_joint(stamp, enc)

        scan = self.lidar.maybe_sample(
            t, {"pose": self.sim.body.pose, "twist": twist_world})
        if scan is not None:
            self._publish_scan(stamp, scan)

        # odom / ground truth at nav rate
        if self.sim.clock.nav_fires():
            self._publish_odom(stamp)
            self._publish_state(t)

    # -- message builders -------------------------------------------------
    def _publish_imu(self, stamp, reading) -> None:
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = "imu_link"
        msg.angular_velocity.z = reading.angular_velocity_z
        msg.linear_acceleration.x = float(reading.linear_acceleration[0])
        msg.linear_acceleration.y = float(reading.linear_acceleration[1])
        self.pub_imu.publish(msg)
        self._pub_counts["imu"] += 1

    def _publish_joint(self, stamp, enc) -> None:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = [f"wheel_{i}" for i in range(len(enc.angle_rad))]
        msg.position = [float(a) for a in enc.angle_rad]
        self.pub_joint.publish(msg)
        self._pub_counts["joint"] += 1

    def _publish_scan(self, stamp, scan) -> None:
        p = self.lidar.lp
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = "laser_link"
        msg.angle_min = float(scan.angles[0])
        msg.angle_increment = float(scan.angles[1] - scan.angles[0])
        msg.angle_max = msg.angle_min + msg.angle_increment * (len(scan.angles) - 1)
        msg.range_min = p.range_min_m
        msg.range_max = p.range_max_m
        msg.ranges = [float(r) for r in scan.ranges]
        self.pub_scan.publish(msg)
        self._pub_counts["scan"] += 1

    def _odom_msg(self, stamp, pose, twist, frame_child) -> Odometry:
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = frame_child
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        qx, qy, qz, qw = yaw_to_quat(pose[2])
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.twist.twist.linear.x = float(twist[0])
        msg.twist.twist.linear.y = float(twist[1])
        msg.twist.twist.angular.z = float(twist[2])
        return msg

    def _publish_odom(self, stamp) -> None:
        gt_pose = self.sim.body.pose
        odom_pose = self.sim.odom_pose
        self.pub_odom.publish(
            self._odom_msg(stamp, odom_pose, self.sim.body.twist_body(), "base_link"))
        gt = self._odom_msg(stamp, gt_pose, self.sim.body.twist_body(), "base_link")
        gt.header.frame_id = "map"
        self.pub_gt.publish(gt)
        self.tf.publish(stamp, gt_pose, odom_pose)
        self._pub_counts["odom"] += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

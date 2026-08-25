"""ROS 2 (Jazzy) node wrapping omni_sim_core.

This is the *only* file allowed to import rclpy. It drives ``RobotSim`` and the
sensor models, publishes /clock in simulation time, the sensor topics, odom and
ground-truth odom, and exposes reset/pause/set_pose services.

Real-time behaviour is controlled by ``real_time_factor``:
    > 0 : advance sim so that it tracks wall-clock * factor
    = 0 : advance a fixed batch each wall tick (as fast as possible)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from omni_sim_core.clock import ClockConfig
from omni_sim_core.simulator import RobotSim, RobotConfig
from omni_sim_core.sensors.imu import Imu as ImuModel, ImuParams
from omni_sim_core.sensors.lidar import Lidar as LidarModel, LidarParams
from omni_sim_core.sensors.encoder import Encoder as EncoderModel, EncoderParams
from omni_sim_core.env.occupancy_grid import OccupancyGrid, generate_rect_field

from .tf_broadcaster import RobotTf, yaw_to_quat


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

        map_yaml = self.get_parameter("map_yaml").value
        if map_yaml:
            self.grid = OccupancyGrid.from_yaml(map_yaml)
        else:
            self.grid = generate_rect_field(6.0, 4.0, 0.05)
            self.get_logger().info("no map_yaml given; using a 6x4 m rect field")
        # place robot roughly in the middle
        self.sim.reset(pose=np.array([1.0, 1.0, 0.0]))

        rng = np.random.default_rng(seed)
        self.imu = ImuModel(ImuParams(), rng)
        self.lidar = LidarModel(LidarParams(), self.grid, rng)
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

        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
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

    def destroy_node(self) -> bool:
        if self._dash is not None:
            self._dash.stop()
        return super().destroy_node()

    # -- callbacks --------------------------------------------------------
    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self._wheel_torque_cmd = None  # cmd_vel takes over

    def _on_wheel_torque(self, msg: Float64MultiArray) -> None:
        self._wheel_torque_cmd = np.asarray(msg.data, dtype=float)

    def _on_disturbance(self, msg: Float64MultiArray) -> None:
        d = np.asarray(msg.data, dtype=float)
        self._ext_wrench = d[:3] if d.size >= 3 else np.pad(d, (0, 3 - d.size))

    def _srv_reset(self, req, resp):
        self.sim.reset(pose=np.array([1.0, 1.0, 0.0]))
        self.imu.reset(); self.lidar.reset(); self.encoder.reset()
        resp.success = True
        resp.message = "sim reset"
        return resp

    def _srv_pause(self, req, resp):
        self._paused = req.data
        resp.success = True
        resp.message = "paused" if req.data else "resumed"
        return resp

    # -- main loop --------------------------------------------------------
    def _wheel_torque(self) -> np.ndarray:
        if self._wheel_torque_cmd is not None:
            return self._wheel_torque_cmd
        return self.sim._wheel_torque_from_cmd_vel(self._cmd_vel, kp=0.2)

    def _on_tick(self) -> None:
        if self._paused:
            return
        dt_sim = self.sim.clock.config.dt_sim
        for _ in range(self._steps_per_tick):
            torque = self._wheel_torque()
            self.sim.step(torque)
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

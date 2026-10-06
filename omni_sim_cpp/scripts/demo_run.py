"""Drive the sim through a short holonomic sequence and record it.

One node that both PUBLISHES phased wheel-torque commands (to show off omni
motion: forward, spin, strafe, brake) and RECORDS /odom + /imu to a CSV. It
stops itself after the sequence and writes the file.

Run sim_node separately (no controller_node needed):
    ros2 run omni_sim_ros sim_node --ros-args -p plant_config:=<plant_true.yaml>
    python3 scripts/demo_run.py <out.csv>

Explicit loops only (project rule).
"""

import csv
import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray


# Wheel-torque patterns for the symmetric X-config (wheels 0..3).
# Each produces a "pure" body wrench; see the plant tests.
PATTERN_FORWARD_X = [-1.0, -1.0, 1.0, 1.0]   # +x body force
PATTERN_STRAFE_Y = [1.0, -1.0, -1.0, 1.0]    # +y body force
PATTERN_SPIN = [1.0, 1.0, 1.0, 1.0]          # +yaw moment


def scaled(pattern, gain):
    result = []
    for value in pattern:
        result.append(value * gain)
    return result


class DemoRun(Node):
    def __init__(self, output_path):
        super().__init__("demo_run")
        self.output_path = output_path

        self.publisher = self.create_publisher(Float64MultiArray, "wheel_torques", 10)
        self.create_subscription(Imu, "imu", self.on_imu, 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 50)

        self.rows = []
        self.start_time = None
        self.latest_imu = None

        # Command sequence: (duration_seconds, wheel_torque_list).
        self.phases = [
            (2.0, scaled(PATTERN_FORWARD_X, 0.15)),  # accelerate forward
            (1.5, scaled(PATTERN_SPIN, 0.06)),       # rotate in place
            (2.5, scaled(PATTERN_STRAFE_Y, 0.15)),   # strafe sideways
            (2.0, scaled(PATTERN_FORWARD_X, -0.15)), # brake / reverse thrust
            (1.0, [0.0, 0.0, 0.0, 0.0]),             # coast
        ]
        self.total_duration = 0.0
        for duration, _ in self.phases:
            self.total_duration += duration

        # Publish commands at 50 Hz off wall time.
        self.create_timer(0.02, self.on_command_timer)

    def on_imu(self, message):
        self.latest_imu = message

    def stamp_seconds(self, message):
        stamp = message.header.stamp
        return stamp.sec + stamp.nanosec * 1e-9

    def elapsed_from(self, now):
        if self.start_time is None:
            self.start_time = now
        return now - self.start_time

    def command_for(self, elapsed):
        boundary = 0.0
        for duration, torques in self.phases:
            boundary += duration
            if elapsed < boundary:
                return torques
        return [0.0, 0.0, 0.0, 0.0]

    def on_command_timer(self):
        # Use sim time so the phases line up with the recorded stamps.
        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = self.elapsed_from(now)

        command = Float64MultiArray()
        command.data = self.command_for(elapsed)
        self.publisher.publish(command)

    def on_odom(self, message):
        elapsed = self.elapsed_from(self.stamp_seconds(message))

        gyro_z = 0.0
        accel_x = 0.0
        accel_y = 0.0
        if self.latest_imu is not None:
            gyro_z = self.latest_imu.angular_velocity.z
            accel_x = self.latest_imu.linear_acceleration.x
            accel_y = self.latest_imu.linear_acceleration.y

        row = {
            "t": elapsed,
            "odom_x": message.pose.pose.position.x,
            "odom_y": message.pose.pose.position.y,
            "vx_body": message.twist.twist.linear.x,
            "vy_body": message.twist.twist.linear.y,
            "omega": message.twist.twist.angular.z,
            "imu_gyro_z": gyro_z,
            "imu_accel_x": accel_x,
            "imu_accel_y": accel_y,
        }
        self.rows.append(row)

        if elapsed >= self.total_duration:
            self.write_csv()
            self.get_logger().info(
                "recorded %d samples over %.2f s -> %s"
                % (len(self.rows), elapsed, self.output_path)
            )
            raise SystemExit

    def write_csv(self):
        field_names = [
            "t", "odom_x", "odom_y", "vx_body", "vy_body",
            "omega", "imu_gyro_z", "imu_accel_x", "imu_accel_y",
        ]
        output_file = open(self.output_path, "w", newline="")
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row)
        output_file.close()


def main():
    output_path = sys.argv[1]

    rclpy.init()
    node = DemoRun(output_path)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Record a short demo run of the omni_sim ROS system into a CSV.

Subscribes to /odom (encoder dead-reckoned pose + body twist) and /imu (noisy
yaw rate and body acceleration), stamped with sim time, and writes one row per
odom sample. Meant to be launched alongside sim.launch.py; it stops itself after
`duration` seconds of sim time.

Deliberately explicit loops, no list comprehensions (project rule).
"""

import csv
import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class DemoRecorder(Node):
    def __init__(self, output_path, duration_seconds):
        super().__init__("demo_recorder")
        self.output_path = output_path
        self.duration_seconds = duration_seconds

        self.rows = []
        self.start_time = None
        self.latest_imu = None

        self.create_subscription(Imu, "imu", self.on_imu, 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 50)

    def on_imu(self, message):
        self.latest_imu = message

    def stamp_seconds(self, message):
        stamp = message.header.stamp
        return stamp.sec + stamp.nanosec * 1e-9

    def on_odom(self, message):
        now = self.stamp_seconds(message)
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time

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

        if elapsed >= self.duration_seconds:
            self.write_csv()
            self.get_logger().info(
                "recorded %d samples over %.2f s -> %s"
                % (len(self.rows), elapsed, self.output_path)
            )
            raise SystemExit

    def write_csv(self):
        field_names = [
            "t",
            "odom_x",
            "odom_y",
            "vx_body",
            "vy_body",
            "omega",
            "imu_gyro_z",
            "imu_accel_x",
            "imu_accel_y",
        ]
        output_file = open(self.output_path, "w", newline="")
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row)
        output_file.close()


def main():
    output_path = sys.argv[1]
    duration_seconds = float(sys.argv[2])

    rclpy.init()
    recorder = DemoRecorder(output_path, duration_seconds)
    try:
        rclpy.spin(recorder)
    except SystemExit:
        pass
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

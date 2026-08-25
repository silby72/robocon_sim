"""TF helpers: map->odom (from ground truth), odom->base_link (from wheel
odometry, so it drifts), and the static sensor frames."""
from __future__ import annotations

import math

from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def make_transform(stamp, parent: str, child: str,
                   x: float, y: float, yaw: float) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = float(x)
    tf.transform.translation.y = float(y)
    tf.transform.translation.z = 0.0
    qx, qy, qz, qw = yaw_to_quat(yaw)
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf


class RobotTf:
    def __init__(self, node, lidar_mount=(0.0, 0.0, 0.0),
                 imu_mount=(0.0, 0.0, 0.0)) -> None:
        self._dyn = TransformBroadcaster(node)
        self._static = StaticTransformBroadcaster(node)
        stamp = node.get_clock().now().to_msg()
        self._static.sendTransform([
            make_transform(stamp, "base_link", "laser_link", *lidar_mount),
            make_transform(stamp, "base_link", "imu_link", *imu_mount),
        ])

    def publish(self, stamp, gt_pose, odom_pose) -> None:
        # map->odom carries the true offset so map->base_link equals ground
        # truth; odom->base_link is the (drifting) wheel odometry.
        gx, gy, gth = gt_pose
        ox, oy, oth = odom_pose
        # map->odom = map->base * (odom->base)^-1  (planar composition)
        c, s = math.cos(-oth), math.sin(-oth)
        dx = gx - (c * ox - s * oy)
        dy = gy - (s * ox + c * oy)
        self._dyn.sendTransform([
            make_transform(stamp, "map", "odom", dx, dy, gth - oth),
            make_transform(stamp, "odom", "base_link", ox, oy, oth),
        ])

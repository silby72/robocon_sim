"""Launch the omni_sim node with use_sim_time and (optionally) RViz2."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("omni_sim_ros")
    rviz_cfg = os.path.join(pkg, "rviz", "omni_sim.rviz")

    map_yaml = LaunchConfiguration("map_yaml")
    chassis_yaml = LaunchConfiguration("chassis_yaml")
    rtf = LaunchConfiguration("real_time_factor")
    use_rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", default_value=""),
        # Empty is fine: sim_node falls back to <repo>/config/robot/chassis.yaml
        # next to the map. Pass this only to run a different chassis.
        DeclareLaunchArgument("chassis_yaml", default_value="",
                              description="config/robot/chassis.yaml; the footprint collision is judged with"),
        DeclareLaunchArgument("real_time_factor", default_value="1.0"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="omni_sim_ros",
            executable="sim_node",
            name="omni_sim",
            output="screen",
            # sim_node is the /clock *source* -- it must NOT set use_sim_time
            # on itself. That parameter makes rclpy's create_timer() wait on
            # /clock before firing, and the only thing that will ever publish
            # /clock is the timer callback it's now waiting on: a permanent,
            # silent deadlock (looks exactly like a hang; no error, 0% CPU).
            # Consumers (RViz2 below) still need use_sim_time to sync to it.
            parameters=[{
                "map_yaml": map_yaml,
                "chassis_yaml": chassis_yaml,
                "real_time_factor": rtf,
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_cfg],
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(use_rviz),
        ),
    ])

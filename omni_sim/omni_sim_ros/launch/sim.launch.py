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
    rtf = LaunchConfiguration("real_time_factor")
    use_rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", default_value=""),
        DeclareLaunchArgument("real_time_factor", default_value="1.0"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="omni_sim_ros",
            executable="sim_node",
            name="omni_sim",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "map_yaml": map_yaml,
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

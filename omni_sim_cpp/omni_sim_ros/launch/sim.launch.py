"""Launch the true-plant sim_node and the nominal-model controller_node.

The two nodes deliberately load two different YAML files:
    sim_node        -> config/plant_true.yaml
    controller_node -> config/plant_nominal.yaml

Both run with use_sim_time:=true; sim_node publishes /clock as the time source.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("omni_sim_ros")
    config_dir = os.path.join(package_share, "config")

    default_true_config = os.path.join(config_dir, "plant_true.yaml")
    default_nominal_config = os.path.join(config_dir, "plant_nominal.yaml")

    true_config_arg = DeclareLaunchArgument(
        "true_config",
        default_value=default_true_config,
        description="Path to the true-plant YAML for sim_node.",
    )
    nominal_config_arg = DeclareLaunchArgument(
        "nominal_config",
        default_value=default_nominal_config,
        description="Path to the nominal-model YAML for controller_node.",
    )

    use_sim_time = {"use_sim_time": True}

    sim_node = Node(
        package="omni_sim_ros",
        executable="sim_node",
        name="sim_node",
        output="screen",
        parameters=[
            use_sim_time,
            {"plant_config": LaunchConfiguration("true_config")},
            {"rate_hz": 200.0},
        ],
    )

    controller_node = Node(
        package="omni_sim_ros",
        executable="controller_node",
        name="controller_node",
        output="screen",
        parameters=[
            use_sim_time,
            {"plant_config": LaunchConfiguration("nominal_config")},
            {"rate_hz": 100.0},
            {"command_torque": 0.5},
        ],
    )

    launch_description = LaunchDescription()
    launch_description.add_action(true_config_arg)
    launch_description.add_action(nominal_config_arg)
    launch_description.add_action(sim_node)
    launch_description.add_action(controller_node)
    return launch_description

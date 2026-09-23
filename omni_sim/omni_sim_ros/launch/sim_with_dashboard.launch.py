"""Launch omni_sim + rosbridge (WebSocket) + rosapi for the browser dashboard.

    ros2 launch omni_sim_ros sim_with_dashboard.launch.py map_yaml:=$PWD/maps/field_2027_ground.yaml

Then open ``omni_sim_web/index.html`` and pick 自動走行 in the sidebar. Opening
the file directly works: the page only needs a WebSocket to rosbridge, and a
file:// origin may open one. Serving it (``python3 -m http.server`` from
omni_sim_web/) buys only the Service Worker, i.e. offline use.

That page is also the chassis/verification console, which runs omni_sim_core
under Pyodide and needs nothing from ROS -- the two halves are independent, so
whichever one is not running simply shows as unavailable.

omni_sim_ros/web/dashboard.html is now a redirect to it.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    map_yaml = LaunchConfiguration("map_yaml")
    chassis_yaml = LaunchConfiguration("chassis_yaml")
    rtf = LaunchConfiguration("real_time_factor")
    port = LaunchConfiguration("port")

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", default_value=""),
        # Empty is fine: sim_node falls back to <repo>/config/robot/chassis.yaml
        # next to the map. Pass this only to run a different chassis.
        DeclareLaunchArgument("chassis_yaml", default_value="",
                              description="config/robot/chassis.yaml; the footprint collision is judged with"),
        DeclareLaunchArgument("real_time_factor", default_value="1.0"),
        DeclareLaunchArgument("port", default_value="9090",
                              description="rosbridge WebSocket port"),
        Node(
            package="omni_sim_ros",
            executable="sim_node",
            name="omni_sim",
            output="screen",
            # Not use_sim_time here: sim_node is the /clock *source*, and
            # that parameter would make its own create_timer() wait on
            # /clock before firing -- a permanent, silent, 0%-CPU deadlock
            # against the very message it's supposed to publish. See
            # sim.launch.py for the longer version of this note.
            parameters=[{
                "map_yaml": map_yaml,
                "chassis_yaml": chassis_yaml,
                "real_time_factor": rtf,
            }],
        ),
        Node(
            package="rosbridge_server",
            executable="rosbridge_websocket",
            name="rosbridge_websocket",
            output="screen",
            parameters=[{"port": port}],
        ),
        Node(
            package="rosapi",
            executable="rosapi_node",
            name="rosapi",
            output="screen",
        ),
    ])

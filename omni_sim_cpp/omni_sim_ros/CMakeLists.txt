cmake_minimum_required(VERSION 3.16)
project(omni_sim_ros LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Pull in the ROS-free core as a subdirectory (single source of truth, no
# separate install needed). Its own tests are turned off in this build.
# ---------------------------------------------------------------------------
set(OMNI_SIM_CORE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../omni_sim_core
                 ${CMAKE_CURRENT_BINARY_DIR}/omni_sim_core_build)

# ---------------------------------------------------------------------------
# ROS 2 dependencies.
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)

set(node_dependencies
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  rosgraph_msgs
)

# ---------------------------------------------------------------------------
# sim_node — runs the true plant.
# ---------------------------------------------------------------------------
add_executable(sim_node src/sim_node.cpp)
target_link_libraries(sim_node omni_sim_core)
ament_target_dependencies(sim_node ${node_dependencies})

# ---------------------------------------------------------------------------
# controller_node — runs on the nominal model.
# ---------------------------------------------------------------------------
add_executable(controller_node src/controller_node.cpp)
target_link_libraries(controller_node omni_sim_core)
ament_target_dependencies(controller_node ${node_dependencies})

# ---------------------------------------------------------------------------
# Install.
# ---------------------------------------------------------------------------
install(TARGETS sim_node controller_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
# Ship the shared config (the true/nominal YAMLs) into the package share so
# launch can find it after a colcon install.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()

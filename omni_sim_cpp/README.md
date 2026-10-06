# omni_sim_cpp

Pure C++ reimplementation of the 4-wheel omni-robot simulator, following the
"True Plant vs. Nominal Model are two separate ROS 2 nodes" design.

This lives *alongside* the existing Python `omni_sim/` project (which stays the
reference for acceptance criteria); it does not replace it.

## Layout

```
omni_sim_cpp/
├── omni_sim_core/   # Pure C++17/20 library (CMake + GoogleTest). No ROS.
├── omni_sim_ros/    # ROS 2 Jazzy package (ament_cmake): sim_node, controller_node
└── scripts/         # Python analysis / sweep / plotting (no TUI, no custom GUI)
```

## Build & test the core (no ROS needed)

```bash
cmake -S omni_sim_core -B omni_sim_core/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build omni_sim_core/build -j
ctest --test-dir omni_sim_core/build --output-on-failure
```

## Progress

- [x] Step 1 — `omni_sim_core` skeleton (CMake, `config.hpp`, `plant.hpp`) + GoogleTest harness
- [x] Step 2 — `plant_true.yaml` / `plant_nominal.yaml` + direct YAML parsing (`config_io`, yaml-cpp)
- [x] Step 3 — RK4 integration + 4-wheel omni dynamics (force map, world/body frames, viscous friction)
- [x] Step 4 — `omni_sim_ros` package: `sim_node` / `controller_node` + `sim.launch.py`
- [x] Sensors — `sensors.hpp`: encoder odometry (quantization) + IMU (bias/white noise), wired into `sim_node`'s `/odom` & `/imu`
- [x] Control — `control.hpp`: PID (deriv-on-measurement, anti-windup) + DOB (bilinear IIR, `d̂ = A(ω) − B(τ)`), ported from the Python `control/`. Wired into `controller_node` as a 3-axis (vx, vy, ω) body-twist loop with per-axis DOB and a wrench→wheel-torque pseudo-inverse.
- [ ] LiDAR 2D raycast; MCL localization (needs LiDAR + map)

## Run the ROS 2 system (Jazzy)

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select omni_sim_ros   # pulls in omni_sim_core as a subdir
source install/setup.bash
ros2 launch omni_sim_ros sim.launch.py
# sim_node -> /clock /odom /imu, subscribes /wheel_torques ; controller_node -> /wheel_torques
```

# omni_sim — omni robot control / trajectory / estimation simulator

A hardware-free Python simulator for studying three things on a 4-wheel
omni-directional robot:

- **Control** — disturbance observers (DOB), two-DOF control, sensitivity,
  robustness.
- **Path & trajectory** — path generation, velocity profiles, tracking.
- **Estimation** — EKF-ready sensor models (IMU / LiDAR / encoder), fusion,
  observation/covariance design.

The guiding design principle: **the true plant and the nominal model the
controller believes in are kept strictly separate**, because that separation is
the whole point of a DOB experiment.

## Layout

```
omni_sim_core/   ROS-independent core (numpy/scipy). Runs under pytest alone.
omni_sim_ros/    ROS 2 Jazzy wrapper (rclpy). The only place rclpy is imported.
config/          plant_true.yaml, plant_nominal.yaml (separate!), scenarios/
maps/            map_server-compatible PGM + YAML
experiments/     headless sweeps, frequency response, plotting
```

## Quick start (no ROS needed)

```bash
PYTHONPATH=omni_sim_core/src pytest omni_sim_core/tests -q      # 18 tests

# run from the project root -- run_sim.py sets up the path for you, no PYTHONPATH needed
python3 run_sim.py --scenario config/scenarios/dob_step_load.yaml           # -> results/*.csv
python3 run_sim.py --scenario config/scenarios/dob_step_load.yaml --ui      # live dashboard (real-time)
python3 run_sim.py --scenario config/scenarios/dob_step_load.yaml --ui --rtf 0   # dashboard, fastest
python experiments/plot_results.py results/dob_step_load.csv                # -> results/*.png
python experiments/sweep.py --n-tau 10 --n-j 5                   # parallel sweep
python experiments/frequency_response.py                         # Bode of S with/without DOB
```

## ROS 2 (Jazzy)

```bash
# in a colcon workspace with omni_sim_core pip-installed:
colcon build --packages-select omni_sim_ros
ros2 launch omni_sim_ros sim.launch.py map_yaml:=$PWD/maps/field.yaml
ros2 run omni_sim_ros sim_node --ros-args -p dashboard:=true   # serial2can-style console dashboard
```

## Real-time matplotlib view

```bash
python3 experiments/live_view.py                       # live window (needs a display)
python3 experiments/live_view.py --save results/live_view.gif --duration 15   # headless -> GIF
```

Robot following a rectangular loop on the field map, with the LiDAR point cloud,
reference vs. travelled path, and the tracking-error trace (spec 10.3). seaborn
is used for styling if installed, otherwise a matplotlib dark theme is used.

## Console UI

`omni_sim_core/ui/` is a terminal live dashboard styled after serial2can
(`serial_bridge/graphical_ui.hpp`, `ros2can/console_ui.py`): the same 256-colour
palette, 100-column `+--+` panels, braille spinner and clear-and-redraw. Used by
the headless runner (`--ui`) and, opt-in, by the ROS node (`dashboard:=true`).

Publishes `/scan /imu/data_raw /joint_states /odom /ground_truth/odom /clock`
and TF `map -> odom -> base_link -> {laser_link, imu_link}`; subscribes
`/cmd_vel /sim/wheel_torque_cmd /sim/disturbance`; services `/sim/reset
/sim/pause`. All nodes assume `use_sim_time: true`.

## Acceptance criteria (spec §11)

All nine are automated in `omni_sim_core/tests/test_acceptance.py`: DOB no-op
under a matched model, step-load rejection, `d_hat -> d` convergence, dt_sim
invariance, bit-exact reproducibility, smooth odometry drift, jacobian roundtrip
identity, rclpy-free import, and LiDAR raycast vs. analytic on a rectangular map.

## Implementation phases

Phase 0 (clock, RK4, motor, nominal, disturbance, PID, DOB) is complete and is
all that control-theory study needs. Phases 1–2 (jacobian/body/trajectory,
occupancy grid/raycast/sensors) and the Phase 3 ROS wrapper and Phase 4
experiment harness are also implemented. See `docs`/spec for the roadmap.

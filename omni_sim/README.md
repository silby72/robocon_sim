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

## Setup

```bash
python setup_env.py            # venv + deps + VS Code, safe to re-run any time
python setup_env.py --check    # show what would change, touch nothing
python setup_env.py --verify   # ...and prove it works: run the tests, regen maps
```

Re-run it whenever dependencies move — it is idempotent and syncs rather than
reinstalls. It reads the dependency list straight out of
`omni_sim_core/pyproject.toml`, so **add new runtime dependencies there**, not
to the setup script; only editor extensions and the default extras list live in
the CONFIG block at the top of `setup_env.py`.

Useful flags:

| flag | when |
|---|---|
| `--system` | inside a sourced ROS 2 env — never stack a venv on top of ROS |
| `--python 3.12 --recreate` | rebuild the venv on another interpreter |
| `--extras test,viz,viz-extra` | pick which pyproject extras to install |
| `--no-vscode` | skip editor setup |

On Windows, a `DLL load failed` on a freshly released Python is usually Smart
App Control refusing a low-reputation wheel, not a broken package; the script
detects this and prints the options. `--python 3.12 --recreate` is the cheap
way out.

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

## Robot description → plant (mechanism layer, Phase A)

Describe the machine in `config/robot/` (chassis geometry, actuator datasheets)
and generate the simulator's plant from it:

```bash
python tools/build_plant.py     # config/robot/* + presets → config/generated/plant_*.yaml
```

- `config/robot/chassis.yaml` — wheel layout, radii, **gear ratios (required)**, CoM.
- `config/robot/actuators.yaml` — motor presets (`config/presets/motors/*.yaml`) + per-unit overrides, or a raw datasheet.
- `config/robot/model_error.yaml` — nominal = true × ratios (`ratio` mode) or a fully hand-written nominal (`manual` mode).
- Output goes to `config/generated/` with a `GENERATED` header; values that were
  estimated (unknown rotor inertia, lumped damping) are tagged `# ESTIMATED`.
  The hand-written `config/plant_*.yaml` are never overwritten.

Scenarios pick which plant to use:

```yaml
plant:
  true_model: config/generated/plant_true.yaml
  nominal_model: config/generated/plant_nominal.yaml
control_rate_hz: 1000.0    # motor loop rate; integration stays at clock.dt_sim
```

The GUI that edits these files (PySide6) is Phase B and installs separately
(`python setup_env.py --extras gui`); nothing here depends on it.

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

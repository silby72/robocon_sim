# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hardware-free simulator for a 4-wheel omni robot, built to study disturbance
observers (DOB), trajectory following, and state estimation. The full build
spec (Japanese) defines the acceptance criteria; those criteria — not any
particular implementation detail — are the contract.

## Commands

Everything in the core runs without ROS. Use `PYTHONPATH=omni_sim_core/src` (or
`pip install -e omni_sim_core[test,viz]`).

```bash
# environment: idempotent, re-run whenever deps move. Reads the dependency list
# from omni_sim_core/pyproject.toml -- add runtime deps THERE, not to the script.
python setup_env.py [--check|--verify|--system|--python 3.12 --recreate]

# tests (18: 9 acceptance in test_acceptance.py + 9 unit in test_components.py)
PYTHONPATH=omni_sim_core/src pytest omni_sim_core/tests -q
PYTHONPATH=omni_sim_core/src pytest omni_sim_core/tests/test_acceptance.py::test_dhat_converges_to_disturbance -q   # single test

# headless scenario -> CSV. run_sim.py (project root) injects the path, so no PYTHONPATH needed.
# --ui = live dashboard, real-time paced (--rtf 1.0 default; --rtf 0 = fastest).
python3 run_sim.py --scenario config/scenarios/dob_step_load.yaml [--ui] [--rtf 1.0]

# experiments (run from repo root; they inject omni_sim_core/src onto sys.path themselves)
python experiments/plot_results.py results/dob_step_load.csv
python experiments/sweep.py --n-tau 10 --n-j 5          # multiprocessing sweep
python experiments/frequency_response.py                # ~30 s: Bode of sensitivity S

# ROS 2 (Jazzy) — needs a colcon workspace with omni_sim_core pip-installed
colcon build --packages-select omni_sim_ros
ros2 launch omni_sim_ros sim.launch.py map_yaml:=$PWD/maps/field.yaml
```

## Architecture — the non-obvious parts

**True plant vs. nominal model are deliberately two separate objects, two
separate YAMLs.** `plant/motor.py` (`MotorArray`, with friction/cogging/
saturation) is the *real* motor; `plant/nominal.py` (`NominalModel`, a bare
`1/(Jn s + Bn)`) is what the controller believes. The DOB's only job is to
reconstruct the gap between them. Never merge these or load them from one file —
`config.py` keeps `plant_true.yaml` and `plant_nominal.yaml` distinct on
purpose. `nominal_from_true(true, j_ratio, b_ratio)` is how experiments inject a
known model error.

**Layered plant, forces flow up, kinematics flow down.** `motor -> jacobian ->
body`. `JacobianLayer` (`plant/jacobian.py`) is a *static* 4×3 map: wheel torque
→ wheel force → body wrench (`J^T`), and body velocity → wheel speed. `RigidBody`
(`plant/body.py`) is a 3-DOF world-frame rigid body driven by that wrench.
Wheels are rigidly coupled to the body (no slip — slip is explicitly out of
scope), so in `RobotSim.step` the true wheel angle is integrated from the body
twist, not from free-spinning motors. This is what makes wheel odometry
physically meaningful (drift comes only from encoder quantization). Future slip
work plugs a `WheelDynamics` block into the jacobian layer — keep its interface
as `wheel-velocity <-> body-velocity`.

**Two simulator entry points** in `simulator.py`: `MotorControlSim` (single-axis
speed loop, PID+DOB — the Phase 0 / control-theory workhorse behind acceptance
tests 1–5) and `RobotSim` (full body + jacobian + no-slip wheels — Phase 1/2,
odometry). The headless `run.py` only wires up `MotorControlSim`.

**Multi-rate, zero-order hold, fixed-step.** `MultiRateClock` (`clock.py`)
advances `dt_sim` and fires the motor loop and nav loop on integer strides;
`dt_sim` must divide the other periods (validated at construction). Integration
is fixed-step RK4 (`integrator.py`) — *never* `solve_ivp`; variable steps break
control-period sync and reproducibility.

**The DOB never realizes `P_n^{-1}` alone** (it's improper). `control/dob.py`
implements two proper, Tustin-discretized filters — `Q·P_n^{-1}` on ω and `Q` on
τ — and subtracts. `tau_q` is runtime-changeable (`set_tau_q`) for robustness
sweeps; `order` selects a 1st-order or 2nd-order Butterworth `Q`.

**Determinism is a hard requirement.** Every random draw goes through an
explicitly passed `numpy.random.Generator` (`disturbance.py`, `sensors/`). No
global RNG. Same seed ⇒ bit-identical output (acceptance test 5).

**rclpy isolation is enforced by test.** Nothing under `omni_sim_core/` may
import `rclpy`; `sim_node.py` in `omni_sim_ros/` is the sole exception.
`test_core_imports_without_rclpy` guards this.

**The console UI mimics serial2can.** `omni_sim_core/ui/console.py` copies the
`serial_bridge`/`ros2can` dashboard style (256-colour palette, 100-col `+--+`
panels, braille spinner, clear-and-redraw, east-asian-width-aware padding). It is
pure stdlib and core-clean. `MotorControlSim.run(observer=...)` feeds it live;
`run.py --ui` and the ROS node's `dashboard:=true` param are the two entry points.
Keep any new dashboard in this exact style — the whole toolchain shares it.
`ui/viewer.py` is the matplotlib real-time view (robot/LiDAR/path/error, spec
10.3); `experiments/live_view.py` launches it (`--save X.gif` for headless).
seaborn is optional styling (imported in a try/except); never a hard dep.

**Trajectory following runs the control at the nav rate, in the body frame.**
`TrajectoryFollower.command` returns a *body-frame* twist (world-frame PID output
rotated by `-theta`) with the linear speed clamped; the wheel jacobian expects
body frame. Feeding a world-frame command, or refiring only once per animation
frame instead of every `dt_nav`, makes the loop diverge — both were real bugs.

**Raycasting is vectorized over beams, looped over DDA steps** (`env/raycast.py`)
— a per-beam Python loop can't sustain 450 beams @ 10 Hz. It returns `inf` on a
miss (LaserScan convention) and the entry distance to the first occupied cell on
a hit.

## Conventions

SI units everywhere, encoded in variable names (`omega_rad_s`, `torque_nm`). All
tuning lives in `@dataclass` configs loaded from YAML via `config.py`; don't bury
magic numbers in code. Prefer numpy vectorization over element loops. Sensors
subclass `sensors/base.py` (rate / latency FIFO / jitter / dropout / quantize)
and implement `_measure`.

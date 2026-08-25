"""Unit tests for individual core components (beyond the acceptance criteria)."""
import numpy as np
import pytest

from omni_sim_core.clock import ClockConfig, MultiRateClock
from omni_sim_core.integrator import rk4_step
from omni_sim_core.plant.motor import MotorParams, MotorArray
from omni_sim_core.plant.body import RigidBody, BodyParams
from omni_sim_core.control.dob import DisturbanceObserver, DOBParams
from omni_sim_core.plant.nominal import NominalModel
from omni_sim_core.control.trajectory import (Trajectory, TrajectoryParams)
from omni_sim_core.disturbance import DisturbanceManager, DisturbanceSpec
from omni_sim_core.sensors.base import Sensor, SensorParams
from omni_sim_core.env.occupancy_grid import (generate_rect_field, OccupancyGrid,
                                              OCCUPIED)


def test_clock_rejects_non_divisor_periods():
    with pytest.raises(ValueError):
        MultiRateClock(ClockConfig(dt_sim=3e-4, dt_motor=1e-3, dt_nav=2e-2))


def test_clock_firing_strides():
    clk = MultiRateClock(ClockConfig(dt_sim=1e-4, dt_motor=1e-3, dt_nav=2e-2))
    motor_fires = nav_fires = 0
    for _ in range(200):
        if clk.motor_fires():
            motor_fires += 1
        if clk.nav_fires():
            nav_fires += 1
        clk.advance()
    assert motor_fires == 20  # every 10 steps over 200
    assert nav_fires == 1     # step 0 only, over 200 (next at 200)


def test_rk4_matches_analytic_exponential():
    # y' = -2y, y(0)=1 -> y(t)=exp(-2t)
    f = lambda t, y: -2.0 * y
    y = np.array([1.0])
    dt = 1e-3
    for _ in range(1000):
        y = rk4_step(f, 0.0, y, dt)
    assert abs(y[0] - np.exp(-2.0)) < 1e-8


def test_motor_electrical_dynamics_runs():
    p = MotorParams(n_motors=1, enable_electrical_dynamics=True)
    m = MotorArray(p)
    assert m.state_size == 3
    for _ in range(100):
        m.state = rk4_step(lambda t, y: m.deriv(t, y, np.array([0.5])),
                           0.0, m.state, 1e-4)
    assert m.omega()[0] > 0  # spins up under positive command


def test_dob_order2_builds_and_estimates_zero_at_rest():
    dob = DisturbanceObserver(DOBParams(enabled=True, tau_q=5e-3, order=2),
                              NominalModel(), dt=1e-3)
    for _ in range(50):
        d = dob.update(0.0, 0.0)
    assert abs(d) < 1e-9


def test_disturbance_is_deterministic_given_generator():
    spec = [DisturbanceSpec(target="motor_torque", waveform="white", sigma=1.0)]
    a = DisturbanceManager(spec, 1.0, np.random.default_rng(7), n_motors=1)
    b = DisturbanceManager(spec, 1.0, np.random.default_rng(7), n_motors=1)
    va = [a.vector("motor_torque", i * 1e-3, 1e-3)[0] for i in range(50)]
    vb = [b.vector("motor_torque", i * 1e-3, 1e-3)[0] for i in range(50)]
    assert va == vb


def test_sensor_latency_delays_output():
    class Passthrough(Sensor):
        def _measure(self, t, truth):
            return truth
    s = Passthrough(SensorParams(rate_hz=100.0, latency_s=0.05),
                    np.random.default_rng(0))
    outputs = []
    dt = 1e-3
    for i in range(200):
        outputs.append(s.maybe_sample(i * dt, i * dt))
    first_release = next(i for i, o in enumerate(outputs) if o is not None)
    # first sample fires at t=0, released ~0.05 s later
    assert 45 <= first_release <= 55


def test_occupancy_grid_roundtrip_and_walls():
    grid = generate_rect_field(2.0, 2.0, 0.05, wall_thickness_cells=1)
    assert grid.grid[0, 0] == OCCUPIED
    r, c = grid.world_to_grid(1.0, 1.0)
    x, y = grid.grid_to_world(r, c)
    assert abs(x - 1.0) < 0.05 and abs(y - 1.0) < 0.05


def test_trajectory_reaches_endpoint():
    traj = Trajectory(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
                      TrajectoryParams(v_max=1.0, a_max=2.0, dt=0.02))
    end = traj.sample(traj.duration)
    assert abs(end.x - 1.0) < 0.05 and abs(end.y - 1.0) < 0.05

"""Acceptance criteria from spec section 11. Passing these is the completion
condition for the project."""
import copy
import sys

import numpy as np
import pytest

from omni_sim_core.clock import ClockConfig
from omni_sim_core.plant.motor import MotorParams
from omni_sim_core.plant.nominal import NominalModel, nominal_from_true
from omni_sim_core.plant.jacobian import JacobianLayer, JacobianParams
from omni_sim_core.plant.body import BodyParams
from omni_sim_core.control.pid import PIDParams
from omni_sim_core.control.dob import DOBParams
from omni_sim_core.disturbance import DisturbanceSpec
from omni_sim_core.simulator import MotorControlSim, RobotSim, RobotConfig
from omni_sim_core.sensors.encoder import Encoder, EncoderParams
from omni_sim_core.env.occupancy_grid import generate_rect_field
from omni_sim_core.env.raycast import raycast


def _make_sim(true_params, nominal, clock_cfg, *, dob_enabled, dob_tau_q=0.005,
              pid=PIDParams(kp=0.02, ki=0.0), disturbances=None, duration=2.0,
              setpoint=50.0, seed=0):
    return MotorControlSim(
        true_params=copy.deepcopy(true_params),
        nominal=nominal,
        pid_params=pid,
        dob_params=DOBParams(enabled=dob_enabled, tau_q=dob_tau_q, order=1),
        clock_cfg=clock_cfg,
        seed=seed,
        disturbances=disturbances or [],
        duration_s=duration,
        setpoint_fn=lambda t: setpoint,
    )


# 1 --------------------------------------------------------------------------
def test_dob_is_noop_when_nominal_equals_true(clean_true_params, clock_cfg):
    nominal = nominal_from_true(clean_true_params, 1.0, 1.0)
    off = _make_sim(clean_true_params, nominal, clock_cfg, dob_enabled=False).run()
    on = _make_sim(clean_true_params, nominal, clock_cfg, dob_enabled=True).run()
    rel = np.linalg.norm(on["omega_true"] - off["omega_true"]) / (
        np.linalg.norm(off["omega_true"]) + 1e-12)
    assert rel < 0.01, f"DOB changed matched-model response, rel={rel:.4e}"


# 2 --------------------------------------------------------------------------
def test_dob_rejects_step_load(clean_true_params, clock_cfg):
    nominal = nominal_from_true(clean_true_params, j_ratio=0.5, b_ratio=1.0)
    load = [DisturbanceSpec(target="motor_torque", index=0, waveform="step",
                            amplitude=0.2, start_s=0.5)]
    off = _make_sim(clean_true_params, nominal, clock_cfg, dob_enabled=False,
                    disturbances=load, duration=1.5).run()
    on = _make_sim(clean_true_params, nominal, clock_cfg, dob_enabled=True,
                   disturbances=load, duration=1.5).run()

    def load_induced_drop(log):
        t = log["t"]
        pre = np.mean(log["omega_true"][(t > 0.45) & (t < 0.5)])
        post = np.mean(log["omega_true"][t > 1.45])
        return abs(post - pre)

    drop_off = load_induced_drop(off)
    drop_on = load_induced_drop(on)
    assert drop_on <= 0.1 * drop_off, (
        f"DOB steady deviation not <=1/10: on={drop_on:.4e} off={drop_off:.4e}")


# 3 --------------------------------------------------------------------------
def test_dhat_converges_to_disturbance(clean_true_params, clock_cfg):
    nominal = nominal_from_true(clean_true_params, 1.0, 1.0)
    amp = 0.2
    tau_q = 0.005
    dist = [DisturbanceSpec(target="motor_torque", index=0, waveform="step",
                            amplitude=amp, start_s=0.5)]
    log = _make_sim(clean_true_params, nominal, clock_cfg, dob_enabled=True,
                    dob_tau_q=tau_q, disturbances=dist, duration=1.0).run()
    t = log["t"]
    # evaluate a comfortable margin after 5*tau_q past the step
    mask = t > 0.5 + 10 * tau_q
    err = np.abs(log["d_hat"][mask] - amp)
    assert np.max(err) < 0.05 * amp, f"d_hat did not converge: max err={np.max(err):.4e}"


# 4 --------------------------------------------------------------------------
def test_dt_sim_invariance_at_fixed_control_rate(clean_true_params):
    # Spec section 5 reading of the original "dt_sim invariance": with the
    # CONTROL RATE held fixed (control_rate_hz -> dt_motor), halving the
    # integration step dt_sim must not change the result. Both runs use a
    # 1 kHz control loop; only dt_sim differs.
    nominal = nominal_from_true(clean_true_params, 0.5, 1.0)
    dist = [DisturbanceSpec(target="motor_torque", index=0, waveform="step",
                            amplitude=0.2, start_s=0.3)]
    coarse_cfg = ClockConfig(dt_sim=1.0e-4, dt_nav=2.0e-2, control_rate_hz=1000.0)
    fine_cfg = ClockConfig(dt_sim=0.5e-4, dt_nav=2.0e-2, control_rate_hz=1000.0)
    coarse = _make_sim(clean_true_params, nominal, coarse_cfg, dob_enabled=True,
                       disturbances=dist, duration=1.0).run()
    fine = _make_sim(clean_true_params, nominal, fine_cfg, dob_enabled=True,
                     disturbances=dist, duration=1.0).run()
    # compare on the shared coarse time grid
    fi = np.searchsorted(fine["t"], coarse["t"])
    fi = np.clip(fi, 0, len(fine["t"]) - 1)
    a, b = coarse["omega_true"], fine["omega_true"][fi]
    rel = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)
    assert rel < 1e-3, f"result changed when halving dt_sim, rel={rel:.4e}"


# 5 --------------------------------------------------------------------------
def test_same_seed_is_bit_identical(clean_true_params, clock_cfg):
    nominal = nominal_from_true(clean_true_params, 1.0, 1.0)
    dist = [DisturbanceSpec(target="motor_torque", index=0, waveform="white",
                            amplitude=0.0, sigma=0.05, start_s=0.0)]
    a = _make_sim(clean_true_params, nominal, clock_cfg, dob_enabled=True,
                  disturbances=dist, duration=0.5, seed=123).run()
    b = _make_sim(clean_true_params, nominal, clock_cfg, dob_enabled=True,
                  disturbances=dist, duration=0.5, seed=123).run()
    for k in a:
        assert np.array_equal(a[k], b[k]), f"signal {k} not bit-identical"


# 6 --------------------------------------------------------------------------
def test_odometry_drifts_smoothly():
    cfg = RobotConfig(clock=ClockConfig(dt_sim=1e-3, dt_motor=1e-3, dt_nav=2e-2))
    sim = RobotSim(cfg, seed=0)
    # coarse encoder so quantization produces a visible, but smooth, drift
    enc = Encoder(EncoderParams(counts_per_rev=360), np.random.default_rng(0))

    torque = np.array([0.05, 0.05, -0.05, -0.05])  # smooth translating motion
    errors = []
    n = int(2.0 / cfg.clock.dt_sim)
    for _ in range(n):
        sim.step(torque)
        reading = enc._measure(sim.clock.t,
                               {"wheel_angle_rad": sim.wheel_angle_true})
        sim.update_odometry(wheel_angle=reading.angle_rad)
        if sim.clock.nav_fires():
            errors.append(np.linalg.norm(sim.body.pose[:2] - sim.odom_pose[:2]))
    errors = np.array(errors)
    jumps = np.abs(np.diff(errors))

    # Odometry must track the truth closely (quantization-only drift) ...
    assert errors[-1] < 0.05, f"odometry drift too large: {errors[-1]:.4e}"
    # ... and accumulate smoothly: no per-sample teleport. A jump far larger
    # than the typical step signals a discontinuity.
    typical = np.median(jumps) + 1e-6
    assert np.max(jumps) < 20 * typical, (
        f"odometry error jumped (teleport): max={np.max(jumps):.4e}, "
        f"median={np.median(jumps):.4e}")


# 7 --------------------------------------------------------------------------
def test_jacobian_roundtrip_is_identity():
    jac = JacobianLayer(JacobianParams())
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.standard_normal(3)
        w = jac.wheel_velocity_from_body(v)
        v2 = jac.body_velocity_from_wheel(w)
        assert np.allclose(v, v2, atol=1e-9), f"roundtrip failed: {v} -> {v2}"


# 8 --------------------------------------------------------------------------
def test_core_imports_without_rclpy():
    import omni_sim_core
    import omni_sim_core.simulator  # noqa: F401
    import omni_sim_core.control.dob  # noqa: F401
    import omni_sim_core.env.raycast  # noqa: F401
    import omni_sim_core.sensors.lidar  # noqa: F401
    assert "rclpy" not in sys.modules, "core pulled in rclpy"


# 9 --------------------------------------------------------------------------
def test_lidar_raycast_matches_rectangular_map():
    res = 0.05
    grid = generate_rect_field(4.0, 4.0, res, wall_thickness_cells=1)
    ox = oy = 2.02  # comfortably inside a free cell near the centre

    # analytic distance to each wall face (inner edge of the border cell)
    right = (4.0 - res) - ox
    left = ox - res
    top = (4.0 - res) - oy
    bottom = oy - res
    cases = {0.0: right, np.pi: left, np.pi / 2: top, -np.pi / 2: bottom}

    for ang, expected in cases.items():
        r = raycast(grid, ox, oy, np.array([ang]), max_range=12.0)[0]
        assert abs(r - expected) < res, (
            f"angle={ang:.2f}: got {r:.4f}, expected {expected:.4f}")

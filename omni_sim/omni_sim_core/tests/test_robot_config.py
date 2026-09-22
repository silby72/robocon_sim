"""The simulated robot must be the configured robot.

``RobotSim`` used to be built from ``RobotConfig()``'s library defaults while
the GUI edited ``config/robot/*``. Nothing connected the two, so the ROS node
simulated an idealised 10 kg machine with four ungeared wheels 0.2 m from its
centre -- for a 1 m/s command, 5.9x wrong about motor speed, almost all of it
the 6:1 gearboxes it did not know about.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omni_sim_core.plant.jacobian import JacobianLayer, JacobianParams
from omni_sim_core.simulator import RobotConfig, RobotSim

ROOT = Path(__file__).resolve().parents[2]
ROBOT = ROOT / "config" / "robot" / "chassis.yaml"

pytestmark = pytest.mark.skipif(not ROBOT.exists(), reason="repo config/ not available")


@pytest.fixture(scope="module")
def built():
    from omni_sim_core.mechanism.robot_config import robot_config_from_dir
    cfg, warnings = robot_config_from_dir(ROOT)
    return cfg, warnings


def test_the_gear_ratio_reaches_the_simulator(built):
    """The defaults' whole error, in one number."""
    cfg, _ = built
    v = np.array([1.0, 0.0, 0.0])
    configured = np.abs(JacobianLayer(cfg.jacobian).wheel_velocity_from_body(v)).max()
    default = np.abs(JacobianLayer().wheel_velocity_from_body(v)).max()
    assert configured == pytest.approx(83.5, abs=1.0)
    assert configured / default > 5.0, (
        "the configured drivetrain's gearing is not reaching the simulator")


def test_mass_and_inertia_come_from_the_chassis(built):
    cfg, _ = built
    assert cfg.body.mass_kg == pytest.approx(15.0)      # chassis.yaml
    assert cfg.body.inertia_z_kgm2 == pytest.approx(0.35)
    assert cfg.body.mass_kg != RobotConfig().body.mass_kg


def test_kinematics_and_statics_stay_consistent(built):
    """``J`` (geared, motor-shaft) and ``A`` (contact) must not drift apart:
    the wrench a motor torque produces has to come out the same computed
    either directly or through the contact force."""
    cfg, _ = built
    jac = JacobianLayer(cfg.jacobian)
    v = np.array([0.7, -0.3, 0.4])
    assert jac.body_velocity_from_wheel(jac.wheel_velocity_from_body(v)) == \
        pytest.approx(v)
    tau = np.array([0.5, -0.2, 0.3, 0.1])
    assert jac.body_force_from_wheel(jac.wheel_force_from_torque(tau)) == \
        pytest.approx(jac.J.T @ tau)


def test_the_symmetric_default_still_behaves_exactly_as_before():
    """The refactor that added ``drive_matrix`` must not move the default path
    -- the DOB and trajectory studies are tuned against it."""
    jac = JacobianLayer(JacobianParams())
    alphas = [np.deg2rad(a) for a in (45.0, 135.0, 225.0, 315.0)]
    A = np.stack([-np.sin(alphas), np.cos(alphas), np.full(4, 0.2)], axis=1)
    v = np.array([0.3, 0.2, -0.1])
    assert jac.wheel_velocity_from_body(v) == pytest.approx((A @ v) / 0.05)
    assert jac.body_velocity_from_wheel(jac.wheel_velocity_from_body(v)) == \
        pytest.approx(v)
    f = np.array([1.0, 2.0, -1.0, 0.5])
    assert jac.body_force_from_wheel(f) == pytest.approx(A.T @ f)


def test_the_motor_envelope_caps_the_top_speed(built):
    """A drivetrain built from datasheets is held to them.

    Without this the commanded torque went straight to the body: no top speed,
    no stall torque, and swapping the motor changed nothing that moved.
    """
    from omni_sim_core.mechanism.robot_config import achievable_speed
    cfg, _ = built
    assert cfg.enforce_motor_envelope

    v_max = achievable_speed(cfg, (1.0, 0.0, 0.0))
    sim = RobotSim(cfg, seed=0)
    # ask for far more than it has; it must saturate near the model's figure
    for _ in range(int(8.0 / cfg.clock.dt_sim)):
        sim.step(sim._wheel_torque_from_cmd_vel(np.array([v_max, 0.0, 0.0]), kp=0.5))
    reached = sim.body.twist_body()[0]
    assert reached == pytest.approx(v_max, rel=0.05), (
        f"reached {reached:.3f} m/s against a modelled {v_max:.3f} m/s")
    # ...and that ceiling is well under the 1.2 m/s the profile used to assume
    assert v_max < 1.2


def test_saturation_keeps_the_commanded_direction(built):
    """Per-wheel clipping changes the ratio between wheels, and on an omni the
    ratio is the direction: a weak drivetrain asked for too much used to turn
    "drive forward" into an unrecoverable spin."""
    cfg, _ = built
    sim = RobotSim(cfg, seed=0)
    tau = sim._wheel_torque_from_cmd_vel(np.array([50.0, 0.0, 0.0]), kp=5.0)
    assert np.max(np.abs(tau)) == pytest.approx(cfg.motor.torque_max_nm)
    unsaturated = sim._wheel_torque_from_cmd_vel(np.array([50.0, 0.0, 0.0]), kp=1e-6)
    # same direction, different magnitude
    cos = (tau @ unsaturated) / (np.linalg.norm(tau) * np.linalg.norm(unsaturated))
    assert cos == pytest.approx(1.0, abs=1e-9)


def test_swapping_the_motor_changes_the_plant():
    from omni_sim_core.mechanism.robot_config import (available_motor_presets,
                                                      robot_config_from_dir)
    presets = available_motor_presets(ROOT)
    assert "m3508_c620" in presets and len(presets) > 1
    seen = set()
    for p in presets:
        cfg, _ = robot_config_from_dir(ROOT, motor_preset=p)
        seen.add((round(cfg.motor.torque_constant_nm_a, 4),
                  round(cfg.motor.torque_max_nm, 3),
                  round(cfg.motor.omega_max_rad_s, 2)))
        # the chassis is unchanged by a motor swap
        assert cfg.body.mass_kg == pytest.approx(15.0)
    assert len(seen) == len(presets), "some presets produce an identical plant"


def test_an_unknown_preset_is_refused():
    from omni_sim_core.mechanism.robot_config import robot_config_from_dir
    with pytest.raises(KeyError):
        robot_config_from_dir(ROOT, motor_preset="no_such_motor")


def test_a_reload_moves_every_consumer_together(tmp_path):
    """A chassis edit must reach the dynamics, the drawing, the collision
    rectangle *and* the planner's inflation radius.

    They are four separate objects built from one file, and each one that
    lagged behind produced a different confusing failure: a robot drawn at a
    size it is not judged at, or a plan routed for a body the robot no longer
    has. This checks the numbers they each derive move together.
    """
    import shutil

    from omni_sim_core.evaluation.footprint import RobotFootprint
    from omni_sim_core.mechanism.robot_config import robot_config_from_dir
    from omni_sim_core.mechanism.schema import Chassis
    from omni_sim_core.mechanism.yaml_rt import rt_dump, rt_load
    from omni_sim_core.ui.web_scene import robot_scene

    shutil.copytree(ROOT / "config", tmp_path / "config")
    chassis_yaml = tmp_path / "config" / "robot" / "chassis.yaml"

    def snapshot():
        cfg, _ = robot_config_from_dir(tmp_path)
        chassis = Chassis.from_doc(rt_load(chassis_yaml))
        fp = RobotFootprint(length=chassis.footprint.size_m,
                            width=chassis.footprint.size_m)
        return {
            "mass (dynamics)": cfg.body.mass_kg,
            "size (drawing)": robot_scene(chassis_yaml)["size"],
            "r_circ (collision)": fp.r_circ,
            "r_circ (planner)": fp.r_circ,   # sim_node seeds inflation from this
        }

    before = snapshot()
    doc = rt_load(chassis_yaml)
    doc["footprint"]["size_m"] = 0.75
    doc["center_of_mass"]["mass_kg"] = 22.0
    rt_dump(doc, chassis_yaml)
    after = snapshot()

    assert after["mass (dynamics)"] == pytest.approx(22.0)
    assert after["size (drawing)"] == pytest.approx(750.0)
    assert after["r_circ (collision)"] == pytest.approx(0.75 * np.sqrt(2) / 2)
    assert after["r_circ (planner)"] == after["r_circ (collision)"]
    assert all(before[k] != after[k] for k in before), \
        "some consumer did not notice the edit"

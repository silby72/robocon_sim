"""Drivetrain selection: what a (gear ratio, wheel radius, mass) triple can do.

Three metrics, each answering a question a kinematic simulator cannot:

``v_max_diag``          the speed the motors can *hold*, not the speed a
                        velocity command asks for. Measured on the diagonal
                        because that is the binding direction on an X-layout
                        omni: with drive axes at +-45 deg, moving diagonally
                        runs two wheels at full speed while the other two idle.
``climb_deg_at_speed``  the slope it can climb *while moving*, which is not the
                        slope it can hold at rest -- available torque falls off
                        with speed, and a drivetrain can pass the standstill
                        test and still be unable to make progress up a ramp.
``track_err_*``         worst position error when mass steps up mid-run (a
                        transport robot picking up cargo), with the existing
                        disturbance observer on and off.

Everything reuses the real plant: ``plant/motor.py``'s saturation, the geared
jacobian from ``mechanism/jacobian.py``, the slope model from ``field/slope.py``
and ``control/dob.py`` as-is. No new dynamics are invented here.

The config is mutated **in memory**: ``load_inputs`` is called once per process
and the chassis dataclass is deep-copied and edited per condition, so a 1000-
condition sweep does no file IO and needs no temporary directories.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..clock import ClockConfig
from ..control.dob import DisturbanceObserver, DOBParams
from ..control.pid import PIDParams
from ..control.trajectory import ProfileLimits, SampledTrajectory, TrajectoryFollower
from ..field.slope import max_climb_angle_rad
from ..plant.nominal import NominalModel
from ..simulator import RobotSim

_INPUTS: dict = {}


def _inputs(repo: str):
    """``BuildInputs`` for ``repo``, loaded once per process."""
    from ..mechanism.build import load_inputs
    key = str(repo)
    if key not in _INPUTS:
        _INPUTS[key] = load_inputs(key)
    return _INPUTS[key]


def build_variant(repo: str, gear_ratio: float, wheel_radius: float,
                  total_mass_kg: float, *, dt_sim: float = 2.0e-3):
    """A ``RobotConfig`` for one point of the parameter space.

    The chassis is edited before deriving, not after, so the motor's reflected
    inertia is re-derived for the new gearing (it scales as 1/n^2 -- applying
    the change afterwards would keep the old robot's number and quietly
    mis-state the acceleration of every geared variant).
    """
    from ..mechanism.robot_config import (body_params_from, jacobian_params_from,
                                          motor_params_from, RobotConfigWarnings)

    inp = copy.deepcopy(_inputs(repo))
    for w in inp.chassis.drive_wheels:
        w.gear_ratio = float(gear_ratio)
        w.radius_m = float(wheel_radius)
    com = inp.chassis.center_of_mass
    # same shape, more mass: scale the inertia with it rather than keeping a
    # number that belonged to a lighter robot
    com.inertia_zz_kgm2 *= float(total_mass_kg) / com.mass_kg
    com.mass_kg = float(total_mass_kg)

    warnings = RobotConfigWarnings()
    from ..simulator import RobotConfig
    return RobotConfig(
        motor=motor_params_from(inp),
        jacobian=jacobian_params_from(inp.chassis),
        body=body_params_from(inp.chassis, warnings),
        clock=ClockConfig(dt_sim=dt_sim, dt_motor=dt_sim, dt_nav=2.0e-2),
        enforce_motor_envelope=True,
    )


def reflected_wheel_dynamics(cfg) -> tuple[float, float]:
    """The ``(J_n, B_n)`` a per-wheel DOB should believe in.

    NOT the motor's rotor inertia. ``RobotSim`` never integrates the motor --
    the wheels are rigidly coupled to the body (no slip), so what resists a
    wheel torque is the *body's* mass seen through the drive jacobian::

        wrench     = J^T tau
        body accel = M^-1 wrench
        wheel accel= J M^-1 J^T tau      ->  J_n,i = 1 / (J M^-1 J^T)_ii

    Using the rotor inertia instead (5.2e-4 here, and pinned by an actuator
    override so it does not even change with the gear ratio) makes the DOB
    gain 1/J_n up to 25x too high at low reduction, where the reflected body
    inertia is 0.013. The observer then fights its own noise and makes
    tracking worse -- which is what the first run of this sweep showed, and
    which would have been read as "the DOB does not help" rather than "it was
    handed the wrong plant".
    """
    from ..plant.jacobian import JacobianLayer

    jac = JacobianLayer(cfg.jacobian)
    J = jac.J
    M_inv = np.diag([1.0 / cfg.body.mass_kg, 1.0 / cfg.body.mass_kg,
                     1.0 / cfg.body.inertia_z_kgm2])
    D_inv = np.diag([1.0 / max(cfg.body.drag_x_ns_m, 1e-9),
                     1.0 / max(cfg.body.drag_y_ns_m, 1e-9),
                     1.0 / max(cfg.body.drag_theta_nms, 1e-9)])
    acc = np.diag(J @ M_inv @ J.T)          # wheel accel per unit wheel torque
    vel = np.diag(J @ D_inv @ J.T)          # wheel speed per unit wheel torque
    return float(1.0 / np.mean(acc)), float(1.0 / np.mean(vel))


def _straight_trajectory(distance: float, v: float, a_max: float):
    n = max(int(distance / 0.01), 2)
    pts = np.stack([np.linspace(0.0, distance, n), np.zeros(n)], axis=1)
    s = np.linspace(0.0, distance, n)
    return SampledTrajectory(pts, s, np.zeros(n),
                             ProfileLimits(v_max=v, a_max=a_max, a_lat_max=4.0))


def load_step_tracking(cfg, *, distance=2.0, mass_steps=(1.6, 2.2),
                       at_fraction=(0.3, 0.6), use_dob=True,
                       kp_wheel=0.5, dob_tau_q=5.0e-3) -> dict:
    """Worst position error while following a straight line, as mass steps up.

    The mass is changed on the *body* mid-run, which is what carrying cargo
    does: the same wheel torque now buys less acceleration. Injecting an
    equivalent force instead would be a different (and easier) problem -- a
    force can be cancelled by a feed-forward term, a mass change cannot.

    The DOB sits per wheel on the speed loop, estimating the extra load torque
    the nominal model does not know about, exactly as ``MotorControlSim`` uses
    it on a single axis. Its nominal inertia is the one derived for the
    *unloaded* robot, so the added mass is precisely the model error it exists
    to reject.
    """
    from ..mechanism.robot_config import achievable_speed

    v_cap = achievable_speed(cfg, (1.0, 0.0, 0.0))
    v_task = min(0.8 * v_cap, 0.6)
    if v_task < 0.02:
        return {"track_err": float("inf"), "v_task": v_task}

    sim = RobotSim(cfg, seed=0)
    dt = cfg.clock.dt_sim
    traj = _straight_trajectory(distance, v_task, a_max=2.0)
    follower = TrajectoryFollower(traj, PIDParams(kp=2.0, ki=0.0, kd=0.0),
                                  dt=cfg.clock.dt_nav, v_max=v_task)

    Jn, Bn = reflected_wheel_dynamics(cfg)
    nominal = NominalModel(inertia_kgm2=Jn, damping_nms=Bn)
    dobs = [DisturbanceObserver(DOBParams(enabled=use_dob, tau_q=dob_tau_q),
                                nominal, dt)
            for _ in range(sim.jac.n_wheels)]
    d_hat = np.zeros(sim.jac.n_wheels)

    base_mass = cfg.body.mass_kg
    duration = traj.duration
    fires = [f * duration for f in at_fraction]
    worst = 0.0
    cmd = np.zeros(3)
    t = 0.0
    n_steps = int((duration + 1.0) / dt)
    for i in range(n_steps):
        for k, when in enumerate(fires):
            if t >= when and sim.body.p.mass_kg < base_mass * mass_steps[k]:
                sim.body.p.mass_kg = base_mass * mass_steps[k]
        if sim.clock.nav_fires():
            cmd = follower.command(min(t, duration), sim.body.pose)
        tau = sim._wheel_torque_from_cmd_vel(cmd, kp=kp_wheel)
        if use_dob:
            tau = tau - d_hat
        sim.step(tau)
        if use_dob:
            omega = sim.jac.wheel_velocity_from_body(sim.body.twist_body())
            d_hat = np.array([d.update(float(omega[j]), float(tau[j]))
                              for j, d in enumerate(dobs)])
        t += dt
        ref = traj.sample(min(t, duration))
        err = float(np.hypot(ref.x - sim.body.pose[0], ref.y - sim.body.pose[1]))
        worst = max(worst, err)
    return {"track_err": worst, "v_task": v_task,
            "d_hat_final": float(np.mean(np.abs(d_hat)))}


def evaluate(params: dict, rng=None, *, repo: str = ".",
             ramp_deg: float = 9.9) -> dict:
    """One condition -> the metric row. Top-level and picklable for the pool."""
    from ..mechanism.robot_config import achievable_speed, climb_limits

    cfg = build_variant(repo, params["gear_ratio"], params["wheel_radius"],
                        params["total_mass_kg"])

    v_diag = achievable_speed(cfg, (1.0, 1.0, 0.0))
    v_x = achievable_speed(cfg, (1.0, 0.0, 0.0))
    lim = climb_limits(cfg, climb_speed=max(0.2 * v_diag, 0.05))

    with_dob = load_step_tracking(cfg, use_dob=True)
    without = load_step_tracking(cfg, use_dob=False)

    return {
        "v_max_diag": v_diag,
        "v_max_x": v_x,
        "hold_deg": lim["hold_deg"],
        "climb_deg_at_speed": lim["climb_deg"],
        "push_climb_n": lim["push_climb_n"],
        "stall_torque_nm": cfg.motor.torque_max_nm,
        "no_load_rad_s": cfg.motor.omega_max_rad_s,
        "reflected_inertia": cfg.motor.inertia_kgm2,
        "track_err_dob": with_dob["track_err"],
        "track_err_nodob": without["track_err"],
        "d_hat_final": with_dob["d_hat_final"],
        "v_task": with_dob["v_task"],
        "climbs_ramp": bool(lim["climb_deg"] >= ramp_deg),
    }


def meets(row: dict, req: dict) -> bool:
    """Does this condition satisfy the requirement set?"""
    return (row.get("_status") == "ok"
            and row["v_max_diag"] >= req["v_max_diag_min"]
            and row["climb_deg_at_speed"] >= req["ramp_deg"]
            and row["track_err_dob"] <= req["track_err_max"])

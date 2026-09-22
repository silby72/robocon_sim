"""Facade that assembles the layers and runs a scenario, headless.

Two entry points:

- ``MotorControlSim`` : single-axis speed control with PID (+optional DOB),
  the workhorse for the control-theory (Phase 0) experiments and acceptance
  tests 1-5. True plant vs. nominal model are kept strictly separate.

- ``RobotSim`` : full body + jacobian + per-wheel motors driving a 3-DOF body,
  used for trajectory following and odometry (Phase 1/2).

Both are deterministic given a seed: every random draw goes through an
explicitly created ``numpy.random.Generator``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .clock import MultiRateClock, ClockConfig
from .integrator import rk4_step
from .plant.motor import MotorArray, MotorParams
from .plant.nominal import NominalModel
from .plant.jacobian import JacobianLayer, JacobianParams
from .plant.odometry import (OdometryParams, apply_slip,
                             odometry_jacobian)
from .plant.body import RigidBody, BodyParams
from .control.pid import PID, PIDParams
from .control.dob import DisturbanceObserver, DOBParams
from .disturbance import DisturbanceManager, DisturbanceSpec


# ---------------------------------------------------------------------------
# Phase 0: single-axis speed control (PID + DOB)
# ---------------------------------------------------------------------------
class MotorControlSim:
    def __init__(self, true_params: MotorParams, nominal: NominalModel,
                 pid_params: PIDParams, dob_params: DOBParams,
                 clock_cfg: ClockConfig, seed: int = 0,
                 disturbances: list[DisturbanceSpec] | None = None,
                 duration_s: float = 10.0,
                 setpoint_fn: Callable[[float], float] | None = None) -> None:
        # force single motor for this experiment
        true_params.n_motors = 1
        self.motor = MotorArray(true_params)
        self.nominal = nominal
        self.clock = MultiRateClock(clock_cfg)
        self.duration_s = duration_s
        self.rng = np.random.default_rng(seed)
        self.pid = PID(pid_params, clock_cfg.dt_motor)
        self.dob = DisturbanceObserver(dob_params, nominal, clock_cfg.dt_motor)
        self.disturbance = DisturbanceManager(
            disturbances or [], duration_s, self.rng, n_motors=1)
        self.setpoint_fn = setpoint_fn or (lambda t: 50.0)
        self._nominal = nominal

    def run(self, observer=None) -> dict[str, np.ndarray]:
        """Run the scenario. If ``observer`` is given it is called each motor
        tick with ``(progress_fraction, state_dict)`` for live visualisation.
        The state dict holds the latest scalar signals (t, omega_*, tau_*, d_*).
        """
        clock = self.clock
        clock.reset()
        self.pid.reset()
        self.dob.reset()
        self.motor.reset()

        n_steps = clock.num_steps(self.duration_s)
        dt_sim = clock.config.dt_sim

        tau_cmd = 0.0          # held command (ZOH) at motor loop rate
        d_hat = 0.0
        log = {k: np.empty(n_steps + 1) for k in
               ("t", "omega_true", "omega_ref", "tau_cmd", "tau_pid",
                "d_hat", "d_true")}

        def record(i: int, omega_ref: float, d_true: float) -> None:
            log["t"][i] = clock.t
            log["omega_true"][i] = self.motor.omega()[0]
            log["omega_ref"][i] = omega_ref
            log["tau_cmd"][i] = tau_cmd
            log["tau_pid"][i] = self._last_tau_pid
            log["d_hat"][i] = d_hat
            log["d_true"][i] = d_true

        self._last_tau_pid = 0.0
        omega_ref = self.setpoint_fn(0.0)
        d_true = float(self.disturbance.vector("motor_torque", 0.0, dt_sim)[0])
        record(0, omega_ref, d_true)

        for i in range(1, n_steps + 1):
            t = clock.t

            if clock.motor_fires():
                omega_ref = self.setpoint_fn(t)
                omega_meas = self.motor.omega()[0]
                tau_pid = self.pid.update(omega_ref, omega_meas)
                self._last_tau_pid = tau_pid
                # DOB compensation uses the previous estimate; then re-estimate.
                tau_cmd = tau_pid - d_hat
                d_hat = self.dob.update(omega_meas, tau_cmd)
                if observer is not None:
                    observer(i / n_steps, {
                        "t": t, "omega_true": omega_meas, "omega_ref": omega_ref,
                        "tau_pid": tau_pid, "tau_cmd": tau_cmd,
                        "d_hat": d_hat, "d_true": d_true,
                        "dob_enabled": self.dob.p.enabled, "tau_q": self.dob.p.tau_q,
                    })

            d_true = float(self.disturbance.vector("motor_torque", t, dt_sim)[0])
            tau_ext = np.array([d_true])

            def deriv(tt, y, _cmd=np.array([tau_cmd]), _ext=tau_ext):
                return self.motor.deriv(tt, y, _cmd, _ext)

            self.motor.state = rk4_step(deriv, t, self.motor.state, dt_sim)
            clock.advance()
            record(i, omega_ref, d_true)

        return log


# ---------------------------------------------------------------------------
# Phase 1/2: full robot (body + jacobian + wheel motors)
# ---------------------------------------------------------------------------
@dataclass
class RobotConfig:
    motor: MotorParams = field(default_factory=lambda: MotorParams(n_motors=4))
    jacobian: JacobianParams = field(default_factory=JacobianParams)
    body: BodyParams = field(default_factory=BodyParams)
    clock: ClockConfig = field(default_factory=ClockConfig)
    # What the estimator believes about the wheels, as distinct from what is
    # true. Defaults are all 1.0 -- perfect odometry, the old behaviour.
    odometry: OdometryParams = field(default_factory=OdometryParams)
    # Clamp wheel torque to what the motor can actually produce at its current
    # speed. Off by default so the existing control studies (whose gains were
    # tuned against an unlimited actuator) keep their exact behaviour; turned
    # on by ``mechanism.robot_config``, because a robot built from real
    # datasheets should be held to them. See RobotSim._motor_limited.
    enforce_motor_envelope: bool = False


class RobotSim:
    """4-wheel omni robot. Accepts either a body velocity command (mapped to
    wheel torques through a simple per-wheel speed P-loop) or direct wheel
    torque commands. Tracks true state and a wheel-odometry estimate."""

    def __init__(self, cfg: RobotConfig, seed: int = 0) -> None:
        cfg.motor.n_motors = 4
        self.cfg = cfg
        self.motor = MotorArray(cfg.motor)
        self.jac = JacobianLayer(cfg.jacobian)
        # A second, deliberately mis-measured jacobian for dead reckoning.
        # Sharing self.jac made odometry exact by construction (0.01 mm of
        # error over 9 m), which quietly handed every estimator the answer.
        self.jac_odom = odometry_jacobian(cfg.jacobian, cfg.odometry)
        self.body = RigidBody(cfg.body)
        self.clock = MultiRateClock(cfg.clock)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, pose: np.ndarray | None = None) -> None:
        self.clock.reset()
        self.motor.reset()
        self.body.reset(pose=pose)
        # true wheel angle obeys the no-slip constraint w.r.t. the body twist
        self._wheel_angle_true = np.zeros(self.jac.n_wheels)
        src = self.cfg.odometry.sources
        # integrated true readings of the standalone units (a dead wheel's
        # angle, an optical sensor's travelled distance, the gyro's heading)
        self._odom_reading_true = np.zeros(src.n_rows if src else 0)
        self._prev_odom_reading = self._odom_reading_true.copy()
        self.odom_pose = self.body.pose
        self._prev_wheel_angle = self._wheel_angle_true.copy()

    @property
    def wheel_angle_true(self) -> np.ndarray:
        return self._wheel_angle_true.copy()

    def _wheel_torque_from_cmd_vel(self, cmd_vel: np.ndarray, kp: float = 0.05,
                                   torque_max: float | None = None) -> np.ndarray:
        """P speed controller mapping a BODY-frame velocity command to wheel
        torque via the no-slip wheel speeds, so cmd_vel exercises the stack.
        Torque is clamped to keep the closed loop well-behaved -- by default to
        the motor's own stall torque rather than a hard-coded 5 Nm, which was
        silently generous for a small motor and tight for a big one."""
        if torque_max is None:
            torque_max = self.cfg.motor.torque_max_nm
        wheel_omega_ref = self.jac.wheel_velocity_from_body(cmd_vel)
        wheel_omega_now = self.jac.wheel_velocity_from_body(self.body.twist_body())
        tau = kp * (wheel_omega_ref - wheel_omega_now)
        # Scale the whole vector down, do not clip wheel by wheel. Clipping
        # changes the *ratio* between wheels, and on an omni the ratio is the
        # direction: ask a weak drivetrain for more than it has and per-wheel
        # clipping turns "drive forward" into a spin it cannot recover from,
        # because a saturated wheel stops reporting how much it wanted. Uniform
        # scaling keeps the commanded wrench pointing where it was asked to.
        peak = float(np.max(np.abs(tau))) if tau.size else 0.0
        if peak > torque_max:
            tau = tau * (torque_max / peak)
        return tau

    def _motor_limited(self, wheel_torque: np.ndarray) -> np.ndarray:
        """Clamp commanded torque to the motor's torque-speed envelope.

        The wheels are rigidly coupled to the body (no slip), so the motor's
        speed is *determined* by the body twist rather than free: the motor
        cannot be integrated as an independent state, but its capability at
        that speed still applies. For the ideal back-emf motor the rest of the
        mechanism layer assumes, that capability is the straight line from
        stall torque at zero speed to zero torque at no-load speed::

            tau_avail(w) = tau_stall * max(0, 1 - |w| / w_noload)

        applied only when the torque is *driving* (same sign as the speed).
        Braking is left at the full stall torque: shorting a spinning motor
        can brake harder than it can accelerate, and capping it at the driving
        limit would invent a robot that cannot stop.

        Without this, ``RobotSim`` applied whatever torque it was handed
        straight to the body -- the ``MotorArray`` it constructs is never
        integrated here -- so the drivetrain had no top speed and no stall
        torque, and swapping the motor changed nothing that moved.
        """
        p = self.cfg.motor
        omega = self.jac.wheel_velocity_from_body(self.body.twist_body())
        avail = p.torque_max_nm * np.clip(
            1.0 - np.abs(omega) / max(p.omega_max_rad_s, 1e-9), 0.0, 1.0)
        driving = np.sign(wheel_torque) == np.sign(omega)
        limit = np.where(driving & (omega != 0.0), avail, p.torque_max_nm)
        return np.clip(wheel_torque, -limit, limit)

    def step(self, wheel_torque: np.ndarray,
             ext_wrench: np.ndarray | None = None) -> None:
        """Advance the true robot one ``dt_sim``.

        Wheel torques become contact forces (f = tau / r), the jacobian maps
        them to a body wrench, and the 3-DOF body integrates. The wheels are
        rigidly coupled to the body (no slip -- slip is out of scope, section
        1.3/13), so the true wheel angle is integrated from the body twist.

        ``ext_wrench`` is any body-frame force the wheels did not produce:
        gravity on a ramp, a shove from ``/sim/disturbance``. There was no way
        to pass one before, so both of those existed only as numbers nobody
        applied -- a ramp cost nothing to climb and a disturbance disturbed
        nothing.
        """
        dt_sim = self.clock.config.dt_sim
        t = self.clock.t

        # wheel forces -> body wrench (static jacobian) -> body dynamics
        if self.cfg.enforce_motor_envelope:
            wheel_torque = self._motor_limited(np.asarray(wheel_torque, dtype=float))
        wheel_force = self.jac.wheel_force_from_torque(wheel_torque)
        wrench_body = self.jac.body_force_from_wheel(wheel_force)
        if ext_wrench is not None:
            wrench_body = wrench_body + np.asarray(ext_wrench, dtype=float)

        def b_deriv(tt, y, _w=wrench_body):
            return self.body.deriv(tt, y, _w)
        # trapezoidal wheel-angle update using pre/post body twist (no slip)
        twist_pre = self.body.twist_body()
        omega_pre = self.jac.wheel_velocity_from_body(twist_pre)
        self.body.state = rk4_step(b_deriv, t, self.body.state, dt_sim)
        omega_post = self.jac.wheel_velocity_from_body(self.body.twist_body())
        self._wheel_angle_true += 0.5 * (omega_pre + omega_post) * dt_sim
        src = self.cfg.odometry.sources
        if src is not None:
            # same trapezoid, same no-slip assumption -- but these units are
            # passive, so they are the ones slip does not corrupt
            self._odom_reading_true += 0.5 * (
                src.readings_from_twist(twist_pre)
                + src.readings_from_twist(self.body.twist_body())) * dt_sim

        self.clock.advance()

    def update_odometry(self, wheel_angle: np.ndarray | None = None,
                        gyro_wz: float | None = None) -> None:
        """Integrate wheel-odometry pose from measured wheel motion.

        Pass a quantised ``wheel_angle`` (e.g. from an ``Encoder``) to make the
        drift grow realistically; otherwise the true wheel angle is used. The
        estimate is always integrated incrementally, so it is smooth with no
        teleport regardless of measurement noise."""
        dt = self.clock.config.dt_sim
        src = self.cfg.odometry.sources
        if src is not None:
            # Standalone units win when they exist: that is the entire reason
            # for bolting them on. They are passive, so apply_slip is NOT
            # applied here -- a wheel with no drive torque has no traction to
            # lose, and modelling it otherwise would erase the one advantage
            # the hardware buys.
            d = self._odom_reading_true - self._prev_odom_reading
            self._prev_odom_reading = self._odom_reading_true.copy()
            rates = d / dt
            if src.uses_gyro and gyro_wz is not None:
                # Substitute a *measured* yaw rate for the true one. Without
                # this the gyro row is perfect, and since it is the only thing
                # observing wz the pod comes out with exactly zero heading
                # error -- which flatters the hardware. Real gyro bias is the
                # dominant long-run heading error of a build like this.
                rates[-1] = float(gyro_wz)
            body_vel = src.twist_from_readings(rates)
        else:
            if wheel_angle is None:
                wheel_angle = self._wheel_angle_true
            dtheta = wheel_angle - self._prev_wheel_angle
            self._prev_wheel_angle = wheel_angle.copy()
            dtheta = apply_slip(dtheta, self.cfg.odometry, self.rng)
            # jac_odom, not jac: the estimate is built from the robot's *belief*
            body_vel = self.jac_odom.body_velocity_from_wheel(dtheta / dt)
        th = self.odom_pose[2]
        c, s = np.cos(th), np.sin(th)
        # rotate body-frame velocity into world and integrate
        self.odom_pose = self.odom_pose + dt * np.array([
            c * body_vel[0] - s * body_vel[1],
            s * body_vel[0] + c * body_vel[1],
            body_vel[2],
        ])

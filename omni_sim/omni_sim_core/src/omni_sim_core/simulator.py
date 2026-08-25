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


class RobotSim:
    """4-wheel omni robot. Accepts either a body velocity command (mapped to
    wheel torques through a simple per-wheel speed P-loop) or direct wheel
    torque commands. Tracks true state and a wheel-odometry estimate."""

    def __init__(self, cfg: RobotConfig, seed: int = 0) -> None:
        cfg.motor.n_motors = 4
        self.cfg = cfg
        self.motor = MotorArray(cfg.motor)
        self.jac = JacobianLayer(cfg.jacobian)
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
        self.odom_pose = self.body.pose
        self._prev_wheel_angle = self._wheel_angle_true.copy()

    @property
    def wheel_angle_true(self) -> np.ndarray:
        return self._wheel_angle_true.copy()

    def _wheel_torque_from_cmd_vel(self, cmd_vel: np.ndarray, kp: float = 0.05,
                                   torque_max: float = 5.0) -> np.ndarray:
        """P speed controller mapping a BODY-frame velocity command to wheel
        torque via the no-slip wheel speeds, so cmd_vel exercises the stack.
        Torque is clamped to keep the closed loop well-behaved."""
        wheel_omega_ref = self.jac.wheel_velocity_from_body(cmd_vel)
        wheel_omega_now = self.jac.wheel_velocity_from_body(self.body.twist_body())
        tau = kp * (wheel_omega_ref - wheel_omega_now)
        return np.clip(tau, -torque_max, torque_max)

    def step(self, wheel_torque: np.ndarray) -> None:
        """Advance the true robot one ``dt_sim``.

        Wheel torques become contact forces (f = tau / r), the jacobian maps
        them to a body wrench, and the 3-DOF body integrates. The wheels are
        rigidly coupled to the body (no slip -- slip is out of scope, section
        1.3/13), so the true wheel angle is integrated from the body twist.
        """
        dt_sim = self.clock.config.dt_sim
        t = self.clock.t

        # wheel forces -> body wrench (static jacobian) -> body dynamics
        wheel_force = self.jac.wheel_force_from_torque(wheel_torque)
        wrench_body = self.jac.body_force_from_wheel(wheel_force)

        def b_deriv(tt, y, _w=wrench_body):
            return self.body.deriv(tt, y, _w)
        # trapezoidal wheel-angle update using pre/post body twist (no slip)
        omega_pre = self.jac.wheel_velocity_from_body(self.body.twist_body())
        self.body.state = rk4_step(b_deriv, t, self.body.state, dt_sim)
        omega_post = self.jac.wheel_velocity_from_body(self.body.twist_body())
        self._wheel_angle_true += 0.5 * (omega_pre + omega_post) * dt_sim

        self.clock.advance()

    def update_odometry(self, wheel_angle: np.ndarray | None = None) -> None:
        """Integrate wheel-odometry pose from measured wheel motion.

        Pass a quantised ``wheel_angle`` (e.g. from an ``Encoder``) to make the
        drift grow realistically; otherwise the true wheel angle is used. The
        estimate is always integrated incrementally, so it is smooth with no
        teleport regardless of measurement noise."""
        if wheel_angle is None:
            wheel_angle = self._wheel_angle_true
        dtheta = wheel_angle - self._prev_wheel_angle
        self._prev_wheel_angle = wheel_angle.copy()
        dt = self.clock.config.dt_sim
        wheel_omega = dtheta / dt
        body_vel = self.jac.body_velocity_from_wheel(wheel_omega)  # vx,vy,wz (body)
        th = self.odom_pose[2]
        c, s = np.cos(th), np.sin(th)
        # rotate body-frame velocity into world and integrate
        self.odom_pose = self.odom_pose + dt * np.array([
            c * body_vel[0] - s * body_vel[1],
            s * body_vel[0] + c * body_vel[1],
            body_vel[2],
        ])

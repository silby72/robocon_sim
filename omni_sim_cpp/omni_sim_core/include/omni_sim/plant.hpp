#pragma once

// True-plant dynamics for the 4-wheel omni robot.
//
// Model (Step 3):
//   * Each wheel sits at mount_radius * (cos mount_angle, sin mount_angle) in
//     the body frame and can push along (cos drive_angle, sin drive_angle); its
//     rollers free-roll perpendicular (no lateral force, no slip).
//   * Wheels are rigidly coupled to the body: the rolling speed of wheel i is
//     the drive-direction component of that wheel's contact-point velocity.
//   * Traction force per wheel is f_i = (tau_i - b * omega_wheel_i) / r, where
//     b is the viscous friction and omega_wheel_i the rolling speed / r. The
//     four forces sum to a body wrench (Fx, Fy, Mz); dividing by mass and yaw
//     inertia gives the body-frame acceleration, which is rotated into the
//     world frame.
//   * The 6-state (x, y, theta, vx, vy, omega) is advanced with fixed-step RK4,
//     holding the wheel torques constant across the step (zero-order hold).
//
// The state's linear velocity is stored in the WORLD frame.

#include "omni_sim/config.hpp"

#include <array>

namespace omni_sim {

// World-frame rigid-body state, 3 DOF (planar).
struct State {
  // Pose in the world frame.
  double x = 0.0;      // [m]
  double y = 0.0;      // [m]
  double theta = 0.0;  // [rad], yaw

  // Body twist expressed in the world frame.
  double vx = 0.0;     // [m/s]
  double vy = 0.0;     // [m/s]
  double omega = 0.0;  // [rad/s], yaw rate
};

// Wheel torques applied at the four drive motors, in wheel order 0..3 [N*m].
using WheelTorques = std::array<double, kWheelCount>;

class Plant {
 public:
  // A Plant is always constructed from a fully-populated configuration; there
  // is no default plant, because "which parameters" is the whole point.
  explicit Plant(const PlantConfig& config);

  // The parameter set this plant integrates with.
  const PlantConfig& config() const { return config_; }

  // Current world-frame state.
  const State& state() const { return state_; }

  // Overwrite the state, e.g. to reset an experiment to a known pose.
  void set_state(const State& state) { state_ = state; }

  // Advance the simulation by dt seconds under the given wheel torques, using
  // fixed-step RK4 with a zero-order hold on the torques.
  void step(const WheelTorques& wheel_torques, double dt);

  // World-frame acceleration {ax, ay, angular_acceleration} the given wheel
  // torques produce at the current state. Exposed so tests can check the
  // kinematics/dynamics in isolation from the integrator.
  std::array<double, 3> body_acceleration(const WheelTorques& wheel_torques) const;

 private:
  // World-frame acceleration {ax, ay, alpha} for an arbitrary state, so RK4 can
  // evaluate the dynamics at its intermediate stages (not just at state_).
  std::array<double, 3> world_acceleration(const State& state,
                                           const WheelTorques& wheel_torques) const;

  // Full time derivative of the 6-state (dx = vx, ..., dvx = ax, ...).
  State derivative(const State& state, const WheelTorques& wheel_torques) const;

  PlantConfig config_;
  State state_ = {};
};

}  // namespace omni_sim

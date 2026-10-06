#include "omni_sim/plant.hpp"

#include <cmath>
#include <cstddef>

namespace omni_sim {
namespace {

// --- small State algebra used only by the RK4 integrator -------------------
// Kept as explicit named functions (no operator overloading, no one-liners) so
// each integrator stage reads as a plain arithmetic step.

State scale(const State& s, double factor) {
  State result;
  result.x = s.x * factor;
  result.y = s.y * factor;
  result.theta = s.theta * factor;
  result.vx = s.vx * factor;
  result.vy = s.vy * factor;
  result.omega = s.omega * factor;
  return result;
}

State add(const State& a, const State& b) {
  State result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.theta = a.theta + b.theta;
  result.vx = a.vx + b.vx;
  result.vy = a.vy + b.vy;
  result.omega = a.omega + b.omega;
  return result;
}

// state + factor * delta
State add_scaled(const State& state, const State& delta, double factor) {
  return add(state, scale(delta, factor));
}

}  // namespace

Plant::Plant(const PlantConfig& config) : config_(config) {}

std::array<double, 3> Plant::world_acceleration(
    const State& state, const WheelTorques& wheel_torques) const {
  const ChassisParams& chassis = config_.chassis;
  const double radius = chassis.wheel_radius;
  const double viscous = config_.motor.viscous_friction;

  // Rotate the world-frame linear velocity into the body frame: v_body =
  // R(-theta) * v_world.
  const double cos_theta = std::cos(state.theta);
  const double sin_theta = std::sin(state.theta);
  const double body_vx = cos_theta * state.vx + sin_theta * state.vy;
  const double body_vy = -sin_theta * state.vx + cos_theta * state.vy;
  const double yaw_rate = state.omega;

  // Accumulate the body-frame wrench (force x/y and yaw moment) from the four
  // wheel traction forces.
  double force_x = 0.0;
  double force_y = 0.0;
  double moment_z = 0.0;

  for (std::size_t wheel = 0; wheel < kWheelCount; ++wheel) {
    // Wheel mounting position in the body frame.
    const double mount_x = chassis.mount_radius * std::cos(chassis.mount_angle[wheel]);
    const double mount_y = chassis.mount_radius * std::sin(chassis.mount_angle[wheel]);

    // Drive direction (rollers free-roll perpendicular to this).
    const double drive_x = std::cos(chassis.drive_angle[wheel]);
    const double drive_y = std::sin(chassis.drive_angle[wheel]);

    // Velocity of this wheel's contact point in the body frame:
    // v_contact = v_body + omega x r  (planar cross product).
    const double contact_vx = body_vx - yaw_rate * mount_y;
    const double contact_vy = body_vy + yaw_rate * mount_x;

    // No-slip rolling speed along the drive direction, and the wheel's angular
    // velocity that implies.
    const double rolling_speed = contact_vx * drive_x + contact_vy * drive_y;
    const double wheel_angular_velocity = rolling_speed / radius;

    // Net wheel torque after viscous friction, converted to a traction force.
    const double net_torque = wheel_torques[wheel] - viscous * wheel_angular_velocity;
    const double traction = net_torque / radius;

    // Contribute to the body wrench.
    force_x += traction * drive_x;
    force_y += traction * drive_y;
    moment_z += mount_x * (traction * drive_y) - mount_y * (traction * drive_x);
  }

  // Body-frame accelerations from Newton/Euler.
  const double body_ax = force_x / chassis.mass;
  const double body_ay = force_y / chassis.mass;
  const double angular_acceleration = moment_z / chassis.inertia_zz;

  // Rotate the linear acceleration back into the world frame.
  const double world_ax = cos_theta * body_ax - sin_theta * body_ay;
  const double world_ay = sin_theta * body_ax + cos_theta * body_ay;

  return {world_ax, world_ay, angular_acceleration};
}

std::array<double, 3> Plant::body_acceleration(
    const WheelTorques& wheel_torques) const {
  return world_acceleration(state_, wheel_torques);
}

State Plant::derivative(const State& state, const WheelTorques& wheel_torques) const {
  const std::array<double, 3> acceleration = world_acceleration(state, wheel_torques);

  State rate;
  // Pose derivatives are the world-frame velocities.
  rate.x = state.vx;
  rate.y = state.vy;
  rate.theta = state.omega;
  // Velocity derivatives are the accelerations.
  rate.vx = acceleration[0];
  rate.vy = acceleration[1];
  rate.omega = acceleration[2];
  return rate;
}

void Plant::step(const WheelTorques& wheel_torques, double dt) {
  // Classic fixed-step RK4. Torques are held constant across the step.
  const State k1 = derivative(state_, wheel_torques);
  const State k2 = derivative(add_scaled(state_, k1, dt / 2.0), wheel_torques);
  const State k3 = derivative(add_scaled(state_, k2, dt / 2.0), wheel_torques);
  const State k4 = derivative(add_scaled(state_, k3, dt), wheel_torques);

  // state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
  State weighted_sum = k1;
  weighted_sum = add_scaled(weighted_sum, k2, 2.0);
  weighted_sum = add_scaled(weighted_sum, k3, 2.0);
  weighted_sum = add(weighted_sum, k4);

  state_ = add_scaled(state_, weighted_sum, dt / 6.0);
}

}  // namespace omni_sim

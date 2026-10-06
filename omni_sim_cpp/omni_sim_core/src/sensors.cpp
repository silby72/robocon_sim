#include "omni_sim/sensors.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace omni_sim {
namespace {

constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

// Solve the 3x3 linear system M * x = rhs by Cramer's rule. Returns false if M
// is (numerically) singular. Written out explicitly rather than pulling in a
// linear-algebra dependency for one small solve.
bool solve_3x3(const std::array<std::array<double, 3>, 3>& m,
               const std::array<double, 3>& rhs, std::array<double, 3>& out) {
  const double det =
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

  if (std::abs(det) < 1e-12) {
    return false;
  }

  // Replace each column with rhs in turn (Cramer's rule).
  auto column_replaced_det = [&](int column) {
    std::array<std::array<double, 3>, 3> c = m;
    for (int row = 0; row < 3; ++row) {
      c[row][column] = rhs[row];
    }
    return c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) -
           c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0]) +
           c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
  };

  out[0] = column_replaced_det(0) / det;
  out[1] = column_replaced_det(1) / det;
  out[2] = column_replaced_det(2) / det;
  return true;
}

}  // namespace

// --- EncoderOdometry -------------------------------------------------------

EncoderOdometry::EncoderOdometry(const ChassisParams& chassis,
                                 double counts_per_revolution, double gear_ratio)
    : chassis_(chassis),
      counts_per_revolution_(counts_per_revolution),
      gear_ratio_(gear_ratio) {}

void EncoderOdometry::reset_pose(double x, double y, double theta) {
  pose_x_ = x;
  pose_y_ = y;
  pose_theta_ = theta;
}

OdometryReading EncoderOdometry::measure(const State& true_state, double dt) {
  const double radius = chassis_.wheel_radius;

  // True body-frame twist of the plant.
  const double cos_theta = std::cos(true_state.theta);
  const double sin_theta = std::sin(true_state.theta);
  const double body_vx = cos_theta * true_state.vx + sin_theta * true_state.vy;
  const double body_vy = -sin_theta * true_state.vx + cos_theta * true_state.vy;
  const double yaw_rate = true_state.omega;

  // Per-wheel: forward-map the twist to a rolling speed, quantize the wheel
  // displacement to encoder ticks, and recover a quantized rolling speed.
  // Also build the 4x3 kinematics rows for the least-squares reconstruction.
  std::array<double, kWheelCount> measured_speed = {};
  std::array<std::array<double, 3>, kWheelCount> rows = {};

  for (std::size_t wheel = 0; wheel < kWheelCount; ++wheel) {
    const double mount_x = chassis_.mount_radius * std::cos(chassis_.mount_angle[wheel]);
    const double mount_y = chassis_.mount_radius * std::sin(chassis_.mount_angle[wheel]);
    const double drive_x = std::cos(chassis_.drive_angle[wheel]);
    const double drive_y = std::sin(chassis_.drive_angle[wheel]);

    // Contact-point velocity along the drive direction (linear rolling speed).
    const double contact_vx = body_vx - yaw_rate * mount_y;
    const double contact_vy = body_vy + yaw_rate * mount_x;
    const double rolling_speed = contact_vx * drive_x + contact_vy * drive_y;

    // Quantize: wheel angle over dt -> motor-shaft ticks -> back to a speed.
    const double wheel_angle_delta = (rolling_speed / radius) * dt;
    const double motor_angle_delta = wheel_angle_delta * gear_ratio_;
    const double ticks =
        std::round(motor_angle_delta * counts_per_revolution_ / kTwoPi);
    const double quantized_motor_angle = ticks * kTwoPi / counts_per_revolution_;
    const double quantized_wheel_angle = quantized_motor_angle / gear_ratio_;
    // Guard dt == 0.
    const double quantized_rolling_speed =
        (dt > 0.0) ? (quantized_wheel_angle * radius / dt) : 0.0;

    measured_speed[wheel] = quantized_rolling_speed;

    // Row of the forward map s_i = drive_x*vbx + drive_y*vby + moment*omega.
    const double moment = mount_x * drive_y - mount_y * drive_x;
    rows[wheel] = {drive_x, drive_y, moment};
  }

  // Least-squares reconstruction: solve (A^T A) u = A^T s.
  std::array<std::array<double, 3>, 3> normal_matrix = {};
  std::array<double, 3> normal_rhs = {};
  for (std::size_t wheel = 0; wheel < kWheelCount; ++wheel) {
    for (int i = 0; i < 3; ++i) {
      normal_rhs[i] += rows[wheel][i] * measured_speed[wheel];
      for (int j = 0; j < 3; ++j) {
        normal_matrix[i][j] += rows[wheel][i] * rows[wheel][j];
      }
    }
  }

  std::array<double, 3> twist = {0.0, 0.0, 0.0};
  solve_3x3(normal_matrix, normal_rhs, twist);

  OdometryReading reading;
  reading.vx_body = twist[0];
  reading.vy_body = twist[1];
  reading.omega = twist[2];

  // Dead-reckon the pose by integrating the measured twist (Euler). Rotate the
  // measured body velocity into the world frame using the accumulated heading.
  const double cos_pose = std::cos(pose_theta_);
  const double sin_pose = std::sin(pose_theta_);
  pose_x_ += (cos_pose * reading.vx_body - sin_pose * reading.vy_body) * dt;
  pose_y_ += (sin_pose * reading.vx_body + cos_pose * reading.vy_body) * dt;
  pose_theta_ += reading.omega * dt;

  reading.x = pose_x_;
  reading.y = pose_y_;
  reading.theta = pose_theta_;
  return reading;
}

// --- ImuSensor -------------------------------------------------------------

ImuSensor::ImuSensor(const ImuParams& params, std::uint64_t seed)
    : params_(params), rng_(seed) {}

ImuReading ImuSensor::measure(double true_omega, double true_body_ax,
                              double true_body_ay) {
  ImuReading reading;

  reading.angular_velocity_z =
      true_omega + params_.gyro_bias +
      params_.gyro_noise_sigma * unit_normal_(rng_);

  reading.linear_acceleration_x =
      true_body_ax + params_.accel_bias +
      params_.accel_noise_sigma * unit_normal_(rng_);

  reading.linear_acceleration_y =
      true_body_ay + params_.accel_bias +
      params_.accel_noise_sigma * unit_normal_(rng_);

  return reading;
}

}  // namespace omni_sim

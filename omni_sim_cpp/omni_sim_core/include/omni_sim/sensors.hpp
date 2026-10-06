#pragma once

// Simple sensor models for the omni simulator.
//
// Two are implemented here:
//
//   * EncoderOdometry — wheel-encoder odometry. Real encoders report integer
//     tick counts, so the model is a DETERMINISTIC quantization: it forward-maps
//     the true body twist to per-wheel angular displacement, rounds that to
//     encoder ticks, then reconstructs the twist (least squares over 4 wheels)
//     and dead-reckons a pose. Drift comes only from quantization — exactly the
//     no-slip odometry the plant supports.
//
//   * ImuSensor — gyro + accelerometer with a constant bias and additive white
//     (Gaussian) noise. This one is stochastic, so it draws from an EXPLICIT
//     std::mt19937_64 seeded at construction: same seed => same stream, which
//     keeps experiments reproducible.
//
// LiDAR raycast (sensors.hpp in the spec also lists it) is a later addition and
// is intentionally not implemented yet.

#include "omni_sim/config.hpp"
#include "omni_sim/plant.hpp"

#include <cstdint>
#include <random>

namespace omni_sim {

// --- odometry --------------------------------------------------------------

// What the odometry reports: a body-frame twist and a dead-reckoned pose.
struct OdometryReading {
  // Body-frame twist (what the encoders directly observe).
  double vx_body = 0.0;
  double vy_body = 0.0;
  double omega = 0.0;

  // Pose accumulated by integrating the measured twist (drifts over time).
  double x = 0.0;
  double y = 0.0;
  double theta = 0.0;
};

class EncoderOdometry {
 public:
  // counts_per_revolution and gear_ratio set the tick resolution; chassis gives
  // the wheel geometry used for the forward/inverse kinematics.
  EncoderOdometry(const ChassisParams& chassis, double counts_per_revolution,
                  double gear_ratio);

  // Measure over a step of dt seconds. Advances the internal dead-reckoned pose
  // and returns the current reading.
  OdometryReading measure(const State& true_state, double dt);

  // Reset the accumulated pose (e.g. between experiment runs).
  void reset_pose(double x, double y, double theta);

 private:
  ChassisParams chassis_;
  double counts_per_revolution_ = 0.0;
  double gear_ratio_ = 1.0;

  // Dead-reckoned pose carried between calls.
  double pose_x_ = 0.0;
  double pose_y_ = 0.0;
  double pose_theta_ = 0.0;
};

// --- IMU -------------------------------------------------------------------

// Noise/bias configuration for the IMU. Zeroing all fields makes the sensor an
// exact passthrough, which the tests rely on.
struct ImuParams {
  double gyro_bias = 0.0;        // [rad/s]
  double gyro_noise_sigma = 0.0; // [rad/s], white noise std-dev
  double accel_bias = 0.0;       // [m/s^2]
  double accel_noise_sigma = 0.0; // [m/s^2], white noise std-dev
};

// What the IMU reports: yaw rate and body-frame linear acceleration.
struct ImuReading {
  double angular_velocity_z = 0.0;
  double linear_acceleration_x = 0.0;
  double linear_acceleration_y = 0.0;
};

class ImuSensor {
 public:
  ImuSensor(const ImuParams& params, std::uint64_t seed);

  // Corrupt the true yaw rate and body-frame linear acceleration with bias and
  // white noise. The inputs are the true values from the plant.
  ImuReading measure(double true_omega, double true_body_ax, double true_body_ay);

 private:
  ImuParams params_;
  std::mt19937_64 rng_;
  std::normal_distribution<double> unit_normal_{0.0, 1.0};
};

}  // namespace omni_sim

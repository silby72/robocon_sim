#include "omni_sim/sensors.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace omni_sim {
namespace {

constexpr double kDeg = M_PI / 180.0;

ChassisParams make_x_chassis() {
  ChassisParams chassis;
  chassis.mass = 12.0;
  chassis.inertia_zz = 0.45;
  chassis.wheel_radius = 0.05;
  chassis.mount_radius = 0.20;
  chassis.mount_angle = {45.0 * kDeg, 135.0 * kDeg, 225.0 * kDeg, 315.0 * kDeg};
  chassis.drive_angle = {135.0 * kDeg, 225.0 * kDeg, 315.0 * kDeg, 45.0 * kDeg};
  return chassis;
}

// --- encoder odometry ------------------------------------------------------

TEST(EncoderOdometryTest, AtRestReportsZeroAndDoesNotDrift) {
  EncoderOdometry odom(make_x_chassis(), 4096.0, 19.0);

  State at_rest;  // everything zero
  OdometryReading reading = odom.measure(at_rest, 0.01);

  EXPECT_NEAR(reading.vx_body, 0.0, 1e-12);
  EXPECT_NEAR(reading.vy_body, 0.0, 1e-12);
  EXPECT_NEAR(reading.omega, 0.0, 1e-12);
  EXPECT_NEAR(reading.x, 0.0, 1e-12);
  EXPECT_NEAR(reading.y, 0.0, 1e-12);
}

// A fine encoder should recover a steady twist to within a small quantization
// error.
TEST(EncoderOdometryTest, FineEncoderRecoversTwist) {
  EncoderOdometry odom(make_x_chassis(), 8192.0, 19.0);

  State moving;
  moving.vx = 1.0;  // 1 m/s along world +x, theta = 0 so body +x too

  OdometryReading reading = odom.measure(moving, 0.005);

  EXPECT_NEAR(reading.vx_body, 1.0, 2e-2);
  EXPECT_NEAR(reading.vy_body, 0.0, 2e-2);
  EXPECT_NEAR(reading.omega, 0.0, 2e-2);
}

// A coarser encoder must show more quantization error than a fine one for the
// same motion. (Absolute error depends on dt and speed vs. tick size, so the
// meaningful, robust assertion is the coarse-vs-fine comparison.)
TEST(EncoderOdometryTest, CoarserEncoderHasMoreQuantizationError) {
  const double true_speed = 0.5;  // m/s, deliberately off any tick boundary
  const double dt = 0.02;

  State moving;
  moving.vx = true_speed;

  EncoderOdometry coarse(make_x_chassis(), 256.0, 1.0);
  EncoderOdometry fine(make_x_chassis(), 8192.0, 1.0);

  const double coarse_error =
      std::abs(coarse.measure(moving, dt).vx_body - true_speed);
  const double fine_error =
      std::abs(fine.measure(moving, dt).vx_body - true_speed);

  // The coarse encoder actually quantizes (non-zero error) ...
  EXPECT_GT(coarse_error, 1e-3);
  // ... and quantizes more than the fine one.
  EXPECT_GT(coarse_error, fine_error);
}

TEST(EncoderOdometryTest, PoseIsDeterministic) {
  EncoderOdometry a(make_x_chassis(), 4096.0, 19.0);
  EncoderOdometry b(make_x_chassis(), 4096.0, 19.0);

  State moving;
  moving.vx = 0.8;
  moving.vy = 0.3;
  moving.omega = 0.2;

  for (int step = 0; step < 50; ++step) {
    OdometryReading ra = a.measure(moving, 0.01);
    OdometryReading rb = b.measure(moving, 0.01);
    EXPECT_DOUBLE_EQ(ra.x, rb.x);
    EXPECT_DOUBLE_EQ(ra.y, rb.y);
    EXPECT_DOUBLE_EQ(ra.theta, rb.theta);
  }
}

// --- IMU -------------------------------------------------------------------

TEST(ImuSensorTest, ZeroNoiseZeroBiasIsPassthrough) {
  ImuParams params;  // all zero
  ImuSensor imu(params, 1);

  ImuReading reading = imu.measure(0.5, 2.0, -1.0);

  EXPECT_DOUBLE_EQ(reading.angular_velocity_z, 0.5);
  EXPECT_DOUBLE_EQ(reading.linear_acceleration_x, 2.0);
  EXPECT_DOUBLE_EQ(reading.linear_acceleration_y, -1.0);
}

TEST(ImuSensorTest, BiasShiftsTheMean) {
  ImuParams params;
  params.gyro_bias = 0.1;
  params.accel_bias = -0.3;
  params.gyro_noise_sigma = 0.05;
  params.accel_noise_sigma = 0.2;

  ImuSensor imu(params, 42);

  double gyro_sum = 0.0;
  double accel_sum = 0.0;
  const int samples = 20000;
  for (int i = 0; i < samples; ++i) {
    ImuReading reading = imu.measure(0.0, 0.0, 0.0);
    gyro_sum += reading.angular_velocity_z;
    accel_sum += reading.linear_acceleration_x;
  }

  const double gyro_mean = gyro_sum / samples;
  const double accel_mean = accel_sum / samples;

  // Mean should sit near the bias (noise averages out).
  EXPECT_NEAR(gyro_mean, 0.1, 0.01);
  EXPECT_NEAR(accel_mean, -0.3, 0.02);
}

TEST(ImuSensorTest, SameSeedGivesSameStream) {
  ImuParams params;
  params.gyro_noise_sigma = 0.05;
  params.accel_noise_sigma = 0.2;

  ImuSensor a(params, 7);
  ImuSensor b(params, 7);

  for (int i = 0; i < 100; ++i) {
    ImuReading ra = a.measure(0.0, 0.0, 0.0);
    ImuReading rb = b.measure(0.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(ra.angular_velocity_z, rb.angular_velocity_z);
    EXPECT_DOUBLE_EQ(ra.linear_acceleration_x, rb.linear_acceleration_x);
  }
}

TEST(ImuSensorTest, DifferentSeedsDiverge) {
  ImuParams params;
  params.gyro_noise_sigma = 0.05;

  ImuSensor a(params, 1);
  ImuSensor b(params, 2);

  ImuReading ra = a.measure(0.0, 0.0, 0.0);
  ImuReading rb = b.measure(0.0, 0.0, 0.0);
  EXPECT_NE(ra.angular_velocity_z, rb.angular_velocity_z);
}

}  // namespace
}  // namespace omni_sim

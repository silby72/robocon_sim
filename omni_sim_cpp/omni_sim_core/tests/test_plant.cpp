#include "omni_sim/plant.hpp"

#include <gtest/gtest.h>

#include <cmath>

// Step 3 tests: the interface checks from Step 1 plus real physics — the omni
// force map, the world/body frame rotation, viscous friction, and RK4 motion.
// The tests assert physical invariants (pure translation, pure rotation, drag
// sign) rather than hand-transcribed numbers, so they stay meaningful even if
// the geometry convention is tweaked.

namespace omni_sim {
namespace {

constexpr double kDeg = M_PI / 180.0;

// A symmetric X-configuration 4-wheel omni. friction lets a test pick the true
// plant (friction > 0) or the nominal model (friction = 0).
PlantConfig make_x_config(double viscous_friction) {
  PlantConfig config;
  config.chassis.mass = 12.0;
  config.chassis.inertia_zz = 0.45;
  config.chassis.wheel_radius = 0.05;
  config.chassis.mount_radius = 0.20;
  config.chassis.mount_angle = {45.0 * kDeg, 135.0 * kDeg, 225.0 * kDeg, 315.0 * kDeg};
  config.chassis.drive_angle = {135.0 * kDeg, 225.0 * kDeg, 315.0 * kDeg, 45.0 * kDeg};
  config.motor.k_t = 0.042;
  config.motor.resistance = 1.2;
  config.motor.viscous_friction = viscous_friction;
  config.motor.gear_ratio = 19.0;
  return config;
}

// --- interface checks carried over from Step 1 -----------------------------

TEST(PlantTest, StoresConfigAndStartsAtRest) {
  Plant plant(make_x_config(0.001));

  EXPECT_DOUBLE_EQ(plant.config().chassis.mass, 12.0);

  const State& state = plant.state();
  EXPECT_DOUBLE_EQ(state.x, 0.0);
  EXPECT_DOUBLE_EQ(state.vx, 0.0);
  EXPECT_DOUBLE_EQ(state.omega, 0.0);
}

TEST(PlantTest, StateCanBeReset) {
  Plant plant(make_x_config(0.001));

  State pose;
  pose.x = 1.0;
  pose.y = -2.0;
  pose.theta = 0.5;
  plant.set_state(pose);

  EXPECT_DOUBLE_EQ(plant.state().x, 1.0);
  EXPECT_DOUBLE_EQ(plant.state().theta, 0.5);
}

// --- dynamics --------------------------------------------------------------

TEST(PlantTest, ZeroTorqueAtRestGivesZeroAcceleration) {
  Plant plant(make_x_config(0.001));
  WheelTorques none = {0.0, 0.0, 0.0, 0.0};

  std::array<double, 3> accel = plant.body_acceleration(none);

  EXPECT_NEAR(accel[0], 0.0, 1e-12);
  EXPECT_NEAR(accel[1], 0.0, 1e-12);
  EXPECT_NEAR(accel[2], 0.0, 1e-12);
}

// Equal torque on all four wheels is a pure yaw moment (net force cancels).
TEST(PlantTest, EqualTorquesProducePureRotation) {
  Plant plant(make_x_config(0.0));
  WheelTorques equal = {0.5, 0.5, 0.5, 0.5};

  std::array<double, 3> accel = plant.body_acceleration(equal);

  EXPECT_NEAR(accel[0], 0.0, 1e-9);   // no x force
  EXPECT_NEAR(accel[1], 0.0, 1e-9);   // no y force
  EXPECT_GT(std::abs(accel[2]), 1e-3);  // real angular acceleration
}

// The torque pattern [-1,-1,1,1] cancels yaw and y-force, leaving pure +x.
TEST(PlantTest, TranslationPatternGivesPureXForce) {
  Plant plant(make_x_config(0.0));
  const double magnitude = 1.0;
  WheelTorques pattern = {-magnitude, -magnitude, magnitude, magnitude};

  std::array<double, 3> accel = plant.body_acceleration(pattern);

  // Independent closed form: four wheels each contribute |tau|/r * cos45 to Fx.
  const double force_x = 4.0 * (magnitude / 0.05) * (std::sqrt(2.0) / 2.0);
  const double expected_ax = force_x / 12.0;

  EXPECT_NEAR(accel[0], expected_ax, 1e-6);
  EXPECT_NEAR(accel[1], 0.0, 1e-9);
  EXPECT_NEAR(accel[2], 0.0, 1e-9);
}

// The body-frame +x force must rotate with the body: at theta = 90 deg it
// should appear as world +y.
TEST(PlantTest, AccelerationRotatesWithBodyPose) {
  Plant plant(make_x_config(0.0));

  State pose;
  pose.theta = 90.0 * kDeg;
  plant.set_state(pose);

  WheelTorques pattern = {-1.0, -1.0, 1.0, 1.0};
  std::array<double, 3> accel = plant.body_acceleration(pattern);

  EXPECT_NEAR(accel[0], 0.0, 1e-6);   // no world-x
  EXPECT_GT(accel[1], 1.0);           // pushed along world +y
}

// With torque held, RK4 must actually move the robot in +x and build up +x
// velocity; y and theta stay put.
TEST(PlantTest, Rk4IntegratesTranslation) {
  Plant plant(make_x_config(0.0));
  WheelTorques pattern = {-1.0, -1.0, 1.0, 1.0};

  const double dt = 0.001;
  for (int stepIndex = 0; stepIndex < 1000; ++stepIndex) {  // 1 s
    plant.step(pattern, dt);
  }

  const State& state = plant.state();
  EXPECT_GT(state.x, 0.1);
  EXPECT_GT(state.vx, 0.1);
  EXPECT_NEAR(state.y, 0.0, 1e-6);
  EXPECT_NEAR(state.theta, 0.0, 1e-9);
  EXPECT_NEAR(state.omega, 0.0, 1e-9);

  // Constant-force kinematics sanity: x should be about 0.5 * a * t^2.
  const double force_x = 4.0 * (1.0 / 0.05) * (std::sqrt(2.0) / 2.0);
  const double accel_x = force_x / 12.0;
  const double expected_x = 0.5 * accel_x * 1.0 * 1.0;
  EXPECT_NEAR(state.x, expected_x, 1e-3);
}

// Viscous friction must act as drag: moving with zero torque, the true plant
// (friction > 0) decelerates, while the nominal model (friction = 0) coasts.
TEST(PlantTest, FrictionDeceleratesButNominalCoasts) {
  State moving;
  moving.vx = 1.0;  // 1 m/s along world +x

  WheelTorques none = {0.0, 0.0, 0.0, 0.0};

  Plant true_plant(make_x_config(0.02));
  true_plant.set_state(moving);
  std::array<double, 3> true_accel = true_plant.body_acceleration(none);
  EXPECT_LT(true_accel[0], -1e-6);  // drag opposes motion

  Plant nominal_plant(make_x_config(0.0));
  nominal_plant.set_state(moving);
  std::array<double, 3> nominal_accel = nominal_plant.body_acceleration(none);
  EXPECT_NEAR(nominal_accel[0], 0.0, 1e-12);  // frictionless: no drag
}

}  // namespace
}  // namespace omni_sim

#include "omni_sim/config.hpp"

#include <gtest/gtest.h>

// Step 1 only checks that the config data layout is usable and defaults are
// sane. Parsing from YAML (and the true-vs-nominal distinction) is tested in
// Step 2.

namespace omni_sim {
namespace {

TEST(ConfigTest, DefaultsAreZeroedAndSafe) {
  PlantConfig config;

  EXPECT_DOUBLE_EQ(config.chassis.mass, 0.0);
  EXPECT_DOUBLE_EQ(config.chassis.wheel_radius, 0.0);
  EXPECT_DOUBLE_EQ(config.motor.k_t, 0.0);
  EXPECT_DOUBLE_EQ(config.motor.gear_ratio, 1.0);
}

TEST(ConfigTest, HoldsFourWheelsWorthOfGeometry) {
  PlantConfig config;

  EXPECT_EQ(config.chassis.mount_angle.size(), kWheelCount);
  EXPECT_EQ(config.chassis.drive_angle.size(), kWheelCount);
  EXPECT_EQ(kWheelCount, 4u);
}

TEST(ConfigTest, FieldsAreIndependentlyAssignable) {
  PlantConfig config;

  config.chassis.mass = 12.5;
  config.motor.k_t = 0.042;

  EXPECT_DOUBLE_EQ(config.chassis.mass, 12.5);
  EXPECT_DOUBLE_EQ(config.motor.k_t, 0.042);
}

}  // namespace
}  // namespace omni_sim

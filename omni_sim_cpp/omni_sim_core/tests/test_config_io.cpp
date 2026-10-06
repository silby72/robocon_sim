#include "omni_sim/config_io.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>

// Tests for direct YAML parsing. OMNI_SIM_CONFIG_DIR is injected by CMake and
// points at the repo's config/ directory, so these run against the real
// plant_true.yaml / plant_nominal.yaml the ROS nodes will use.

namespace omni_sim {
namespace {

std::string config_path(const std::string& file_name) {
  return std::string(OMNI_SIM_CONFIG_DIR) + "/" + file_name;
}

TEST(ConfigIoTest, LoadsTruePlant) {
  PlantConfig config = load_plant_config(config_path("plant_true.yaml"));

  EXPECT_DOUBLE_EQ(config.chassis.mass, 12.0);
  EXPECT_DOUBLE_EQ(config.chassis.wheel_radius, 0.05);
  EXPECT_DOUBLE_EQ(config.motor.k_t, 0.042);
  EXPECT_DOUBLE_EQ(config.motor.viscous_friction, 0.001);
  EXPECT_STREQ(config.label.data(), "true");
}

TEST(ConfigIoTest, AnglesAreConvertedToRadians) {
  PlantConfig config = load_plant_config(config_path("plant_true.yaml"));

  // First wheel sits at 45 degrees.
  const double expected_first = 45.0 * M_PI / 180.0;
  EXPECT_NEAR(config.chassis.mount_angle[0], expected_first, 1e-12);

  // Full array, mount angles 45/135/225/315.
  const double degrees[kWheelCount] = {45.0, 135.0, 225.0, 315.0};
  for (std::size_t wheel_index = 0; wheel_index < kWheelCount; ++wheel_index) {
    const double expected = degrees[wheel_index] * M_PI / 180.0;
    EXPECT_NEAR(config.chassis.mount_angle[wheel_index], expected, 1e-12);
  }
}

// The whole point of the two-file split: nominal is deliberately different.
TEST(ConfigIoTest, NominalDiffersFromTrueInDocumentedWays) {
  PlantConfig true_plant = load_plant_config(config_path("plant_true.yaml"));
  PlantConfig nominal = load_plant_config(config_path("plant_nominal.yaml"));

  // mass is 10% light in the nominal model.
  EXPECT_LT(nominal.chassis.mass, true_plant.chassis.mass);
  EXPECT_NEAR(nominal.chassis.mass, 0.9 * true_plant.chassis.mass, 1e-9);

  // nominal assumes no friction.
  EXPECT_DOUBLE_EQ(nominal.motor.viscous_friction, 0.0);
  EXPECT_GT(true_plant.motor.viscous_friction, 0.0);

  EXPECT_STREQ(nominal.label.data(), "nominal");
}

TEST(ConfigIoTest, MissingFileThrowsConfigError) {
  EXPECT_THROW(load_plant_config(config_path("does_not_exist.yaml")),
               ConfigError);
}

TEST(ConfigIoTest, MissingRequiredKeyThrowsConfigError) {
  // Write a temp file that omits chassis.wheel_radius.
  const std::string temp = std::string(OMNI_SIM_CONFIG_DIR) + "/_tmp_bad.yaml";
  std::ofstream out(temp);
  out << "chassis:\n";
  out << "  mass: 12.0\n";
  out << "  inertia_zz: 0.45\n";
  out << "  mount_radius: 0.20\n";
  out << "  mount_angle_deg: [45, 135, 225, 315]\n";
  out << "  drive_angle_deg: [135, 225, 315, 45]\n";
  out << "motor:\n";
  out << "  k_t: 0.04\n";
  out << "  resistance: 1.2\n";
  out << "  viscous_friction: 0.0\n";
  out << "  gear_ratio: 19.0\n";
  out.close();

  try {
    load_plant_config(temp);
    FAIL() << "expected ConfigError for the missing wheel_radius";
  } catch (const ConfigError& error) {
    // Message should name the offending key so it is actionable.
    EXPECT_NE(std::string(error.what()).find("wheel_radius"),
              std::string::npos);
  }

  std::remove(temp.c_str());
}

TEST(ConfigIoTest, WrongLengthAngleArrayThrows) {
  const std::string temp = std::string(OMNI_SIM_CONFIG_DIR) + "/_tmp_short.yaml";
  std::ofstream out(temp);
  out << "chassis:\n";
  out << "  mass: 12.0\n";
  out << "  inertia_zz: 0.45\n";
  out << "  wheel_radius: 0.05\n";
  out << "  mount_radius: 0.20\n";
  out << "  mount_angle_deg: [45, 135, 225]\n";  // only 3
  out << "  drive_angle_deg: [135, 225, 315, 45]\n";
  out << "motor:\n";
  out << "  k_t: 0.04\n";
  out << "  resistance: 1.2\n";
  out << "  viscous_friction: 0.0\n";
  out << "  gear_ratio: 19.0\n";
  out.close();

  EXPECT_THROW(load_plant_config(temp), ConfigError);

  std::remove(temp.c_str());
}

}  // namespace
}  // namespace omni_sim

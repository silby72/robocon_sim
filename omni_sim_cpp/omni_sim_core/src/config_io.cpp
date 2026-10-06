#include "omni_sim/config_io.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

// Direct YAML parsing with explicit validation. Every helper reports the file
// and the key it failed on, so a bad config produces a message a human can act
// on instead of a bare yaml-cpp exception.

namespace omni_sim {
namespace {

constexpr double kPi = 3.14159265358979323846;

double degrees_to_radians(double degrees) {
  return degrees * kPi / 180.0;
}

// Fetch a required child node, or throw a ConfigError that names the path.
YAML::Node require_child(const YAML::Node& parent, const std::string& key,
                         const std::string& file, const std::string& context) {
  YAML::Node child = parent[key];
  if (!child) {
    throw ConfigError(file + ": missing required key '" + context + key + "'");
  }
  return child;
}

// Read a required scalar as double, with a clear message on the wrong type.
double require_double(const YAML::Node& parent, const std::string& key,
                      const std::string& file, const std::string& context) {
  YAML::Node child = require_child(parent, key, file, context);
  try {
    return child.as<double>();
  } catch (const YAML::Exception&) {
    throw ConfigError(file + ": key '" + context + key +
                      "' is not a number");
  }
}

// Read a required 4-element angle array (degrees) into a radians array.
std::array<double, kWheelCount> require_angle_array(
    const YAML::Node& parent, const std::string& key, const std::string& file,
    const std::string& context) {
  YAML::Node child = require_child(parent, key, file, context);

  if (!child.IsSequence()) {
    throw ConfigError(file + ": key '" + context + key +
                      "' must be a list of angles");
  }
  if (child.size() != kWheelCount) {
    throw ConfigError(file + ": key '" + context + key + "' must have " +
                      std::to_string(kWheelCount) + " entries, got " +
                      std::to_string(child.size()));
  }

  std::array<double, kWheelCount> radians = {};
  for (std::size_t wheel_index = 0; wheel_index < kWheelCount; ++wheel_index) {
    double degrees = 0.0;
    try {
      degrees = child[wheel_index].as<double>();
    } catch (const YAML::Exception&) {
      throw ConfigError(file + ": key '" + context + key + "[" +
                        std::to_string(wheel_index) + "]' is not a number");
    }
    radians[wheel_index] = degrees_to_radians(degrees);
  }
  return radians;
}

// Copy the optional human-readable label into the fixed-size buffer, truncating
// if needed and always leaving it null-terminated.
void copy_label(const YAML::Node& root, PlantConfig& config) {
  YAML::Node label_node = root["label"];
  if (!label_node) {
    return;  // keeps the "unset" default from config.hpp
  }

  std::string label = label_node.as<std::string>();
  const std::size_t capacity = config.label.size() - 1;
  const std::size_t count = std::min(label.size(), capacity);

  std::size_t position = 0;
  for (; position < count; ++position) {
    config.label[position] = label[position];
  }
  config.label[position] = '\0';
}

}  // namespace

PlantConfig load_plant_config(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::BadFile&) {
    throw ConfigError(path + ": cannot open file");
  } catch (const YAML::Exception& error) {
    throw ConfigError(path + ": YAML parse error: " + error.what());
  }

  PlantConfig config;

  YAML::Node chassis = require_child(root, "chassis", path, "");
  config.chassis.mass = require_double(chassis, "mass", path, "chassis.");
  config.chassis.inertia_zz =
      require_double(chassis, "inertia_zz", path, "chassis.");
  config.chassis.wheel_radius =
      require_double(chassis, "wheel_radius", path, "chassis.");
  config.chassis.mount_radius =
      require_double(chassis, "mount_radius", path, "chassis.");
  config.chassis.mount_angle =
      require_angle_array(chassis, "mount_angle_deg", path, "chassis.");
  config.chassis.drive_angle =
      require_angle_array(chassis, "drive_angle_deg", path, "chassis.");

  YAML::Node motor = require_child(root, "motor", path, "");
  config.motor.k_t = require_double(motor, "k_t", path, "motor.");
  config.motor.resistance = require_double(motor, "resistance", path, "motor.");
  config.motor.viscous_friction =
      require_double(motor, "viscous_friction", path, "motor.");
  config.motor.gear_ratio = require_double(motor, "gear_ratio", path, "motor.");

  copy_label(root, config);

  return config;
}

}  // namespace omni_sim

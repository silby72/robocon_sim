#pragma once

// Plain-data configuration structs for the omni simulator core.
//
// These structs mirror the fields in config/plant_true.yaml and
// config/plant_nominal.yaml. In Step 1 they only define the data layout with
// sensible defaults; the actual YAML parsing is added in Step 2
// (load_plant_config), and the dynamics that consume them arrive in Step 3.
//
// Design rule: the "true plant" and the "nominal model" are two independent
// PlantConfig values loaded from two separate YAML files. The nominal file is
// allowed to hold deliberately wrong values (e.g. mass 10% light, friction 0)
// so the controller's model error can be studied.

#include <array>
#include <cstddef>

namespace omni_sim {

// A 4-wheel omni chassis. Wheels are numbered 0..3.
inline constexpr std::size_t kWheelCount = 4;

// Electrical / mechanical model of one drive motor (all four share the same
// type here; per-wheel differences can be added later if needed).
struct MotorParams {
  // Torque constant [N*m/A].
  double k_t = 0.0;
  // Winding resistance [Ohm].
  double resistance = 0.0;
  // Viscous friction coefficient at the wheel [N*m/(rad/s)].
  double viscous_friction = 0.0;
  // Gear reduction from motor shaft to wheel (motor_speed / wheel_speed).
  double gear_ratio = 1.0;
};

// Geometry and inertia of the chassis body.
struct ChassisParams {
  // Total robot mass [kg].
  double mass = 0.0;
  // Yaw moment of inertia about the body center [kg*m^2].
  double inertia_zz = 0.0;
  // Wheel radius [m], shared by all four wheels.
  double wheel_radius = 0.0;
  // Distance from body center to each wheel contact point [m].
  double mount_radius = 0.0;

  // Mounting angle of each wheel's position vector, measured from the body
  // +x axis [rad]. Together with roller_angle these define the 4x3 kinematics
  // built in Step 3. Defaults describe a symmetric X-configuration.
  std::array<double, kWheelCount> mount_angle = {0.0, 0.0, 0.0, 0.0};

  // Direction the wheel can drive (the roller/free-roll axis is perpendicular),
  // measured from the body +x axis [rad].
  std::array<double, kWheelCount> drive_angle = {0.0, 0.0, 0.0, 0.0};
};

// One complete parameter set: either the true plant or the nominal model.
struct PlantConfig {
  ChassisParams chassis;
  MotorParams motor;

  // A human-readable tag copied from the YAML ("true" / "nominal") so logs and
  // errors can say which model produced a value. Purely informational.
  std::array<char, 16> label = {'u', 'n', 's', 'e', 't', '\0'};
};

}  // namespace omni_sim

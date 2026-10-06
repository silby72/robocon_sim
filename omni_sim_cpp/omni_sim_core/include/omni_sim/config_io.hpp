#pragma once

// Loading PlantConfig from a YAML file.
//
// The spec calls for parsing YAML directly in C++ (no code generation). We use
// yaml-cpp for the parse and do all validation and unit conversion here, so the
// rest of the core only ever sees a fully-checked PlantConfig.
//
// The true plant and the nominal model are two separate files; call this twice
// with the two paths. Nothing in here special-cases "true" vs "nominal" — the
// distinction lives entirely in which file you hand it.

#include "omni_sim/config.hpp"

#include <stdexcept>
#include <string>

namespace omni_sim {

// Thrown when a file is missing, malformed, missing a required key, or holds a
// value that cannot be a valid plant (e.g. a wheel-angle array whose length is
// not kWheelCount). The message names the file and the offending key.
class ConfigError : public std::runtime_error {
 public:
  explicit ConfigError(const std::string& message) : std::runtime_error(message) {}
};

// Parse the YAML at `path` into a PlantConfig. Angle arrays are read in degrees
// and returned in radians. Throws ConfigError on any problem.
PlantConfig load_plant_config(const std::string& path);

}  // namespace omni_sim

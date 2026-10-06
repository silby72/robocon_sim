cmake_minimum_required(VERSION 3.16)
project(omni_sim_core LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language standard: C++20, required, no compiler extensions.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ---------------------------------------------------------------------------
# Dependencies. yaml-cpp is used for direct YAML parsing (Step 2). It ships a
# CMake config package on this system; ROS Jazzy also vendors it.
# ---------------------------------------------------------------------------
find_package(yaml-cpp REQUIRED)

# ---------------------------------------------------------------------------
# The ROS-free core library.
# ---------------------------------------------------------------------------
add_library(omni_sim_core
  src/plant.cpp
  src/config_io.cpp
  src/sensors.cpp
  src/control.cpp
)
target_include_directories(omni_sim_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(omni_sim_core PUBLIC yaml-cpp::yaml-cpp)
target_compile_options(omni_sim_core PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# Tests (GoogleTest). Prefer a system-installed GTest; if it is missing, fall
# back to fetching it so the tests still build on a bare machine.
# ---------------------------------------------------------------------------
option(OMNI_SIM_CORE_BUILD_TESTS "Build the omni_sim_core unit tests" ON)

if(OMNI_SIM_CORE_BUILD_TESTS)
  enable_testing()

  find_package(GTest QUIET)

  if(NOT GTest_FOUND)
    message(STATUS "System GoogleTest not found; fetching via FetchContent.")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(omni_sim_core_tests
    tests/test_config.cpp
    tests/test_config_io.cpp
    tests/test_plant.cpp
    tests/test_sensors.cpp
    tests/test_control.cpp
  )
  target_link_libraries(omni_sim_core_tests PRIVATE
    omni_sim_core
    GTest::gtest_main
  )
  target_compile_options(omni_sim_core_tests PRIVATE -Wall -Wextra)
  # Point the config-loading tests at the repo's config/ directory (absolute).
  target_compile_definitions(omni_sim_core_tests PRIVATE
    OMNI_SIM_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../config"
  )

  include(GoogleTest)
  gtest_discover_tests(omni_sim_core_tests)
endif()

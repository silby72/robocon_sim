// controller_node: runs on the NOMINAL model, closes a body-twist loop.
//
// It loads plant_nominal.yaml (never the true plant), tracks a target body twist
// (vx, vy, omega) with three PID channels, and augments each channel with a
// disturbance observer. The DOB reconstructs everything the nominal model does
// NOT contain — the friction and the 10% mass error between nominal and true —
// and feeds it forward, so the loop holds the setpoint despite the mismatch.
//
// The three body-axis wrenches (Fx, Fy, Mz) are mapped to four wheel torques
// through the pseudo-inverse of the omni force map, and published on
// /wheel_torques. Feedback is the /odom body twist.

#include "omni_sim/config_io.hpp"
#include "omni_sim/control.hpp"

#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

namespace omni_sim {
namespace {

// Invert a 3x3 matrix (row-major). Returns false if singular.
bool invert_3x3(const std::array<std::array<double, 3>, 3>& m,
                std::array<std::array<double, 3>, 3>& inverse) {
  const double det =
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (std::abs(det) < 1e-12) {
    return false;
  }
  const double inv_det = 1.0 / det;

  inverse[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
  inverse[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
  inverse[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
  inverse[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
  inverse[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
  inverse[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
  inverse[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
  inverse[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
  inverse[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
  return true;
}

}  // namespace

class ControllerNode : public rclcpp::Node {
 public:
  ControllerNode() : rclcpp::Node("controller_node") {
    const std::string config_path =
        this->declare_parameter<std::string>("plant_config", "");
    const double rate_hz = this->declare_parameter<double>("rate_hz", 100.0);
    if (config_path.empty()) {
      throw std::runtime_error(
          "controller_node requires the 'plant_config' parameter (path to "
          "plant_nominal.yaml)");
    }

    // Target body twist (setpoint).
    target_vx_ = this->declare_parameter<double>("target_vx", 1.0);
    target_vy_ = this->declare_parameter<double>("target_vy", 0.0);
    target_omega_ = this->declare_parameter<double>("target_omega", 0.0);

    // Load the NOMINAL model.
    nominal_ = load_plant_config(config_path);
    RCLCPP_INFO(
        this->get_logger(),
        "controller_node loaded nominal model '%s' (mass=%.3f kg, Izz=%.3f)",
        nominal_.label.data(), nominal_.chassis.mass, nominal_.chassis.inertia_zz);

    build_force_map();
    build_controllers(1.0 / rate_hz);

    odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom", 10, [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          this->on_odom(msg);
        });
    torque_publisher_ =
        this->create_publisher<std_msgs::msg::Float64MultiArray>("wheel_torques",
                                                                 10);

    const auto period = std::chrono::duration<double>(1.0 / rate_hz);
    timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        [this]() { this->on_timer(); });

    RCLCPP_INFO(this->get_logger(),
                "controller_node tracking body twist (%.2f, %.2f, %.2f) with "
                "PID+DOB at %.1f Hz",
                target_vx_, target_vy_, target_omega_, rate_hz);
  }

 private:
  // Precompute the per-wheel force-map columns and (B B^T)^{-1} for the wrench
  // -> wheel-force pseudo-inverse.
  void build_force_map() {
    const ChassisParams& chassis = nominal_.chassis;
    std::array<std::array<double, 3>, 3> gram = {};

    for (std::size_t wheel = 0; wheel < kWheelCount; ++wheel) {
      const double mount_x = chassis.mount_radius * std::cos(chassis.mount_angle[wheel]);
      const double mount_y = chassis.mount_radius * std::sin(chassis.mount_angle[wheel]);
      const double drive_x = std::cos(chassis.drive_angle[wheel]);
      const double drive_y = std::sin(chassis.drive_angle[wheel]);
      const double moment = mount_x * drive_y - mount_y * drive_x;

      column_[wheel] = {drive_x, drive_y, moment};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          gram[i][j] += column_[wheel][i] * column_[wheel][j];
        }
      }
    }

    if (!invert_3x3(gram, gram_inverse_)) {
      throw std::runtime_error(
          "controller_node: omni force map is singular (check wheel geometry)");
    }
  }

  void build_controllers(double dt) {
    // Linear axes: nominal plant is m_n * v_dot = F (no nominal friction).
    PidParams linear_pid;
    linear_pid.kp = this->declare_parameter<double>("kp_linear", 40.0);
    linear_pid.ki = this->declare_parameter<double>("ki_linear", 80.0);
    linear_pid.kd = this->declare_parameter<double>("kd_linear", 0.0);

    PidParams angular_pid;
    angular_pid.kp = this->declare_parameter<double>("kp_angular", 6.0);
    angular_pid.ki = this->declare_parameter<double>("ki_angular", 12.0);
    angular_pid.kd = this->declare_parameter<double>("kd_angular", 0.0);

    pid_vx_ = std::make_unique<Pid>(linear_pid, dt);
    pid_vy_ = std::make_unique<Pid>(linear_pid, dt);
    pid_omega_ = std::make_unique<Pid>(angular_pid, dt);

    DobParams dob_params;
    dob_params.enabled = this->declare_parameter<bool>("dob_enabled", true);
    dob_params.tau_q = this->declare_parameter<double>("dob_tau_q", 0.02);
    dob_params.order = this->declare_parameter<int>("dob_order", 1);

    // Nominal "inertia" per body axis: mass for vx/vy, yaw inertia for omega.
    // No nominal damping (nominal friction is zero) -> Bn = 0.
    NominalMotor linear_nominal{nominal_.chassis.mass, 0.0};
    NominalMotor angular_nominal{nominal_.chassis.inertia_zz, 0.0};

    dob_vx_ = std::make_unique<DisturbanceObserver>(dob_params, linear_nominal, dt);
    dob_vy_ = std::make_unique<DisturbanceObserver>(dob_params, linear_nominal, dt);
    dob_omega_ =
        std::make_unique<DisturbanceObserver>(dob_params, angular_nominal, dt);
  }

  void on_odom(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // /odom twist is already body-frame (REP-105 child frame).
    measured_vx_ = msg->twist.twist.linear.x;
    measured_vy_ = msg->twist.twist.linear.y;
    measured_omega_ = msg->twist.twist.angular.z;
    have_odom_ = true;
  }

  // One control channel: PID + DOB feedforward. Returns the axis wrench.
  double control_axis(Pid& pid, DisturbanceObserver& dob, double setpoint,
                      double measurement) {
    const double pid_output = pid.update(setpoint, measurement);
    const double compensated = pid_output - dob.d_hat();
    dob.update(measurement, compensated);
    return compensated;
  }

  void on_timer() {
    if (!have_odom_) {
      return;  // wait for the first measurement
    }

    const double wrench_x =
        control_axis(*pid_vx_, *dob_vx_, target_vx_, measured_vx_);
    const double wrench_y =
        control_axis(*pid_vy_, *dob_vy_, target_vy_, measured_vy_);
    const double wrench_z =
        control_axis(*pid_omega_, *dob_omega_, target_omega_, measured_omega_);

    // wheel forces f = B^T (B B^T)^{-1} W ; wheel torque = f * wheel_radius.
    const std::array<double, 3> wrench = {wrench_x, wrench_y, wrench_z};
    std::array<double, 3> intermediate = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        intermediate[i] += gram_inverse_[i][j] * wrench[j];
      }
    }

    const double radius = nominal_.chassis.wheel_radius;
    std_msgs::msg::Float64MultiArray command;
    command.data.resize(kWheelCount);
    for (std::size_t wheel = 0; wheel < kWheelCount; ++wheel) {
      double wheel_force = 0.0;
      for (int i = 0; i < 3; ++i) {
        wheel_force += column_[wheel][i] * intermediate[i];
      }
      command.data[wheel] = wheel_force * radius;
    }
    torque_publisher_->publish(command);
  }

  PlantConfig nominal_;

  // Force map: per-wheel column [drive_x, drive_y, moment] and (B B^T)^{-1}.
  std::array<std::array<double, 3>, kWheelCount> column_ = {};
  std::array<std::array<double, 3>, 3> gram_inverse_ = {};

  double target_vx_ = 1.0;
  double target_vy_ = 0.0;
  double target_omega_ = 0.0;

  bool have_odom_ = false;
  double measured_vx_ = 0.0;
  double measured_vy_ = 0.0;
  double measured_omega_ = 0.0;

  std::unique_ptr<Pid> pid_vx_;
  std::unique_ptr<Pid> pid_vy_;
  std::unique_ptr<Pid> pid_omega_;
  std::unique_ptr<DisturbanceObserver> dob_vx_;
  std::unique_ptr<DisturbanceObserver> dob_vy_;
  std::unique_ptr<DisturbanceObserver> dob_omega_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace omni_sim

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<omni_sim::ControllerNode>());
  rclcpp::shutdown();
  return 0;
}

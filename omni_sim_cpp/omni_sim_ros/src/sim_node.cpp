// sim_node: runs the TRUE plant.
//
// It loads plant_true.yaml, integrates the omni_sim_core Plant on a fixed-rate
// timer, and behaves like the "hardware": it publishes the sensor topics a real
// robot would (/odom, /imu) and subscribes to the wheel-torque command. It is
// also the time authority — it publishes /clock so every other node running
// with use_sim_time:=true is driven by simulated time.
//
// The plant is the true dynamics; the published /odom and /imu are run through
// the omni_sim_core sensor models (encoder quantization on odom, bias + white
// noise on imu), so downstream nodes see realistically corrupted measurements.
// LiDAR raycast is a later addition.

#include "omni_sim/config_io.hpp"
#include "omni_sim/plant.hpp"
#include "omni_sim/sensors.hpp"

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace omni_sim {

class SimNode : public rclcpp::Node {
 public:
  SimNode() : rclcpp::Node("sim_node") {
    // Parameters.
    const std::string config_path =
        this->declare_parameter<std::string>("plant_config", "");
    rate_hz_ = this->declare_parameter<double>("rate_hz", 200.0);

    if (config_path.empty()) {
      throw std::runtime_error(
          "sim_node requires the 'plant_config' parameter (path to "
          "plant_true.yaml)");
    }

    // Load the TRUE plant. A bad file fails loudly at startup.
    PlantConfig config = load_plant_config(config_path);
    RCLCPP_INFO(this->get_logger(),
                "sim_node loaded true plant '%s' (mass=%.3f kg, friction=%.4f)",
                config.label.data(), config.chassis.mass,
                config.motor.viscous_friction);
    plant_ = std::make_unique<Plant>(config);

    // Sensor models. Parameters default to a modest, realistic amount of
    // corruption; set the resolution high / noise to 0 for an ideal sensor.
    const double encoder_counts =
        this->declare_parameter<double>("encoder_counts_per_rev", 4096.0);
    odometry_ = std::make_unique<EncoderOdometry>(
        config.chassis, encoder_counts, config.motor.gear_ratio);

    ImuParams imu_params;
    imu_params.gyro_bias = this->declare_parameter<double>("imu_gyro_bias", 0.01);
    imu_params.gyro_noise_sigma =
        this->declare_parameter<double>("imu_gyro_noise", 0.02);
    imu_params.accel_bias = this->declare_parameter<double>("imu_accel_bias", 0.05);
    imu_params.accel_noise_sigma =
        this->declare_parameter<double>("imu_accel_noise", 0.1);
    const int imu_seed = this->declare_parameter<int>("imu_seed", 12345);
    imu_ = std::make_unique<ImuSensor>(imu_params,
                                       static_cast<std::uint64_t>(imu_seed));

    // Publishers.
    clock_publisher_ =
        this->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
    odom_publisher_ =
        this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
    imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);

    // Subscriber: latest wheel-torque command (4 values).
    torque_subscription_ =
        this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "wheel_torques", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
              this->on_wheel_torques(msg);
            });

    // Fixed-step timer. dt_ is the integration/publish period.
    dt_ = 1.0 / rate_hz_;
    const auto period = std::chrono::duration<double>(dt_);
    timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        [this]() { this->on_timer(); });

    RCLCPP_INFO(this->get_logger(), "sim_node running at %.1f Hz", rate_hz_);
  }

 private:
  void on_wheel_torques(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size() != kWheelCount) {
      RCLCPP_WARN(this->get_logger(),
                  "wheel_torques has %zu values, expected %zu; ignoring",
                  msg->data.size(), kWheelCount);
      return;
    }
    for (std::size_t wheel = 0; wheel < kWheelCount; ++wheel) {
      latest_torques_[wheel] = msg->data[wheel];
    }
  }

  void on_timer() {
    // Advance simulated time and the plant.
    sim_time_seconds_ += dt_;
    plant_->step(latest_torques_, dt_);

    publish_clock();
    publish_odometry();
    publish_imu();
  }

  // Body-frame linear acceleration the plant is currently producing (for the
  // IMU). world_acceleration is exposed via body_acceleration(); rotate it into
  // the body frame here.
  std::array<double, 2> current_body_acceleration() const {
    const State& state = plant_->state();
    const std::array<double, 3> world = plant_->body_acceleration(latest_torques_);
    const double cos_theta = std::cos(state.theta);
    const double sin_theta = std::sin(state.theta);
    const double body_ax = cos_theta * world[0] + sin_theta * world[1];
    const double body_ay = -sin_theta * world[0] + cos_theta * world[1];
    return {body_ax, body_ay};
  }

  rclcpp::Time sim_now() const {
    return rclcpp::Time(static_cast<int64_t>(sim_time_seconds_ * 1e9));
  }

  void publish_clock() {
    rosgraph_msgs::msg::Clock message;
    message.clock = sim_now();
    clock_publisher_->publish(message);
  }

  void publish_odometry() {
    // Encoder odometry: quantized twist + dead-reckoned (drifting) pose.
    const OdometryReading reading = odometry_->measure(plant_->state(), dt_);

    nav_msgs::msg::Odometry message;
    message.header.stamp = sim_now();
    message.header.frame_id = "odom";
    message.child_frame_id = "base_link";

    // Pose is the odometry's own dead reckoning, not the true state.
    message.pose.pose.position.x = reading.x;
    message.pose.pose.position.y = reading.y;
    message.pose.pose.orientation.z = std::sin(reading.theta / 2.0);
    message.pose.pose.orientation.w = std::cos(reading.theta / 2.0);

    // Twist is reported in the child (body) frame, per REP-105.
    message.twist.twist.linear.x = reading.vx_body;
    message.twist.twist.linear.y = reading.vy_body;
    message.twist.twist.angular.z = reading.omega;

    odom_publisher_->publish(message);
  }

  void publish_imu() {
    const State& state = plant_->state();
    const std::array<double, 2> body_accel = current_body_acceleration();

    // IMU: bias + white noise on the true yaw rate and body acceleration.
    const ImuReading reading =
        imu_->measure(state.omega, body_accel[0], body_accel[1]);

    sensor_msgs::msg::Imu message;
    message.header.stamp = sim_now();
    message.header.frame_id = "base_link";
    // Orientation is left as the true heading (a real IMU fuses this itself).
    message.orientation.z = std::sin(state.theta / 2.0);
    message.orientation.w = std::cos(state.theta / 2.0);
    message.angular_velocity.z = reading.angular_velocity_z;
    message.linear_acceleration.x = reading.linear_acceleration_x;
    message.linear_acceleration.y = reading.linear_acceleration_y;

    imu_publisher_->publish(message);
  }

  double rate_hz_ = 200.0;
  double dt_ = 0.005;
  double sim_time_seconds_ = 0.0;

  std::unique_ptr<Plant> plant_;
  std::unique_ptr<EncoderOdometry> odometry_;
  std::unique_ptr<ImuSensor> imu_;
  WheelTorques latest_torques_ = {0.0, 0.0, 0.0, 0.0};

  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
      torque_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace omni_sim

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<omni_sim::SimNode>());
  rclcpp::shutdown();
  return 0;
}

#include "omni_sim/control.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace omni_sim {
namespace {

// Bilinear (Tustin) transform of a continuous transfer function num(s)/den(s),
// coefficients given highest-power-first, into a discrete IIR at sample time dt.
// s = c (z-1)/(z+1), c = 2/dt. Handled explicitly for order 1 and order 2 so
// each filter stays minimal (no spurious pole at z = -1).
IirFilter make_bilinear(std::vector<double> num, const std::vector<double>& den,
                        double dt) {
  const double c = 2.0 / dt;

  // Left-pad the numerator with zeros so it matches the denominator's order.
  while (num.size() < den.size()) {
    num.insert(num.begin(), 0.0);
  }

  if (den.size() == 2) {
    // H(s) = (b1 s + b0) / (a1 s + a0).
    const double b1 = num[0];
    const double b0 = num[1];
    const double a1 = den[0];
    const double a0 = den[1];

    std::vector<double> b = {b1 * c + b0, b0 - b1 * c};
    std::vector<double> a = {a1 * c + a0, a0 - a1 * c};
    return IirFilter(b, a);
  }

  if (den.size() == 3) {
    // H(s) = (b2 s^2 + b1 s + b0) / (a2 s^2 + a1 s + a0).
    const double b2 = num[0];
    const double b1 = num[1];
    const double b0 = num[2];
    const double a2 = den[0];
    const double a1 = den[1];
    const double a0 = den[2];

    const double c2 = c * c;
    std::vector<double> b = {b2 * c2 + b1 * c + b0, -2.0 * b2 * c2 + 2.0 * b0,
                             b2 * c2 - b1 * c + b0};
    std::vector<double> a = {a2 * c2 + a1 * c + a0, -2.0 * a2 * c2 + 2.0 * a0,
                             a2 * c2 - a1 * c + a0};
    return IirFilter(b, a);
  }

  throw std::invalid_argument("make_bilinear supports order 1 and 2 only");
}

}  // namespace

// --- PID -------------------------------------------------------------------

Pid::Pid(const PidParams& params, double dt) : params_(params), dt_(dt) {
  reset();
}

void Pid::reset() {
  integral_ = 0.0;
  has_previous_ = false;
  previous_measurement_ = 0.0;
  derivative_filtered_ = 0.0;
}

double Pid::update(double setpoint, double measurement) {
  const double error = setpoint - measurement;

  // Derivative on measurement (no setpoint-step derivative kick).
  double raw_derivative = 0.0;
  if (has_previous_) {
    raw_derivative = -(measurement - previous_measurement_) / dt_;
  }
  previous_measurement_ = measurement;
  has_previous_ = true;

  double derivative = raw_derivative;
  if (params_.derivative_lpf_tau > 0.0) {
    const double alpha = dt_ / (params_.derivative_lpf_tau + dt_);
    derivative_filtered_ += alpha * (raw_derivative - derivative_filtered_);
    derivative = derivative_filtered_;
  }

  const double integral_candidate = integral_ + params_.ki * error * dt_;
  const double unsaturated =
      params_.kp * error + integral_candidate + params_.kd * derivative;
  const double output =
      std::clamp(unsaturated, params_.output_min, params_.output_max);

  // Anti-windup: back-calculation if enabled, else clamp-only (freeze on sat).
  if (params_.anti_windup_kb > 0.0) {
    integral_ = integral_candidate +
                params_.anti_windup_kb * (output - unsaturated) * dt_;
  } else if (output == unsaturated) {
    integral_ = integral_candidate;
  }
  // else: saturated with clamp-only -> freeze the integral.

  return output;
}

// --- IIR filter ------------------------------------------------------------

IirFilter::IirFilter(std::vector<double> b, std::vector<double> a) {
  const double a0 = a[0];
  b_.resize(b.size());
  a_.resize(a.size());
  for (std::size_t i = 0; i < b.size(); ++i) {
    b_[i] = b[i] / a0;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    a_[i] = a[i] / a0;
  }
  const std::size_t state_size = std::max(a_.size(), b_.size()) - 1;
  z_.assign(state_size, 0.0);
}

void IirFilter::reset() {
  for (std::size_t i = 0; i < z_.size(); ++i) {
    z_[i] = 0.0;
  }
}

double IirFilter::step(double x) {
  const double y = b_[0] * x + (z_.empty() ? 0.0 : z_[0]);
  for (std::size_t i = 1; i < b_.size(); ++i) {
    double zi = b_[i] * x - a_[i] * y;
    if (i < z_.size()) {
      zi += z_[i];
    }
    z_[i - 1] = zi;
  }
  return y;
}

// --- Disturbance observer --------------------------------------------------

DisturbanceObserver::DisturbanceObserver(const DobParams& params,
                                         const NominalMotor& nominal, double dt)
    : params_(params), nominal_(nominal), dt_(dt) {
  build();
}

std::vector<double> DisturbanceObserver::q_denominator() const {
  const double tq = params_.tau_q;
  if (params_.order == 1) {
    return {tq, 1.0};
  }
  if (params_.order == 2) {
    return {tq * tq, std::sqrt(2.0) * tq, 1.0};
  }
  throw std::invalid_argument("DOB order must be 1 or 2");
}

void DisturbanceObserver::build() {
  const std::vector<double> dq = q_denominator();
  // A(s) = (Jn s + Bn) / D_Q(s), applied to omega.
  filter_omega_ = make_bilinear({nominal_.Jn, nominal_.Bn}, dq, dt_);
  // B(s) = 1 / D_Q(s), applied to tau_cmd.
  filter_tau_ = make_bilinear({1.0}, dq, dt_);
  d_hat_ = 0.0;
}

void DisturbanceObserver::reset() {
  filter_omega_.reset();
  filter_tau_.reset();
  d_hat_ = 0.0;
}

void DisturbanceObserver::set_tau_q(double tau_q) {
  params_.tau_q = tau_q;
  build();
}

double DisturbanceObserver::update(double omega_meas, double tau_cmd) {
  if (!params_.enabled) {
    d_hat_ = 0.0;
    return 0.0;
  }
  d_hat_ = filter_omega_.step(omega_meas) - filter_tau_.step(tau_cmd);
  return d_hat_;
}

}  // namespace omni_sim

#pragma once

// Control primitives: a PID controller and a disturbance observer (DOB).
//
// Ported from the Python omni_sim (control/pid.py, control/dob.py) and kept
// numerically faithful — same PID structure (derivative-on-measurement, LPF,
// anti-windup) and the same DOB: two proper, Tustin (bilinear) discretized IIR
// filters whose difference reconstructs the disturbance.
//
//     d_hat = Q(s) [ P_n^{-1}(s) omega - tau_cmd ]
//           = A(omega) - B(tau_cmd),
//     A(s) = (Jn s + Bn) / D_Q(s),   B(s) = 1 / D_Q(s)
//
// P_n^{-1} alone is improper, so it is never realized directly. Q is a low-pass
// of order 1 or 2 (Butterworth) with cutoff time-constant tau_q.
//
// All loops run under zero-order hold at a fixed control period dt.

#include <limits>
#include <vector>

namespace omni_sim {

// --- PID -------------------------------------------------------------------

struct PidParams {
  double kp = 1.0;
  double ki = 0.0;
  double kd = 0.0;
  double output_min = -std::numeric_limits<double>::infinity();
  double output_max = std::numeric_limits<double>::infinity();
  double derivative_lpf_tau = 0.0;  // 0 disables the derivative filter
  double anti_windup_kb = 0.0;      // back-calculation gain (0 => clamp only)
};

class Pid {
 public:
  Pid(const PidParams& params, double dt);

  void reset();
  double update(double setpoint, double measurement);

 private:
  PidParams params_;
  double dt_;

  double integral_ = 0.0;
  double previous_measurement_ = 0.0;
  bool has_previous_ = false;
  double derivative_filtered_ = 0.0;
};

// --- IIR filter (used by the DOB) ------------------------------------------

// Single-input single-output IIR, Direct Form II transposed. b and a are the
// discrete numerator/denominator; a[0] is normalized to 1 on construction.
class IirFilter {
 public:
  IirFilter() = default;
  IirFilter(std::vector<double> b, std::vector<double> a);

  void reset();
  double step(double x);

 private:
  std::vector<double> b_;
  std::vector<double> a_;
  std::vector<double> z_;
};

// --- Disturbance observer --------------------------------------------------

// First-order nominal model P_n(s) = 1 / (Jn s + Bn): what the controller
// believes the (single-axis) plant is.
struct NominalMotor {
  double Jn = 5.0e-4;  // nominal inertia
  double Bn = 1.0e-4;  // nominal damping
};

struct DobParams {
  bool enabled = true;
  double tau_q = 5.0e-3;
  int order = 1;  // 1 or 2 (Butterworth Q)
};

class DisturbanceObserver {
 public:
  DisturbanceObserver(const DobParams& params, const NominalMotor& nominal,
                      double dt);

  void reset();
  // Change the cutoff at runtime (rebuilds and resets the filters).
  void set_tau_q(double tau_q);

  double d_hat() const { return d_hat_; }

  // One control step: measured velocity and the commanded input. Returns d_hat.
  double update(double omega_meas, double tau_cmd);

 private:
  void build();
  std::vector<double> q_denominator() const;

  DobParams params_;
  NominalMotor nominal_;
  double dt_;
  double d_hat_ = 0.0;

  IirFilter filter_omega_;
  IirFilter filter_tau_;
};

}  // namespace omni_sim

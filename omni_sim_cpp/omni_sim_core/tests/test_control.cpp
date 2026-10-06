#include "omni_sim/control.hpp"

#include <gtest/gtest.h>

#include <cmath>

// Ports the spirit of the Python acceptance tests: the PID drives a first-order
// motor to its setpoint, and the DOB's d_hat converges to an injected load
// disturbance. A single-axis first-order plant is simulated inline.

namespace omni_sim {
namespace {

// Jn * omega_dot = tau + d - Bn * omega, integrated with small fixed Euler
// substeps over one control period.
double advance_first_order(double omega, double tau, double disturbance,
                           double Jn, double Bn, double control_dt) {
  const int substeps = 50;
  const double h = control_dt / substeps;
  double state = omega;
  for (int i = 0; i < substeps; ++i) {
    const double omega_dot = (tau + disturbance - Bn * state) / Jn;
    state += omega_dot * h;
  }
  return state;
}

// --- IIR / bilinear sanity -------------------------------------------------

TEST(DobFilterTest, TauFilterHasUnitDcGain) {
  // B(s) = 1 / D_Q has DC gain 1: a sustained constant input settles to itself.
  DobParams params;
  params.tau_q = 5.0e-3;
  NominalMotor nominal{5.0e-4, 1.0e-4};
  DisturbanceObserver dob(params, nominal, 1.0e-3);

  // Feed omega = 0 and tau = 1 for long enough to settle; d_hat -> -B(1) = -1.
  double d_hat = 0.0;
  for (int i = 0; i < 2000; ++i) {
    d_hat = dob.update(0.0, 1.0);
  }
  EXPECT_NEAR(d_hat, -1.0, 1e-3);
}

// --- PID -------------------------------------------------------------------

TEST(PidTest, DrivesFirstOrderPlantToSetpoint) {
  const double Jn = 5.0e-4;
  const double Bn = 1.0e-4;
  const double dt = 1.0e-3;

  PidParams params;
  params.kp = 0.02;
  params.ki = 0.5;
  params.kd = 0.0;
  Pid pid(params, dt);

  const double setpoint = 10.0;  // rad/s
  double omega = 0.0;
  for (int i = 0; i < 5000; ++i) {  // 5 s
    const double tau = pid.update(setpoint, omega);
    omega = advance_first_order(omega, tau, 0.0, Jn, Bn, dt);
  }
  EXPECT_NEAR(omega, setpoint, 0.05);
}

TEST(PidTest, RespectsOutputSaturation) {
  PidParams params;
  params.kp = 100.0;
  params.output_min = -0.5;
  params.output_max = 0.5;
  Pid pid(params, 1.0e-3);

  const double output = pid.update(1000.0, 0.0);
  EXPECT_LE(output, 0.5);
  EXPECT_GE(output, -0.5);
}

// --- DOB convergence -------------------------------------------------------

TEST(DobTest, DHatConvergesToStepDisturbance) {
  const double Jn = 5.0e-4;
  const double Bn = 1.0e-4;
  const double dt = 1.0e-3;

  DobParams params;
  params.tau_q = 3.0e-3;
  params.order = 1;
  NominalMotor nominal{Jn, Bn};
  DisturbanceObserver dob(params, nominal, dt);

  const double disturbance = 0.05;  // step load [N*m]
  const double tau_cmd = 0.02;      // constant drive
  double omega = 0.0;
  double d_hat = 0.0;
  for (int i = 0; i < 4000; ++i) {  // 4 s
    d_hat = dob.update(omega, tau_cmd);
    omega = advance_first_order(omega, tau_cmd, disturbance, Jn, Bn, dt);
  }
  // The DOB reconstructs the load it cannot see in the nominal model.
  EXPECT_NEAR(d_hat, disturbance, 2e-3);
}

TEST(DobTest, SecondOrderAlsoConverges) {
  const double Jn = 5.0e-4;
  const double Bn = 1.0e-4;
  const double dt = 1.0e-3;

  DobParams params;
  params.tau_q = 3.0e-3;
  params.order = 2;
  NominalMotor nominal{Jn, Bn};
  DisturbanceObserver dob(params, nominal, dt);

  const double disturbance = -0.03;
  const double tau_cmd = 0.01;
  double omega = 0.0;
  double d_hat = 0.0;
  for (int i = 0; i < 4000; ++i) {
    d_hat = dob.update(omega, tau_cmd);
    omega = advance_first_order(omega, tau_cmd, disturbance, Jn, Bn, dt);
  }
  EXPECT_NEAR(d_hat, disturbance, 2e-3);
}

TEST(DobTest, DisabledReturnsZero) {
  DobParams params;
  params.enabled = false;
  NominalMotor nominal{5.0e-4, 1.0e-4};
  DisturbanceObserver dob(params, nominal, 1.0e-3);

  EXPECT_DOUBLE_EQ(dob.update(3.0, 1.5), 0.0);
}

TEST(DobTest, SetTauQReconvergesAtNewCutoff) {
  const double Jn = 5.0e-4;
  const double Bn = 1.0e-4;
  const double dt = 1.0e-3;

  DobParams params;
  NominalMotor nominal{Jn, Bn};
  DisturbanceObserver dob(params, nominal, dt);

  dob.set_tau_q(8.0e-3);  // slower cutoff

  const double disturbance = 0.04;
  const double tau_cmd = 0.015;
  double omega = 0.0;
  double d_hat = 0.0;
  for (int i = 0; i < 6000; ++i) {
    d_hat = dob.update(omega, tau_cmd);
    omega = advance_first_order(omega, tau_cmd, disturbance, Jn, Bn, dt);
  }
  EXPECT_NEAR(d_hat, disturbance, 2e-3);
}

}  // namespace
}  // namespace omni_sim

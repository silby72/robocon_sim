import numpy as np
import pytest

from omni_sim_core.clock import ClockConfig
from omni_sim_core.plant.motor import MotorParams
from omni_sim_core.plant.nominal import NominalModel, nominal_from_true
from omni_sim_core.control.pid import PIDParams
from omni_sim_core.control.dob import DOBParams


@pytest.fixture
def clean_true_params():
    """A clean (friction-free) true motor for the control-theory tests."""
    return MotorParams(
        n_motors=1,
        inertia_kgm2=5.0e-4,
        damping_nms=1.0e-4,
        torque_constant_nm_a=0.5,
        friction_coulomb_nm=0.0,
        friction_static_nm=0.0,
        enable_cogging=False,
    )


@pytest.fixture
def clock_cfg():
    return ClockConfig(dt_sim=1.0e-4, dt_motor=1.0e-3, dt_nav=2.0e-2)

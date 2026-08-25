"""Nominal model -- what the controller *believes* the motor is.

    P_n(s) = 1 / (J_n s + B_n)

It deliberately contains no friction, cogging or saturation: those live only in
the true plant and are precisely what the DOB is meant to reconstruct as a
disturbance. Read from a *separate* YAML from the true plant.
"""
from __future__ import annotations

from dataclasses import dataclass

from .motor import MotorParams


@dataclass
class NominalModel:
    """First-order nominal motor model used by the controller/DOB."""

    inertia_kgm2: float = 5.0e-4  # J_n
    damping_nms: float = 1.0e-4   # B_n

    @property
    def Jn(self) -> float:
        return self.inertia_kgm2

    @property
    def Bn(self) -> float:
        return self.damping_nms


def nominal_from_true(true_params: MotorParams, j_ratio: float = 1.0,
                      b_ratio: float = 1.0) -> NominalModel:
    """Build a nominal model as a scaled version of the true parameters.

    ``J_n = j_ratio * J_true`` and ``B_n = b_ratio * B_true``. This is the
    primary knob for injecting a known amount of model error into experiments.
    """
    return NominalModel(
        inertia_kgm2=j_ratio * true_params.inertia_kgm2,
        damping_nms=b_ratio * true_params.damping_nms,
    )

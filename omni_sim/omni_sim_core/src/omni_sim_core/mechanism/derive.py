"""Datasheet -> physical constants.

Input : a motor ``Datasheet`` (+ optional measured ``Physical`` overrides) and
the drivetrain figures (gear ratio, wheel-side load inertia).
Output: a ``DerivedMotor`` holding SI constants (Kt, Ke, R, tau_stall,
omega_noload, reflected inertia, viscous damping) and a record of *which* fields
were estimated rather than taken/derived from the datasheet.

Estimated and measured values are never silently mixed: every estimated field is
listed in ``DerivedMotor.estimated`` and gets a ``# ESTIMATED`` comment when
written out by ``build.py``. A rough rotor-inertia estimate also emits a warning.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

from .schema import Datasheet, Physical

# Rough spin-up time assumed when the rotor inertia is unknown. This is a
# deliberately crude placeholder (a typical small BLDC free-accelerates in tens
# of ms); it exists only so an unknown-J motor still simulates. Any value
# derived from it is flagged ESTIMATED.
ROTOR_SPINUP_TIME_S = 0.05

TWO_PI = 2.0 * math.pi
RPM_TO_RAD_S = TWO_PI / 60.0   # [rev/min] -> [rad/s]


@dataclass
class DerivedMotor:
    torque_constant_nm_a: float          # Kt
    back_emf_v_s: float                  # Ke  (== Kt in SI)
    resistance_ohm: float                # R
    inductance_h: Optional[float]        # L   (None unless measured)
    stall_torque_nm: float               # tau_stall
    no_load_speed_rad_s: float           # omega_noload
    rotor_inertia_kgm2: float            # J_rotor
    reflected_inertia_kgm2: float        # J_reflected (motor-shaft)
    load_contribution_kgm2: float        # (J_wheel + m_share r^2) / n^2
    damping_nms: float                   # B  (viscous, lumped)
    estimated: set[str] = field(default_factory=set)


def derive_motor(datasheet: Datasheet, physical: Physical, *,
                 gear_ratio: float,
                 load_inertia_wheel_side_kgm2: float,
                 spinup_time_s: float = ROTOR_SPINUP_TIME_S) -> DerivedMotor:
    """Derive motor-shaft physical constants.

    ``load_inertia_wheel_side_kgm2`` is the wheel-side load inertia
    (J_wheel + m_share * r_wheel^2); it is reflected to the motor shaft by 1/n^2.
    """
    ds = datasheet
    est: set[str] = set()

    # Kt = 60 / (2*pi*KV)   [Nm/A], KV in rpm/V.
    #   Derivation: Ke_SI [V*s/rad] = 1 / (KV * 2*pi/60); in SI Kt == Ke.
    kt = 60.0 / (TWO_PI * ds.kv_rpm_per_v)
    ke = kt                                   # SI: numerically identical
    if physical.torque_constant_nm_a is not None:
        kt = physical.torque_constant_nm_a
    if physical.back_emf_v_s is not None:
        ke = physical.back_emf_v_s

    # R = V_rated / I_stall   [ohm]  (terminal resistance at stall, L*di/dt = 0)
    r = ds.rated_voltage_v / ds.stall_current_a
    if physical.resistance_ohm is not None:
        r = physical.resistance_ohm

    inductance = physical.inductance_h        # only if measured

    # tau_stall = Kt * (I_stall - I_noload)   [Nm]
    tau_stall = kt * (ds.stall_current_a - ds.no_load_current_a)

    # omega_noload = KV * V_rated * 2*pi/60   [rad/s]
    omega_noload = ds.kv_rpm_per_v * ds.rated_voltage_v * RPM_TO_RAD_S

    # Rotor inertia: measured > datasheet > rough estimate.
    if physical.rotor_inertia_kgm2 is not None:
        j_rotor = physical.rotor_inertia_kgm2
    elif ds.rotor_inertia_kgm2 is not None:
        j_rotor = ds.rotor_inertia_kgm2
    else:
        # J_rotor ~ tau_stall / (omega_noload / t_acc): torque over angular
        # acceleration needed to reach no-load speed in t_acc. Crude by design.
        j_rotor = tau_stall / (omega_noload / spinup_time_s)
        est.add("rotor_inertia_kgm2")
        warnings.warn(
            f"rotor inertia unknown; estimated J_rotor={j_rotor:.3e} kg m^2 "
            f"from stall torque and a {spinup_time_s*1e3:.0f} ms spin-up "
            "assumption -- provide 'rotor_inertia_kgm2' for accuracy",
            stacklevel=2)

    # Reflected to the motor shaft: J_ref = J_rotor + load_wheel_side / n^2.
    load_contribution = load_inertia_wheel_side_kgm2 / (gear_ratio ** 2)
    j_reflected = j_rotor + load_contribution
    if "rotor_inertia_kgm2" in est:
        est.add("reflected_inertia_kgm2")

    # Viscous damping lumped from the no-load operating point: at no load the
    # electromagnetic torque Kt*I_noload is spent on friction; attributing it
    # all to viscous damping gives B ~ Kt*I_noload / omega_noload. This is an
    # approximation (ignores Coulomb friction), hence flagged estimated.
    damping = kt * ds.no_load_current_a / omega_noload
    est.add("damping_nms")

    return DerivedMotor(
        torque_constant_nm_a=kt,
        back_emf_v_s=ke,
        resistance_ohm=r,
        inductance_h=inductance,
        stall_torque_nm=tau_stall,
        no_load_speed_rad_s=omega_noload,
        rotor_inertia_kgm2=j_rotor,
        reflected_inertia_kgm2=j_reflected,
        load_contribution_kgm2=load_contribution,
        damping_nms=damping,
        estimated=est,
    )

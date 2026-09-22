"""Hand-off formats for tools outside omni_sim.

Nothing here is used by the simulator itself. These are one-way exports, so
that a number derived once -- from a datasheet, through a gear ratio, past a
saturation limit -- reaches the other tool as that number rather than being
re-typed, re-derived, or defaulted.
"""
from .gazebo_actuator import (ExportError, WheelLimits, export_actuator_config,
                              wheel_limits)

__all__ = ["ExportError", "WheelLimits", "export_actuator_config",
           "wheel_limits"]

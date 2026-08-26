"""Mechanism parameter layer.

Turns a human-authored robot description (chassis geometry + actuator
datasheets) into the physical plant parameters the simulator consumes. Inputs
live in ``config/robot/*.yaml`` and ``config/presets/motors/*.yaml``; the output
is ``config/generated/plant_*.yaml`` plus a drive jacobian for cross-checking.

This package is ROS-free and GUI-free. It uses ``ruamel.yaml`` (a runtime
dependency of omni_sim_core) so authored comments and key order survive a
load/save round-trip. Importing this subpackage is optional: ``import
omni_sim_core`` never pulls it in, so the rest of the core still imports with
neither ruamel nor a GUI toolkit present.
"""

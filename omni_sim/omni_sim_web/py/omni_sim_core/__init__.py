"""Omni-directional robot simulator core (ROS-independent).

This package deliberately imports *no* rclpy. It runs standalone under pytest
and parameter sweeps. See docs/spec for the layered architecture:

    body  <- jacobian <- motor    (true plant, numeric integration)
    nominal model                 (what the controller believes)
    disturbance / sensors / env   (measurement + environment)

The single most important invariant: the *true* plant and the *nominal* model
are configured from independent files and never share state.
"""

__version__ = "1.0.0"

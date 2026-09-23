"""Self-localization: correcting the dead reckoning that now genuinely drifts.

``plant/odometry.py`` gave odometry a real error model (1.86 % of path length
at 2 % drive slip). This package is the other half -- a LiDAR likelihood field
and a particle filter that consume the maps ``field/slicer.py`` already
produces for the purpose.

rclpy-free and deterministic given a seed, like the rest of the core.
"""
from .likelihood_field import LikelihoodField, LikelihoodFieldParams
from .mcl import MCLParams, ParticleFilter

__all__ = ["LikelihoodField", "LikelihoodFieldParams", "MCLParams",
           "ParticleFilter"]

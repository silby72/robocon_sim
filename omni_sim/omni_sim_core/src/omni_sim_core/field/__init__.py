"""Field definition layer (依頼 §3): one spec -> occupancy grids + true geometry.

No 3D. Height-annotated 2D primitives are sliced directly into map_server grids
(for planning) and kept as exact shapes (for evaluation). rclpy/GUI free.
"""
from __future__ import annotations

from .primitives import Box, Cylinder, Primitive, Segment, FREE, OCC, UNKNOWN
from .spec import FieldSpec, LayerSpec
from .layered_field import LayeredField

__all__ = [
    "Primitive", "Box", "Cylinder", "Segment", "FREE", "OCC", "UNKNOWN",
    "FieldSpec", "LayerSpec", "LayeredField",
]

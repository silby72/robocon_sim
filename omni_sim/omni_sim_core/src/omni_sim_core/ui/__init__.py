"""Terminal live dashboard, styled after serial2can's graphical_ui / console_ui.

Pure stdlib (no rclpy, no numpy dependency for rendering). Reuses the same
ANSI 256-colour palette, fixed 100-column panels, braille spinner and
east-asian-width-aware padding as the serial_bridge / ros2can dashboards so the
whole toolchain looks consistent.
"""
from .console import ConsoleDashboard, colorize, C

__all__ = ["ConsoleDashboard", "colorize", "C"]

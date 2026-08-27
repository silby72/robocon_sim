"""Qt binding shim.

Prefer PySide6 (the spec's chosen binding); fall back to PyQt5 if PySide6 is not
installed. Application code imports Qt only through this module, so the binding
is swappable without touching any GUI logic.
"""
from __future__ import annotations

try:  # spec target
    from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401
    from PySide6.QtCore import Signal, Slot        # noqa: F401
    BINDING = "PySide6"
except ImportError:  # pragma: no cover - depends on what is installed
    from PyQt5 import QtCore, QtGui, QtWidgets      # noqa: F401
    from PyQt5.QtCore import pyqtSignal as Signal   # noqa: F401
    from PyQt5.QtCore import pyqtSlot as Slot       # noqa: F401
    BINDING = "PyQt5"

"""QGraphicsView placement canvas for the chassis (top-down plan view).

Renders the footprint, drive wheels (with a drive-axis arrow and, when selected,
a rotate handle), standalone odometry units, the centre of mass and reference
axes. Wheels and odometry units are draggable and snap to a grid.

Signals let the numeric side panel stay in sync (bidirectional editing) and let
the page record undo checkpoints:

    itemSelected(kind, index)         kind in {"wheel", "odom", "none"}
    itemMoved(kind, index, x_m, y_m)
    axisChanged(kind, index, theta_rad)   # from the rotate handle
    interactionFinished()             # mouse released -> commit an undo step

Scene units are millimetres; world +y (left, REP-103) is drawn upward, so
``world (x_m, y_m) -> scene (x_m*1000, -y_m*1000)``.
"""
from __future__ import annotations

import math

from ..qt import QtCore, QtGui, QtWidgets, Signal

MM = 1000.0
GRID_MM = 10.0

_WHEEL_PEN = QtGui.QPen(QtGui.QColor("#00afd7"), 3)
_WHEEL_BRUSH = QtGui.QBrush(QtGui.QColor(0, 175, 215, 60))
_ODOM_PEN = QtGui.QPen(QtGui.QColor("#ff5fd7"), 3)
_ODOM_BRUSH = QtGui.QBrush(QtGui.QColor(255, 95, 215, 55))
_ARROW_PEN = QtGui.QPen(QtGui.QColor("#00d75f"), 3)
_HANDLE_PEN = QtGui.QPen(QtGui.QColor("#00d75f"), 2)
_HANDLE_BRUSH = QtGui.QBrush(QtGui.QColor("#00d75f"))
_COM_PEN = QtGui.QPen(QtGui.QColor("#ff5f5f"), 3)


def _snap(v_mm: float) -> float:
    return round(v_mm / GRID_MM) * GRID_MM


class _MovableItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, canvas, kind: str, index: int, radius_mm: float,
                 pen, brush) -> None:
        super().__init__(-radius_mm, -radius_mm, 2 * radius_mm, 2 * radius_mm)
        self._canvas = canvas
        self.kind = kind
        self.index = index
        self.setPen(pen)
        self.setBrush(brush)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            return QtCore.QPointF(_snap(value.x()), _snap(value.y()))
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._canvas._item_moved(self.kind, self.index, self.pos())
        return super().itemChange(change, value)


class _RotateHandle(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, canvas, r=7.0) -> None:
        super().__init__(-r, -r, 2 * r, 2 * r)
        self._canvas = canvas
        self.setPen(_HANDLE_PEN)
        self.setBrush(_HANDLE_BRUSH)
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable
                      | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(20)
        self.setCursor(QtCore.Qt.CrossCursor)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._canvas._handle_moved(self.pos())
        return super().itemChange(change, value)


class ChassisCanvas(QtWidgets.QGraphicsView):
    itemSelected = Signal(str, int)
    itemMoved = Signal(str, int, float, float)
    axisChanged = Signal(str, int, float)
    interactionFinished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setBackgroundBrush(QtGui.QColor("#1c1c1c"))
        self._doc = None
        self._wheel_items: list[_MovableItem] = []
        self._arrow_items: list[QtWidgets.QGraphicsLineItem] = []
        self._odom_items: list[_MovableItem] = []
        self._handle: _RotateHandle | None = None
        self._sel_kind = "none"
        self._sel_index = -1
        self._scene.selectionChanged.connect(self._on_selection)

    # -- coordinate helpers ----------------------------------------------
    @staticmethod
    def w2s(x_m, y_m):
        return QtCore.QPointF(x_m * MM, -y_m * MM)

    @staticmethod
    def s2w(p):
        return (p.x() / MM, -p.y() / MM)

    # -- build -----------------------------------------------------------
    def set_doc(self, doc) -> None:
        self._doc = doc
        self.rebuild()

    def rebuild(self) -> None:
        self._scene.blockSignals(True)
        self._scene.clear()
        self._wheel_items, self._arrow_items, self._odom_items = [], [], []
        self._handle = None
        if self._doc is None:
            self._scene.blockSignals(False)
            return

        size_m = float(self._doc["footprint"]["size_m"])
        half = size_m * MM / 2.0
        self._draw_grid(half)
        self._scene.addRect(-half, -half, size_m * MM, size_m * MM,
                            QtGui.QPen(QtGui.QColor("#8a8a8a"), 2))
        self._scene.addLine(-half, 0, half, 0, QtGui.QPen(QtGui.QColor("#3a3a3a"), 1))
        self._scene.addLine(0, -half, 0, half, QtGui.QPen(QtGui.QColor("#3a3a3a"), 1))

        for i, w in enumerate(self._doc["drive_wheels"]):
            self._add_wheel(i, w)
        for i, o in enumerate(self._doc.get("odometry", []) or []):
            self._add_odom(i, o)
        self._add_com(self._doc["center_of_mass"])

        margin = half * 0.3 + 60
        self._scene.setSceneRect(-half - margin, -half - margin,
                                 size_m * MM + 2 * margin, size_m * MM + 2 * margin)
        self._scene.blockSignals(False)
        self._apply_selection()
        self.fit()

    def _draw_grid(self, half: float) -> None:
        pen = QtGui.QPen(QtGui.QColor("#2a2a2a"), 1)
        step, n = 50.0, int(half // 50.0) + 2
        for k in range(-n, n + 1):
            x = k * step
            self._scene.addLine(x, -half, x, half, pen)
            self._scene.addLine(-half, x, half, x, pen)

    def _add_wheel(self, index: int, w) -> None:
        r_mm = float(w["radius_m"]) * MM
        item = _MovableItem(self, "wheel", index, r_mm, _WHEEL_PEN, _WHEEL_BRUSH)
        item.setPos(self.w2s(float(w["position_m"][0]), float(w["position_m"][1])))
        theta = float(w["drive_axis_rad"])
        length = r_mm * 1.8
        arrow = QtWidgets.QGraphicsLineItem(
            0, 0, length * math.cos(theta), -length * math.sin(theta), item)
        arrow.setPen(_ARROW_PEN)
        label = QtWidgets.QGraphicsSimpleTextItem(str(w.get("id", index)), item)
        label.setBrush(QtGui.QBrush(QtGui.QColor("#d0d0d0")))
        label.setPos(r_mm * 0.4, -r_mm * 0.4)
        self._scene.addItem(item)
        self._wheel_items.append(item)
        self._arrow_items.append(arrow)

    def _add_odom(self, index: int, o) -> None:
        r_mm = float(o["radius_m"]) * MM if o.get("radius_m") else 25.0
        item = _MovableItem(self, "odom", index, r_mm, _ODOM_PEN, _ODOM_BRUSH)
        item.setPos(self.w2s(float(o["position_m"][0]), float(o["position_m"][1])))
        if o.get("measure_axis_rad") is not None:  # dead_wheel: draw its roll axis
            theta = float(o["measure_axis_rad"])
            length = r_mm * 1.6
            tick = QtWidgets.QGraphicsLineItem(
                -length * math.cos(theta), length * math.sin(theta),
                length * math.cos(theta), -length * math.sin(theta), item)
            tick.setPen(_ODOM_PEN)
        label = QtWidgets.QGraphicsSimpleTextItem(str(o.get("id", index)), item)
        label.setBrush(QtGui.QBrush(QtGui.QColor("#ffa8e6")))
        label.setPos(r_mm * 0.4, -r_mm * 0.4)
        self._scene.addItem(item)
        self._odom_items.append(item)

    def _add_com(self, com) -> None:
        p = self.w2s(float(com["position_m"][0]), float(com["position_m"][1]))
        s = 12.0
        self._scene.addLine(p.x() - s, p.y(), p.x() + s, p.y(), _COM_PEN)
        self._scene.addLine(p.x(), p.y() - s, p.x(), p.y() + s, _COM_PEN)
        self._scene.addEllipse(p.x() - s, p.y() - s, 2 * s, 2 * s, _COM_PEN)

    # -- rotate handle for the selected drive wheel ----------------------
    def _update_handle(self) -> None:
        if self._handle is not None:
            self._scene.removeItem(self._handle)
            self._handle = None
        if self._sel_kind != "wheel" or self._sel_index < 0:
            return
        item = self._wheel_items[self._sel_index]
        w = self._doc["drive_wheels"][self._sel_index]
        theta = float(w["drive_axis_rad"])
        r_mm = float(w["radius_m"]) * MM * 1.8
        c = item.pos()
        self._handle = _RotateHandle(self)
        self._handle.setPos(c.x() + r_mm * math.cos(theta),
                            c.y() - r_mm * math.sin(theta))
        self._scene.addItem(self._handle)

    def _handle_moved(self, pos) -> None:
        if self._sel_kind != "wheel" or self._sel_index < 0:
            return
        c = self._wheel_items[self._sel_index].pos()
        dx, dy = pos.x() - c.x(), pos.y() - c.y()
        theta = math.atan2(-dy, dx)              # scene y is flipped
        # live-update the arrow so rotation is visible while dragging
        arrow = self._arrow_items[self._sel_index]
        length = arrow.line().length() or (float(
            self._doc["drive_wheels"][self._sel_index]["radius_m"]) * MM * 1.8)
        arrow.setLine(0, 0, length * math.cos(theta), -length * math.sin(theta))
        self.axisChanged.emit("wheel", self._sel_index, theta)

    # -- interaction -----------------------------------------------------
    def _item_moved(self, kind, index, scene_pos) -> None:
        x_m, y_m = self.s2w(scene_pos)
        if kind == "wheel" and self._handle is not None and index == self._sel_index:
            self._update_handle()
        self.itemMoved.emit(kind, index, x_m, y_m)

    def _on_selection(self) -> None:
        sel = [it for it in self._scene.selectedItems()
               if isinstance(it, _MovableItem)]
        if sel:
            self._sel_kind, self._sel_index = sel[0].kind, sel[0].index
        else:
            self._sel_kind, self._sel_index = "none", -1
        self._update_handle()
        self.itemSelected.emit(self._sel_kind, self._sel_index)

    def _apply_selection(self) -> None:
        items = self._wheel_items if self._sel_kind == "wheel" else self._odom_items
        for it in items:
            if it.index == self._sel_index:
                it.setSelected(True)
        self._update_handle()

    def select(self, kind: str, index: int) -> None:
        self._sel_kind, self._sel_index = kind, index
        self._apply_selection()

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self.interactionFinished.emit()

    def fit(self) -> None:
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit()

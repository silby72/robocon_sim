"""Chassis page: edits config/robot/chassis.yaml.

Left: the placement canvas (drag wheels/odometry, rotate the drive axis with the
green handle). Right: numeric fields kept in sync with the canvas. Top: a palette
to add/delete units, mirror or rotate-copy the selected wheel, and undo/redo.
A live readout shows the drive jacobian condition number. Saving shows a diff and
writes SI-unit YAML with comments preserved.

Undo/redo is snapshot based: each committed edit pushes the whole document (as
comment-preserving YAML text) onto a history stack, which is simple and robust.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from omni_sim_core.mechanism.jacobian import (
    build_drive_jacobian, CONDITION_NUMBER_LIMIT)
from omni_sim_core.mechanism.schema import SchemaError

from ..qt import QtCore, QtGui, QtWidgets
from ..io import yaml_io
from ..widgets.chassis_canvas import ChassisCanvas


def _spin(lo, hi, decimals=1, step=1.0, suffix="") -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    return s


class ChassisPage(QtWidgets.QWidget):
    def __init__(self, path, parent=None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.doc = yaml_io.load_chassis_doc(self.path)
        # actuator ids from the sibling actuators.yaml (for the ref dropdown +
        # dangling-reference checks). gui->core one-way, read-only here.
        self._actuator_names = yaml_io.load_actuator_names(
            self.path.parent / "actuators.yaml")
        self._disk_text = yaml_io.render_doc(self.doc)
        self._history = [self._disk_text]
        self._hp = 0
        self._updating = False
        self._sel_kind, self._sel_index = "none", -1

        self.canvas = ChassisCanvas()
        self.canvas.itemSelected.connect(self._on_selected)
        self.canvas.itemMoved.connect(self._on_moved)
        self.canvas.axisChanged.connect(self._on_axis)
        self.canvas.interactionFinished.connect(self._commit)

        outer = QtWidgets.QVBoxLayout(self)
        outer.addLayout(self._build_toolbar())
        content = QtWidgets.QHBoxLayout()
        content.addWidget(self.canvas, 3)
        content.addWidget(self._build_panel(), 2)
        outer.addLayout(content)

        self.canvas.set_doc(self.doc)
        self._refresh_validation()
        self._update_history_buttons()

    # -- toolbar ---------------------------------------------------------
    def _build_toolbar(self) -> QtWidgets.QHBoxLayout:
        bar = QtWidgets.QHBoxLayout()

        def button(text, slot, tip=""):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            if tip:
                b.setToolTip(tip)
            bar.addWidget(b)
            return b

        button("＋Wheel", self._add_wheel, "add a drive wheel")
        button("＋Odom", self._add_odom, "add a standalone odometry (dead) wheel")
        self._del_btn = button("Delete", self._delete_selected, "delete selected unit")
        bar.addSpacing(12)
        self._mx = button("Mirror X", lambda: self._mirror("x"), "mirror selected wheel across x")
        self._my = button("Mirror Y", lambda: self._mirror("y"), "mirror selected wheel across y")
        self._r90 = button("Rot-copy 90°", lambda: self._rotate_copy(math.pi / 2))
        self._r120 = button("Rot-copy 120°", lambda: self._rotate_copy(2 * math.pi / 3))
        bar.addSpacing(12)
        self._undo_btn = button("↶ Undo", self._undo)
        self._redo_btn = button("↷ Redo", self._redo)
        bar.addStretch(1)

        QtWidgets.QShortcut(QtGui.QKeySequence.Undo, self, self._undo)
        QtWidgets.QShortcut(QtGui.QKeySequence.Redo, self, self._redo)
        return bar

    # -- side panel ------------------------------------------------------
    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)

        form = QtWidgets.QFormLayout()
        self.f_size = _spin(50, 3000, 0, 10, " mm")
        self.f_size.setValue(float(self.doc["footprint"]["size_m"]) * 1000)
        self.f_size.editingFinished.connect(self._apply_footprint)
        form.addRow("footprint size", self.f_size)

        com = self.doc["center_of_mass"]
        self.f_mass = _spin(0.1, 200, 2, 0.5, " kg"); self.f_mass.setValue(float(com["mass_kg"]))
        self.f_izz = _spin(0.001, 50, 3, 0.01, " kg·m²"); self.f_izz.setValue(float(com["inertia_zz_kgm2"]))
        self.f_comx = _spin(-1500, 1500, 0, 10, " mm"); self.f_comx.setValue(float(com["position_m"][0]) * 1000)
        self.f_comy = _spin(-1500, 1500, 0, 10, " mm"); self.f_comy.setValue(float(com["position_m"][1]) * 1000)
        for w in (self.f_mass, self.f_izz, self.f_comx, self.f_comy):
            w.editingFinished.connect(self._apply_com)
        form.addRow("CoM mass", self.f_mass)
        form.addRow("CoM Izz", self.f_izz)
        form.addRow("CoM x", self.f_comx)
        form.addRow("CoM y", self.f_comy)
        v.addLayout(form)

        # drive wheel group
        self._wheel_box = QtWidgets.QGroupBox("selected drive wheel")
        wf = QtWidgets.QFormLayout(self._wheel_box)
        self.w_id = QtWidgets.QLineEdit()
        self.w_x = _spin(-1500, 1500, 0, 10, " mm")
        self.w_y = _spin(-1500, 1500, 0, 10, " mm")
        self.w_axis = _spin(-360, 360, 1, 5, " °")
        self.w_radius = _spin(5, 300, 1, 1, " mm")
        self.w_gear = _spin(0.1, 100, 2, 0.5, " :1")
        self.w_ref = QtWidgets.QComboBox()
        self.w_ref.setEditable(True)          # dropdown of known actuators, still typeable
        self.w_ref.addItems(self._actuator_names)
        self.w_reverse = QtWidgets.QCheckBox("reverse")
        for lbl, wdg in (("id", self.w_id), ("x", self.w_x), ("y", self.w_y),
                         ("drive axis", self.w_axis), ("radius", self.w_radius),
                         ("gear ratio", self.w_gear), ("actuator_ref", self.w_ref),
                         ("", self.w_reverse)):
            wf.addRow(lbl, wdg)
        for wdg in (self.w_x, self.w_y, self.w_axis, self.w_radius, self.w_gear):
            wdg.editingFinished.connect(self._apply_wheel)
        self.w_id.editingFinished.connect(self._apply_wheel)
        self.w_ref.activated.connect(self._apply_wheel)
        self.w_ref.lineEdit().editingFinished.connect(self._apply_wheel)
        self.w_reverse.toggled.connect(self._apply_wheel)
        self._wheel_box.setVisible(False)
        v.addWidget(self._wheel_box)

        # odometry group
        self._odom_box = QtWidgets.QGroupBox("selected odometry")
        of = QtWidgets.QFormLayout(self._odom_box)
        self.o_id = QtWidgets.QLineEdit()
        self.o_x = _spin(-1500, 1500, 0, 10, " mm")
        self.o_y = _spin(-1500, 1500, 0, 10, " mm")
        self.o_axis = _spin(-360, 360, 1, 5, " °")
        self.o_radius = _spin(5, 200, 1, 1, " mm")
        self.o_cpr = QtWidgets.QSpinBox(); self.o_cpr.setRange(1, 1_000_000); self.o_cpr.setSuffix(" cpr")
        for lbl, wdg in (("id", self.o_id), ("x", self.o_x), ("y", self.o_y),
                         ("measure axis", self.o_axis), ("radius", self.o_radius),
                         ("encoder", self.o_cpr)):
            of.addRow(lbl, wdg)
        for wdg in (self.o_x, self.o_y, self.o_axis, self.o_radius):
            wdg.editingFinished.connect(self._apply_odom)
        self.o_cpr.editingFinished.connect(self._apply_odom)
        self.o_id.editingFinished.connect(self._apply_odom)
        self._odom_box.setVisible(False)
        v.addWidget(self._odom_box)

        self.validation = QtWidgets.QLabel(); self.validation.setWordWrap(True)
        v.addWidget(self.validation)
        v.addStretch(1)
        self.save_btn = QtWidgets.QPushButton("Save chassis.yaml…")
        self.save_btn.clicked.connect(self._on_save)
        v.addWidget(self.save_btn)
        return panel

    # -- helpers ---------------------------------------------------------
    def _wheels(self):
        return self.doc["drive_wheels"]

    def _odos(self):
        return self.doc.get("odometry", []) or []

    def _unique_id(self, prefix, existing):
        n = len(existing)
        ids = {str(e.get("id")) for e in existing}
        while f"{prefix}{n}" in ids:
            n += 1
        return f"{prefix}{n}"

    def _ensure_odometry_list(self):
        if "odometry" not in self.doc or self.doc["odometry"] is None:
            from ruamel.yaml.comments import CommentedSeq
            self.doc["odometry"] = CommentedSeq()
        return self.doc["odometry"]

    # -- palette actions -------------------------------------------------
    def _add_wheel(self):
        ref = self._wheels()[0] if len(self._wheels()) else None
        radius = float(ref["radius_m"]) if ref else 0.0508
        gear = float(ref["gear_ratio"]) if ref else 1.0
        wid = self._unique_id("w", self._wheels())
        self._wheels().append(yaml_io.new_wheel(wid, 0.0, 0.0, 0.0, radius, gear, ""))
        self._after_structural("wheel", len(self._wheels()) - 1)

    def _add_odom(self):
        odos = self._ensure_odometry_list()
        oid = self._unique_id("odo", odos)
        odos.append(yaml_io.new_odometry(oid, 0.0, 0.0, 0.0, 0.029, 4096))
        self._after_structural("odom", len(odos) - 1)

    def _delete_selected(self):
        if self._sel_kind == "wheel" and len(self._wheels()) > 1:
            del self._wheels()[self._sel_index]
        elif self._sel_kind == "odom":
            del self.doc["odometry"][self._sel_index]
        else:
            return
        self._after_structural("none", -1)

    def _mirror(self, axis):
        if self._sel_kind != "wheel":
            return
        src = self._wheels()[self._sel_index]
        x, y = float(src["position_m"][0]), float(src["position_m"][1])
        th = float(src["drive_axis_rad"])
        if axis == "x":           # mirror across x-axis: y -> -y, theta -> -theta
            x2, y2, th2 = x, -y, -th
        else:                      # mirror across y-axis: x -> -x, theta -> pi-theta
            x2, y2, th2 = -x, y, math.pi - th
        wid = self._unique_id("w", self._wheels())
        self._wheels().append(yaml_io.new_wheel(
            wid, x2, y2, th2, float(src["radius_m"]), float(src["gear_ratio"]), ""))
        self._after_structural("wheel", len(self._wheels()) - 1)

    def _rotate_copy(self, angle):
        if self._sel_kind != "wheel":
            return
        src = self._wheels()[self._sel_index]
        x, y = float(src["position_m"][0]), float(src["position_m"][1])
        c, s = math.cos(angle), math.sin(angle)
        x2, y2 = c * x - s * y, s * x + c * y
        th2 = float(src["drive_axis_rad"]) + angle
        wid = self._unique_id("w", self._wheels())
        self._wheels().append(yaml_io.new_wheel(
            wid, x2, y2, th2, float(src["radius_m"]), float(src["gear_ratio"]), ""))
        self._after_structural("wheel", len(self._wheels()) - 1)

    def _after_structural(self, kind, index):
        self.canvas.set_doc(self.doc)
        self.canvas.select(kind, index)
        self._refresh_validation()
        self._commit()

    # -- canvas -> model -------------------------------------------------
    def _on_selected(self, kind, index):
        self._sel_kind, self._sel_index = kind, index
        self._wheel_box.setVisible(kind == "wheel")
        self._odom_box.setVisible(kind == "odom")
        self._del_btn.setEnabled(index >= 0)
        for b in (self._mx, self._my, self._r90, self._r120):
            b.setEnabled(kind == "wheel")
        if kind == "wheel":
            self._populate_wheel(self._wheels()[index])
        elif kind == "odom":
            self._populate_odom(self._odos()[index])

    def _on_moved(self, kind, index, x_m, y_m):
        node = (self._wheels() if kind == "wheel" else self.doc["odometry"])[index]
        node["position_m"][0] = round(x_m, 6)
        node["position_m"][1] = round(y_m, 6)
        if kind == self._sel_kind and index == self._sel_index:
            self._updating = True
            (self.w_x if kind == "wheel" else self.o_x).setValue(x_m * 1000)
            (self.w_y if kind == "wheel" else self.o_y).setValue(y_m * 1000)
            self._updating = False
        self._refresh_validation()

    def _on_axis(self, kind, index, theta):
        self._wheels()[index]["drive_axis_rad"] = round(theta, 6)
        if kind == self._sel_kind and index == self._sel_index:
            self._updating = True
            self.w_axis.setValue(math.degrees(theta))
            self._updating = False

    # -- fields -> model -------------------------------------------------
    def _populate_wheel(self, w):
        self._updating = True
        self.w_id.setText(str(w.get("id", "")))
        self.w_x.setValue(float(w["position_m"][0]) * 1000)
        self.w_y.setValue(float(w["position_m"][1]) * 1000)
        self.w_axis.setValue(math.degrees(float(w["drive_axis_rad"])))
        self.w_radius.setValue(float(w["radius_m"]) * 1000)
        self.w_gear.setValue(float(w["gear_ratio"]))
        self.w_ref.setCurrentText(str(w["actuator_ref"]))
        self.w_reverse.setChecked(bool(w.get("reverse", False)))
        self._updating = False

    def _populate_odom(self, o):
        self._updating = True
        self.o_id.setText(str(o.get("id", "")))
        self.o_x.setValue(float(o["position_m"][0]) * 1000)
        self.o_y.setValue(float(o["position_m"][1]) * 1000)
        self.o_axis.setValue(math.degrees(float(o.get("measure_axis_rad", 0.0) or 0.0)))
        self.o_radius.setValue(float(o.get("radius_m", 0.029) or 0.029) * 1000)
        self.o_cpr.setValue(int(o.get("encoder_cpr", 4096) or 4096))
        self._updating = False

    def _apply_footprint(self):
        if self._updating:
            return
        self.doc["footprint"]["size_m"] = round(self.f_size.value() / 1000, 6)
        self._after_field()

    def _apply_com(self):
        if self._updating:
            return
        com = self.doc["center_of_mass"]
        com["mass_kg"] = round(self.f_mass.value(), 4)
        com["inertia_zz_kgm2"] = round(self.f_izz.value(), 5)
        com["position_m"][0] = round(self.f_comx.value() / 1000, 6)
        com["position_m"][1] = round(self.f_comy.value() / 1000, 6)
        self._after_field()

    def _apply_wheel(self):
        if self._updating or self._sel_kind != "wheel":
            return
        w = self._wheels()[self._sel_index]
        w["id"] = self.w_id.text()
        w["position_m"][0] = round(self.w_x.value() / 1000, 6)
        w["position_m"][1] = round(self.w_y.value() / 1000, 6)
        w["drive_axis_rad"] = round(math.radians(self.w_axis.value()), 6)
        w["radius_m"] = round(self.w_radius.value() / 1000, 6)
        w["gear_ratio"] = round(self.w_gear.value(), 4)
        w["actuator_ref"] = self.w_ref.currentText()
        w["reverse"] = bool(self.w_reverse.isChecked())
        self._after_field()

    def _apply_odom(self):
        if self._updating or self._sel_kind != "odom":
            return
        o = self.doc["odometry"][self._sel_index]
        o["id"] = self.o_id.text()
        o["position_m"][0] = round(self.o_x.value() / 1000, 6)
        o["position_m"][1] = round(self.o_y.value() / 1000, 6)
        o["measure_axis_rad"] = round(math.radians(self.o_axis.value()), 6)
        o["radius_m"] = round(self.o_radius.value() / 1000, 6)
        o["encoder_cpr"] = int(self.o_cpr.value())
        self._after_field()

    def _after_field(self):
        self.canvas.set_doc(self.doc)
        self.canvas.select(self._sel_kind, self._sel_index)
        self._refresh_validation()
        self._commit()

    # -- undo / redo -----------------------------------------------------
    def _commit(self):
        text = yaml_io.render_doc(self.doc)
        if text == self._history[self._hp]:
            return
        self._history = self._history[: self._hp + 1]
        self._history.append(text)
        self._hp += 1
        self._update_history_buttons()

    def _undo(self):
        if self._hp > 0:
            self._hp -= 1
            self._load_state(self._history[self._hp])

    def _redo(self):
        if self._hp < len(self._history) - 1:
            self._hp += 1
            self._load_state(self._history[self._hp])

    def _load_state(self, text):
        self.doc = yaml_io.loads_doc(text)
        self.canvas.set_doc(self.doc)
        self._updating = True
        self.f_size.setValue(float(self.doc["footprint"]["size_m"]) * 1000)
        com = self.doc["center_of_mass"]
        self.f_mass.setValue(float(com["mass_kg"]))
        self.f_izz.setValue(float(com["inertia_zz_kgm2"]))
        self.f_comx.setValue(float(com["position_m"][0]) * 1000)
        self.f_comy.setValue(float(com["position_m"][1]) * 1000)
        self._updating = False
        self._sel_kind, self._sel_index = "none", -1
        self._wheel_box.setVisible(False)
        self._odom_box.setVisible(False)
        self._refresh_validation()
        self._update_history_buttons()

    def _update_history_buttons(self):
        self._undo_btn.setEnabled(self._hp > 0)
        self._redo_btn.setEnabled(self._hp < len(self._history) - 1)

    # -- validation ------------------------------------------------------
    def _refresh_validation(self):
        try:
            chassis = yaml_io.parse_chassis(self.doc)
        except SchemaError as e:
            self._set_validation([("err", f"⚠ invalid: {e}")])
            return

        msgs: list[tuple[str, str]] = []

        # drive jacobian conditioning
        cond = float(np.linalg.cond(build_drive_jacobian(chassis)))
        if not np.isfinite(cond) or cond > CONDITION_NUMBER_LIMIT:
            msgs.append(("err", f"⚠ singular layout (cond {cond:.3g}): wheels don't span 3 DOF"))
        elif cond > 50:
            msgs.append(("warn", f"△ ill-conditioned drive jacobian: cond {cond:.2f}"))
        else:
            msgs.append(("ok", f"✓ drive jacobian cond {cond:.2f}"))

        # actuator assignment integrity (chassis <-> actuators.yaml)
        for w in chassis.drive_wheels:
            ref = w.actuator_ref.strip()
            if not ref:
                msgs.append(("warn", f"△ wheel '{w.id}': no actuator assigned"))
            elif self._actuator_names and ref not in self._actuator_names:
                msgs.append(("warn", f"△ wheel '{w.id}': actuator '{ref}' not in actuators.yaml"))

        # placement warnings (not errors -- real machines can overhang)
        half = chassis.footprint.size_m / 2.0
        for w in chassis.drive_wheels:
            if abs(w.position_m[0]) > half or abs(w.position_m[1]) > half:
                msgs.append(("warn", f"△ wheel '{w.id}' is outside the footprint"))
        if self._com_outside_hull(chassis) is True:
            msgs.append(("warn", "△ centre of mass is outside the wheels' convex hull"))

        if self._actuator_names == [] and any(w.actuator_ref for w in chassis.drive_wheels):
            msgs.append(("info", "· actuators.yaml not found; actuator refs unchecked"))

        self._set_validation(msgs)

    @staticmethod
    def _com_outside_hull(chassis):
        """True if CoM is outside the wheel convex hull, False if inside, None if
        undetermined (fewer than 3 non-collinear wheels)."""
        pts = np.array([w.position_m for w in chassis.drive_wheels], float)
        if len(pts) < 3:
            return None
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
        except Exception:
            return None
        com = np.append(np.array(chassis.center_of_mass.position_m, float), 1.0)
        # inside iff on the interior side of every facet hyperplane
        return not bool(np.all(hull.equations @ com <= 1e-9))

    _SEVERITY_RANK = {"info": 0, "ok": 1, "warn": 2, "err": 3}
    _SEVERITY_COLOR = {"info": "#8a8a8a", "ok": "#00d75f",
                       "warn": "#ffaf00", "err": "#ff5f5f"}

    def _set_validation(self, msgs):
        n_odo = len(self._odos())
        if n_odo:
            msgs = msgs + [("info", f"· {n_odo} standalone odometry unit(s)")]
        worst = max((self._SEVERITY_RANK[s] for s, _ in msgs), default=1)
        color = {v: k for k, v in self._SEVERITY_RANK.items()}[worst]
        self.validation.setText("\n".join(m for _, m in msgs))
        self.validation.setStyleSheet(
            f"color: {self._SEVERITY_COLOR[color]}; font-weight: bold;")

    # -- save ------------------------------------------------------------
    def _on_save(self):
        updated = yaml_io.render_doc(self.doc)
        diff = yaml_io.diff_text(self._disk_text, updated)
        if not diff.strip():
            QtWidgets.QMessageBox.information(self, "No changes", "chassis.yaml is unchanged.")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Save chassis.yaml")
        box.setText(f"Write changes to {self.path.name}?")
        box.setDetailedText(diff)
        box.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
        if box.exec() == QtWidgets.QMessageBox.Save:
            yaml_io.save_chassis_doc(self.doc, self.path)
            self._disk_text = updated
            QtWidgets.QMessageBox.information(self, "Saved", f"Wrote {self.path}")

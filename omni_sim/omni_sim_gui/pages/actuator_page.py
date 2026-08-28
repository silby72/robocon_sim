"""Actuators page: edits config/robot/actuators.yaml.

Left: the list of actuators. Right: a preset dropdown that fills the datasheet
fields, per-field override highlighting with a revert (↺) button, optional
physical overrides, and a live read-only panel of derived constants
(Kt, Ke, R, tau_stall, omega_noload, J_reflected). J_reflected depends on the
drivetrain, so it is computed with the gear ratio and load of the wheel that
references this actuator (from chassis.yaml); unassigned actuators show n/a.
A usage line cross-checks assignment against the chassis. Saving writes only the
overrides, with comments preserved.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from omni_sim_core.mechanism.build import (
    DATASHEET_OVERRIDE_KEYS, load_presets)
from omni_sim_core.mechanism.derive import derive_motor
from omni_sim_core.mechanism.schema import Datasheet, Physical

from ..qt import QtCore, QtGui, QtWidgets
from ..io import yaml_io

# (key, label, decimals, step, suffix, is_int)
_DATASHEET_FIELDS = [
    ("rated_voltage_v", "rated voltage", 1, 0.5, " V", False),
    ("kv_rpm_per_v", "KV", 2, 1.0, " rpm/V", False),
    ("stall_torque_nm", "stall torque", 3, 0.1, " N·m", False),
    ("stall_current_a", "stall current", 2, 0.5, " A", False),
    ("no_load_speed_rpm", "no-load speed", 0, 10, " rpm", False),
    ("no_load_current_a", "no-load current", 2, 0.1, " A", False),
    ("encoder_cpr", "encoder CPR", 0, 1, "", True),
]
# physical overrides (applied to the final MotorParams, after derivation)
_PHYSICAL_FIELDS = [
    ("inertia_kgm2", "inertia (reflected)", 6, 1e-4, " kg·m²"),
    ("torque_constant_nm_a", "Kt override", 4, 0.01, " N·m/A"),
    ("resistance_ohm", "R override", 3, 0.01, " Ω"),
]
_OVERRIDE_BG = "background:#3a2f10;"       # amber tint for overridden fields


class ActuatorPage(QtWidgets.QWidget):
    def __init__(self, config_dir, parent=None) -> None:
        super().__init__(parent)
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / "actuators.yaml"
        self.doc = yaml_io.load_actuators_doc(self.path)
        self.presets = load_presets(self.config_dir.parent / "presets" / "motors")
        try:
            self.chassis_doc = yaml_io.load_chassis_doc(self.config_dir / "chassis.yaml")
        except Exception:
            self.chassis_doc = None

        self._disk_text = yaml_io.render_doc(self.doc)
        self._history = [self._disk_text]
        self._hp = 0
        self._updating = False
        self._name: str | None = None

        root = QtWidgets.QHBoxLayout(self)
        root.addWidget(self._build_left(), 1)
        root.addWidget(self._build_right(), 2)
        self._reload_list()
        self._update_history_buttons()

    # -- left: actuator list --------------------------------------------
    def _build_left(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.listw = QtWidgets.QListWidget()
        self.listw.currentTextChanged.connect(self._on_select)
        v.addWidget(self.listw)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("＋Add", self._add), ("Delete", self._delete),
                           ("↶", self._undo), ("↷", self._redo)):
            b = QtWidgets.QPushButton(text); b.clicked.connect(slot); row.addWidget(b)
            if text == "↶":
                self._undo_btn = b
            if text == "↷":
                self._redo_btn = b
        v.addLayout(row)
        return w

    # -- right: detail form ---------------------------------------------
    def _build_right(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("(custom / none)")
        self.preset_combo.addItems(sorted(self.presets.keys()))
        self.preset_combo.activated.connect(self._on_preset_changed)
        pform = QtWidgets.QFormLayout(); pform.addRow("preset", self.preset_combo)
        v.addLayout(pform)

        # datasheet fields
        ds_box = QtWidgets.QGroupBox("datasheet")
        ds_form = QtWidgets.QFormLayout(ds_box)
        self._ds_spins: dict[str, QtWidgets.QAbstractSpinBox] = {}
        self._ds_revert: dict[str, QtWidgets.QToolButton] = {}
        for key, label, dec, step, suffix, is_int in _DATASHEET_FIELDS:
            spin = QtWidgets.QSpinBox() if is_int else QtWidgets.QDoubleSpinBox()
            spin.setRange(0, 1_000_000)
            if not is_int:
                spin.setDecimals(dec)
            spin.setSingleStep(step)
            if suffix:
                spin.setSuffix(suffix)
            spin.valueChanged.connect(lambda _v, k=key: self._on_ds_changed(k))
            revert = QtWidgets.QToolButton(); revert.setText("↺")
            revert.setToolTip("revert to preset"); revert.setAutoRaise(True)
            revert.clicked.connect(lambda _c=False, k=key: self._revert_ds(k))
            self._ds_spins[key], self._ds_revert[key] = spin, revert
            roww = QtWidgets.QWidget(); rl = QtWidgets.QHBoxLayout(roww)
            rl.setContentsMargins(0, 0, 0, 0); rl.addWidget(spin, 1); rl.addWidget(revert)
            ds_form.addRow(label, roww)
        # rotor inertia (nullable -> checkbox controls known/estimated)
        self.rotor_known = QtWidgets.QCheckBox("known")
        self.rotor_spin = QtWidgets.QDoubleSpinBox()
        self.rotor_spin.setDecimals(7); self.rotor_spin.setRange(0, 10); self.rotor_spin.setSingleStep(1e-5)
        self.rotor_spin.setSuffix(" kg·m²")
        self.rotor_known.toggled.connect(lambda _c: self._on_ds_changed("rotor_inertia_kgm2"))
        self.rotor_spin.valueChanged.connect(lambda _v: self._on_ds_changed("rotor_inertia_kgm2"))
        rrow = QtWidgets.QWidget(); rhl = QtWidgets.QHBoxLayout(rrow)
        rhl.setContentsMargins(0, 0, 0, 0); rhl.addWidget(self.rotor_known); rhl.addWidget(self.rotor_spin, 1)
        ds_form.addRow("rotor inertia", rrow)
        v.addWidget(ds_box)

        # physical overrides
        ph_box = QtWidgets.QGroupBox("physical overrides (optional)")
        ph_form = QtWidgets.QFormLayout(ph_box)
        self._ph_check: dict[str, QtWidgets.QCheckBox] = {}
        self._ph_spin: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, label, dec, step, suffix in _PHYSICAL_FIELDS:
            chk = QtWidgets.QCheckBox()
            spin = QtWidgets.QDoubleSpinBox(); spin.setDecimals(dec); spin.setRange(0, 1e6)
            spin.setSingleStep(step); spin.setSuffix(suffix); spin.setEnabled(False)
            chk.toggled.connect(lambda on, k=key: self._on_ph_toggled(k, on))
            spin.valueChanged.connect(lambda _v, k=key: self._on_ph_changed(k))
            self._ph_check[key], self._ph_spin[key] = chk, spin
            rw = QtWidgets.QWidget(); hl = QtWidgets.QHBoxLayout(rw)
            hl.setContentsMargins(0, 0, 0, 0); hl.addWidget(chk); hl.addWidget(spin, 1)
            ph_form.addRow(label, rw)
        v.addWidget(ph_box)

        # derived (read-only)
        self.derived_lbl = QtWidgets.QLabel(); self.derived_lbl.setWordWrap(True)
        self.derived_lbl.setStyleSheet("font-family: monospace;")
        dv = QtWidgets.QGroupBox("derived (read-only, live)")
        dvl = QtWidgets.QVBoxLayout(dv); dvl.addWidget(self.derived_lbl)
        v.addWidget(dv)

        self.usage_lbl = QtWidgets.QLabel(); self.usage_lbl.setWordWrap(True)
        v.addWidget(self.usage_lbl)
        v.addStretch(1)
        self.save_btn = QtWidgets.QPushButton("Save actuators.yaml…")
        self.save_btn.clicked.connect(self._on_save)
        v.addWidget(self.save_btn)
        self._form = w
        return w

    # -- data helpers ----------------------------------------------------
    def _actuators(self):
        if "actuators" not in self.doc or self.doc["actuators"] is None:
            from ruamel.yaml.comments import CommentedMap
            self.doc["actuators"] = CommentedMap()
        return self.doc["actuators"]

    def _current(self):
        return self._actuators().get(self._name) if self._name else None

    def _overrides(self, a):
        if "overrides" not in a or a["overrides"] is None:
            from ruamel.yaml.comments import CommentedMap
            a["overrides"] = CommentedMap()
        return a["overrides"]

    def _baseline_datasheet(self, a) -> Datasheet:
        preset = a.get("preset")
        if preset:
            return self.presets[str(preset)].datasheet
        return Datasheet.from_doc(a["datasheet"], f"actuator '{self._name}'")

    # -- list / selection ------------------------------------------------
    def _reload_list(self):
        self._updating = True
        self.listw.clear()
        self.listw.addItems(list(self._actuators().keys()))
        self._updating = False
        if self.listw.count():
            self.listw.setCurrentRow(0)

    def _on_select(self, name: str):
        if self._updating or not name:
            return
        self._name = name
        self._populate()

    def _populate(self):
        a = self._current()
        if a is None:
            return
        self._updating = True
        preset = a.get("preset")
        self.preset_combo.setCurrentText(str(preset) if preset else "(custom / none)")
        base = self._baseline_datasheet(a)
        ov = a.get("overrides", {}) or {}
        for key, *_ in _DATASHEET_FIELDS:
            spin = self._ds_spins[key]
            val = ov[key] if key in ov else getattr(base, key)
            spin.setValue(val)
            self._mark_override(spin, self._ds_revert[key], key in ov)
        # rotor inertia
        rot = ov["rotor_inertia_kgm2"] if "rotor_inertia_kgm2" in ov else base.rotor_inertia_kgm2
        self.rotor_known.setChecked(rot is not None)
        self.rotor_spin.setEnabled(rot is not None)
        self.rotor_spin.setValue(rot if rot is not None else 0.0)
        # physical overrides
        for key, *_ in _PHYSICAL_FIELDS:
            on = key in ov
            self._ph_check[key].setChecked(on)
            self._ph_spin[key].setEnabled(on)
            if on:
                self._ph_spin[key].setValue(float(ov[key]))
        self._updating = False
        self._refresh_derived()

    def _mark_override(self, spin, revert, overridden):
        spin.setStyleSheet(_OVERRIDE_BG if overridden else "")
        revert.setEnabled(overridden)

    # -- edits -----------------------------------------------------------
    def _on_preset_changed(self, _idx=0):
        if self._updating:
            return
        a = self._current()
        if a is None:
            return
        text = self.preset_combo.currentText()
        if text == "(custom / none)":
            # snapshot current effective datasheet into an own datasheet block
            base = self._baseline_datasheet(a)
            from ruamel.yaml.comments import CommentedMap
            ds = CommentedMap()
            for key, *_ in _DATASHEET_FIELDS:
                ds[key] = getattr(base, key)
            ds["rotor_inertia_kgm2"] = base.rotor_inertia_kgm2
            a["preset"] = None
            a["datasheet"] = ds
        else:
            a["preset"] = text
            if "datasheet" in a:
                del a["datasheet"]
        self._populate()
        self._commit()

    def _on_ds_changed(self, key):
        if self._updating:
            return
        a = self._current()
        base = self._baseline_datasheet(a)
        if key == "rotor_inertia_kgm2":
            val = self.rotor_spin.value() if self.rotor_known.isChecked() else None
            self.rotor_spin.setEnabled(self.rotor_known.isChecked())
        else:
            val = self._ds_spins[key].value()
        self._set_field(a, base, key, val)
        self._after_edit()

    def _revert_ds(self, key):
        a = self._current()
        ov = a.get("overrides", {}) or {}
        if key in ov:
            del ov[key]
            self._cleanup_overrides(a)
        self._populate()
        self._commit()

    def _set_field(self, a, base, key, val):
        """Write ``val`` for a datasheet key: as an override (preset actuator) or
        into its own datasheet (preset: null); dropping it when it equals the
        preset baseline."""
        if a.get("preset"):
            ov = self._overrides(a)
            baseline = getattr(base, key)
            if val == baseline:
                if key in ov:
                    del ov[key]
            else:
                ov[key] = val
            self._cleanup_overrides(a)
        else:
            a["datasheet"][key] = val

    def _on_ph_toggled(self, key, on):
        if self._updating:
            return
        a = self._current()
        self._ph_spin[key].setEnabled(on)
        ov = self._overrides(a)
        if on:
            ov[key] = round(self._ph_spin[key].value(), 8)
        elif key in ov:
            del ov[key]
        self._cleanup_overrides(a)
        self._after_edit()

    def _on_ph_changed(self, key):
        if self._updating:
            return
        a = self._current()
        if self._ph_check[key].isChecked():
            self._overrides(a)[key] = round(self._ph_spin[key].value(), 8)
            self._after_edit()

    def _cleanup_overrides(self, a):
        if "overrides" in a and not a["overrides"]:
            del a["overrides"]

    def _after_edit(self):
        self._populate_marks()
        self._refresh_derived()
        self._commit()

    def _populate_marks(self):
        a = self._current()
        if a is None:
            return
        ov = a.get("overrides", {}) or {}
        for key, *_ in _DATASHEET_FIELDS:
            self._mark_override(self._ds_spins[key], self._ds_revert[key], key in ov)

    # -- derived + usage -------------------------------------------------
    def _ref_wheel(self):
        if self.chassis_doc is None or not self._name:
            return None
        for w in self.chassis_doc.get("drive_wheels", []) or []:
            if str(w.get("actuator_ref")) == self._name:
                return w
        return None

    def _refresh_derived(self):
        a = self._current()
        if a is None:
            self.derived_lbl.setText(""); return
        base = self._baseline_datasheet(a)
        ov = a.get("overrides", {}) or {}
        ds_over = {k: ov[k] for k in DATASHEET_OVERRIDE_KEYS if k in ov}
        eff = replace(base, **ds_over)
        phys = Physical(
            torque_constant_nm_a=ov.get("torque_constant_nm_a"),
            resistance_ohm=ov.get("resistance_ohm"),
            rotor_inertia_kgm2=ov.get("rotor_inertia_kgm2"))

        wheel = self._ref_wheel()
        if wheel is not None:
            gear = float(wheel["gear_ratio"])
            n = len(self.chassis_doc["drive_wheels"])
            mass = float(self.chassis_doc["center_of_mass"]["mass_kg"])
            load = (mass / n) * float(wheel["radius_m"]) ** 2
        else:
            gear, load = 1.0, 0.0

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = derive_motor(eff, phys, gear_ratio=gear, load_inertia_wheel_side_kgm2=load)

        j_reflected = ov.get("inertia_kgm2", d.reflected_inertia_kgm2)
        est = " (est.)" if "rotor_inertia_kgm2" in d.estimated else ""
        jr = "n/a (not assigned to a wheel)" if wheel is None else f"{float(j_reflected):.3e} kg·m²{est}"
        self.derived_lbl.setText(
            f"Kt        {d.torque_constant_nm_a:.4f} N·m/A\n"
            f"Ke        {d.back_emf_v_s:.4f} V·s\n"
            f"R         {d.resistance_ohm:.4f} Ω\n"
            f"tau_stall {d.stall_torque_nm:.4f} N·m\n"
            f"w_noload  {d.no_load_speed_rad_s:.2f} rad/s "
            f"({d.no_load_speed_rad_s * 60 / (2 * math.pi):.0f} rpm)\n"
            f"J_reflect {jr}   [wheel gear {gear:.1f}:1]")
        self._refresh_usage(wheel)

    def _refresh_usage(self, wheel):
        if self.chassis_doc is None:
            self.usage_lbl.setText("· chassis.yaml not found; assignment unchecked")
            self.usage_lbl.setStyleSheet("color:#8a8a8a;")
            return
        users = [str(w.get("id")) for w in (self.chassis_doc.get("drive_wheels") or [])
                 if str(w.get("actuator_ref")) == self._name]
        if users:
            self.usage_lbl.setText(f"✓ used by wheel(s): {', '.join(users)}")
            self.usage_lbl.setStyleSheet("color:#00d75f;")
        else:
            self.usage_lbl.setText("△ not assigned to any drive wheel")
            self.usage_lbl.setStyleSheet("color:#ffaf00;")

    # -- add / delete ----------------------------------------------------
    def _add(self):
        base = sorted(self.presets.keys())[0] if self.presets else None
        n = self._actuators()
        i = len(n)
        while f"motor_{i}" in n:
            i += 1
        n[f"motor_{i}"] = yaml_io.new_actuator(base)
        self._reload_list()
        self.listw.setCurrentRow(self.listw.count() - 1)
        self._commit()

    def _delete(self):
        if self._name and self._name in self._actuators():
            del self._actuators()[self._name]
            self._name = None
            self._reload_list()
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
        keep = self._name
        self._reload_list()
        if keep and keep in self._actuators():
            found = self.listw.findItems(keep, QtCore.Qt.MatchExactly)
            if found:
                self.listw.setCurrentItem(found[0])
        self._update_history_buttons()

    def _update_history_buttons(self):
        self._undo_btn.setEnabled(self._hp > 0)
        self._redo_btn.setEnabled(self._hp < len(self._history) - 1)

    # -- save ------------------------------------------------------------
    def _on_save(self):
        updated = yaml_io.render_doc(self.doc)
        diff = yaml_io.diff_text(self._disk_text, updated)
        if not diff.strip():
            QtWidgets.QMessageBox.information(self, "No changes", "actuators.yaml is unchanged.")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Save actuators.yaml")
        box.setText(f"Write changes to {self.path.name}?")
        box.setDetailedText(diff)
        box.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
        if box.exec() == QtWidgets.QMessageBox.Save:
            yaml_io.save_chassis_doc(self.doc, self.path)   # generic atomic ruamel write
            self._disk_text = updated
            QtWidgets.QMessageBox.information(self, "Saved", f"Wrote {self.path}")

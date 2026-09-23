"""Mechanism schema: dataclasses + validation for the robot description.

Input : parsed YAML mappings (from ``config/robot/*.yaml``).
Output: validated dataclasses (``Chassis``, ``ActuatorSet``, ``ModelError``).

Everything here is SI (metres, radians, kg, N, A, V, s); the GUI is responsible
for mm/degree <-> SI conversion and must never leak display units into this
layer. The wheel count is deliberately *not* fixed -- 3-wheel / other omni
layouts are a future addition, so ``drive_wheels`` is just a list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Raised when a robot description is structurally invalid."""


def _require(mapping: Any, key: str, ctx: str) -> Any:
    if mapping is None or key not in mapping:
        raise SchemaError(f"{ctx}: required key '{key}' is missing")
    return mapping[key]


def _xy(value: Any, ctx: str) -> tuple[float, float]:
    if value is None or len(value) != 2:
        raise SchemaError(f"{ctx}: expected [x, y] in metres, got {value!r}")
    return (float(value[0]), float(value[1]))


# --------------------------------------------------------------------------- #
# chassis.yaml
# --------------------------------------------------------------------------- #
@dataclass
class Footprint:
    shape: str = "square"
    size_m: float = 0.5

    def validate(self) -> None:
        if self.shape != "square":
            raise SchemaError(f"footprint.shape '{self.shape}' unsupported "
                              "(only 'square' for now)")
        if self.size_m <= 0:
            raise SchemaError("footprint.size_m must be > 0")


@dataclass
class CenterOfMass:
    position_m: tuple[float, float] = (0.0, 0.0)
    mass_kg: float = 0.0
    inertia_zz_kgm2: float = 0.0

    def validate(self) -> None:
        if self.mass_kg <= 0:
            raise SchemaError("center_of_mass.mass_kg must be > 0")
        if self.inertia_zz_kgm2 <= 0:
            raise SchemaError("center_of_mass.inertia_zz_kgm2 must be > 0")


@dataclass
class DriveWheel:
    id: str
    position_m: tuple[float, float]   # ground-contact point, base_link frame
    drive_axis_rad: float             # direction the wheel produces force
    radius_m: float
    gear_ratio: float                 # motor:wheel (REQUIRED -- no default!)
    actuator_ref: str
    reverse: bool = False

    def validate(self) -> None:
        if self.radius_m <= 0:
            raise SchemaError(f"wheel '{self.id}': radius_m must be > 0")
        if self.gear_ratio <= 0:
            raise SchemaError(f"wheel '{self.id}': gear_ratio must be > 0")


ODOMETRY_TYPES = ("dead_wheel", "optical")


@dataclass
class OdometrySource:
    """A standalone odometry unit, separate from the drive-wheel encoders.

    ``dead_wheel``: a passive (non-driven) measurement wheel rolling along
    ``measure_axis_rad``; ``optical``: a 2D optical-flow / mouse sensor that
    reports planar motion directly (no wheel, so radius/axis may be null).
    """
    id: str
    type: str                                    # dead_wheel | optical
    position_m: tuple[float, float]
    measure_axis_rad: Optional[float] = None     # required for dead_wheel
    radius_m: Optional[float] = None             # required for dead_wheel
    encoder_cpr: Optional[int] = None

    def validate(self) -> None:
        if self.type not in ODOMETRY_TYPES:
            raise SchemaError(f"odometry '{self.id}': type '{self.type}' invalid "
                              f"(expected one of {ODOMETRY_TYPES})")
        if self.type == "dead_wheel":
            if self.radius_m is None or self.radius_m <= 0:
                raise SchemaError(f"odometry '{self.id}': dead_wheel needs radius_m > 0")
            if self.measure_axis_rad is None:
                raise SchemaError(f"odometry '{self.id}': dead_wheel needs measure_axis_rad")


@dataclass
class Chassis:
    schema_version: int
    footprint: Footprint
    center_of_mass: CenterOfMass
    drive_wheels: list[DriveWheel]
    odometry: list[OdometrySource] = field(default_factory=list)

    @staticmethod
    def from_doc(doc: Any) -> "Chassis":
        _check_version(doc, "chassis.yaml")
        fp_raw = _require(doc, "footprint", "chassis.yaml")
        footprint = Footprint(shape=str(fp_raw.get("shape", "square")),
                              size_m=float(_require(fp_raw, "size_m", "footprint")))
        com_raw = _require(doc, "center_of_mass", "chassis.yaml")
        com = CenterOfMass(
            position_m=_xy(com_raw.get("position_m", [0.0, 0.0]), "center_of_mass"),
            mass_kg=float(_require(com_raw, "mass_kg", "center_of_mass")),
            inertia_zz_kgm2=float(_require(com_raw, "inertia_zz_kgm2",
                                           "center_of_mass")),
        )
        wheels_raw = _require(doc, "drive_wheels", "chassis.yaml")
        wheels: list[DriveWheel] = []
        for w in wheels_raw:
            wid = str(_require(w, "id", "drive_wheel"))
            wheels.append(DriveWheel(
                id=wid,
                position_m=_xy(_require(w, "position_m", f"wheel '{wid}'"),
                               f"wheel '{wid}'"),
                drive_axis_rad=float(_require(w, "drive_axis_rad", f"wheel '{wid}'")),
                radius_m=float(_require(w, "radius_m", f"wheel '{wid}'")),
                # gear_ratio is required on purpose: an implicit 1.0 hides a
                # reflected-inertia error that scales with n^2 (see derive.py).
                gear_ratio=float(_require(w, "gear_ratio", f"wheel '{wid}'")),
                actuator_ref=str(_require(w, "actuator_ref", f"wheel '{wid}'")),
                reverse=bool(w.get("reverse", False)),
            ))
        # odometry is optional (standalone units, in addition to drive encoders)
        odo: list[OdometrySource] = []
        for o in (doc.get("odometry", []) or []):
            oid = str(_require(o, "id", "odometry"))
            odo.append(OdometrySource(
                id=oid,
                type=str(_require(o, "type", f"odometry '{oid}'")),
                position_m=_xy(_require(o, "position_m", f"odometry '{oid}'"),
                               f"odometry '{oid}'"),
                measure_axis_rad=(None if o.get("measure_axis_rad", None) is None
                                  else float(o["measure_axis_rad"])),
                radius_m=(None if o.get("radius_m", None) is None
                          else float(o["radius_m"])),
                encoder_cpr=(None if o.get("encoder_cpr", None) is None
                             else int(o["encoder_cpr"])),
            ))

        chassis = Chassis(schema_version=int(doc["schema_version"]),
                          footprint=footprint, center_of_mass=com,
                          drive_wheels=wheels, odometry=odo)
        chassis.validate()
        return chassis

    def validate(self) -> None:
        self.footprint.validate()
        self.center_of_mass.validate()
        if not self.drive_wheels:
            raise SchemaError("chassis has no drive_wheels")
        ids = [w.id for w in self.drive_wheels]
        if len(ids) != len(set(ids)):
            raise SchemaError("drive_wheels have duplicate ids")
        for w in self.drive_wheels:
            w.validate()
        odo_ids = [o.id for o in self.odometry]
        if len(odo_ids) != len(set(odo_ids)):
            raise SchemaError("odometry sources have duplicate ids")
        for o in self.odometry:
            o.validate()

    def actuator_refs(self) -> list[str]:
        return [w.actuator_ref for w in self.drive_wheels]


# --------------------------------------------------------------------------- #
# actuators.yaml
# --------------------------------------------------------------------------- #
@dataclass
class Datasheet:
    rated_voltage_v: float
    kv_rpm_per_v: float
    stall_torque_nm: float
    stall_current_a: float
    no_load_speed_rpm: float
    no_load_current_a: float
    encoder_cpr: int
    rotor_inertia_kgm2: Optional[float] = None  # None => estimated in derive.py

    @staticmethod
    def from_doc(d: Any, ctx: str) -> "Datasheet":
        return Datasheet(
            rated_voltage_v=float(_require(d, "rated_voltage_v", ctx)),
            kv_rpm_per_v=float(_require(d, "kv_rpm_per_v", ctx)),
            stall_torque_nm=float(_require(d, "stall_torque_nm", ctx)),
            stall_current_a=float(_require(d, "stall_current_a", ctx)),
            no_load_speed_rpm=float(_require(d, "no_load_speed_rpm", ctx)),
            no_load_current_a=float(_require(d, "no_load_current_a", ctx)),
            encoder_cpr=int(_require(d, "encoder_cpr", ctx)),
            rotor_inertia_kgm2=(None if d.get("rotor_inertia_kgm2", None) is None
                                else float(d["rotor_inertia_kgm2"])),
        )


@dataclass
class Physical:
    """Directly-measured electrical/mechanical constants (SI). Optional; when
    present they take precedence over datasheet-derived values."""
    torque_constant_nm_a: Optional[float] = None   # Kt
    back_emf_v_s: Optional[float] = None            # Ke
    resistance_ohm: Optional[float] = None          # R
    inductance_h: Optional[float] = None            # L
    rotor_inertia_kgm2: Optional[float] = None      # J

    @staticmethod
    def from_doc(d: Any) -> "Physical":
        if d is None:
            return Physical()
        g = lambda k: (None if d.get(k, None) is None else float(d[k]))
        return Physical(
            torque_constant_nm_a=g("torque_constant_nm_a"),
            back_emf_v_s=g("back_emf_v_s"),
            resistance_ohm=g("resistance_ohm"),
            inductance_h=g("inductance_h"),
            rotor_inertia_kgm2=g("rotor_inertia_kgm2"),
        )


@dataclass
class ActuatorDef:
    """One actuator instance: a preset reference plus overrides, or a
    self-contained datasheet when ``preset`` is null."""
    name: str
    preset: Optional[str] = None
    overrides: dict[str, Any] = field(default_factory=dict)
    datasheet: Optional[Datasheet] = None

    def validate(self) -> None:
        if self.preset is None and self.datasheet is None:
            raise SchemaError(
                f"actuator '{self.name}': needs either 'preset' or a "
                "'datasheet' block")


@dataclass
class ActuatorSet:
    schema_version: int
    actuators: dict[str, ActuatorDef]

    @staticmethod
    def from_doc(doc: Any) -> "ActuatorSet":
        _check_version(doc, "actuators.yaml")
        raw = _require(doc, "actuators", "actuators.yaml")
        out: dict[str, ActuatorDef] = {}
        for name, body in raw.items():
            body = body or {}
            preset = body.get("preset", None)
            preset = None if preset is None else str(preset)
            ds_raw = body.get("datasheet", None)
            out[str(name)] = ActuatorDef(
                name=str(name),
                preset=preset,
                overrides=dict(body.get("overrides", {}) or {}),
                datasheet=(Datasheet.from_doc(ds_raw, f"actuator '{name}'.datasheet")
                           if ds_raw is not None else None),
            )
        aset = ActuatorSet(schema_version=int(doc["schema_version"]), actuators=out)
        aset.validate()
        return aset

    def validate(self) -> None:
        if not self.actuators:
            raise SchemaError("actuators.yaml defines no actuators")
        for a in self.actuators.values():
            a.validate()


# --------------------------------------------------------------------------- #
# presets/motors/*.yaml
# --------------------------------------------------------------------------- #
@dataclass
class MotorPreset:
    name: str
    datasheet: Datasheet
    physical: Physical = field(default_factory=Physical)

    @staticmethod
    def from_doc(doc: Any, name: str) -> "MotorPreset":
        ds = Datasheet.from_doc(_require(doc, "datasheet", f"preset '{name}'"),
                                f"preset '{name}'.datasheet")
        phys = Physical.from_doc(doc.get("physical", None))
        return MotorPreset(name=name, datasheet=ds, physical=phys)


# --------------------------------------------------------------------------- #
# model_error.yaml
# --------------------------------------------------------------------------- #
@dataclass
class ModelError:
    schema_version: int
    ratios: dict[str, float]
    mode: str = "ratio"              # 'ratio' | 'manual'
    manual_file: Optional[str] = None

    @staticmethod
    def from_doc(doc: Any) -> "ModelError":
        _check_version(doc, "model_error.yaml")
        mode = str(doc.get("mode", "ratio"))
        if mode not in ("ratio", "manual"):
            raise SchemaError(f"model_error.mode '{mode}' invalid "
                              "(expected 'ratio' or 'manual')")
        me = ModelError(
            schema_version=int(doc["schema_version"]),
            ratios={str(k): float(v) for k, v in (doc.get("ratios", {}) or {}).items()},
            mode=mode,
            manual_file=(None if doc.get("manual_file", None) is None
                         else str(doc["manual_file"])),
        )
        if me.mode == "manual" and me.manual_file is None:
            raise SchemaError("model_error.mode is 'manual' but manual_file is null")
        return me


def _check_version(doc: Any, ctx: str) -> None:
    v = _require(doc, "schema_version", ctx)
    if int(v) != SCHEMA_VERSION:
        raise SchemaError(f"{ctx}: schema_version {v} unsupported "
                          f"(this build expects {SCHEMA_VERSION})")

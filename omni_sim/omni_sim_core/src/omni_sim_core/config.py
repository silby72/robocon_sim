"""Dataclass configuration + YAML loaders.

All tuning lives in YAML; no magic numbers in code. The true plant and the
nominal model are loaded from *separate* files (see spec 9.1) to prevent
accidental coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .clock import ClockConfig
from .plant.motor import MotorParams
from .plant.nominal import NominalModel, nominal_from_true
from .plant.jacobian import JacobianParams
from .plant.body import BodyParams
from .control.pid import PIDParams
from .control.dob import DOBParams
from .disturbance import DisturbanceSpec


def _filter_kwargs(cls, data: dict) -> dict:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_motor_params(path: str | Path) -> MotorParams:
    return MotorParams(**_filter_kwargs(MotorParams, load_yaml(path)))


def load_nominal_model(path: str | Path) -> NominalModel:
    return NominalModel(**_filter_kwargs(NominalModel, load_yaml(path)))


@dataclass
class ControllerConfig:
    type: str = "pid_with_dob"
    setpoint_omega_rad_s: float = 50.0
    pid: PIDParams = field(default_factory=PIDParams)
    dob: DOBParams = field(default_factory=DOBParams)


@dataclass
class NominalErrorConfig:
    j_ratio: float = 1.0
    b_ratio: float = 1.0


@dataclass
class LoggingConfig:
    output: str = "results/run.csv"
    signals: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    name: str = "unnamed"
    seed: int = 0
    duration_s: float = 10.0
    clock: ClockConfig = field(default_factory=ClockConfig)
    plant_true: str = "config/plant_true.yaml"
    plant_nominal: str = "config/plant_nominal.yaml"
    nominal_error: NominalErrorConfig = field(default_factory=NominalErrorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    disturbance: list[DisturbanceSpec] = field(default_factory=list)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    base_dir: Path = field(default_factory=lambda: Path("."))

    # -- resolved objects -------------------------------------------------
    def resolve_true_params(self) -> MotorParams:
        return load_motor_params(self._path(self.plant_true))

    def resolve_nominal(self) -> NominalModel:
        """Nominal model: from file, then overridden by nominal_error ratios
        relative to the true plant if ratios are not 1.0."""
        ne = self.nominal_error
        if ne.j_ratio != 1.0 or ne.b_ratio != 1.0:
            return nominal_from_true(self.resolve_true_params(),
                                     j_ratio=ne.j_ratio, b_ratio=ne.b_ratio)
        return load_nominal_model(self._path(self.plant_nominal))

    def _path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.base_dir / p


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path).resolve()
    raw = load_yaml(path)
    # scenario files live in <root>/config/scenarios/; plant/log paths in them
    # are written relative to <root>.
    if path.parent.name == "scenarios" and path.parents[1].name == "config":
        base_dir = path.parents[2]
    else:
        base_dir = path.parent

    clock = ClockConfig(**_filter_kwargs(ClockConfig, raw.get("clock", {})))
    ne = NominalErrorConfig(**_filter_kwargs(NominalErrorConfig,
                                             raw.get("nominal_error", {})))

    ctrl_raw = raw.get("controller", {})
    controller = ControllerConfig(
        type=ctrl_raw.get("type", "pid_with_dob"),
        setpoint_omega_rad_s=ctrl_raw.get("setpoint_omega_rad_s", 50.0),
        pid=PIDParams(**_filter_kwargs(PIDParams, ctrl_raw.get("pid", {}))),
        dob=DOBParams(**_filter_kwargs(DOBParams, ctrl_raw.get("dob", {}))),
    )

    dist = []
    for d in raw.get("disturbance", []):
        d = dict(d)
        # accept human-friendly aliases from scenario files
        if "motor_index" in d:
            d["index"] = d.pop("motor_index")
        if "component_index" in d:
            d["index"] = d.pop("component_index")
        dist.append(DisturbanceSpec(**_filter_kwargs(DisturbanceSpec, d)))

    logging_cfg = LoggingConfig(**_filter_kwargs(LoggingConfig,
                                                 raw.get("logging", {})))

    return Scenario(
        name=raw.get("name", "unnamed"),
        seed=int(raw.get("seed", 0)),
        duration_s=float(raw.get("duration_s", 10.0)),
        clock=clock,
        plant_true=raw.get("plant_true", "config/plant_true.yaml"),
        plant_nominal=raw.get("plant_nominal", "config/plant_nominal.yaml"),
        nominal_error=ne,
        controller=controller,
        disturbance=dist,
        logging=logging_cfg,
        base_dir=base_dir,
    )

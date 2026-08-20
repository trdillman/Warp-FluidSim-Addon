"""Strict immutable schemas for Custom Effects v2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    FieldAccess,
    ParameterKind,
)

_EFFECT_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?$")


def _strict_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")


def _enum(enum_type, value, label: str):
    try:
        return value if isinstance(value, enum_type) else enum_type(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _normalized_sequence(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numeric values")
    return tuple(_finite_number(component, f"{label}[{index}]") for index, component in enumerate(value))


@dataclass(frozen=True)
class EffectIssue:
    code: str
    message: str
    severity: str = "ERROR"
    effect_id: str = ""
    instance_id: str = ""
    kernel_id: str = ""
    line: int = 0
    column: int = 0
    hint: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "effect_id": self.effect_id,
            "instance_id": self.instance_id,
            "kernel_id": self.kernel_id,
            "line": int(self.line),
            "column": int(self.column),
            "hint": self.hint,
        }


@dataclass(frozen=True)
class ParameterDefinitionV2:
    key: str
    label: str
    kind: ParameterKind
    default: object
    minimum: float | int | None = None
    maximum: float | int | None = None
    soft_minimum: float | int | None = None
    soft_maximum: float | int | None = None
    unit: str = "NONE"
    precision: int = 3
    description: str = ""
    enum_items: tuple[tuple[str, str], ...] = ()
    runtime_bindable: bool = True
    animatable: bool = True

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.fullmatch(self.key):
            raise ValueError(f"invalid parameter identifier {self.key!r}")
        object.__setattr__(self, "kind", _enum(ParameterKind, self.kind, "parameter kind"))
        enum_items = tuple((str(key), str(label)) for key, label in self.enum_items)
        object.__setattr__(self, "enum_items", enum_items)
        keys = [key for key, _label in enum_items]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate enum item in parameter {self.key}")
        normalized = self.validate_value(self.default, label=f"default for {self.key}")
        object.__setattr__(self, "default", normalized)
        if self.minimum is not None:
            object.__setattr__(self, "minimum", _finite_number(self.minimum, f"minimum for {self.key}"))
        if self.maximum is not None:
            object.__setattr__(self, "maximum", _finite_number(self.maximum, f"maximum for {self.key}"))
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"minimum exceeds maximum for parameter {self.key}")
        self._validate_range(normalized, label=f"default for {self.key}")

    def _validate_range(self, value: object, *, label: str) -> None:
        if self.kind not in {ParameterKind.FLOAT, ParameterKind.INT}:
            return
        numeric = float(value)
        if self.minimum is not None and numeric < float(self.minimum):
            raise ValueError(f"{label} below minimum {self.minimum}")
        if self.maximum is not None and numeric > float(self.maximum):
            raise ValueError(f"{label} above maximum {self.maximum}")

    def validate_value(self, value: object, *, label: str | None = None) -> object:
        label = label or f"parameter {self.key}"
        if self.kind is ParameterKind.FLOAT:
            result: object = _finite_number(value, label)
        elif self.kind is ParameterKind.INT:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be an integer")
            numeric = _finite_number(value, label)
            if not numeric.is_integer():
                raise ValueError(f"{label} must be an integer")
            result = int(numeric)
        elif self.kind is ParameterKind.BOOL:
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be boolean")
            result = bool(value)
        elif self.kind is ParameterKind.VECTOR:
            result = _normalized_sequence(value, 3, label)
        elif self.kind is ParameterKind.COLOR:
            result = _normalized_sequence(value, 4, label)
        elif self.kind is ParameterKind.ENUM:
            result = str(value)
            legal = {key for key, _item_label in self.enum_items}
            if not legal:
                raise ValueError(f"enum parameter {self.key} has no enum_items")
            if result not in legal:
                raise ValueError(f"{label} is not one of {sorted(legal)}")
        elif self.kind is ParameterKind.OBJECT:
            if value is None:
                result = ""
            elif not isinstance(value, str):
                raise ValueError(f"{label} must be a stable object identifier string")
            else:
                result = value
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unsupported parameter kind {self.kind}")
        self._validate_range(result, label=label)
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind.value,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "soft_minimum": self.soft_minimum,
            "soft_maximum": self.soft_maximum,
            "unit": self.unit,
            "precision": int(self.precision),
            "description": self.description,
            "enum_items": [list(item) for item in self.enum_items],
            "runtime_bindable": bool(self.runtime_bindable),
            "animatable": bool(self.animatable),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterDefinitionV2":
        allowed = {
            "key", "label", "kind", "default", "minimum", "maximum",
            "soft_minimum", "soft_maximum", "unit", "precision", "description",
            "enum_items", "runtime_bindable", "animatable",
        }
        _strict_keys(payload, allowed, "ParameterDefinitionV2")
        required = {"key", "label", "kind", "default"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing ParameterDefinitionV2 field(s): {', '.join(missing)}")
        return cls(
            key=str(payload["key"]),
            label=str(payload["label"]),
            kind=_enum(ParameterKind, payload["kind"], "parameter kind"),
            default=payload["default"],
            minimum=payload.get("minimum"),
            maximum=payload.get("maximum"),
            soft_minimum=payload.get("soft_minimum"),
            soft_maximum=payload.get("soft_maximum"),
            unit=str(payload.get("unit", "NONE")),
            precision=int(payload.get("precision", 3)),
            description=str(payload.get("description", "")),
            enum_items=tuple(tuple(item) for item in payload.get("enum_items", ())),
            runtime_bindable=bool(payload.get("runtime_bindable", True)),
            animatable=bool(payload.get("animatable", True)),
        )


@dataclass(frozen=True)
class FieldBindingV2:
    field: str
    access: FieldAccess = FieldAccess.READ

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.fullmatch(self.field):
            raise ValueError(f"invalid field identifier {self.field!r}")
        object.__setattr__(self, "access", _enum(FieldAccess, self.access, "field access"))

    def to_dict(self) -> dict[str, object]:
        return {"field": self.field, "access": self.access.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FieldBindingV2":
        _strict_keys(payload, {"field", "access"}, "FieldBindingV2")
        if "field" not in payload:
            raise ValueError("missing FieldBindingV2 field: field")
        return cls(
            field=str(payload["field"]),
            access=_enum(FieldAccess, payload.get("access", "READ"), "field access"),
        )


@dataclass(frozen=True)
class KernelDefinitionV2:
    kernel_id: str
    label: str
    domain: EffectDomain
    stage: EffectStage
    cadence: EffectCadence
    source: str
    fields: tuple[FieldBindingV2, ...] = ()
    parameter_keys: tuple[str, ...] = ()
    safe_mode: bool = True
    frame_start: int = 0
    frame_end: int = 0
    frame_step: int = 1
    description: str = ""
    prerequisites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.fullmatch(self.kernel_id):
            raise ValueError(f"invalid kernel identifier {self.kernel_id!r}")
        object.__setattr__(self, "domain", _enum(EffectDomain, self.domain, "effect domain"))
        object.__setattr__(self, "stage", _enum(EffectStage, self.stage, "effect stage"))
        object.__setattr__(self, "cadence", _enum(EffectCadence, self.cadence, "effect cadence"))
        object.__setattr__(self, "fields", tuple(
            field if isinstance(field, FieldBindingV2) else FieldBindingV2.from_dict(field)
            for field in self.fields
        ))
        object.__setattr__(self, "parameter_keys", tuple(str(key) for key in self.parameter_keys))
        object.__setattr__(self, "prerequisites", tuple(sorted({str(item) for item in self.prerequisites})))
        if not str(self.source).strip():
            raise ValueError(f"kernel {self.kernel_id} source is empty")
        if self.frame_start < 0 or self.frame_end < 0:
            raise ValueError("kernel frame bounds cannot be negative")
        if self.frame_end and self.frame_start and self.frame_end < self.frame_start:
            raise ValueError("kernel frame_end precedes frame_start")
        if self.frame_step < 1:
            raise ValueError("kernel frame_step must be at least 1")
        field_names = [binding.field for binding in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError(f"duplicate field binding in kernel {self.kernel_id}")
        if len(self.parameter_keys) != len(set(self.parameter_keys)):
            raise ValueError(f"duplicate parameter binding in kernel {self.kernel_id}")

    def active_on_frame(self, frame: int) -> bool:
        if self.frame_start and frame < self.frame_start:
            return False
        if self.frame_end and frame > self.frame_end:
            return False
        base = self.frame_start if self.frame_start else 0
        return (frame - base) % max(1, self.frame_step) == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "label": self.label,
            "domain": self.domain.value,
            "stage": self.stage.value,
            "cadence": self.cadence.value,
            "source": self.source,
            "fields": [binding.to_dict() for binding in self.fields],
            "parameter_keys": list(self.parameter_keys),
            "safe_mode": bool(self.safe_mode),
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "frame_step": int(self.frame_step),
            "description": self.description,
            "prerequisites": list(self.prerequisites),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KernelDefinitionV2":
        allowed = {
            "kernel_id", "label", "domain", "stage", "cadence", "source",
            "fields", "parameter_keys", "safe_mode", "frame_start", "frame_end",
            "frame_step", "description", "prerequisites",
        }
        _strict_keys(payload, allowed, "KernelDefinitionV2")
        required = {"kernel_id", "label", "domain", "stage", "cadence", "source"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing KernelDefinitionV2 field(s): {', '.join(missing)}")
        return cls(
            kernel_id=str(payload["kernel_id"]),
            label=str(payload["label"]),
            domain=_enum(EffectDomain, payload["domain"], "effect domain"),
            stage=_enum(EffectStage, payload["stage"], "effect stage"),
            cadence=_enum(EffectCadence, payload["cadence"], "effect cadence"),
            source=str(payload["source"]),
            fields=tuple(FieldBindingV2.from_dict(item) for item in payload.get("fields", ())),
            parameter_keys=tuple(str(item) for item in payload.get("parameter_keys", ())),
            safe_mode=bool(payload.get("safe_mode", True)),
            frame_start=int(payload.get("frame_start", 0)),
            frame_end=int(payload.get("frame_end", 0)),
            frame_step=int(payload.get("frame_step", 1)),
            description=str(payload.get("description", "")),
            prerequisites=tuple(str(item) for item in payload.get("prerequisites", ())),
        )


@dataclass(frozen=True)
class EffectDefinitionV2:
    effect_id: str
    version: str
    name: str
    summary: str
    description: str
    category: str
    author: str
    license: str
    parameters: tuple[ParameterDefinitionV2, ...] = ()
    kernels: tuple[KernelDefinitionV2, ...] = ()
    schema_version: int = 2
    effect_api_version: int = 2
    effect_abi_version: int = 1
    min_fluidsim_version: str = "1.0.251"
    built_in: bool = False
    package_origin: str = ""
    preview_asset: str = ""
    tags: tuple[str, ...] = ()
    cost_hint: str = "LOW"

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"unsupported effect schema version {self.schema_version}")
        if self.effect_api_version != 2:
            raise ValueError(f"unsupported effect API version {self.effect_api_version}")
        if self.effect_abi_version < 1:
            raise ValueError("effect ABI version must be positive")
        if not _EFFECT_ID_RE.fullmatch(self.effect_id):
            raise ValueError(f"invalid effect_id {self.effect_id!r}")
        if not _VERSION_RE.fullmatch(self.version):
            raise ValueError(f"invalid effect version {self.version!r}")
        parameters = tuple(
            item if isinstance(item, ParameterDefinitionV2) else ParameterDefinitionV2.from_dict(item)
            for item in self.parameters
        )
        kernels = tuple(
            item if isinstance(item, KernelDefinitionV2) else KernelDefinitionV2.from_dict(item)
            for item in self.kernels
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "tags", tuple(sorted({str(tag) for tag in self.tags})))
        parameter_keys = [item.key for item in parameters]
        kernel_ids = [item.kernel_id for item in kernels]
        if len(parameter_keys) != len(set(parameter_keys)):
            raise ValueError("duplicate parameter key in effect definition")
        if len(kernel_ids) != len(set(kernel_ids)):
            raise ValueError("duplicate kernel id in effect definition")
        declared = set(parameter_keys)
        for kernel in kernels:
            unknown = sorted(set(kernel.parameter_keys) - declared)
            if unknown:
                raise ValueError(
                    f"kernel {kernel.kernel_id} references unknown parameter(s): {', '.join(unknown)}"
                )
        if not kernels:
            raise ValueError("effect definition must contain at least one kernel")

    def parameter_map(self) -> dict[str, ParameterDefinitionV2]:
        return {item.key: item for item in self.parameters}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            "effect_api_version": int(self.effect_api_version),
            "effect_abi_version": int(self.effect_abi_version),
            "effect_id": self.effect_id,
            "version": self.version,
            "name": self.name,
            "summary": self.summary,
            "description": self.description,
            "category": self.category,
            "author": self.author,
            "license": self.license,
            "min_fluidsim_version": self.min_fluidsim_version,
            "built_in": bool(self.built_in),
            "package_origin": self.package_origin,
            "preview_asset": self.preview_asset,
            "tags": list(self.tags),
            "cost_hint": self.cost_hint,
            "parameters": [item.to_dict() for item in self.parameters],
            "kernels": [item.to_dict() for item in self.kernels],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffectDefinitionV2":
        allowed = {
            "schema_version", "effect_api_version", "effect_abi_version", "effect_id",
            "version", "name", "summary", "description", "category", "author",
            "license", "min_fluidsim_version", "built_in", "package_origin",
            "preview_asset", "tags", "cost_hint", "parameters", "kernels",
        }
        _strict_keys(payload, allowed, "EffectDefinitionV2")
        required = {
            "effect_id", "version", "name", "summary", "description", "category",
            "author", "license", "kernels",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing EffectDefinitionV2 field(s): {', '.join(missing)}")
        return cls(
            schema_version=int(payload.get("schema_version", 2)),
            effect_api_version=int(payload.get("effect_api_version", 2)),
            effect_abi_version=int(payload.get("effect_abi_version", 1)),
            effect_id=str(payload["effect_id"]),
            version=str(payload["version"]),
            name=str(payload["name"]),
            summary=str(payload["summary"]),
            description=str(payload["description"]),
            category=str(payload["category"]),
            author=str(payload["author"]),
            license=str(payload["license"]),
            min_fluidsim_version=str(payload.get("min_fluidsim_version", "1.0.251")),
            built_in=bool(payload.get("built_in", False)),
            package_origin=str(payload.get("package_origin", "")),
            preview_asset=str(payload.get("preview_asset", "")),
            tags=tuple(str(item) for item in payload.get("tags", ())),
            cost_hint=str(payload.get("cost_hint", "LOW")),
            parameters=tuple(ParameterDefinitionV2.from_dict(item) for item in payload.get("parameters", ())),
            kernels=tuple(KernelDefinitionV2.from_dict(item) for item in payload.get("kernels", ())),
        )


@dataclass(frozen=True)
class EffectInstanceV2:
    instance_id: str
    definition: EffectDefinitionV2
    enabled: bool = True
    muted: bool = False
    solo: bool = False
    order: int = 0
    parameter_values: Mapping[str, object] = field(default_factory=dict)
    controller_bindings: Mapping[str, object] = field(default_factory=dict)
    frame_start: int = 0
    frame_end: int = 0
    frame_step: int = 1
    safe_mode: bool = True
    expert_acknowledged: bool = False
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.instance_id or len(self.instance_id) > 128:
            raise ValueError("instance_id must contain 1-128 characters")
        if not isinstance(self.definition, EffectDefinitionV2):
            object.__setattr__(self, "definition", EffectDefinitionV2.from_dict(self.definition))
        if self.frame_start < 0 or self.frame_end < 0:
            raise ValueError("effect frame bounds cannot be negative")
        if self.frame_end and self.frame_start and self.frame_end < self.frame_start:
            raise ValueError("effect frame_end precedes frame_start")
        if self.frame_step < 1:
            raise ValueError("effect frame_step must be at least 1")
        definitions = self.definition.parameter_map()
        unknown = sorted(set(self.parameter_values) - set(definitions))
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
        normalized: dict[str, object] = {}
        for key, value in self.parameter_values.items():
            normalized[key] = definitions[key].validate_value(value)
        object.__setattr__(self, "parameter_values", normalized)
        object.__setattr__(self, "controller_bindings", _normalize_plain_mapping(self.controller_bindings))

    def effective_parameter_values(self) -> dict[str, object]:
        values = {item.key: item.default for item in self.definition.parameters}
        values.update(self.parameter_values)
        return values

    def active_on_frame(self, frame: int) -> bool:
        if not self.enabled or self.muted:
            return False
        if self.frame_start and frame < self.frame_start:
            return False
        if self.frame_end and frame > self.frame_end:
            return False
        base = self.frame_start if self.frame_start else 0
        return (frame - base) % max(1, self.frame_step) == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "definition": self.definition.to_dict(),
            "enabled": bool(self.enabled),
            "muted": bool(self.muted),
            "solo": bool(self.solo),
            "order": int(self.order),
            "parameter_values": dict(self.parameter_values),
            "controller_bindings": dict(self.controller_bindings),
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "frame_step": int(self.frame_step),
            "safe_mode": bool(self.safe_mode),
            "expert_acknowledged": bool(self.expert_acknowledged),
            "display_name": self.display_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffectInstanceV2":
        allowed = {
            "instance_id", "definition", "enabled", "muted", "solo", "order",
            "parameter_values", "controller_bindings", "frame_start", "frame_end",
            "frame_step", "safe_mode", "expert_acknowledged", "display_name",
        }
        _strict_keys(payload, allowed, "EffectInstanceV2")
        required = {"instance_id", "definition"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing EffectInstanceV2 field(s): {', '.join(missing)}")
        return cls(
            instance_id=str(payload["instance_id"]),
            definition=EffectDefinitionV2.from_dict(payload["definition"]),
            enabled=bool(payload.get("enabled", True)),
            muted=bool(payload.get("muted", False)),
            solo=bool(payload.get("solo", False)),
            order=int(payload.get("order", 0)),
            parameter_values=dict(payload.get("parameter_values", {})),
            controller_bindings=dict(payload.get("controller_bindings", {})),
            frame_start=int(payload.get("frame_start", 0)),
            frame_end=int(payload.get("frame_end", 0)),
            frame_step=int(payload.get("frame_step", 1)),
            safe_mode=bool(payload.get("safe_mode", True)),
            expert_acknowledged=bool(payload.get("expert_acknowledged", False)),
            display_name=str(payload.get("display_name", "")),
        )


def _normalize_plain_mapping(value: Mapping[str, object] | object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("controller_bindings must be a mapping")

    def normalize(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return item
        if isinstance(item, (int, float)):
            return _finite_number(item, "controller binding number")
        raise ValueError(f"unsupported controller binding value {type(item).__name__}")

    return {str(key): normalize(item) for key, item in value.items()}

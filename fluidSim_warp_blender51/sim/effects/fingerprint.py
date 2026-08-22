"""Canonical JSON and SHA-256 fingerprints for custom effects."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import is_dataclass
from enum import Enum
from typing import Mapping

from .schema import EffectDefinitionV2, EffectInstanceV2


def _canonical(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, EffectDefinitionV2):
        return _canonical(value.to_dict())
    if isinstance(value, EffectInstanceV2):
        return _canonical(value.to_dict())
    if is_dataclass(value) and hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical effect data cannot contain NaN or Infinity")
        if value == 0.0:
            return 0.0
        return value
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def effect_snapshot_dict(instance: EffectInstanceV2) -> dict[str, object]:
    return _canonical(instance.to_dict())


def effect_fingerprint(instance: EffectInstanceV2) -> str:
    return sha256_hex(effect_snapshot_dict(instance))


def stack_fingerprint(
    instances: tuple[EffectInstanceV2, ...] | list[EffectInstanceV2],
) -> tuple[str, tuple[dict[str, object], ...]]:
    inventory = tuple({
        "instance_id": instance.instance_id,
        "effect_id": instance.definition.effect_id,
        "effect_version": instance.definition.version,
        "enabled": bool(instance.enabled),
        "muted": bool(instance.muted),
        "solo": bool(instance.solo),
        "order": int(instance.order),
        "fingerprint": effect_fingerprint(instance),
    } for instance in instances)
    return sha256_hex({"effect_stack_api": 2, "inventory": inventory}), inventory


def effective_stack_instances(
    instances: tuple[EffectInstanceV2, ...] | list[EffectInstanceV2],
) -> tuple[EffectInstanceV2, ...]:
    """Return the exact deterministic execution stack after mute/solo filtering."""

    indexed = list(enumerate(instances))
    identifiers = [item.instance_id for _index, item in indexed]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
        raise ValueError(f"duplicate effect instance_id: {', '.join(duplicates)}")
    enabled = [
        (index, item)
        for index, item in indexed
        if bool(item.enabled) and not bool(item.muted)
    ]
    solos = [(index, item) for index, item in enabled if bool(item.solo)]
    selected = solos or enabled
    selected.sort(key=lambda row: (int(row[1].order), int(row[0])))
    return tuple(item for _index, item in selected)


def execution_stack_fingerprint(
    instances: tuple[EffectInstanceV2, ...] | list[EffectInstanceV2],
) -> tuple[str, tuple[dict[str, object], ...]]:
    """Fingerprint only code that will execute; disabled/bypassed drafts are excluded."""

    effective = effective_stack_instances(instances)
    if not effective:
        return "", ()
    return stack_fingerprint(effective)

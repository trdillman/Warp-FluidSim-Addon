"""Strict deterministic Effect Package v2 import, export, and migration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from .enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    FieldAccess,
    ParameterKind,
)
from .fields import FIELD_SPECS
from .schema import (
    EffectDefinitionV2,
    FieldBindingV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)

PACKAGE_SCHEMA_VERSION = 2
MAX_PACKAGE_MEMBERS = 32
MAX_PACKAGE_MEMBER_BYTES = 1 * 1024 * 1024
MAX_PACKAGE_TOTAL_BYTES = 8 * 1024 * 1024
MAX_PACKAGE_COMPRESSION_RATIO = 200.0
_PACKAGE_SUFFIX = ".fseffect"
_FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_ID_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PackageReport:
    path: Path
    sha256: str
    size: int
    effect_id: str
    effect_version: str
    members: tuple[str, ...]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json(data: bytes, label: str):
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _safe_member(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"unsafe archive member {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member {name!r}")
    return str(path)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _validate_archive_limits(infos: list[zipfile.ZipInfo]) -> None:
    """Reject resource-exhaustion archives before any member is decompressed."""

    if len(infos) > MAX_PACKAGE_MEMBERS:
        raise ValueError(
            f"effect package has too many archive members: {len(infos)} > "
            f"{MAX_PACKAGE_MEMBERS}"
        )
    total = 0
    for info in infos:
        size = int(info.file_size)
        compressed = int(info.compress_size)
        if size < 0 or compressed < 0:
            raise ValueError(f"invalid archive size metadata for {info.filename!r}")
        if size > MAX_PACKAGE_MEMBER_BYTES:
            raise ValueError(
                f"archive member {info.filename!r} exceeds the uncompressed size limit: "
                f"{size} > {MAX_PACKAGE_MEMBER_BYTES}"
            )
        total += size
        if total > MAX_PACKAGE_TOTAL_BYTES:
            raise ValueError(
                "effect package exceeds the total uncompressed size limit: "
                f"{total} > {MAX_PACKAGE_TOTAL_BYTES}"
            )
        if size:
            ratio = size / max(1, compressed)
            if ratio > MAX_PACKAGE_COMPRESSION_RATIO:
                raise ValueError(
                    f"archive member {info.filename!r} exceeds the compression ratio limit: "
                    f"{ratio:.1f} > {MAX_PACKAGE_COMPRESSION_RATIO:.1f}"
                )


def _source_members(definition: EffectDefinitionV2) -> dict[str, bytes]:
    members = {}
    for kernel in definition.kernels:
        name = f"sources/{kernel.kernel_id}.py"
        body = kernel.source.rstrip("\n") + "\n"
        members[name] = body.encode("utf-8")
    return members


def _package_payload(definition: EffectDefinitionV2, sources: Mapping[str, bytes]) -> bytes:
    payload = {
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "effect_api_version": definition.effect_api_version,
        "effect_abi_version": definition.effect_abi_version,
        "definition": definition.to_dict(),
        "files": {name: _digest(data) for name, data in sorted(sources.items())},
    }
    return (json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def write_effect_package(path: str | Path, definition: EffectDefinitionV2) -> PackageReport:
    """Write one deterministic package through an fsync-backed atomic replace."""

    path = Path(path)
    if path.suffix.lower() != _PACKAGE_SUFFIX:
        path = path.with_suffix(_PACKAGE_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = _source_members(definition)
    manifest = _package_payload(definition, sources)
    members = {"effect.json": manifest, **sources}
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(members):
                archive.writestr(_zip_info(name), members[name])
        with temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except Exception:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise
    payload = path.read_bytes()
    return PackageReport(
        path=path,
        sha256=_digest(payload),
        size=len(payload),
        effect_id=definition.effect_id,
        effect_version=definition.version,
        members=tuple(sorted(members)),
    )


def read_effect_package(path: str | Path) -> EffectDefinitionV2:
    """Read and fully validate an Effect Package v2 without importing trust."""

    path = Path(path)
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid effect package: {exc}") from exc
    with archive:
        infos = archive.infolist()
        _validate_archive_limits(infos)
        names: list[str] = []
        seen: set[str] = set()
        for info in infos:
            name = _safe_member(info.filename)
            if name in seen:
                raise ValueError(f"duplicate archive member {name!r}")
            seen.add(name)
            names.append(name)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"symlink archive member is forbidden: {name}")
            if info.is_dir():
                raise ValueError(f"directory archive member is forbidden: {name}")
        if "effect.json" not in seen:
            raise ValueError("effect package is missing effect.json")
        raw = {name: archive.read(name) for name in names}
    payload = _strict_json(raw["effect.json"], "effect.json")
    if not isinstance(payload, dict):
        raise ValueError("effect.json must be a JSON object")
    expected = {
        "package_schema_version", "effect_api_version", "effect_abi_version",
        "definition", "files",
    }
    if set(payload) != expected:
        raise ValueError(
            f"invalid effect package keys; missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )
    if payload["package_schema_version"] != PACKAGE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported effect package schema {payload['package_schema_version']}"
        )
    definition = EffectDefinitionV2.from_dict(payload["definition"])
    if int(payload["effect_api_version"]) != definition.effect_api_version:
        raise ValueError("effect package API version does not match definition")
    if int(payload["effect_abi_version"]) != definition.effect_abi_version:
        raise ValueError("effect package ABI version does not match definition")
    files = payload["files"]
    if not isinstance(files, dict):
        raise ValueError("effect package files must be an object")
    expected_members = {"effect.json", *files}
    if set(raw) != expected_members:
        raise ValueError(
            f"effect package member inventory mismatch; expected {sorted(expected_members)}, "
            f"found {sorted(raw)}"
        )
    for name, expected_digest in files.items():
        safe_name = _safe_member(name)
        if safe_name == "effect.json" or not safe_name.startswith("sources/"):
            raise ValueError(f"unsupported effect package member {safe_name!r}")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError(f"invalid digest for {safe_name}")
        actual = _digest(raw[safe_name])
        if actual != expected_digest:
            raise ValueError(
                f"digest mismatch for {safe_name}: expected {expected_digest}, found {actual}"
            )
    for kernel in definition.kernels:
        name = f"sources/{kernel.kernel_id}.py"
        if name not in files:
            raise ValueError(f"effect package is missing source for {kernel.kernel_id}")
        packaged_source = raw[name].decode("utf-8").rstrip("\n")
        if packaged_source != kernel.source.rstrip("\n"):
            raise ValueError(f"definition/source mismatch for kernel {kernel.kernel_id}")
    return definition


def _slug(value: str) -> str:
    result = _ID_RE.sub("-", str(value).strip().lower()).strip("-")
    return result or "legacy-effect"


def migrate_legacy_preset(payload: Mapping[str, object]):
    """Convert one strict legacy snippet payload without mutating the source payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("legacy preset must be an object")
    allowed = {
        "schema_version", "name", "kernel_name", "run_over", "order", "code",
        "arguments", "frame_start", "frame_end", "frame_step", "enabled",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown legacy preset field(s): {', '.join(sorted(unknown))}")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("unsupported legacy preset schema")
    name = str(payload.get("name") or payload.get("kernel_name") or "Legacy Effect")
    kernel_name = str(payload.get("kernel_name") or _slug(name).replace("-", "_"))
    run_over = str(payload.get("run_over", "VOXEL_LINEAR"))
    warnings: list[str] = []
    if run_over in {"VOXEL_LINEAR", "VOXEL_TILE"}:
        domain = EffectDomain.ACTIVE_VOXELS
        if run_over == "VOXEL_TILE":
            warnings.append(
                "Legacy VOXEL_TILE had no distinct tiled semantics and was mapped to ACTIVE_VOXELS."
            )
    elif run_over in {"PARTICLE", "PARTICLES"}:
        domain = EffectDomain.PARTICLES
    else:
        raise ValueError(f"unsupported legacy run_over {run_over!r}")
    stage_map = {
        "FIRST_STEP": EffectStage.BEFORE_EMISSION,
        "BEFORE_SOURCING": EffectStage.BEFORE_EMISSION,
        "BEFORE_FORCES": EffectStage.AFTER_PARTICLE_TRANSFER,
        "BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION": EffectStage.BEFORE_PRESSURE,
        "FINAL_STEP": EffectStage.AFTER_PARTICLE_UPDATE,
    }
    order = str(payload.get("order", "BEFORE_FORCES"))
    if order not in stage_map:
        raise ValueError(f"unsupported legacy order {order!r}")
    bindings: list[FieldBindingV2] = []
    parameters: list[ParameterDefinitionV2] = []
    parameter_keys: list[str] = []
    arguments = payload.get("arguments", ())
    if not isinstance(arguments, (tuple, list)):
        raise ValueError("legacy arguments must be a list")
    for item in arguments:
        if not isinstance(item, Mapping):
            raise ValueError("legacy argument must be an object")
        mode = str(item.get("mode", "FIELD"))
        if mode == "FIELD":
            field = str(item.get("field", ""))
            spec = FIELD_SPECS.get(field)
            if spec is None:
                raise ValueError(f"unknown legacy field {field!r}")
            access = FieldAccess.READ if spec.structural else FieldAccess.READ_WRITE_CURRENT
            bindings.append(FieldBindingV2(field, access))
        elif mode == "CONSTANT":
            key = str(item.get("constant_name", "value"))
            value = float(item.get("constant_value", 0.0))
            parameters.append(ParameterDefinitionV2(
                key=key,
                label=key.replace("_", " ").title(),
                kind=ParameterKind.FLOAT,
                default=value,
                minimum=-1.0e12,
                maximum=1.0e12,
                unit="NONE",
                description="Migrated legacy numeric constant.",
            ))
            parameter_keys.append(key)
        else:
            raise ValueError(f"unsupported legacy argument mode {mode!r}")
    definition = EffectDefinitionV2(
        effect_id=f"legacy.{_slug(name)}",
        version="1.0.0",
        name=name,
        summary=f"Migrated legacy custom Warp snippet {name}.",
        description=(
            "Imported from the legacy Custom Warp Snippets system without source rewriting. "
            "Review and trust the exact content digest before enabling it."
        ),
        category="Migrated",
        author="Unknown legacy author",
        license="NOASSERTION",
        built_in=False,
        package_origin="legacy://custom-warp-snippet",
        parameters=tuple(parameters),
        kernels=(KernelDefinitionV2(
            kernel_id=kernel_name,
            label=name,
            domain=domain,
            stage=stage_map[order],
            cadence=EffectCadence.EVERY_SUBSTEP,
            source=str(payload.get("code", "pass")),
            fields=tuple(bindings),
            parameter_keys=tuple(parameter_keys),
            frame_start=int(payload.get("frame_start", 0)),
            frame_end=int(payload.get("frame_end", 0)),
            frame_step=int(payload.get("frame_step", 1)),
            description="Migrated legacy kernel body.",
        ),),
    )
    return definition, tuple(warnings)

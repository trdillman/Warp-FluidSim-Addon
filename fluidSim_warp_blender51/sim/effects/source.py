"""Deterministic generated Warp source and source-line mapping."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

from .enums import EffectDomain, ParameterKind
from .fields import FIELD_SPECS
from .schema import EffectInstanceV2


def addon_root_package(package: str) -> str:
    marker = ".sim.effects"
    if marker not in package:
        raise ValueError(f"expected a .sim.effects package, got {package!r}")
    return package.split(marker, 1)[0]


_ADDON_ROOT = addon_root_package(__package__)
_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class GeneratedKernelSource:
    source: str
    source_digest: str
    module_name: str
    kernel_name: str
    user_line_start: int
    user_line_count: int
    kernel_index: int
    effect_id: str
    kernel_id: str

    def original_line_for_generated(self, line: int) -> int:
        offset = int(line) - int(self.user_line_start)
        return offset + 1 if 0 <= offset < self.user_line_count else 0

    def cache_path(self, directory: str | Path | None = None) -> Path:
        base = Path(directory) if directory is not None else Path(
            tempfile.gettempdir(), "fluidsim_warp_effects_v2"
        )
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{self.module_name}.py"

    def write_cache_file(self, directory: str | Path | None = None) -> Path:
        path = self.cache_path(directory)
        if path.is_file() and path.read_text(encoding="utf-8") == self.source:
            return path
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        return path


_DTYPE_DECL = {
    "float": "wp.array(dtype=float)",
    "int": "wp.array(dtype=wp.int32)",
    "vec3": "wp.array(dtype=wp.vec3)",
}
_PARAMETER_DECL = {
    ParameterKind.FLOAT: "float",
    ParameterKind.INT: "int",
    ParameterKind.BOOL: "int",
    ParameterKind.VECTOR: "wp.vec3",
    ParameterKind.COLOR: "wp.vec4",
    ParameterKind.ENUM: "int",
    ParameterKind.OBJECT: "wp.vec3",
}


def _safe_identifier(value: str) -> str:
    result = _IDENTIFIER_RE.sub("_", value)
    if not result or result[0].isdigit():
        result = "effect_" + result
    return result


def parameter_runtime_inputs(instance: EffectInstanceV2, kernel_index: int):
    kernel = instance.definition.kernels[kernel_index]
    definitions = instance.definition.parameter_map()
    values = instance.effective_parameter_values()
    result: list[object] = []
    for key in kernel.parameter_keys:
        definition = definitions[key]
        value = values[key]
        if definition.kind is ParameterKind.FLOAT:
            result.append(float(value))
        elif definition.kind is ParameterKind.INT:
            result.append(int(value))
        elif definition.kind is ParameterKind.BOOL:
            result.append(1 if value else 0)
        elif definition.kind is ParameterKind.VECTOR:
            result.append(tuple(float(item) for item in value))
        elif definition.kind is ParameterKind.COLOR:
            result.append(tuple(float(item) for item in value))
        elif definition.kind is ParameterKind.ENUM:
            keys = [item[0] for item in definition.enum_items]
            result.append(keys.index(str(value)))
        elif definition.kind is ParameterKind.OBJECT:
            binding = instance.controller_bindings.get(key, {})
            position = (0.0, 0.0, 0.0)
            if isinstance(binding, dict):
                if "position" in binding:
                    raw = binding["position"]
                    if isinstance(raw, (tuple, list)) and len(raw) >= 3:
                        position = tuple(float(item) for item in raw[:3])
                elif "matrix_world" in binding:
                    raw = binding["matrix_world"]
                    if isinstance(raw, (tuple, list)) and len(raw) == 16:
                        position = (float(raw[3]), float(raw[7]), float(raw[11]))
            result.append(position)
        else:  # pragma: no cover
            raise ValueError(f"unsupported parameter kind {definition.kind}")
    return result


def _signature_lines(instance: EffectInstanceV2, kernel_index: int) -> list[str]:
    kernel = instance.definition.kernels[kernel_index]
    definitions = instance.definition.parameter_map()
    lines = [f"{binding.field}: {_DTYPE_DECL[FIELD_SPECS[binding.field].dtype]}" for binding in kernel.fields]
    lines.extend(f"{key}: {_PARAMETER_DECL[definitions[key].kind]}" for key in kernel.parameter_keys)
    if not lines:
        lines.append("_unused_pad: float")
    return lines


def _base_template(instance: EffectInstanceV2, kernel_index: int, kernel_name: str):
    kernel = instance.definition.kernels[kernel_index]
    extra = _signature_lines(instance, kernel_index)
    common = [
        "voxel_size: float",
        "frame: int",
        "time: float",
        "real_time_step: float",
        "solver_time_step: float",
        "time_scale: float",
        "substeps: int",
    ]
    if kernel.domain is EffectDomain.PARTICLES:
        params = [
            "_positions: wp.array(dtype=wp.vec3)",
            "_velocities: wp.array(dtype=wp.vec3)",
            "_alive: wp.array(dtype=wp.int32)",
            "_count: wp.array(dtype=wp.int32)",
            *common,
            *extra,
        ]
        prelude = [
            "    _tid = wp.tid()",
            "    if _tid >= _count[0]:",
            "        return",
            "    if _alive[_tid] == 0:",
            "        return",
            "    particle_index = _tid",
            "    position = _positions[_tid]",
            "    velocity = _velocities[_tid]",
        ]
        imports = "import warp as wp\nwp.set_module_options({\"enable_backward\": False})\n"
    else:
        params = [
            "_leaf_coords: wp.array(dtype=wp.vec3i)",
            "_num_leaves: wp.array(dtype=wp.int32)",
            "_voxel_type: wp.array(dtype=wp.int32)",
            "_origin: wp.vec3",
            *common,
            *extra,
        ]
        prelude = [
            "    _tid = wp.tid()",
            "    if _tid >= _num_leaves[0] * C.LEAF_VOLUME:",
            "        return",
            "    if (_voxel_type[_tid] & C.VOXEL_COLLISION) != 0:",
            "        return",
            "    flat_voxel_index = _tid",
            "    leaf_index = _tid // C.LEAF_VOLUME",
            "    _ijk = voxel_coord_of_flat(_leaf_coords, _tid)",
            "    global_x = _ijk[0]",
            "    global_y = _ijk[1]",
            "    global_z = _ijk[2]",
            "    local_x = global_x & C.LEAF_MASK",
            "    local_y = global_y & C.LEAF_MASK",
            "    local_z = global_z & C.LEAF_MASK",
            "    world_position = voxel_center_world(_ijk, _origin, voxel_size)",
        ]
        imports = (
            "import warp as wp\n"
            "wp.set_module_options({\"enable_backward\": False})\n"
            f"from {_ADDON_ROOT}.sim import constants as C\n"
            f"from {_ADDON_ROOT}.sim.fields import voxel_coord_of_flat, voxel_center_world\n"
        )
    lines = [imports.rstrip("\n"), "", "@wp.kernel", f"def {kernel_name}("]
    for index, declaration in enumerate(params):
        comma = "," if index < len(params) - 1 else ""
        lines.append(f"    {declaration}{comma}")
    lines.extend(["):", *prelude])
    return lines


def generate_kernel_source(instance: EffectInstanceV2, kernel_index: int) -> GeneratedKernelSource:
    kernel = instance.definition.kernels[kernel_index]
    provisional_name = _safe_identifier(f"effect_{instance.definition.effect_id}_{kernel.kernel_id}")
    lines = _base_template(instance, kernel_index, provisional_name)
    user_line_start = len("\n".join(lines).splitlines()) + 1
    body = textwrap.dedent(kernel.source).strip("\n") or "pass"
    user_lines = body.splitlines() or ["pass"]
    lines.extend("    " + line if line.strip() else "" for line in user_lines)
    provisional = "\n".join(lines) + "\n"
    digest = hashlib.sha256(provisional.encode("utf-8")).hexdigest()
    module_name = f"fluidsim_effect_{digest[:24]}"
    kernel_name = _safe_identifier(f"effect_{kernel.kernel_id}_{digest[:12]}")
    lines = _base_template(instance, kernel_index, kernel_name)
    user_line_start = len("\n".join(lines).splitlines()) + 1
    lines.extend("    " + line if line.strip() else "" for line in user_lines)
    source = "\n".join(lines) + "\n"
    final_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if final_digest != digest:
        module_name = f"fluidsim_effect_{final_digest[:24]}"
    return GeneratedKernelSource(
        source=source,
        source_digest=final_digest,
        module_name=module_name,
        kernel_name=kernel_name,
        user_line_start=user_line_start,
        user_line_count=len(user_lines),
        kernel_index=kernel_index,
        effect_id=instance.definition.effect_id,
        kernel_id=kernel.kernel_id,
    )

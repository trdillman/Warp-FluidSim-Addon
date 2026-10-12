"""Blender scene adapters for the versioned Custom Effects workflow.

The PropertyGroups in this module are authoring state only. Every compile,
apply, cache, and report boundary converts them to immutable v2 snapshots.
"""

from __future__ import annotations

import json
import struct
import uuid
from dataclasses import replace
from pathlib import Path

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from ..sim.effects.builtins import builtin_effect_definitions
from ..sim.effects.compiler import CandidateBuildResult, build_candidate_effects
from ..sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    EffectStatus,
    FieldAccess,
    ParameterKind,
    TrustState,
)
from ..sim.effects.fingerprint import (
    effect_fingerprint,
    effective_stack_instances,
    execution_stack_fingerprint,
)
from ..sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    FieldBindingV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)
from ..sim.effects.trust import TrustStore, content_trust_digest


_PARAMETER_KIND_ITEMS = tuple((item.value, item.name.replace("_", " ").title(), "") for item in ParameterKind)
_STATUS_ITEMS = tuple((item.value, item.name.replace("_", " ").title(), "") for item in EffectStatus)
_TRUST_ITEMS = tuple((item.value, item.name.replace("_", " ").title(), "") for item in TrustState)
_DOMAIN_ITEMS = (("PARTICLES", "Particles", "Run over live liquid particles"), ("ACTIVE_VOXELS", "Active Voxels", "Run over active sparse-grid voxels"))
_STAGE_ITEMS = (
    ("BEFORE_EMISSION", "Before Emission",
     "Start of the substep, before emission and particle-to-grid transfer; use for particle velocity forces"),
    ("AFTER_PARTICLE_TRANSFER", "After Particle Transfer",
     "After particle-to-grid transfer; edit grid fields here (the grid transfer replaces particle velocity edits)"),
    ("BEFORE_PRESSURE", "Before Pressure", "Immediately before pressure projection"),
    ("AFTER_PARTICLE_UPDATE", "After Particle Update", "After advection and lifecycle"),
)
_CADENCE_ITEMS = (
    ("EVERY_SUBSTEP", "Every Substep", "Run on each solver substep"),
    ("ONCE_PER_FRAME", "Once per Frame", "Run only on the first substep of each frame"),
    ("INITIALIZATION", "Initialization Only", "Run once after the stack is applied"),
)
_ACCESS_ITEMS = (
    ("READ", "Read", "Read-only access"),
    ("WRITE_CURRENT", "Write Current", "Write only the current particle or voxel"),
    ("READ_WRITE_CURRENT", "Read/Write Current", "Read and write only the current element"),
    ("ATOMIC_SCATTER", "Atomic Scatter", "Expert atomic writes"),
    ("UNSAFE_ARBITRARY", "Unsafe Arbitrary", "Expert unrestricted indexing"),
)
_POLICY_ITEMS = (
    ("DENY", "Deny", "Reject every custom effect in background execution"),
    ("BUILTINS_ONLY", "Built-ins Only", "Allow only bundled FluidSim effects"),
    ("TRUSTED", "Trusted", "Allow built-ins and exact locally trusted content digests"),
)


def _root_package() -> str:
    return __package__.rsplit(".blender", 1)[0]


def effect_user_root(*, create: bool = True) -> Path:
    try:
        return Path(bpy.utils.extension_path_user(
            _root_package(), path="custom_effects", create=create
        ))
    except Exception:
        fallback = Path.home() / ".fluidsim" / "custom_effects"
        if create:
            fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def trust_store_for_scene(_scene=None) -> TrustStore:
    return TrustStore(effect_user_root(create=True) / "trust.json")


def _mark_dirty(_self=None, context=None):
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is not None and hasattr(scene, "fluidsim_effects"):
        try:
            if getattr(_self, "key", "") and _try_live_parameter_update(scene):
                return
            refresh_stack_state(scene)
        except Exception as exc:  # authoring feedback, not an execution path
            scene.fluidsim_effects.blocking_reason = f"Effect snapshot error: {exc}"


_BUILDING_INSTANCE = False


def _mark_structure_dirty(_self=None, context=None):
    """Refresh after an edit to a parameter's definition (key/kind/limits/...).

    Unlike ``_mark_dirty`` this never takes the live-value path, and it is a
    no-op while ``add_instance_to_scene`` is still populating the group.
    """
    scene = getattr(context, "scene", None) if context is not None else None
    if _BUILDING_INSTANCE or scene is None or not hasattr(scene, "fluidsim_effects"):
        return
    try:
        refresh_stack_state(scene)
    except Exception as exc:  # authoring feedback, not an execution path
        scene.fluidsim_effects.blocking_reason = f"Effect snapshot error: {exc}"


def _parameter_keys_edited(self, context=None):
    self.parameter_keys_authored = True
    _mark_dirty(self, context)


class FLUIDSIM_PG_effect_parameter(bpy.types.PropertyGroup):
    key: StringProperty(update=_mark_structure_dirty)
    label: StringProperty(update=_mark_structure_dirty)
    kind: EnumProperty(items=_PARAMETER_KIND_ITEMS, default="FLOAT", update=_mark_structure_dirty)
    unit: StringProperty(default="NONE", update=_mark_structure_dirty)
    description: StringProperty(update=_mark_structure_dirty)
    minimum: FloatProperty(default=0.0, update=_mark_structure_dirty)
    maximum: FloatProperty(default=0.0, update=_mark_structure_dirty)
    has_minimum: BoolProperty(default=False, update=_mark_structure_dirty)
    has_maximum: BoolProperty(default=False, update=_mark_structure_dirty)
    float_value: FloatProperty(update=_mark_dirty)
    int_value: IntProperty(update=_mark_dirty)
    bool_value: BoolProperty(update=_mark_dirty)
    vector_value: FloatVectorProperty(size=3, update=_mark_dirty)
    color_value: FloatVectorProperty(size=4, subtype="COLOR", min=0.0, max=1.0, update=_mark_dirty)
    enum_value: StringProperty(update=_mark_dirty)
    enum_items_json: StringProperty(
        default="[]",
        description='Enum choices as a JSON list of [id, label] pairs, e.g. [["LOW", "Low"], ["HIGH", "High"]]',
        update=_mark_structure_dirty,
    )
    object_value: PointerProperty(type=bpy.types.Object, update=_mark_dirty)


class FLUIDSIM_PG_effect_field(bpy.types.PropertyGroup):
    field: StringProperty()
    access: EnumProperty(items=_ACCESS_ITEMS, default="READ", update=_mark_dirty)


class FLUIDSIM_PG_effect_kernel(bpy.types.PropertyGroup):
    kernel_id: StringProperty()
    label: StringProperty()
    domain: EnumProperty(items=_DOMAIN_ITEMS, default="PARTICLES", update=_mark_dirty)
    stage: EnumProperty(items=_STAGE_ITEMS, default="AFTER_PARTICLE_TRANSFER", update=_mark_dirty)
    cadence: EnumProperty(items=_CADENCE_ITEMS, default="EVERY_SUBSTEP", update=_mark_dirty)
    source_text: PointerProperty(type=bpy.types.Text, update=_mark_dirty)
    fields: CollectionProperty(type=FLUIDSIM_PG_effect_field)
    safe_mode: BoolProperty(default=True, update=_mark_dirty)
    frame_start: IntProperty(default=0, min=0, update=_mark_dirty)
    frame_end: IntProperty(default=0, min=0, update=_mark_dirty)
    frame_step: IntProperty(default=1, min=1, update=_mark_dirty)
    parameter_keys_csv: StringProperty(default="", update=_parameter_keys_edited)
    parameter_keys_authored: BoolProperty(default=False)
    error_line: IntProperty(default=0)
    error_column: IntProperty(default=0)
    generated_source_digest: StringProperty()
    generated_text_name: StringProperty()


class FLUIDSIM_PG_effect_diagnostic(bpy.types.PropertyGroup):
    code: StringProperty()
    message: StringProperty()
    severity: StringProperty(default="ERROR")
    instance_id: StringProperty()
    kernel_id: StringProperty()
    line: IntProperty(default=0)
    column: IntProperty(default=0)
    hint: StringProperty()


class FLUIDSIM_PG_effect_instance(bpy.types.PropertyGroup):
    instance_id: StringProperty()
    effect_id: StringProperty()
    effect_version: StringProperty()
    definition_json: StringProperty()
    display_name: StringProperty(update=_mark_dirty)
    enabled: BoolProperty(default=True, update=_mark_dirty)
    muted: BoolProperty(default=False, update=_mark_dirty)
    solo: BoolProperty(default=False, update=_mark_dirty)
    order: IntProperty(default=0, update=_mark_dirty)
    frame_start: IntProperty(default=0, min=0, update=_mark_dirty)
    frame_end: IntProperty(default=0, min=0, update=_mark_dirty)
    frame_step: IntProperty(default=1, min=1, update=_mark_dirty)
    safe_mode: BoolProperty(default=True, update=_mark_dirty)
    expert_acknowledged: BoolProperty(default=False, update=_mark_dirty)
    parameters: CollectionProperty(type=FLUIDSIM_PG_effect_parameter)
    kernels: CollectionProperty(type=FLUIDSIM_PG_effect_kernel)
    diagnostics: CollectionProperty(type=FLUIDSIM_PG_effect_diagnostic)
    status: EnumProperty(items=_STATUS_ITEMS, default="DRAFT")
    trust_state: EnumProperty(items=_TRUST_ITEMS, default="UNTRUSTED_IMPORTED")
    content_digest: StringProperty()
    draft_fingerprint: StringProperty()
    tested_fingerprint: StringProperty()
    applied_fingerprint: StringProperty()
    compile_report_json: StringProperty(default="{}")
    sandbox_report_json: StringProperty(default="{}")
    runtime_error: StringProperty()
    calls: IntProperty(default=0)
    elements: IntProperty(default=0)
    submission_ms: FloatProperty(default=0.0)


class FLUIDSIM_PG_effect_stack(bpy.types.PropertyGroup):
    items: CollectionProperty(type=FLUIDSIM_PG_effect_instance)
    active_index: IntProperty(default=0, min=0)
    draft_signature: StringProperty()
    tested_signature: StringProperty()
    applied_signature: StringProperty()
    applied_inventory_json: StringProperty(default="[]")
    applied_snapshot_json: StringProperty(default="[]")
    status: EnumProperty(items=_STATUS_ITEMS, default="DRAFT")
    blocking_reason: StringProperty()
    custom_code_policy: EnumProperty(items=_POLICY_ITEMS, default="BUILTINS_ONLY")
    artist_mode: BoolProperty(default=True)
    expert_acknowledged: BoolProperty(default=False)
    library_query: StringProperty()
    library_category: StringProperty()
    selected_library_effect: StringProperty(default="fluidsim.vortex_force")
    last_compile_device: StringProperty()
    last_warp_version: StringProperty()
    last_compile_seconds: FloatProperty(default=0.0)
    last_sandbox_seconds: FloatProperty(default=0.0)


CLASSES = (
    FLUIDSIM_PG_effect_parameter,
    FLUIDSIM_PG_effect_field,
    FLUIDSIM_PG_effect_kernel,
    FLUIDSIM_PG_effect_diagnostic,
    FLUIDSIM_PG_effect_instance,
    FLUIDSIM_PG_effect_stack,
)


def _parameter_value(pg):
    kind = ParameterKind(pg.kind)
    if kind is ParameterKind.FLOAT:
        return float(pg.float_value)
    if kind is ParameterKind.INT:
        return int(pg.int_value)
    if kind is ParameterKind.BOOL:
        return bool(pg.bool_value)
    if kind is ParameterKind.VECTOR:
        return tuple(float(value) for value in pg.vector_value)
    if kind is ParameterKind.COLOR:
        return tuple(float(value) for value in pg.color_value)
    if kind is ParameterKind.ENUM:
        return str(pg.enum_value)
    if kind is ParameterKind.OBJECT:
        return str(pg.object_value.name) if pg.object_value is not None else ""
    raise ValueError(f"unsupported parameter kind {pg.kind}")


def _controller_snapshot(pg):
    obj = pg.object_value
    if obj is None:
        return {}
    matrix_world = tuple(float(value) for row in obj.matrix_world for value in row)
    return {
        "object_name": str(obj.name),
        "matrix_world": matrix_world,
        "position": tuple(float(value) for value in obj.matrix_world.translation),
    }


def unique_id(prefix: str, existing) -> str:
    """Return ``prefix_N`` not present in ``existing`` (``len()+1`` collides after removals)."""
    taken = set(existing)
    number = len(taken) + 1
    while f"{prefix}_{number}" in taken:
        number += 1
    return f"{prefix}_{number}"


def _as_float32(value: float) -> float | None:
    try:
        return struct.unpack("f", struct.pack("f", float(value)))[0]
    except OverflowError:
        return None


def _pg_limit(enabled: bool, pg_value: float, base_value):
    """A parameter min/max from its float32 PG storage.

    Unticked clears the limit.  An unchanged limit keeps the exact definition
    value, so float32 rounding never perturbs the content digest.
    """
    if not enabled:
        return None
    if base_value is not None and _as_float32(base_value) == float(pg_value):
        return base_value
    return float(pg_value)


def _pg_default(pg_value, base_default):
    """A parameter default from its PG storage; an unedited value keeps the exact definition value."""
    if base_default is None:
        return pg_value
    if isinstance(pg_value, float) and not isinstance(base_default, bool):
        if isinstance(base_default, (int, float)) and _as_float32(base_default) == pg_value:
            return base_default
    elif isinstance(pg_value, tuple) and isinstance(base_default, (tuple, list)):
        if len(base_default) == len(pg_value) and all(
            _as_float32(b) == p for b, p in zip(base_default, pg_value, strict=True)
        ):
            return base_default
    return pg_value


def parameter_definition_from_pg(item, base: ParameterDefinitionV2 | None = None) -> ParameterDefinitionV2:
    kind = ParameterKind(item.kind)
    if base is not None and base.kind is not kind:
        base = None  # kind changed: the old default/enum items no longer apply
    enum_items = base.enum_items if base is not None else ()
    if item.enum_items_json:
        try:
            parsed = json.loads(item.enum_items_json)
            if parsed:
                enum_items = tuple(tuple(pair) for pair in parsed)
        except Exception:
            pass
    val = _parameter_value(item)
    min_val = _pg_limit(item.has_minimum, item.minimum, base.minimum if base is not None else None)
    max_val = _pg_limit(item.has_maximum, item.maximum, base.maximum if base is not None else None)
    return ParameterDefinitionV2(
        key=str(item.key),
        label=str(item.label or item.key),
        kind=kind,
        default=base.default if base is not None else val,
        minimum=min_val,
        maximum=max_val,
        unit=str(item.unit or (base.unit if base is not None else "NONE")),
        description=str(item.description or (base.description if base is not None else "")),
        enum_items=enum_items,
        runtime_bindable=base.runtime_bindable if base is not None else True,
        animatable=base.animatable if base is not None else True,
    )


def _kernel_parameter_keys(kernel_pg, base: KernelDefinitionV2 | None) -> tuple[str, ...]:
    csv = str(getattr(kernel_pg, "parameter_keys_csv", ""))
    if base is None or csv.strip() or getattr(kernel_pg, "parameter_keys_authored", True):
        # The PG is authoritative once authored, so an emptied field really unbinds.
        return tuple(k.strip() for k in csv.split(",") if k.strip())
    return base.parameter_keys  # .blend saved before the field existed


def unbind_parameter_key(pg, key: str) -> None:
    """Drop ``key`` from every kernel's bound parameters (before removing the parameter)."""
    original = {}
    if pg.definition_json:
        definition = EffectDefinitionV2.from_dict(json.loads(pg.definition_json))
        original = {kernel.kernel_id: kernel for kernel in definition.kernels}
    for kernel_pg in pg.kernels:
        keys = _kernel_parameter_keys(kernel_pg, original.get(kernel_pg.kernel_id))
        if key in keys:
            kernel_pg.parameter_keys_csv = ", ".join(k for k in keys if k != key)
            kernel_pg.parameter_keys_authored = True


def kernel_definition_from_pg(kernel_pg, base: KernelDefinitionV2 | None = None) -> KernelDefinitionV2:
    source = (
        kernel_pg.source_text.as_string()
        if kernel_pg.source_text is not None
        else (base.source if base is not None else "pass")
    )
    fields = tuple(FieldBindingV2(item.field, FieldAccess(item.access)) for item in kernel_pg.fields)
    param_keys = _kernel_parameter_keys(kernel_pg, base)

    if base is not None:
        return replace(
            base,
            label=str(kernel_pg.label or base.label),
            domain=EffectDomain(kernel_pg.domain),
            stage=EffectStage(kernel_pg.stage),
            cadence=EffectCadence(kernel_pg.cadence),
            source=source.rstrip("\n") or "pass",
            fields=fields,
            parameter_keys=param_keys,
            safe_mode=bool(kernel_pg.safe_mode),
            frame_start=int(kernel_pg.frame_start),
            frame_end=int(kernel_pg.frame_end),
            frame_step=max(1, int(kernel_pg.frame_step)),
        )

    return KernelDefinitionV2(
        kernel_id=str(kernel_pg.kernel_id),
        label=str(kernel_pg.label or kernel_pg.kernel_id),
        domain=EffectDomain(kernel_pg.domain),
        stage=EffectStage(kernel_pg.stage),
        cadence=EffectCadence(kernel_pg.cadence),
        source=source.rstrip("\n") or "pass",
        fields=fields,
        parameter_keys=param_keys,
        safe_mode=bool(kernel_pg.safe_mode),
        frame_start=int(kernel_pg.frame_start),
        frame_end=int(kernel_pg.frame_end),
        frame_step=max(1, int(kernel_pg.frame_step)),
    )


def rebuild_definition_json(scene, pg) -> EffectDefinitionV2:
    """Rebuild pg.definition_json and content_digest from the live RNA authoring properties."""
    raw_def = json.loads(pg.definition_json)
    definition = EffectDefinitionV2.from_dict(raw_def)
    param_map = {p.key: p for p in definition.parameters}
    param_defs = [parameter_definition_from_pg(item, base=param_map.get(item.key)) for item in pg.parameters]
    original = {kernel.kernel_id: kernel for kernel in definition.kernels}
    kernels = [kernel_definition_from_pg(kernel_pg, original.get(kernel_pg.kernel_id)) for kernel_pg in pg.kernels]

    definition = replace(
        definition,
        name=str(pg.display_name or definition.name),
        parameters=tuple(param_defs),
        kernels=tuple(kernels) if kernels else definition.kernels,
    )
    pg.definition_json = json.dumps(definition.to_dict(), sort_keys=True, allow_nan=False)
    digest = content_trust_digest(definition)
    pg.content_digest = digest
    pg.display_name = definition.name
    store = trust_store_for_scene(scene)
    state = store.status(digest, built_in=definition.built_in)
    pg.trust_state = state.value
    if pg.status != "TESTED":
        pg.status = "DRAFT" if state in {TrustState.BUILT_IN_TRUSTED, TrustState.LOCAL_TRUSTED} else "UNTRUSTED"
    refresh_stack_state(scene)
    return definition


def add_custom_effect_from_template(scene, template_key: str, *, effect_name: str | None = None):
    from ..sim.effects.templates import get_template_definition
    definition = get_template_definition(template_key)
    uid = uuid.uuid4().hex[:12]
    custom_id = f"custom.{template_key.lower()}_{uid[:6]}"
    display_name = effect_name or definition.name
    definition = replace(
        definition,
        effect_id=custom_id,
        name=display_name,
        built_in=False,
    )
    instance = EffectInstanceV2(
        instance_id=f"effect-{uid}",
        definition=definition,
        order=len(scene.fluidsim_effects.items),
        display_name=display_name,
    )
    pg = add_instance_to_scene(scene, instance)
    object_params = [p for p in definition.parameters if p.kind is ParameterKind.OBJECT]
    for p_def in object_params:
        suffix = f" {p_def.label}" if len(object_params) > 1 else ""
        empty = bpy.data.objects.new(f"FluidSim {display_name}{suffix} Controller", None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.5
        empty["fluidsim_owned"] = True
        empty["fluidsim_effect_controller"] = instance.instance_id
        empty["fluidsim_effect_parameter"] = p_def.key
        target_collection = getattr(scene, "collection", None)
        if target_collection is not None:
            target_collection.objects.link(empty)
        empty.location = (0.0, 0.0, 1.0)
        for param_pg in pg.parameters:
            if param_pg.key == p_def.key:
                param_pg.object_value = empty
                break
    if object_params:
        rebuild_definition_json(scene, pg)
    return pg


def convert_instance_to_custom(scene, pg):
    """Detach a built-in or imported package instance into an editable custom effect."""
    uid = uuid.uuid4().hex[:8]
    old_def = json.loads(pg.definition_json)
    old_id = old_def.get("effect_id", "custom")
    clean_id = old_id.split(".")[-1]
    new_id = f"custom.{clean_id}_{uid}"
    pg.effect_id = new_id
    pg.display_name = f"{pg.display_name} (Custom)"
    instance = effect_instance_from_pg(scene, pg)
    new_def = replace(
        instance.definition,
        effect_id=new_id,
        name=pg.display_name,
        built_in=False,
    )
    pg.definition_json = json.dumps(new_def.to_dict(), sort_keys=True, allow_nan=False)
    digest = content_trust_digest(new_def)
    pg.content_digest = digest
    pg.trust_state = TrustState.UNTRUSTED_IMPORTED.value
    pg.status = "DRAFT"
    refresh_stack_state(scene)
    return pg


def exportable_definition(scene, pg) -> EffectDefinitionV2:
    """The definition to package, with each parameter's current value as its default.

    Only export bakes values into defaults: at run time the PG value is the live,
    possibly animated value, and folding it into the definition would change the
    trust digest on every edit.
    """
    definition = effect_instance_from_pg(scene, pg).definition
    by_key = {item.key: item for item in pg.parameters}
    parameters = tuple(
        replace(p, default=_pg_default(_parameter_value(by_key[p.key]), p.default))
        if p.key in by_key and p.kind is not ParameterKind.OBJECT
        else p
        for p in definition.parameters
    )
    return replace(definition, parameters=parameters, built_in=False)


def effect_instance_from_pg(scene, pg) -> EffectInstanceV2:
    """Build one immutable execution snapshot from Blender authoring state."""

    definition = EffectDefinitionV2.from_dict(json.loads(pg.definition_json))
    original = {kernel.kernel_id: kernel for kernel in definition.kernels}
    kernels = []
    for kernel_pg in pg.kernels:
        base = original.get(kernel_pg.kernel_id)
        kernels.append(kernel_definition_from_pg(kernel_pg, base))

    definition = replace(definition, kernels=tuple(kernels))
    if not definition.built_in:
        # Authored effects: the PG is the source of truth for name and parameters.
        # Built-ins keep their shipped definition so the trust digest stays exact.
        param_map = {p.key: p for p in definition.parameters}
        definition = replace(
            definition,
            name=str(pg.display_name or definition.name),
            parameters=tuple(
                parameter_definition_from_pg(item, param_map.get(item.key)) for item in pg.parameters
            ),
        )
    parameter_values = {item.key: _parameter_value(item) for item in pg.parameters}
    controllers = {
        item.key: _controller_snapshot(item)
        for item in pg.parameters
        if item.kind == ParameterKind.OBJECT.value and item.object_value is not None
    }
    return EffectInstanceV2(
        instance_id=str(pg.instance_id),
        definition=definition,
        enabled=bool(pg.enabled),
        muted=bool(pg.muted),
        solo=bool(pg.solo),
        order=int(pg.order),
        parameter_values=parameter_values,
        controller_bindings=controllers,
        frame_start=int(pg.frame_start),
        frame_end=int(pg.frame_end),
        frame_step=max(1, int(pg.frame_step)),
        safe_mode=bool(pg.safe_mode),
        expert_acknowledged=bool(pg.expert_acknowledged),
        display_name=str(pg.display_name),
    )


def scene_effect_instances(scene) -> tuple[EffectInstanceV2, ...]:
    indexed = [
        (int(item.order), index, effect_instance_from_pg(scene, item))
        for index, item in enumerate(scene.fluidsim_effects.items)
    ]
    indexed.sort(key=lambda row: (row[0], row[1]))
    return tuple(row[2] for row in indexed)


def _set_parameter_pg(target, definition, value, scene=None):
    target.key = definition.key
    target.label = definition.label
    target.kind = definition.kind.value
    target.unit = definition.unit
    target.description = definition.description
    target.has_minimum = definition.minimum is not None
    target.has_maximum = definition.maximum is not None
    target.minimum = float(definition.minimum or 0.0)
    target.maximum = float(definition.maximum or 0.0)
    target.enum_items_json = json.dumps(definition.enum_items)
    if definition.kind is ParameterKind.FLOAT:
        target.float_value = float(value)
    elif definition.kind is ParameterKind.INT:
        target.int_value = int(value)
    elif definition.kind is ParameterKind.BOOL:
        target.bool_value = bool(value)
    elif definition.kind is ParameterKind.VECTOR:
        target.vector_value = tuple(value)
    elif definition.kind is ParameterKind.COLOR:
        target.color_value = tuple(value)
    elif definition.kind is ParameterKind.ENUM:
        target.enum_value = str(value)
    elif definition.kind is ParameterKind.OBJECT and scene is not None and value:
        target.object_value = scene.objects.get(str(value))


def add_instance_to_scene(scene, instance: EffectInstanceV2, *, select: bool = True):
    global _BUILDING_INSTANCE
    _BUILDING_INSTANCE = True
    try:
        return _add_instance_to_scene(scene, instance, select=select)
    finally:
        _BUILDING_INSTANCE = False


def _add_instance_to_scene(scene, instance: EffectInstanceV2, *, select: bool = True):
    stack = scene.fluidsim_effects
    pg = stack.items.add()
    pg.instance_id = instance.instance_id
    pg.effect_id = instance.definition.effect_id
    pg.effect_version = instance.definition.version
    pg.definition_json = json.dumps(instance.definition.to_dict(), sort_keys=True, allow_nan=False)
    pg.display_name = instance.display_name or instance.definition.name
    pg.enabled = bool(instance.enabled)
    pg.muted = bool(instance.muted)
    pg.solo = bool(instance.solo)
    pg.order = int(instance.order)
    pg.frame_start = int(instance.frame_start)
    pg.frame_end = int(instance.frame_end)
    pg.frame_step = int(instance.frame_step)
    pg.safe_mode = bool(instance.safe_mode)
    pg.expert_acknowledged = bool(instance.expert_acknowledged)
    values = instance.effective_parameter_values()
    for definition in instance.definition.parameters:
        parameter_pg = pg.parameters.add()
        _set_parameter_pg(parameter_pg, definition, values[definition.key], scene)
        binding = instance.controller_bindings.get(definition.key, {})
        if definition.kind is ParameterKind.OBJECT and isinstance(binding, dict):
            object_name = binding.get("object_name")
            if object_name:
                parameter_pg.object_value = scene.objects.get(str(object_name))
    for kernel in instance.definition.kernels:
        kernel_pg = pg.kernels.add()
        kernel_pg.kernel_id = kernel.kernel_id
        kernel_pg.label = kernel.label
        kernel_pg.domain = kernel.domain.value
        kernel_pg.stage = kernel.stage.value
        kernel_pg.cadence = kernel.cadence.value
        kernel_pg.safe_mode = kernel.safe_mode
        kernel_pg.frame_start = kernel.frame_start
        kernel_pg.frame_end = kernel.frame_end
        kernel_pg.frame_step = kernel.frame_step
        kernel_pg.parameter_keys_csv = ", ".join(kernel.parameter_keys)
        kernel_pg.parameter_keys_authored = True
        text_name = f"FluidSim Effect - {instance.instance_id} - {kernel.kernel_id}.py"
        text = bpy.data.texts.get(text_name) or bpy.data.texts.new(text_name)
        text.clear()
        text.write(kernel.source.rstrip("\n") + "\n")
        text["fluidsim_effect_instance"] = instance.instance_id
        text["fluidsim_effect_kernel"] = kernel.kernel_id
        kernel_pg.source_text = text
        for binding in kernel.fields:
            field_pg = kernel_pg.fields.add()
            field_pg.field = binding.field
            field_pg.access = binding.access.value
    digest = content_trust_digest(instance.definition)
    pg.content_digest = digest
    state = trust_store_for_scene(scene).status(digest, built_in=instance.definition.built_in)
    pg.trust_state = state.value
    pg.status = "DRAFT" if state in {TrustState.BUILT_IN_TRUSTED, TrustState.LOCAL_TRUSTED} else "UNTRUSTED"
    if select:
        stack.active_index = len(stack.items) - 1
    refresh_stack_state(scene)
    return pg


def add_builtin_effect(scene, effect_id: str, *, instance_id: str | None = None):
    definitions = {item.effect_id: item for item in builtin_effect_definitions()}
    definition = definitions[str(effect_id)]
    identifier = instance_id or f"effect-{uuid.uuid4().hex[:12]}"
    instance = EffectInstanceV2(
        identifier,
        definition,
        order=len(scene.fluidsim_effects.items),
        display_name=definition.name,
    )
    return add_instance_to_scene(scene, instance)


def clear_effects(scene, *, remove_owned_texts: bool = False):
    stack = scene.fluidsim_effects
    if remove_owned_texts:
        names = [
            kernel.source_text.name
            for item in stack.items
            for kernel in item.kernels
            if kernel.source_text is not None
        ]
        for name in names:
            text = bpy.data.texts.get(name)
            if text is not None and text.users <= 1:
                bpy.data.texts.remove(text)
    stack.items.clear()
    stack.active_index = 0
    refresh_stack_state(scene)


def _publish_applied_state(scene, instances) -> None:
    """Persist one already-validated applied stack snapshot into Blender state."""

    stack = scene.fluidsim_effects
    effective_ids = {item.instance_id for item in effective_stack_instances(instances)}
    signature, inventory = execution_stack_fingerprint(instances)
    stack.applied_signature = signature
    stack.applied_inventory_json = json.dumps(inventory, sort_keys=True, allow_nan=False)
    stack.applied_snapshot_json = json.dumps(
        [instance.to_dict() for instance in instances], sort_keys=True, allow_nan=False
    )
    stack.tested_signature = signature
    instances_by_id = {instance.instance_id: instance for instance in instances}
    for pg in stack.items:
        fingerprint = effect_fingerprint(instances_by_id[str(pg.instance_id)])
        pg.applied_fingerprint = fingerprint
        pg.tested_fingerprint = fingerprint
        pg.status = "APPLIED" if pg.instance_id in effective_ids else "DISABLED"
    stack.status = "APPLIED"
    stack.blocking_reason = ""


def _try_live_parameter_update(scene) -> bool:
    """Publish runtime-bindable parameter edits into the active solver in place."""

    stack = scene.fluidsim_effects
    if not stack.applied_signature:
        return False
    from . import production_bake, scene_bridge

    if production_bake.any_bake_active():
        # Held for reapply_after_bake: the baking solver keeps the effect
        # parameters its cache identity was recorded with (decision D2).
        return False
    session = scene_bridge.session()
    if not session.active or not session.owns_scene(scene):
        return False
    try:
        instances = build_effect_stack_for_admission(scene)
    except (RuntimeError, ValueError):
        return False
    if not session.solver.update_custom_effect_runtime_bindings(instances):
        return False
    _publish_applied_state(scene, instances)
    return True


def _live_bindable_values(pg, applied: EffectInstanceV2, *,
                          animated_only: frozenset[str] | None = None) -> EffectInstanceV2:
    """``applied`` with its runtime-bindable values re-read from the scene.

    F-curve/driver evaluation and controller-object motion change these
    values without an RNA update callback, so they are read every step.
    Non-bindable values keep their applied snapshot. With ``animated_only``
    (a Production Bake) only those keyframed or driven keys and the motion of
    the admitted controller objects are read; any other edit stays held.
    """
    by_key = {str(item.key): item for item in pg.parameters}
    values = dict(applied.parameter_values)
    controllers = dict(applied.controller_bindings)
    for definition in applied.definition.parameters:
        item = by_key.get(definition.key)
        if item is None or not definition.runtime_bindable:
            continue
        if definition.kind is ParameterKind.OBJECT:
            snapshot = _controller_snapshot(item)
            if animated_only is not None and (
                    snapshot.get("object_name")
                    != controllers.get(definition.key, {}).get("object_name")):
                continue  # a different controller object is an edit, not motion
            values[definition.key] = _parameter_value(item)
            if snapshot:
                controllers[definition.key] = snapshot
            else:
                controllers.pop(definition.key, None)
        elif animated_only is None or definition.key in animated_only:
            values[definition.key] = _parameter_value(item)
    return replace(applied, parameter_values=values, controller_bindings=controllers)


# The value property each parameter kind stores; F-curves and drivers name it.
_PARAMETER_VALUE_PROPS = {
    ParameterKind.FLOAT.value: "float_value",
    ParameterKind.INT.value: "int_value",
    ParameterKind.BOOL.value: "bool_value",
    ParameterKind.VECTOR.value: "vector_value",
    ParameterKind.COLOR.value: "color_value",
    ParameterKind.ENUM.value: "enum_value",
}


def _action_data_paths(action, slot=None) -> set[str]:
    """F-curve data paths of ``action`` for ``slot`` (all slots when unknown)."""
    paths = set()
    legacy = getattr(action, "fcurves", None)  # pre-slotted Action API
    if legacy is not None:
        paths.update(str(fcurve.data_path) for fcurve in legacy)
    handle = getattr(slot, "handle", None)
    for layer in getattr(action, "layers", ()) or ():
        for strip in getattr(layer, "strips", ()) or ():
            for bag in getattr(strip, "channelbags", ()) or ():
                bag_handle = getattr(bag, "slot_handle", None)
                if handle is not None and bag_handle is not None and bag_handle != handle:
                    continue
                paths.update(str(fcurve.data_path) for fcurve in bag.fcurves)
    return paths


def _animated_effect_paths(scene) -> frozenset[str]:
    """Effect data paths on ``scene`` that an F-curve, NLA strip or driver sets."""
    anim = getattr(scene, "animation_data", None)
    if anim is None:
        return frozenset()
    paths = set()
    action = getattr(anim, "action", None)
    if action is not None:
        paths |= _action_data_paths(action, getattr(anim, "action_slot", None))
    for track in getattr(anim, "nla_tracks", ()) or ():
        if getattr(track, "mute", False):
            continue
        for strip in getattr(track, "strips", ()) or ():
            if getattr(strip, "action", None) is not None and not getattr(strip, "mute", False):
                paths |= _action_data_paths(
                    strip.action, getattr(strip, "action_slot", None))
    paths.update(str(driver.data_path) for driver in getattr(anim, "drivers", ()) or ())
    return frozenset(path for path in paths if path.startswith("fluidsim_effects."))


def animated_parameter_keys(scene) -> dict[str, frozenset[str]]:
    """Per instance id, the parameter keys a keyframe or driver sets."""
    stack = getattr(scene, "fluidsim_effects", None)
    paths = _animated_effect_paths(scene) if stack is not None else frozenset()
    if not paths:
        return {}
    result = {}
    for index, pg in enumerate(stack.items):
        keys = frozenset(
            str(item.key) for slot, item in enumerate(pg.parameters)
            if f"fluidsim_effects.items[{index}].parameters[{slot}]."
               f"{_PARAMETER_VALUE_PROPS.get(str(item.kind), '')}" in paths)
        if keys:
            result[str(pg.instance_id)] = keys
    return result


def effect_frame_stamp(scene) -> str:
    """Digest of the running effect values animation can change at this frame.

    Covers keyframed/driven parameter values and controller object transforms
    of the effects that execute. ``""`` when there are none, so a scene without
    animated effects keeps the per-frame cache stamp it always had.
    """
    stack = getattr(scene, "fluidsim_effects", None)
    if stack is None or not len(stack.items):
        return ""
    from ..sim.effects.fingerprint import sha256_hex

    try:
        running = {instance.instance_id for instance in
                   effective_stack_instances(scene_effect_instances(scene))}
    except (RuntimeError, ValueError):
        return ""
    animated = animated_parameter_keys(scene)
    parts = []
    for pg in stack.items:
        instance_id = str(pg.instance_id)
        if instance_id not in running:
            continue
        keys = animated.get(instance_id, frozenset())
        for item in pg.parameters:
            if str(item.key) in keys:
                parts.append([instance_id, str(item.key), _parameter_value(item)])
            elif (getattr(item, "kind", "") == ParameterKind.OBJECT.value
                  and getattr(item, "object_value", None) is not None):
                parts.append([instance_id, str(item.key), _controller_snapshot(item)])
    return sha256_hex(parts) if parts else ""


def _effect_values_may_change(scene, stack) -> bool:
    """Cheap gate: can an effect value change without an update callback?

    Effect parameters live on the Scene, so their F-curves, NLA strips and
    drivers are on the Scene's animation data; controllers are objects that
    can move. With neither, a step reads nothing else.
    """
    anim = getattr(scene, "animation_data", None)
    if anim is not None and (
            getattr(anim, "action", None) is not None
            or len(getattr(anim, "drivers", ()) or ())
            or len(getattr(anim, "nla_tracks", ()) or ())):
        return True
    return any(
        getattr(item, "object_value", None) is not None
        for pg in stack.items
        for item in pg.parameters
        if getattr(item, "kind", "") == ParameterKind.OBJECT.value)


def sync_animated_effect_bindings(scene, session) -> bool:
    """Rebind animated or moved runtime-bindable effect values before a step.

    Preview follows every runtime-bindable value. A Production Bake follows
    only keyframed/driven values and the motion of its controller objects, and
    holds any other edit (decision D2); its cache identity is taken at the bake
    start frame and these per-frame values go into each frame's stamp
    (:func:`effect_frame_stamp`), so it publishes nothing. Costs no rebinding
    unless a value actually changed since the last step.
    """
    stack = getattr(scene, "fluidsim_effects", None)
    if stack is None or not stack.applied_signature:
        return False
    runtime = getattr(getattr(session.solver, "features", None), "effects", None)
    if runtime is None or not runtime.has_any():
        return False
    from . import production_bake

    if not _effect_values_may_change(scene, stack):
        return False
    baking = production_bake.any_bake_active()
    animated = animated_parameter_keys(scene) if baking else None
    by_id = {str(pg.instance_id): pg for pg in stack.items}
    applied = runtime.applied_instances()
    updated = []
    try:
        for instance in applied:
            pg = by_id.get(instance.instance_id)
            live = instance if pg is None else _live_bindable_values(
                pg, instance, animated_only=None if animated is None
                else animated.get(instance.instance_id, frozenset()))
            updated.append(live)
    except (ReferenceError, ValueError):
        # A deleted controller or an animated value outside its limits keeps
        # the applied values; the next apply reports the authoring problem.
        return False
    if all(live is instance or (
            live.parameter_values == instance.parameter_values
            and live.controller_bindings == instance.controller_bindings)
           for live, instance in zip(updated, applied, strict=True)):
        return False
    if not runtime.update_runtime_bindings(tuple(updated)):
        return False
    if baking:
        return True
    try:
        instances = scene_effect_instances(scene)
    except (RuntimeError, ValueError):
        return True
    if execution_stack_fingerprint(instances)[0] == runtime.applied_signature:
        # The draft matches what now runs: keep the published applied state
        # (status, cache identity) in step with the animated values.
        _publish_applied_state(scene, instances)
    return True


def _running_effect_trust_revoked(stack, store) -> bool:
    """Is code the solver may be running no longer trusted?

    Judged on the applied snapshot, not the draft: an edited draft has a new
    content digest (an edited built-in is no longer the built-in) and so reads
    untrusted, yet the solver still runs the applied code, which was trusted
    when it was admitted. Only a trust change to that applied code revokes it.
    An unreadable snapshot behind an applied stack fails closed.
    """
    if not getattr(stack, "applied_signature", ""):
        return False
    try:
        payload = json.loads(getattr(stack, "applied_snapshot_json", "") or "[]")
        applied = tuple(EffectInstanceV2.from_dict(item) for item in payload)
    except (AttributeError, KeyError, TypeError, ValueError):
        return True
    trusted = {TrustState.BUILT_IN_TRUSTED, TrustState.LOCAL_TRUSTED}
    return any(
        store.status(content_trust_digest(instance.definition),
                     built_in=instance.definition.built_in) not in trusted
        for instance in effective_stack_instances(applied))


def refresh_stack_state(scene):
    stack = scene.fluidsim_effects
    try:
        instances = scene_effect_instances(scene)
        effective = effective_stack_instances(instances)
        effective_ids = {instance.instance_id for instance in effective}
        signature, inventory = execution_stack_fingerprint(instances)
        stack.draft_signature = signature
        store = trust_store_for_scene(scene)
        blocking = []
        instances_by_id = {instance.instance_id: instance for instance in instances}
        for pg in stack.items:
            instance = instances_by_id[str(pg.instance_id)]
            fingerprint = effect_fingerprint(instance)
            pg.draft_fingerprint = fingerprint
            digest = content_trust_digest(instance.definition)
            pg.content_digest = digest
            trust = store.status(digest, built_in=instance.definition.built_in)
            pg.trust_state = trust.value
            if instance.instance_id not in effective_ids:
                pg.status = "DISABLED"
            elif trust not in {TrustState.BUILT_IN_TRUSTED, TrustState.LOCAL_TRUSTED}:
                pg.status = "UNTRUSTED"
                blocking.append(f"{pg.display_name}: exact source is not trusted")
            elif pg.runtime_error:
                pg.status = "RUNTIME_ERROR"
                blocking.append(f"{pg.display_name}: {pg.runtime_error}")
            elif fingerprint == pg.applied_fingerprint and fingerprint:
                pg.status = "APPLIED"
            elif fingerprint == pg.tested_fingerprint and fingerprint:
                pg.status = "TESTED"
            elif pg.applied_fingerprint:
                pg.status = "DIRTY"
                if instance.enabled:
                    blocking.append(f"{pg.display_name}: draft differs from applied effect")
            else:
                pg.status = "DRAFT"
                if instance.enabled:
                    blocking.append(f"{pg.display_name}: draft has not been tested and applied")
        stack.status = (
            "APPLIED"
            if signature == stack.applied_signature and not blocking
            else "DIRTY" if stack.applied_signature else "DRAFT"
        )
        stack.blocking_reason = "; ".join(blocking)
        # Playback keeps running the admitted stack: drafts apply on rewind or
        # Apply Changes. Only revoked trust in running code stops the solver.
        if _running_effect_trust_revoked(stack, store):
            from .playback import stop_revoked_effects

            stop_revoked_effects(scene)
        return signature, inventory
    except Exception as exc:
        stack.status = "INVALID"
        stack.blocking_reason = str(exc)
        raise


def build_effect_candidate(scene, *, device: str, trust_store: TrustStore | None = None) -> CandidateBuildResult:
    instances = scene_effect_instances(scene)
    store = trust_store or trust_store_for_scene(scene)
    return build_candidate_effects(instances, device=device, trust_store=store)


def _clear_diagnostics(pg):
    pg.diagnostics.clear()
    pg.runtime_error = ""
    for kernel in pg.kernels:
        kernel.error_line = 0
        kernel.error_column = 0


def _mirror_issues(stack, issues):
    by_instance = {item.instance_id: item for item in stack.items}
    by_effect: dict[str, list[object]] = {}
    for item in stack.items:
        by_effect.setdefault(item.effect_id, []).append(item)
    for issue in issues:
        pg = by_instance.get(issue.instance_id)
        if pg is None:
            matches = by_effect.get(issue.effect_id, [])
            pg = matches[0] if len(matches) == 1 else None
        if pg is None:
            continue
        diagnostic = pg.diagnostics.add()
        diagnostic.code = issue.code
        diagnostic.message = issue.message
        diagnostic.severity = issue.severity
        diagnostic.instance_id = issue.instance_id
        diagnostic.kernel_id = issue.kernel_id
        diagnostic.line = issue.line
        diagnostic.column = issue.column
        diagnostic.hint = issue.hint
        if issue.severity != "ERROR":
            continue  # advisory: shown, but neither blocks nor marks the code line
        for kernel in pg.kernels:
            if kernel.kernel_id == issue.kernel_id:
                kernel.error_line = issue.line
                kernel.error_column = issue.column
        pg.status = "INVALID"


def test_draft(scene, *, device: str) -> CandidateBuildResult:
    stack = scene.fluidsim_effects
    refresh_stack_state(scene)
    for pg in stack.items:
        _clear_diagnostics(pg)
    candidate = build_effect_candidate(scene, device=device)
    stack.last_compile_device = str(device)
    if not candidate.success:
        _mirror_issues(stack, candidate.issues)
        stack.status = "INVALID"
        stack.blocking_reason = candidate.issues[0].message if candidate.issues else "Effect validation failed"
        return candidate
    instances = scene_effect_instances(scene)
    effective_ids = {item.instance_id for item in effective_stack_instances(instances)}
    signature, _inventory = execution_stack_fingerprint(instances)
    stack.tested_signature = signature
    for pg in stack.items:
        if pg.instance_id in effective_ids:
            pg.tested_fingerprint = pg.draft_fingerprint
            pg.status = "TESTED"
        else:
            pg.status = "DISABLED"
    # A successful candidate carries only advisory warnings; show them.
    _mirror_issues(stack, candidate.issues)
    compile_seconds = 0.0
    sandbox_seconds = 0.0
    warp_version = ""
    for compiled, sandbox in zip(candidate.compiled_effects, candidate.sandbox_reports):
        target = next((item for item in stack.items if item.instance_id == compiled.instance.instance_id), None)
        reports = []
        for kernel in compiled.kernels:
            report = kernel.compile_report
            compile_seconds += report.duration_seconds
            warp_version = report.warp_version
            reports.append({
                "kernel_id": kernel.kernel_definition.kernel_id,
                "compiled": report.compiled,
                "device": report.device,
                "warp_version": report.warp_version,
                "module_file": report.module_file,
                "source_digest": report.source_digest,
                "duration_seconds": report.duration_seconds,
            })
            if target is not None:
                for kernel_pg in target.kernels:
                    if kernel_pg.kernel_id == kernel.kernel_definition.kernel_id:
                        kernel_pg.generated_source_digest = report.source_digest
        if target is not None:
            target.compile_report_json = json.dumps(reports, sort_keys=True)
            target.sandbox_report_json = json.dumps({
                "passed": sandbox.passed,
                "kernels": [vars(item) for item in sandbox.kernels],
            }, sort_keys=True)
        sandbox_seconds += sum(item.duration_seconds for item in sandbox.kernels)
    stack.last_compile_seconds = compile_seconds
    stack.last_sandbox_seconds = sandbox_seconds
    stack.last_warp_version = warp_version
    stack.status = "TESTED"
    stack.blocking_reason = "Draft tested; Compile & Apply before Production Bake."
    return candidate


def apply_stack(scene, *, device: str) -> CandidateBuildResult:
    """Compile the complete draft and atomically publish it as the applied stack."""

    stack = scene.fluidsim_effects
    previous_signature = stack.applied_signature
    previous_inventory = stack.applied_inventory_json
    previous_snapshot = stack.applied_snapshot_json
    candidate = test_draft(scene, device=device)
    if not candidate.success:
        stack.applied_signature = previous_signature
        stack.applied_inventory_json = previous_inventory
        stack.applied_snapshot_json = previous_snapshot
        return candidate
    from . import scene_bridge

    session = scene_bridge.session()
    if session.active and session.owns_scene(scene):
        # Test for the method, not ``except AttributeError``: an error raised
        # inside installation must not publish the stack as APPLIED.
        install = getattr(session.solver, "set_custom_effect_candidate", None)
        if install is not None and not install(candidate):
            raise RuntimeError("live solver rejected the complete candidate stack")
    instances = scene_effect_instances(scene)
    _publish_applied_state(scene, instances)
    return candidate


def revert_draft(scene) -> None:
    stack = scene.fluidsim_effects
    payload = json.loads(stack.applied_snapshot_json or "[]")
    if not isinstance(payload, list):
        raise ValueError("applied effect snapshot is invalid")
    applied_signature = stack.applied_signature
    applied_inventory = stack.applied_inventory_json
    clear_effects(scene, remove_owned_texts=True)
    for item in payload:
        instance = EffectInstanceV2.from_dict(item)
        pg = add_instance_to_scene(scene, instance, select=False)
        fingerprint = effect_fingerprint(instance)
        pg.tested_fingerprint = fingerprint
        pg.applied_fingerprint = fingerprint
        pg.status = "APPLIED" if instance.enabled and not instance.muted else "DISABLED"
    stack.applied_signature = applied_signature
    stack.tested_signature = applied_signature
    stack.applied_inventory_json = applied_inventory
    stack.applied_snapshot_json = json.dumps(payload, sort_keys=True, allow_nan=False)
    stack.status = "APPLIED"
    stack.blocking_reason = ""
    refresh_stack_state(scene)


def effect_identity(scene) -> tuple[str, tuple[dict[str, object], ...]]:
    """Recompute and return the exact code that will execute at bake/resume admission."""

    stack = getattr(scene, "fluidsim_effects", None)
    if stack is None or not stack.items:
        return "", ()
    current_signature, current_inventory = execution_stack_fingerprint(
        scene_effect_instances(scene)
    )
    if not current_signature:
        return "", ()
    if not stack.applied_signature:
        raise ValueError("enabled Custom Effects have not been compiled and applied")
    # Keyframed/driven values and controller motion legitimately differ from
    # the applied snapshot, which was published at some other frame: compare
    # with those taken from the snapshot. The identity returned is still the
    # scene's own, so a bake records it at its start frame.
    compared_signature, compared_inventory = current_signature, current_inventory
    if current_signature != stack.applied_signature:
        compared_signature, compared_inventory = execution_stack_fingerprint(
            _with_applied_frame_values(scene, stack))
    if compared_signature != stack.applied_signature:
        raise ValueError(
            "Custom Effects source, parameters, controllers, timing, or stack order "
            "differ from the applied stack"
        )
    try:
        applied_inventory = tuple(
            dict(item) for item in json.loads(stack.applied_inventory_json or "[]")
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("applied Custom Effects inventory is invalid") from exc
    if applied_inventory != compared_inventory:
        raise ValueError("Custom Effects applied inventory differs from the executable stack")
    return current_signature, current_inventory


def _with_applied_frame_values(scene, stack) -> tuple[EffectInstanceV2, ...]:
    """The draft, with its per-frame values replaced by the applied snapshot's.

    Per-frame values are keyframed/driven parameters and the transform of a
    controller object that is still the applied one. Anything else (code,
    other values, a different controller object) is left as drafted.
    """
    try:
        payload = json.loads(getattr(stack, "applied_snapshot_json", "") or "[]")
        applied = {item.instance_id: item for item in
                   (EffectInstanceV2.from_dict(entry) for entry in payload)}
    except (AttributeError, KeyError, TypeError, ValueError):
        return scene_effect_instances(scene)
    animated = animated_parameter_keys(scene)
    aligned = []
    for instance in scene_effect_instances(scene):
        old = applied.get(instance.instance_id)
        if old is None:
            aligned.append(instance)
            continue
        values = dict(instance.parameter_values)
        for key in animated.get(instance.instance_id, ()):
            if key in old.parameter_values:
                values[key] = old.parameter_values[key]
        controllers = dict(instance.controller_bindings)
        for key, binding in instance.controller_bindings.items():
            previous = old.controller_bindings.get(key)
            if previous and previous.get("object_name") == binding.get("object_name"):
                controllers[key] = previous
        aligned.append(replace(
            instance, parameter_values=values, controller_bindings=controllers))
    return tuple(aligned)


def production_blocking_reason(scene) -> str:
    try:
        effect_identity(scene)
    except ValueError as exc:
        return str(exc)
    return ""




def effect_report_state(scene) -> dict[str, object]:
    """Return strict support-report state without compiling or executing code."""

    stack = getattr(scene, "fluidsim_effects", None)
    if stack is None:
        return {
            "draft_signature": "", "tested_signature": "", "applied_signature": "",
            "policy": "BUILTINS_ONLY", "blocking_reason": "", "inventory": [], "items": [],
        }
    try:
        inventory = json.loads(stack.applied_inventory_json or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        inventory = []
    items = []
    for item in stack.items:
        entry = {
            "instance_id": item.instance_id,
            "effect_id": item.effect_id,
            "effect_version": item.effect_version,
            "effect_api_version": 0,
            "effect_abi_version": 0,
            "definition_error": "",
            "enabled": bool(item.enabled),
            "muted": bool(item.muted),
            "solo": bool(item.solo),
            "status": item.status,
            "trust": item.trust_state,
            "content_digest": item.content_digest,
            "draft_fingerprint": item.draft_fingerprint,
            "tested_fingerprint": item.tested_fingerprint,
            "applied_fingerprint": item.applied_fingerprint,
            "runtime_error": item.runtime_error,
            "calls": int(item.calls),
            "elements": int(item.elements),
            "submission_ms": float(item.submission_ms),
        }
        try:
            definition = EffectDefinitionV2.from_dict(json.loads(item.definition_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            entry["definition_error"] = f"{type(exc).__name__}: {exc}"
        else:
            entry["effect_api_version"] = definition.effect_api_version
            entry["effect_abi_version"] = definition.effect_abi_version
        items.append(entry)
    return {
        "draft_signature": stack.draft_signature,
        "tested_signature": stack.tested_signature,
        "applied_signature": stack.applied_signature,
        "policy": stack.custom_code_policy,
        "blocking_reason": stack.blocking_reason,
        "compile_device": stack.last_compile_device,
        "warp_version": stack.last_warp_version,
        "compile_seconds": float(stack.last_compile_seconds),
        "sandbox_seconds": float(stack.last_sandbox_seconds),
        "inventory": inventory,
        "items": items,
    }

def build_effect_stack_for_admission(scene) -> tuple[EffectInstanceV2, ...]:
    """Capture eligible authoring state; admission validates and installs it.

    Unlike production_blocking_reason, capture does not require a previously
    applied draft. Cache identity checks remain strict at their own boundary.
    Keep disabled instances so successful admission can publish full evidence.
    """

    from .file_compat import UNMIGRATED_SNIPPETS_MESSAGE, has_unmigrated_snippets

    if has_unmigrated_snippets(scene):
        raise RuntimeError(UNMIGRATED_SNIPPETS_MESSAGE)
    instances = scene_effect_instances(scene)
    store = trust_store_for_scene(scene)
    for instance in effective_stack_instances(instances):
        digest = content_trust_digest(instance.definition)
        trust = store.status(digest, built_in=instance.definition.built_in)
        if trust not in {TrustState.BUILT_IN_TRUSTED, TrustState.LOCAL_TRUSTED}:
            raise RuntimeError(
                f"Custom Effects admission blocked: {instance.definition.name}: "
                "exact source is not trusted")
    return instances


def publish_admitted_effects(scene, instances: tuple[EffectInstanceV2, ...]) -> None:
    """Publish the full immutable snapshot installed by Simulation Admission."""
    for item in scene.fluidsim_effects.items:
        _clear_diagnostics(item)
    _publish_applied_state(scene, instances)

def trust_effect(scene, index: int) -> None:
    stack = scene.fluidsim_effects
    pg = stack.items[int(index)]
    instance = effect_instance_from_pg(scene, pg)
    digest = content_trust_digest(instance.definition)
    trust_store_for_scene(scene).trust(digest, pg.display_name)
    pg.trust_state = TrustState.LOCAL_TRUSTED.value
    pg.status = "DRAFT"
    refresh_stack_state(scene)


def untrust_effect(scene, index: int) -> None:
    stack = scene.fluidsim_effects
    pg = stack.items[int(index)]
    instance = effect_instance_from_pg(scene, pg)
    digest = content_trust_digest(instance.definition)
    trust_store_for_scene(scene).revoke(digest)
    refresh_stack_state(scene)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fluidsim_effects = PointerProperty(type=FLUIDSIM_PG_effect_stack)


def unregister():
    if hasattr(bpy.types.Scene, "fluidsim_effects"):
        del bpy.types.Scene.fluidsim_effects
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"FluidSim: could not unregister {cls.__name__}: {exc!r}")

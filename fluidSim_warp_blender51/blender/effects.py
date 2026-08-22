"""Blender scene adapters for the versioned Custom Effects workflow.

The PropertyGroups in this module are authoring state only. Every compile,
apply, cache, and report boundary converts them to immutable v2 snapshots.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, replace
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
from ..sim.effects.enums import EffectStatus, ParameterKind, TrustState
from ..sim.effects.fingerprint import (
    effect_fingerprint,
    effective_stack_instances,
    execution_stack_fingerprint,
)
from ..sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    FieldBindingV2,
)
from ..sim.effects.trust import TrustStore, content_trust_digest


_PARAMETER_KIND_ITEMS = tuple((item.value, item.name.replace("_", " ").title(), "") for item in ParameterKind)
_STATUS_ITEMS = tuple((item.value, item.name.replace("_", " ").title(), "") for item in EffectStatus)
_TRUST_ITEMS = tuple((item.value, item.name.replace("_", " ").title(), "") for item in TrustState)
_DOMAIN_ITEMS = (("PARTICLES", "Particles", "Run over live liquid particles"), ("ACTIVE_VOXELS", "Active Voxels", "Run over active sparse-grid voxels"))
_STAGE_ITEMS = (
    ("BEFORE_EMISSION", "Before Emission", "Before sources emit this substep"),
    ("AFTER_PARTICLE_TRANSFER", "After Particle Transfer", "After particle-to-grid transfer"),
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
            refresh_stack_state(scene)
        except Exception as exc:  # authoring feedback, not an execution path
            scene.fluidsim_effects.blocking_reason = f"Effect snapshot error: {exc}"


class FLUIDSIM_PG_effect_parameter(bpy.types.PropertyGroup):
    key: StringProperty()
    label: StringProperty()
    kind: EnumProperty(items=_PARAMETER_KIND_ITEMS, default="FLOAT")
    unit: StringProperty(default="NONE")
    description: StringProperty()
    minimum: FloatProperty(default=0.0)
    maximum: FloatProperty(default=0.0)
    has_minimum: BoolProperty(default=False)
    has_maximum: BoolProperty(default=False)
    float_value: FloatProperty(update=_mark_dirty)
    int_value: IntProperty(update=_mark_dirty)
    bool_value: BoolProperty(update=_mark_dirty)
    vector_value: FloatVectorProperty(size=3, update=_mark_dirty)
    color_value: FloatVectorProperty(size=4, subtype="COLOR", min=0.0, max=1.0, update=_mark_dirty)
    enum_value: StringProperty(update=_mark_dirty)
    enum_items_json: StringProperty(default="[]")
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
    migration_report: StringProperty()
    migration_complete: BoolProperty(default=False)
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


def effect_instance_from_pg(scene, pg) -> EffectInstanceV2:
    """Build one immutable execution snapshot from Blender authoring state."""

    definition = EffectDefinitionV2.from_dict(json.loads(pg.definition_json))
    original = {kernel.kernel_id: kernel for kernel in definition.kernels}
    kernels = []
    for kernel_pg in pg.kernels:
        base = original[kernel_pg.kernel_id]
        source = (
            kernel_pg.source_text.as_string()
            if kernel_pg.source_text is not None
            else base.source
        )
        fields = tuple(FieldBindingV2(item.field, item.access) for item in kernel_pg.fields)
        kernels.append(replace(
            base,
            domain=kernel_pg.domain,
            stage=kernel_pg.stage,
            cadence=kernel_pg.cadence,
            source=source.rstrip("\n") or "pass",
            fields=fields,
            safe_mode=bool(kernel_pg.safe_mode),
            frame_start=int(kernel_pg.frame_start),
            frame_end=int(kernel_pg.frame_end),
            frame_step=max(1, int(kernel_pg.frame_step)),
        ))
    definition = replace(definition, kernels=tuple(kernels))
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
    try:
        from . import scene_bridge

        session = scene_bridge.session()
        if session.active and session.owns_scene(scene):
            if not session.solver.set_custom_effect_candidate(candidate):
                raise RuntimeError("live solver rejected the complete candidate stack")
    except AttributeError:
        pass
    instances = scene_effect_instances(scene)
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
        instance = instances_by_id[str(pg.instance_id)]
        fingerprint = effect_fingerprint(instance)
        pg.applied_fingerprint = fingerprint
        pg.tested_fingerprint = fingerprint
        pg.status = "APPLIED" if pg.instance_id in effective_ids else "DISABLED"
    stack.status = "APPLIED"
    stack.blocking_reason = ""
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
    if current_signature != stack.applied_signature:
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
    if applied_inventory != current_inventory:
        raise ValueError("Custom Effects applied inventory differs from the executable stack")
    return current_signature, current_inventory


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
    """Return the exact applied stack or reject initialization before allocation."""

    reason = production_blocking_reason(scene)
    if reason:
        raise RuntimeError(f"Custom Effects admission blocked: {reason}")
    return effective_stack_instances(scene_effect_instances(scene))

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


@dataclass(frozen=True)
class MigrationReport:
    migrated: int
    warnings: tuple[str, ...]


def migrate_legacy_snippets(scene) -> MigrationReport:
    """Migrate legacy snippets without rewriting source or trusting content.

    Legacy VOXEL_TILE entries are explicitly reported and mapped by the v1
    compatibility converter. Every unknown scene-code effect is created with
    enabled=False and UNTRUSTED_IMPORTED until the exact digest is reviewed.
    """

    stack = scene.fluidsim_effects
    if stack.migration_complete:
        return MigrationReport(0, ())
    legacy = tuple(getattr(getattr(scene, "fluidsim", None), "snippets", ()))
    if not legacy:
        stack.migration_complete = True
        return MigrationReport(0, ())
    from .snippets import snippet_from_pg
    from ..sim.custom import legacy_snippet_to_effect

    warnings = []
    migrated = 0
    for index, legacy_pg in enumerate(legacy):
        legacy_instance, item_warnings = legacy_snippet_to_effect(
            snippet_from_pg(legacy_pg), instance_id=f"legacy-scene-{index}"
        )
        migrated_instance = replace(legacy_instance, enabled=False)
        pg = add_instance_to_scene(scene, migrated_instance, select=False)
        pg.trust_state = TrustState.UNTRUSTED_IMPORTED.value
        pg.status = "UNTRUSTED"
        warnings.extend(item_warnings)
        migrated += 1
    stack.migration_complete = True
    stack.migration_report = (
        f"Migrated {migrated} legacy snippet(s); disabled until reviewed and trusted. "
        + " ".join(warnings)
    )
    refresh_stack_state(scene)
    return MigrationReport(migrated, tuple(warnings))


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
        except (RuntimeError, ValueError):
            pass

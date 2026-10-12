"""Guided selected-object setup for a runnable FluidSim scene."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .scene_access import active_domain_object

OWNER_TAG = "fluidsim_owned"
OWNER_ID_TAG = "fluidsim_owner_id"
KIND_TAG = "fluidsim_kind"
SCENE_SETUP_ID = "fluidsim_setup_id"

SETUP_MODES = (
    "QUICK_LIQUID",
    "QUICK_CONTAINER",
    "QUICK_VISCOUS",
    "ADD_COLLIDER",
    "ADD_INFLOW",
    "ADD_OUTFLOW",
)

class _SetupRollbackJournal:
    """Small LIFO unwind journal scoped to one Guided Setup invocation."""

    def __init__(self) -> None:
        self._undo: list[Callable[[], None]] = []
        self._closed = False

    def record(self, undo: Callable[[], None]) -> None:
        if self._closed:
            raise RuntimeError("setup rollback journal is already closed")
        self._undo.append(undo)

    def commit(self) -> None:
        self._undo.clear()
        self._closed = True

    def rollback(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        for undo in reversed(self._undo):
            try:
                undo()
            except Exception as exc:  # noqa: BLE001 - finish the remaining unwind
                errors.append(str(exc))
        self._undo.clear()
        if errors:
            raise RuntimeError("; ".join(errors))


@dataclass(frozen=True)
class SetupResult:
    mode: str
    preset_id: str
    domain_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    created_names: tuple[str, ...]
    warnings: tuple[str, ...]
    capacity_message: str
    message: str


def _triple(value, label: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} must be finite")
    return result


def _rounded(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == -0.0 else result


def padded_bounds(
    minimum: Sequence[float],
    maximum: Sequence[float],
    voxel_size: float,
    padding_voxels: int | Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Pad bounds in voxel units without imposing grid alignment."""
    minimum = _triple(minimum, "minimum")
    maximum = _triple(maximum, "maximum")
    voxel_size = float(voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if isinstance(padding_voxels, (int, float)):
        padding = (float(padding_voxels),) * 3
    else:
        padding = _triple(padding_voxels, "padding_voxels")
    if any(value < 0 for value in padding):
        raise ValueError("padding_voxels must be non-negative")
    if any(hi < lo for lo, hi in zip(minimum, maximum)):
        raise ValueError("maximum must not be below minimum")
    pad = tuple(value * voxel_size for value in padding)
    return (
        tuple(_rounded(value - amount) for value, amount in zip(minimum, pad)),
        tuple(_rounded(value + amount) for value, amount in zip(maximum, pad)),
    )


def align_bounds(
    minimum: Sequence[float], maximum: Sequence[float], voxel_size: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Round outward to deterministic world-origin-aligned voxel planes."""
    minimum = _triple(minimum, "minimum")
    maximum = _triple(maximum, "maximum")
    voxel_size = float(voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    low = tuple(
        _rounded(math.floor(value / voxel_size + 1.0e-12) * voxel_size)
        for value in minimum
    )
    high = tuple(
        _rounded(math.ceil(value / voxel_size - 1.0e-12) * voxel_size)
        for value in maximum
    )
    return low, high


def gravity_padding(
    gravity: Sequence[float],
    *,
    lateral: int = 8,
    below: int = 4,
    above: int = 12,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return minimum/maximum padding with gravity-aware above/below sides."""
    gravity = _triple(gravity, "gravity")
    axis = max(range(3), key=lambda index: abs(gravity[index]))
    min_pad = [int(lateral)] * 3
    max_pad = [int(lateral)] * 3
    # 'Below' follows gravity; 'above' opposes it.
    if gravity[axis] < 0.0:
        min_pad[axis], max_pad[axis] = int(below), int(above)
    else:
        min_pad[axis], max_pad[axis] = int(above), int(below)
    return tuple(min_pad), tuple(max_pad)


def directional_padded_bounds(
    minimum: Sequence[float],
    maximum: Sequence[float],
    voxel_size: float,
    gravity: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    minimum = _triple(minimum, "minimum")
    maximum = _triple(maximum, "maximum")
    min_pad, max_pad = gravity_padding(gravity)
    low = tuple(
        value - amount * voxel_size for value, amount in zip(minimum, min_pad)
    )
    high = tuple(
        value + amount * voxel_size for value, amount in zip(maximum, max_pad)
    )
    return align_bounds(low, high, voxel_size)


def _world_bounds(objects, depsgraph=None):
    import bpy
    from mathutils import Vector

    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    count = 0
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        matrix = evaluated.matrix_world
        for corner in evaluated.bound_box:
            point = matrix @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], float(point[axis]))
                maximum[axis] = max(maximum[axis], float(point[axis]))
            count += 1
    if count == 0:
        raise ValueError("selected objects have no evaluated bounds")
    return tuple(minimum), tuple(maximum)


def compute_domain_bounds(
    objects,
    voxel_size: float,
    padding_voxels: int | None = None,
    *,
    gravity=(0.0, 0.0, -9.81),
    depsgraph=None,
):
    minimum, maximum = _world_bounds(objects, depsgraph)
    if padding_voxels is not None:
        return align_bounds(
            *padded_bounds(minimum, maximum, voxel_size, padding_voxels),
            voxel_size,
        )
    return directional_padded_bounds(minimum, maximum, voxel_size, gravity)


def _scene_owner_id(scene) -> str:
    value = str(scene.get(SCENE_SETUP_ID, "") or "")
    if not value:
        value = uuid.uuid4().hex
        scene[SCENE_SETUP_ID] = value
    return value


def _owned(datablock, owner_id: str, kind: str) -> bool:
    try:
        return bool(datablock.get(OWNER_TAG)) and (
            datablock.get(OWNER_ID_TAG) == owner_id
            and datablock.get(KIND_TAG) == kind
        )
    except AttributeError:
        return False


def _mark_owned(datablock, owner_id: str, kind: str) -> None:
    datablock[OWNER_TAG] = True
    datablock[OWNER_ID_TAG] = owner_id
    datablock[KIND_TAG] = kind


def _ensure_child_collection(parent, name: str, owner_id: str, kind: str):
    import bpy

    for collection in parent.children:
        if _owned(collection, owner_id, kind):
            return collection, False
    collection = bpy.data.collections.new(name)
    _mark_owned(collection, owner_id, kind)
    parent.children.link(collection)
    return collection, True


def ensure_collection_tree(scene):
    owner_id = _scene_owner_id(scene)
    root, root_created = _ensure_child_collection(
        scene.collection, "FluidSim", owner_id, "ROOT_COLLECTION"
    )
    inputs, inputs_created = _ensure_child_collection(
        root, "Inputs", owner_id, "INPUT_COLLECTION"
    )
    outputs, outputs_created = _ensure_child_collection(
        root, "Outputs", owner_id, "OUTPUT_COLLECTION"
    )
    helpers, helpers_created = _ensure_child_collection(
        root, "Helpers", owner_id, "HELPER_COLLECTION"
    )
    created = tuple(
        collection.name
        for collection, was_created in (
            (root, root_created),
            (inputs, inputs_created),
            (outputs, outputs_created),
            (helpers, helpers_created),
        )
        if was_created
    )
    return owner_id, root, inputs, outputs, helpers, created


def _link_if_needed(collection, obj) -> None:
    if obj.name not in collection.objects:
        collection.objects.link(obj)


def _cube_mesh(name: str):
    import bpy

    mesh = bpy.data.meshes.new(name)
    vertices = (
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    )
    faces = (
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7),
    )
    mesh.from_pydata(vertices, (), faces)
    mesh.update()
    return mesh


def _ensure_domain(helpers, owner_id: str):
    """Find or create the wizard-owned domain object (not yet designated)."""
    import bpy

    domain = None
    for obj in helpers.objects:
        if _owned(obj, owner_id, "DOMAIN"):
            domain = obj
            break
    created = False
    if domain is None:
        domain = bpy.data.objects.new("FluidSim Domain", _cube_mesh("FluidSim Domain"))
        _mark_owned(domain, owner_id, "DOMAIN")
        helpers.objects.link(domain)
        created = True
    return domain, created


def present_as_domain(obj) -> None:
    """Show a domain as the simulation boundary: wireframe, in front, unrendered.

    Every path that designates a domain (Guided Setup, Add > Fluid Domain, Use
    Active as Domain) uses this, so a domain never renders as a solid box that
    hides the liquid it contains.
    """
    obj.display_type = "WIRE"
    obj.hide_render = True
    obj.show_in_front = True


def _fit_domain(scene, domain, bounds) -> None:
    """Size the domain box to ``bounds``, then designate it for the scene."""
    minimum, maximum = bounds
    center = tuple((lo + hi) * 0.5 for lo, hi in zip(minimum, maximum))
    dimensions = tuple(max(hi - lo, 1.0e-6) for lo, hi in zip(minimum, maximum))
    domain.location = center
    domain.rotation_euler = (0.0, 0.0, 0.0)
    domain.scale = dimensions
    present_as_domain(domain)
    scene.fluidsim.domain_object = domain


def _wizard_cell_size(domain_fs, preset) -> float:
    """Cell size Guided Setup fits the domain to.

    A preset's ``voxel_size`` is written through the derived property, which
    quantizes it against the domain's *current* (pre-fit) size: 0.12 on the
    1 m placeholder reads back 0.125, and on a reused 0.3 m domain the
    8-division minimum turns it into 0.0375. Fitting uses the preset's authored
    value instead, so the cell size does not depend on the box being replaced.
    """
    settings = getattr(preset, "settings", None) or {}
    authored = settings.get("voxel_size") if hasattr(settings, "get") else None
    if authored is not None:
        value = float(authored)
        if math.isfinite(value) and value > 0.0:
            return value
    return float(domain_fs.voxel_size)


def _fit_domain_keeping_voxel(context, scene, domain, domain_fs, bounds, voxel_size) -> None:
    """Fit the domain, then keep the cell size the bounds were aligned to.

    The cell size is derived from the domain's longest axis, so resizing the box
    changes it. Writing ``voxel_size`` back re-derives the divisions that give the
    wizard's grid. The world matrix must be current first, because the derivation
    reads it.
    """
    _fit_domain(scene, domain, bounds)
    view_layer = getattr(context, "view_layer", None)
    if view_layer is not None:
        view_layer.update()
    domain_fs.voxel_size = voxel_size


def _tag_source_mesh(obj, *, continuous: bool) -> None:
    from . import sources

    sources.tag_source(obj, mode="PARTICLES")
    pg = obj.fluidsim
    pg.use_mesh_shape = True
    pg.geometry_mode = "MESH"
    pg.source_region_kind = "MESH_SOLID"
    pg.mesh_object = obj
    pg.mesh_evaluation_mode = "EVERY_SIM_FRAME" if continuous else "FIRST_SIM_FRAME"
    pg.emission_mode = "CONTINUOUS_INFLOW" if continuous else "INITIAL_FILL"


def _tag_sink_mesh(obj) -> None:
    from . import sources

    sources.tag_sink(obj)
    pg = obj.fluidsim
    pg.use_mesh_shape = True
    pg.geometry_mode = "MESH"
    pg.sink_region_kind = "MESH_SOLID"
    pg.mesh_object = obj


def _mesh_watertight_warning(obj) -> str:
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "edges"):
        return ""
    try:
        if any(not edge.is_manifold for edge in mesh.edges):
            return f"{obj.name}: solid source/container mesh is not manifold"
    except AttributeError:
        # Blender MeshEdge does not expose is_manifold outside edit BMesh on
        # every supported build. The solver performs authoritative validation.
        return ""
    return ""


def _capacity_summary(scene) -> str:
    try:
        from .scene_bridge import refresh_capacity_plan

        plan = refresh_capacity_plan(scene, operation="INITIALIZE")
    except Exception as exc:
        return f"Capacity preflight deferred: {exc}"
    blockers = tuple(getattr(plan, "blockers", ()) or ())
    if blockers:
        from .simulation_admission import capacity_blocking_message

        return capacity_blocking_message(blockers)
    estimate = int(getattr(plan, "estimated_bytes", 0) or 0)
    return (
        f"Capacity preflight ready; estimated peak {estimate / 1024**3:.2f} GiB"
        if estimate
        else "Capacity preflight ready"
    )


def reserved_object_reason(scene, obj) -> str:
    """Why ``obj`` must never become a simulation input, or ``""``.

    The designated domain, a wizard-owned domain and the add-on's own output
    objects are simulation boundaries and results; tagging them as sources,
    sinks or colliders would feed the simulation its own container or output.
    """
    if obj is None:
        return ""
    if obj is active_domain_object(scene):
        return "the FluidSim domain"
    try:
        kind = obj.get(KIND_TAG)
    except (AttributeError, TypeError):
        kind = None
    if kind == "DOMAIN":
        return "a FluidSim domain"
    from .output_publication import is_output_object

    if kind == "OUTPUT" or is_output_object(obj):
        return "a FluidSim output"
    return ""


def _selected_meshes(context):
    accepted_roles = {"NONE", "SOURCE", "COLLIDER", "SINK"}
    scene = getattr(context, "scene", None)
    return tuple(
        obj
        for obj in getattr(context, "selected_objects", ())
        if getattr(obj, "type", None) == "MESH"
        and getattr(getattr(obj, "fluidsim", None), "role", "NONE")
        in accepted_roles
        and not reserved_object_reason(scene, obj)
    )


def _record_attributes(journal, owner, names, *, optional: bool = True) -> None:
    """Record ``names`` on ``owner`` so the journal can restore them.

    ``optional=True`` is for Blender-native attributes (``Object.display_type``
    and friends), where an attribute missing on an unusual ID type is skipped.
    Settings fields go through ``_record_settings``: a name the settings group
    does not declare is a wrong-group read and must raise instead of silently
    leaving the field out of the rollback.
    """
    values = {}
    for name in names:
        if optional and not hasattr(owner, name):
            continue
        value = getattr(owner, name)
        if hasattr(value, "copy") and not hasattr(value, "as_pointer"):
            value = value.copy()
        values[name] = value

    def restore():
        for name, value in values.items():
            setattr(owner, name, value)

    journal.record(restore)


def _record_settings(journal, group, names) -> None:
    _record_attributes(journal, group, names, optional=False)


_MISSING = object()


def _id_property_values(owner) -> dict:
    try:
        return dict(owner.items())
    except (AttributeError, TypeError):
        return {}


def _record_property_keys(journal, owner, keys) -> None:
    values = {key: owner.get(key, _MISSING) for key in keys}

    def restore():
        for key, value in values.items():
            if value is _MISSING:
                if key in owner:
                    del owner[key]
            else:
                owner[key] = value

    journal.record(restore)


class _CreatedSetupIDs:
    """Exact datablocks returned as newly created by this setup."""

    def __init__(self, journal) -> None:
        self._items = []
        journal.record(self.remove)

    def track(self, registry, datablock, order: int) -> None:
        pointer = datablock.as_pointer()
        if any(item[3] == pointer for item in self._items):
            return
        self._items.append(
            (order, registry, datablock, pointer, str(getattr(datablock, "name", "")))
        )

    def remove(self) -> None:
        errors = []
        for _order, registry, _datablock, pointer, name in sorted(
            self._items, key=lambda item: item[0]
        ):
            candidate = next(
                (
                    item for item in tuple(registry)
                    if item.as_pointer() == pointer
                    and str(getattr(item, "name", "")) == name
                ),
                None,
            )
            if candidate is None:
                continue
            try:
                registry.remove(candidate, do_unlink=True)
            except TypeError:
                try:
                    registry.remove(candidate)
                except Exception as exc:  # noqa: BLE001 - continue exact cleanup
                    errors.append(str(exc))
            except Exception as exc:  # noqa: BLE001 - continue exact cleanup
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))


def _track_object_ids(created, bpy, obj) -> None:
    created.track(bpy.data.objects, obj, 0)
    data = getattr(obj, "data", None)
    if data is None:
        return
    for name in ("meshes", "pointclouds"):
        registry = getattr(bpy.data, name, None)
        if registry is not None and any(
            item.as_pointer() == data.as_pointer() for item in tuple(registry)
        ):
            created.track(registry, data, 2)
            return


def _record_object_links(journal, obj) -> None:
    previous = tuple(obj.users_collection)

    def restore():
        for collection in tuple(obj.users_collection):
            if collection not in previous:
                collection.objects.unlink(obj)
        for collection in previous:
            if collection not in tuple(obj.users_collection):
                collection.objects.link(obj)

    journal.record(restore)


def _record_selected_state(journal, selected) -> None:
    setting_names = (
        "role", "shape", "source_mode", "source_ui_schema_version",
        "source_sampling_mode", "source_sampling_multiplier", "source_region_kind",
        "sink_region_kind", "emission_mode", "use_mesh_shape", "geometry_mode",
        "mesh_object", "mesh_evaluation_mode", "collider_friction",
        "collider_restitution", "transfer_velocity",
    )
    for obj in selected:
        _record_object_links(journal, obj)
        _record_attributes(journal, obj, ("display_type", "hide_render"))
        _record_settings(journal, obj.fluidsim, setting_names)


def _record_domain_state(journal, domain, preset_fields, mode) -> None:
    # ``voxel_size`` is derived from ``resolution_divisions`` and the domain's
    # world size, and the wizard rewrites both the transform and the divisions.
    # Restoring the derived value against the fitted box re-derives the wrong
    # divisions, so the authoritative divisions are recorded instead -- always,
    # since fitting changes them even without a preset -- and recorded *before*
    # the transform so the LIFO rollback puts the transform back first.
    fields = [name for name in preset_fields if name != "voxel_size"]
    if mode == "QUICK_CONTAINER":
        fields.extend(("height_enabled", "height_emission_mode", "height_base"))
    if "resolution_divisions" in fields:
        fields.remove("resolution_divisions")
    fields.append("resolution_divisions")
    _record_settings(journal, domain.fluidsim_domain, fields)
    _record_attributes(
        journal, domain,
        ("location", "rotation_euler", "scale", "display_type", "hide_render", "show_in_front"),
    )


def _record_output_state(journal, obj, property_keys):
    _record_property_keys(journal, obj, property_keys)
    _record_attributes(journal, obj, ("color", "display_type"))
    materials = tuple(getattr(obj.data, "materials", ()))

    def restore_materials():
        slots = getattr(obj.data, "materials", None)
        if slots is not None:
            slots.clear()
            for material in materials:
                slots.append(material)

    journal.record(restore_materials)
    before = {
        modifier.as_pointer(): (
            modifier,
            getattr(modifier, "node_group", _MISSING),
            _id_property_values(modifier),
        )
        for modifier in obj.modifiers
    }
    created_modifier_ids = set()
    changed_modifiers = []

    def capture_modifier_changes():
        for modifier in obj.modifiers:
            pointer = modifier.as_pointer()
            if pointer not in before:
                created_modifier_ids.add(pointer)
                continue
            _original, node_group, values = before[pointer]
            current = _id_property_values(modifier)
            keys = {
                key for key in values.keys() | current.keys()
                if values.get(key, _MISSING) != current.get(key, _MISSING)
            }
            group_changed = (
                node_group is not _MISSING
                and getattr(modifier, "node_group", _MISSING) is not node_group
            )
            if keys or group_changed:
                changed_modifiers.append(
                    (pointer, node_group, values, keys, group_changed)
                )

    def restore_modifiers():
        for modifier in tuple(obj.modifiers):
            if modifier.as_pointer() in created_modifier_ids:
                obj.modifiers.remove(modifier)
        current = {modifier.as_pointer(): modifier for modifier in obj.modifiers}
        for pointer, node_group, values, keys, group_changed in changed_modifiers:
            modifier = current.get(pointer)
            if modifier is None:
                continue
            if group_changed and hasattr(modifier, "node_group"):
                modifier.node_group = node_group
            for key in keys:
                value = values.get(key, _MISSING)
                if value is _MISSING:
                    if key in modifier:
                        del modifier[key]
                else:
                    modifier[key] = value

    journal.record(restore_modifiers)
    return capture_modifier_changes


def _setup_selected_transaction(context, mode: str, preset_id: str = "") -> SetupResult:
    """Apply one guided setup transaction to selected mesh objects."""
    mode = str(mode).upper()
    if mode not in SETUP_MODES:
        raise ValueError(f"unsupported setup mode {mode!r}")
    scene = context.scene
    from . import production_bake

    if production_bake.bake_is_active(scene):
        raise RuntimeError("Production Bake owns this scene; finish or cancel it before setup")
    selected = _selected_meshes(context)
    if not selected:
        raise ValueError(
            "Select at least one mesh object other than the FluidSim domain "
            "and outputs"
        )

    from ..presets import library
    from . import colliders, geometry_nodes, output_publication

    default_preset = "thick_liquid" if mode == "QUICK_VISCOUS" else "preview_water"
    preset_id = preset_id or default_preset
    apply_preset = mode in {"QUICK_LIQUID", "QUICK_CONTAINER", "QUICK_VISCOUS"}
    preset = library.load_preset(preset_id) if apply_preset else None
    import bpy

    journal = _SetupRollbackJournal()
    created_ids = _CreatedSetupIDs(journal)
    _record_property_keys(journal, scene, (SCENE_SETUP_ID,))
    scene_fields = [
        "output_collection_name", "domain_object", "live", "cache_enabled",
        "bake_start", "bake_end", "setup_mode", "active_preset_id",
    ]
    if preset is not None:
        scene_fields.extend(
            key for key in preset.settings if key in library.SCENE_OWNED_SETTINGS
        )
    _record_settings(journal, scene.fluidsim, scene_fields)
    _record_selected_state(journal, selected)
    try:
        owner_id, _root, inputs, outputs, helpers, collection_names = (
            ensure_collection_tree(scene)
        )
        for collection in (_root, inputs, outputs, helpers):
            if collection.name in collection_names:
                created_ids.track(bpy.data.collections, collection, 1)

        # Preset application invokes the normal settings callbacks, which may
        # refresh materials on already-configured outputs.  Capture those
        # outputs before the first setup-owned property write so rollback has
        # the genuine pre-transaction state rather than the refreshed state.
        existing_output_names = {obj.name for obj in outputs.objects}
        existing_output_ids = {obj.as_pointer() for obj in outputs.objects}
        point_type = (
            "POINTCLOUD" if output_publication._supports_pointcloud() else "MESH"
        )
        expected_types = {
            output_publication.SURFACE_NAME: "MESH",
            output_publication.PARTICLES_NAME: point_type,
            output_publication.WHITEWATER_NAME: point_type,
            output_publication.FOAM_NAME: point_type,
            output_publication.BUBBLES_NAME: point_type,
            output_publication.SPRAY_NAME: point_type,
        }
        output_state_captures = []
        output_property_keys = (
            OWNER_TAG,
            OWNER_ID_TAG,
            KIND_TAG,
            output_publication.OUTPUT_TAG,
            *geometry_nodes._MODIFIER_OWNER_KEYS.values(),
        )
        for obj in tuple(outputs.objects):
            kind = str(obj.get(output_publication.OUTPUT_TAG, "") or "")
            candidate = kind if kind in expected_types else obj.name
            if candidate not in expected_types:
                continue
            if obj.type != expected_types[candidate]:
                raise ValueError(
                    f"{obj.name}: existing configured output is {obj.type}, "
                    f"expected {expected_types[candidate]}"
                )
            output_state_captures.append(
                _record_output_state(journal, obj, output_property_keys)
            )
        existing_material_ids = {
            material.as_pointer() for material in bpy.data.materials
        }
        scene.fluidsim.output_collection_name = outputs.name
        for obj in selected:
            _link_if_needed(inputs, obj)

        warnings: list[str] = []
        created = list(collection_names)
        # Quick setups (and any setup on a scene without a domain) size the domain
        # from the preset's gravity and voxel size. Those are domain-owned, so the
        # domain object has to exist before the preset can be written to it.
        fit_domain = apply_preset or active_domain_object(scene) is None
        domain = None
        if fit_domain:
            domain, domain_created = _ensure_domain(helpers, owner_id)
            if domain_created:
                created.append(domain.name)
                _track_object_ids(created_ids, bpy, domain)
            else:
                preset_fields = (
                    key for key in preset.settings
                    if key not in library.SCENE_OWNED_SETTINGS
                ) if preset is not None else ()
                _record_domain_state(journal, domain, preset_fields, mode)
        applied = (
            library.apply_preset(scene, preset, target=domain.fluidsim_domain)
            if apply_preset else None
        )
        if mode in {"QUICK_LIQUID", "QUICK_VISCOUS", "ADD_INFLOW"}:
            continuous = mode == "ADD_INFLOW"
            for obj in selected:
                _tag_source_mesh(obj, continuous=continuous)
                warning = _mesh_watertight_warning(obj)
                if warning:
                    warnings.append(warning)
        elif mode == "QUICK_CONTAINER":
            for obj in selected:
                colliders.tag_collider(obj)
            warnings.extend(
                warning for obj in selected
                if (warning := _mesh_watertight_warning(obj))
            )
        elif mode == "ADD_COLLIDER":
            for obj in selected:
                colliders.tag_collider(obj)
        elif mode == "ADD_OUTFLOW":
            for obj in selected:
                _tag_sink_mesh(obj)
        if fit_domain:
            domain_fs = domain.fluidsim_domain
            gravity = (
                tuple(scene.gravity)
                if bool(domain_fs.use_scene_gravity)
                else tuple(domain_fs.gravity)
            )
            wizard_voxel = _wizard_cell_size(domain_fs, preset)
            bounds = compute_domain_bounds(
                selected,
                wizard_voxel,
                gravity=gravity,
                depsgraph=context.evaluated_depsgraph_get(),
            )
            _fit_domain_keeping_voxel(
                context, scene, domain, domain_fs, bounds, wizard_voxel)
            if mode == "QUICK_CONTAINER":
                minimum, maximum = bounds
                # The wizard seeds the same height fields the Height panel edits, so
                # it has to write the owner the solver reads -- otherwise Quick
                # Container produces a scene whose height fill never runs.
                height_owner = domain_fs
                height_owner.height_enabled = True
                height_owner.height_emission_mode = "INITIAL_FILL"
                height_owner.height_base = (
                    minimum[2] + (maximum[2] - minimum[2]) * 0.25
                )
        output_objects = output_publication.ensure_output_shells(scene)
        for obj in output_objects:
            if obj.as_pointer() not in existing_output_ids:
                _track_object_ids(created_ids, bpy, obj)
            for material in getattr(obj.data, "materials", ()):
                if material.as_pointer() not in existing_material_ids:
                    created_ids.track(bpy.data.materials, material, 4)
            _mark_owned(obj, owner_id, "OUTPUT")
            if obj.name not in existing_output_names:
                created.append(obj.name)
        existing_node_group_ids = {
            group.as_pointer() for group in bpy.data.node_groups
        }
        try:
            render_result = geometry_nodes.install_render_nodes(
                scene,
                output_objects={
                    str(obj.get(output_publication.OUTPUT_TAG, obj.name)): obj
                    for obj in output_objects
                },
            )
            warnings.extend(render_result.warnings)
        except Exception as exc:  # noqa: BLE001 - rendering is optional
            warnings.append(f"Render-node installation deferred: {exc}")
        finally:
            for capture in output_state_captures:
                capture()
            for group in bpy.data.node_groups:
                if (
                    group.as_pointer() not in existing_node_group_ids
                    and group.get(geometry_nodes._ASSET_KEY_TAG) in geometry_nodes.GROUP_BY_OUTPUT.values()
                ):
                    created_ids.track(bpy.data.node_groups, group, 3)
        scene.fluidsim.live = False
        scene.fluidsim.cache_enabled = False
        scene.fluidsim.bake_start = int(scene.frame_start)
        scene.fluidsim.bake_end = int(scene.frame_end)
        scene.fluidsim.setup_mode = mode
        if applied is not None:
            scene.fluidsim.active_preset_id = applied.preset_id
        capacity = _capacity_summary(scene)
        message = (
            f"{mode.replace('_', ' ').title()} configured from {len(selected)} selected "
            f"mesh object(s); press Play to simulate"
        )
        result = SetupResult(
            mode=mode,
            preset_id=applied.preset_id if applied else "",
            domain_name=getattr(domain, "name", ""),
            input_names=tuple(obj.name for obj in selected),
            output_names=tuple(obj.name for obj in output_objects),
            created_names=tuple(created),
            warnings=tuple(warnings),
            capacity_message=capacity,
            message=message,
        )
    except Exception as setup_error:
        from . import pending_settings

        try:
            with pending_settings.suppress_callbacks(scene):
                journal.rollback()
        except Exception as rollback_error:
            raise RuntimeError(
                f"{setup_error}; setup rollback was incomplete: {rollback_error}"
            ) from setup_error
        raise
    journal.commit()
    return result


def setup_selected(context, mode: str, preset_id: str = "") -> SetupResult:
    """Apply Guided Setup without publishing intermediate callback state."""
    from . import pending_settings

    with pending_settings.suppress_callbacks(context.scene):
        return _setup_selected_transaction(context, mode, preset_id)

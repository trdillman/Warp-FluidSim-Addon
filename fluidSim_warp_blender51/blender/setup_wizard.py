"""Guided selected-object setup for a runnable FluidSim scene."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import uuid

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


def _ensure_domain(scene, helpers, owner_id: str, bounds):
    import bpy

    fs = scene.fluidsim
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
    minimum, maximum = bounds
    center = tuple((lo + hi) * 0.5 for lo, hi in zip(minimum, maximum))
    dimensions = tuple(max(hi - lo, 1.0e-6) for lo, hi in zip(minimum, maximum))
    domain.location = center
    domain.rotation_euler = (0.0, 0.0, 0.0)
    domain.scale = dimensions
    domain.display_type = "WIRE"
    domain.hide_render = True
    domain.show_in_front = True
    fs.domain_object = domain
    return domain, created


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


def _selected_meshes(context):
    accepted_roles = {"NONE", "SOURCE", "COLLIDER", "SINK"}
    return tuple(
        obj
        for obj in getattr(context, "selected_objects", ())
        if getattr(obj, "type", None) == "MESH"
        and getattr(getattr(obj, "fluidsim", None), "role", "NONE")
        in accepted_roles
    )


def setup_selected(context, mode: str, preset_id: str = "") -> SetupResult:
    """Apply one guided setup transaction to selected mesh objects."""
    mode = str(mode).upper()
    if mode not in SETUP_MODES:
        raise ValueError(f"unsupported setup mode {mode!r}")
    scene = context.scene
    selected = _selected_meshes(context)
    if not selected:
        raise ValueError("Select at least one mesh object")

    from ..presets import library
    from . import colliders, geometry_nodes, output_publication

    default_preset = "thick_liquid" if mode == "QUICK_VISCOUS" else "preview_water"
    preset_id = preset_id or default_preset
    apply_preset = mode in {"QUICK_LIQUID", "QUICK_CONTAINER", "QUICK_VISCOUS"}
    applied = library.apply_preset(scene, library.load_preset(preset_id)) if apply_preset else None

    owner_id, _root, inputs, outputs, helpers, collection_names = (
        ensure_collection_tree(scene)
    )
    scene.fluidsim.output_collection_name = outputs.name
    for obj in selected:
        _link_if_needed(inputs, obj)

    warnings: list[str] = []
    created = list(collection_names)
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

    domain = getattr(scene.fluidsim, "domain_object", None)
    if mode in {"QUICK_LIQUID", "QUICK_CONTAINER", "QUICK_VISCOUS"} or domain is None:
        gravity = (
            tuple(scene.gravity)
            if bool(getattr(scene.fluidsim, "use_scene_gravity", True))
            else tuple(scene.fluidsim.gravity)
        )
        bounds = compute_domain_bounds(
            selected,
            float(scene.fluidsim.voxel_size),
            gravity=gravity,
            depsgraph=context.evaluated_depsgraph_get(),
        )
        domain, domain_created = _ensure_domain(scene, helpers, owner_id, bounds)
        if domain_created:
            created.append(domain.name)
        if mode == "QUICK_CONTAINER":
            minimum, maximum = bounds
            # The wizard seeds the same height fields the Height panel edits, so
            # it has to write the owner the solver reads -- otherwise Quick
            # Container produces a scene whose height fill never runs.
            from .scene_bridge import active_domain_settings

            height_owner = active_domain_settings(scene) or scene.fluidsim
            height_owner.height_enabled = True
            height_owner.height_emission_mode = "INITIAL_FILL"
            height_owner.height_base = minimum[2] + (maximum[2] - minimum[2]) * 0.25

    existing_output_names = {obj.name for obj in outputs.objects}
    output_objects = output_publication.ensure_output_shells(scene)
    for obj in output_objects:
        _mark_owned(obj, owner_id, "OUTPUT")
        if obj.name not in existing_output_names:
            created.append(obj.name)
    try:
        render_result = geometry_nodes.install_render_nodes(
            scene,
            output_objects={
                str(obj.get(output_publication.OUTPUT_TAG, obj.name)): obj
                for obj in output_objects
            },
        )
        warnings.extend(render_result.warnings)
    except Exception as exc:  # noqa: BLE001 - setup remains recoverable
        warnings.append(f"Render-node installation deferred: {exc}")

    scene.fluidsim.live = False
    scene.fluidsim.cache_enabled = False
    scene.fluidsim.bake_start = int(scene.frame_start)
    scene.fluidsim.bake_end = int(scene.frame_end)
    if hasattr(scene.fluidsim, "setup_mode"):
        scene.fluidsim.setup_mode = mode
    if hasattr(scene.fluidsim, "active_preset_id") and applied is not None:
        scene.fluidsim.active_preset_id = applied.preset_id
    capacity = _capacity_summary(scene)
    message = (
        f"{mode.replace('_', ' ').title()} configured from {len(selected)} selected "
        f"mesh object(s); arm Live and press Play, or initialize for manual stepping"
    )
    return SetupResult(
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

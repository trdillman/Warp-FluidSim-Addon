"""Install and attach bundled Geometry Nodes render groups.

The bundled ``.blend`` asset is immutable release content.  Installation
appends a tagged node group into the current file and attaches it through one
owned modifier per FluidSim output object.  Existing user node groups are never
mutated or removed; an explicit ``force=True`` only repoints FluidSim-owned
modifiers to the current bundled revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ASSET_REVISION = "fluidsim-render-p3-v1"
ASSET_RELATIVE_PATH = Path("resources") / "geometry_nodes" / "fluidsim_render_nodes.blend"

SURFACE_GROUP = "FluidSim Surface Render [P3 v1]"
PARTICLE_GROUP = "FluidSim Liquid Particles [P3 v1]"
WHITEWATER_GROUP = "FluidSim Whitewater [P3 v1]"

GROUP_BY_OUTPUT = {
    "FluidSim Surface": SURFACE_GROUP,
    "FluidSim Particles": PARTICLE_GROUP,
    "FluidSim Whitewater": WHITEWATER_GROUP,
}
MODIFIER_BY_OUTPUT = {
    "FluidSim Surface": "FluidSim Surface Render",
    "FluidSim Particles": "FluidSim Liquid Particles",
    "FluidSim Whitewater": "FluidSim Whitewater",
}
MATERIAL_SOCKET = "Material"
RADIUS_SOCKET = "Radius Scale"

_OWNER_TAG = "fluidsim_owned"
_ASSET_KEY_TAG = "fluidsim_asset_key"
_REVISION_TAG = "fluidsim_source_revision"
_MODIFIER_OWNER_KEYS = {
    "FluidSim Surface": "fluidsim_surface_render_modifier",
    "FluidSim Particles": "fluidsim_particles_render_modifier",
    "FluidSim Whitewater": "fluidsim_whitewater_render_modifier",
}


@dataclass(frozen=True)
class InstallResult:
    installed_groups: tuple[str, ...]
    attached_objects: tuple[str, ...]
    skipped_objects: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def message(self) -> str:
        return (
            f"FluidSim render nodes ready on {len(self.attached_objects)} output(s); "
            f"{len(self.installed_groups)} group(s) appended"
        )


def asset_path() -> Path:
    return Path(__file__).resolve().parents[1] / ASSET_RELATIVE_PATH


def _require_bpy():
    try:
        import bpy
    except ModuleNotFoundError as exc:  # pragma: no cover - Blender-only path
        raise RuntimeError("Geometry Nodes installation requires Blender") from exc
    return bpy


def _tagged_group(bpy, asset_key: str):
    for group in bpy.data.node_groups:
        try:
            if (
                group.get(_ASSET_KEY_TAG) == asset_key
                and group.get(_REVISION_TAG) == ASSET_REVISION
            ):
                return group
        except AttributeError:
            continue
    return None


def _append_group(bpy, group_name: str):
    path = asset_path()
    if not path.is_file():
        raise FileNotFoundError(f"bundled Geometry Nodes asset is missing: {path}")
    with bpy.data.libraries.load(str(path), link=False) as (source, target):
        if group_name not in source.node_groups:
            raise RuntimeError(
                f"bundled Geometry Nodes asset does not contain {group_name!r}"
            )
        target.node_groups = [group_name]
    group = target.node_groups[0]
    if group is None:
        raise RuntimeError(f"Blender did not append node group {group_name!r}")
    if group.get(_REVISION_TAG) != ASSET_REVISION:
        bpy.data.node_groups.remove(group)
        raise RuntimeError(f"node group {group_name!r} has an unexpected revision")
    return group


def ensure_node_group(asset_key: str):
    """Return the current immutable bundled group, appending it if needed."""
    bpy = _require_bpy()
    if asset_key not in GROUP_BY_OUTPUT.values():
        raise ValueError(f"unknown FluidSim node-group key {asset_key!r}")
    group = _tagged_group(bpy, asset_key)
    return group if group is not None else _append_group(bpy, asset_key)


def _interface_input_identifier(group, name: str) -> str | None:
    for item in group.interface.items_tree:
        if (
            getattr(item, "item_type", "") == "SOCKET"
            and getattr(item, "in_out", "") == "INPUT"
            and item.name == name
        ):
            return str(item.identifier)
    return None


def _set_modifier_input(modifier, group, socket_name: str, value) -> None:
    identifier = _interface_input_identifier(group, socket_name)
    if not identifier:
        return
    # Blender 5.2 exposes Geometry Nodes modifier values through a typed
    # ``modifier.properties.inputs.<identifier>.value`` interface.  Older
    # supported builds used ID properties keyed by the interface identifier.
    # Prefer the typed API, then fall back so the release asset remains usable
    # across the certified Blender line without version-string branching.
    try:
        inputs = modifier.properties.inputs
        socket = getattr(inputs, identifier)
        socket.value = value
        return
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    try:
        modifier[identifier] = value
    except (KeyError, TypeError, ValueError):
        # A missing optional socket in a future asset revision must not make
        # the entire installation transaction fail.
        return


def _modifier_input_value(modifier, group, socket_name: str):
    """Return an assigned interface value for validation and diagnostics."""
    identifier = _interface_input_identifier(group, socket_name)
    if not identifier:
        raise KeyError(socket_name)
    try:
        return getattr(modifier.properties.inputs, identifier).value
    except (AttributeError, KeyError, TypeError):
        return modifier[identifier]


def _object_material(obj):
    materials = getattr(getattr(obj, "data", None), "materials", None)
    return materials[0] if materials and len(materials) else None


def _owned_modifier(obj, output_name: str, modifier_name: str):
    modifier = obj.modifiers.get(modifier_name)
    if modifier is None:
        return None
    try:
        owned_name = str(obj.get(_MODIFIER_OWNER_KEYS[output_name], ""))
    except (AttributeError, TypeError):
        return None
    return modifier if owned_name == modifier.name else None


def _attach_group(
    obj, group, output_name: str, *, force: bool, radius_scale: float
) -> tuple[bool, str]:
    modifier_name = MODIFIER_BY_OUTPUT[output_name]
    existing = obj.modifiers.get(modifier_name)
    owned = _owned_modifier(obj, output_name, modifier_name)
    if existing is not None and owned is None:
        return False, f"{obj.name}: modifier {modifier_name!r} is user-owned"
    if owned is None:
        owned = obj.modifiers.new(name=modifier_name, type="NODES")
        obj[_MODIFIER_OWNER_KEYS[output_name]] = owned.name
    current = getattr(owned, "node_group", None)
    current_revision = (
        current.get(_REVISION_TAG) if current is not None and hasattr(current, "get") else None
    )
    if current is not None and current is not group and current_revision != ASSET_REVISION:
        if not force:
            return False, (
                f"{obj.name}: FluidSim modifier uses another revision; "
                "use the explicit reinstall action to update it"
            )
    owned.node_group = group
    obj[_MODIFIER_OWNER_KEYS[output_name]] = owned.name
    _set_modifier_input(owned, group, MATERIAL_SOCKET, _object_material(obj))
    _set_modifier_input(owned, group, RADIUS_SOCKET, float(radius_scale))
    return True, ""


def install_render_nodes(
    scene,
    *,
    force: bool = False,
    output_objects: Mapping[str, object] | None = None,
) -> InstallResult:
    """Install groups and attach them idempotently to FluidSim outputs."""
    _require_bpy()
    from . import output_publication

    if output_objects is None:
        output_objects = {
            output_name: output_publication.output_object(scene, output_name)
            for output_name in GROUP_BY_OUTPUT
        }
    groups = {}
    appended: list[str] = []
    for output_name, group_name in GROUP_BY_OUTPUT.items():
        before = _tagged_group(_require_bpy(), group_name)
        group = ensure_node_group(group_name)
        groups[output_name] = group
        if before is None:
            appended.append(group.name)

    fs = scene.fluidsim
    radius_scales = {
        "FluidSim Surface": 1.0,
        "FluidSim Particles": float(getattr(fs, "particle_display_scale", 1.0)),
        "FluidSim Whitewater": float(getattr(fs, "whitewater_display_scale", 1.0)),
    }
    attached: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []
    for output_name, group in groups.items():
        obj = output_objects.get(output_name)
        if obj is None:
            skipped.append(output_name)
            warnings.append(f"{output_name}: output object is missing")
            continue
        ok, warning = _attach_group(
            obj,
            group,
            output_name,
            force=bool(force),
            radius_scale=radius_scales[output_name],
        )
        if ok:
            attached.append(obj.name)
        else:
            skipped.append(obj.name)
            warnings.append(warning)
    return InstallResult(
        installed_groups=tuple(appended),
        attached_objects=tuple(attached),
        skipped_objects=tuple(skipped),
        warnings=tuple(warnings),
    )


def sync_render_inputs(scene) -> None:
    """Refresh material and radius modifier inputs without changing groups."""
    from . import output_publication

    fs = scene.fluidsim
    radius_scales = {
        "FluidSim Surface": 1.0,
        "FluidSim Particles": float(getattr(fs, "particle_display_scale", 1.0)),
        "FluidSim Whitewater": float(getattr(fs, "whitewater_display_scale", 1.0)),
    }
    for output_name, modifier_name in MODIFIER_BY_OUTPUT.items():
        obj = output_publication.output_object(scene, output_name)
        modifier = (
            _owned_modifier(obj, output_name, modifier_name)
            if obj is not None else None
        )
        group = getattr(modifier, "node_group", None) if modifier is not None else None
        if group is None or group.get(_REVISION_TAG) != ASSET_REVISION:
            continue
        _set_modifier_input(modifier, group, MATERIAL_SOCKET, _object_material(obj))
        _set_modifier_input(modifier, group, RADIUS_SOCKET, radius_scales[output_name])


def installed_state(scene) -> dict[str, str]:
    """Return output-name to current FluidSim render-node revision."""
    from . import output_publication

    state = {}
    for output_name, modifier_name in MODIFIER_BY_OUTPUT.items():
        obj = output_publication.output_object(scene, output_name)
        modifier = obj.modifiers.get(modifier_name) if obj is not None else None
        group = getattr(modifier, "node_group", None)
        state[output_name] = str(group.get(_REVISION_TAG, "")) if group else ""
    return state

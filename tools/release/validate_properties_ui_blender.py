"""Validate FluidSim's native Properties workflow in the supplied Blender runtime."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy

# The Scene tab registers scene-owned panels only; every domain control is a
# Physics tab mirror (see PHYSICS_MIRROR_SOURCES below).
EXPECTED_SCENE_ROOTS = {
    "FLUIDSIM_PT_scene_lifecycle": (1, True),
    "FLUIDSIM_PT_scene_control": (2, False),
    "FLUIDSIM_PT_scene_setup": (10, False),
    "FLUIDSIM_PT_custom_effects": (50, True),
    "FLUIDSIM_PT_scene_outputs": (80, False),
    "FLUIDSIM_PT_scene_diagnostics": (100, True),
}
# Source panel ids (the Scene-tab draw functions) mirrored into Physics
# Properties, per mirror group. A mirror id is
# ``FLUIDSIM_PT_physics_<group>_<source id without FLUIDSIM_PT_>``.
PHYSICS_MIRROR_SOURCES = {
    "domain": (
        "FLUIDSIM_PT_scene_domain",
        "FLUIDSIM_PT_scene_capacity",
        "FLUIDSIM_PT_scene_height",
        "FLUIDSIM_PT_scene_simulation",
        "FLUIDSIM_PT_sim_time",
        "FLUIDSIM_PT_sim_gravity",
        "FLUIDSIM_PT_sim_pressure",
        "FLUIDSIM_PT_sim_viscosity",
        "FLUIDSIM_PT_sim_surface_tension",
        "FLUIDSIM_PT_sim_color_mixing",
        "FLUIDSIM_PT_sim_relax",
        "FLUIDSIM_PT_sim_limits",
        "FLUIDSIM_PT_sim_forces",
        "FLUIDSIM_PT_sim_dissipation",
        "FLUIDSIM_PT_sim_attraction",
        "FLUIDSIM_PT_output_attributes",
        "FLUIDSIM_PT_attrs_surface",
        "FLUIDSIM_PT_attrs_particles",
        "FLUIDSIM_PT_attrs_whitewater",
        "FLUIDSIM_PT_diagnostics_profiling",
    ),
    "surface": (
        "FLUIDSIM_PT_scene_surface",
        "FLUIDSIM_PT_surface_resolution",
        "FLUIDSIM_PT_surface_reconstruct",
        "FLUIDSIM_PT_surface_filters",
    ),
    "whitewater": (
        "FLUIDSIM_PT_scene_whitewater",
        "FLUIDSIM_PT_whitewater_emission",
        "FLUIDSIM_PT_whitewater_classification",
        "FLUIDSIM_PT_whitewater_life",
        "FLUIDSIM_PT_whitewater_dynamics",
    ),
}


def _mirror_id(group: str, source_id: str) -> str:
    return f"FLUIDSIM_PT_physics_{group}_{source_id[len('FLUIDSIM_PT_'):]}"
EXPECTED_OBJECT_ROOTS = {
    "FLUIDSIM_PT_object_role": 0,
    "FLUIDSIM_PT_object_region": 10,
    "FLUIDSIM_PT_object_source": 20,
    "FLUIDSIM_PT_object_blending": 30,
    "FLUIDSIM_PT_object_collider": 40,
    "FLUIDSIM_PT_object_force": 50,
    "FLUIDSIM_PT_object_field": 60,
    "FLUIDSIM_PT_object_volume": 70,
}
EXPECTED_EFFECT_CHILDREN = {
    "FLUIDSIM_PT_effect_packages",
    "FLUIDSIM_PT_effect_warp_code",
}

ALL_AUTHORED_SCENE_PANELS = (
    "FLUIDSIM_PT_scene_lifecycle",
    "FLUIDSIM_PT_scene_control",
    "FLUIDSIM_PT_scene_live_timeline",
    "FLUIDSIM_PT_scene_setup",
    "FLUIDSIM_PT_scene_presets",
    "FLUIDSIM_PT_scene_outputs",
    "FLUIDSIM_PT_output_display",
    "FLUIDSIM_PT_output_materials",
    "FLUIDSIM_PT_scene_diagnostics",
    "FLUIDSIM_PT_diagnostics_benchmark",
    "FLUIDSIM_PT_diagnostics_report",
    "FLUIDSIM_PT_custom_effects",
    "FLUIDSIM_PT_effect_packages",
    "FLUIDSIM_PT_effect_engineering",
    "FLUIDSIM_PT_effect_warp_code",
    "FLUIDSIM_PT_viewport_status",
    "FLUIDSIM_PT_header_popover",
)

# Panels the sidebar redesign deleted outright: transport and cache now live
# only in the pinned launcher strip and the Bake & Export stage, and the
# Live Tuning shelf's knobs are Simulate / Look stage controls.
REMOVED_PANEL_IDS = (
    "FLUIDSIM_VIEW3D_PT_live_tuning",
    "FLUIDSIM_VIEW3D_PT_full_settings",
    "FLUIDSIM_PT_scene_transport",
    "FLUIDSIM_PT_scene_cache",
    "FLUIDSIM_PT_cache_playback",
    "FLUIDSIM_PT_cache_range",
    "FLUIDSIM_PT_cache_storage",
    "FLUIDSIM_PT_cache_bake",
    "FLUIDSIM_PT_cache_manage",
    "FLUIDSIM_STAGE_PT_bake_advanced",
)

# Staged-workflow sidebar shelves: panel id -> (bl_order, bl_parent_id, closed).
EXPECTED_STAGE_PANELS = {
    "FLUIDSIM_STAGE_PT_setup": (10, "", False),
    "FLUIDSIM_STAGE_PT_setup_advanced": (0, "FLUIDSIM_STAGE_PT_setup", True),
    "FLUIDSIM_STAGE_PT_simulate": (20, "", False),
    "FLUIDSIM_STAGE_PT_look": (30, "", False),
    "FLUIDSIM_STAGE_PT_look_materials": (0, "FLUIDSIM_STAGE_PT_look", True),
    "FLUIDSIM_STAGE_PT_bake": (40, "", False),
    "FLUIDSIM_STAGE_PT_bake_storage": (1, "FLUIDSIM_STAGE_PT_bake", True),
    "FLUIDSIM_STAGE_PT_bake_manage": (2, "FLUIDSIM_STAGE_PT_bake", True),
}

OBJECT_ROLE_CASES = (
    ("SOURCE_ANALYTIC", "SOURCE", "source_region_kind", "ANALYTIC"),
    ("SOURCE_MESH_SHELL", "SOURCE", "source_region_kind", "MESH_SHELL"),
    ("SOURCE_POINTS", "SOURCE", "source_region_kind", "POINTS"),
    ("SOURCE_VOLUME_GRID", "SOURCE", "source_region_kind", "VOLUME_GRID"),
    ("SINK_ANALYTIC", "SINK", "sink_region_kind", "ANALYTIC"),
    ("SINK_VOLUME_GRID", "SINK", "sink_region_kind", "VOLUME_GRID"),
    ("COLLIDER", "COLLIDER", "", ""),
    ("FORCE", "FORCE", "", ""),
    ("FIELD_ANALYTIC", "FIELD", "field_region_kind", "ANALYTIC"),
    ("FIELD_VOLUME_GRID", "FIELD", "field_region_kind", "VOLUME_GRID"),
)

OBJECT_PANEL_IDS = (
    "FLUIDSIM_PT_object_role",
    "FLUIDSIM_PT_object_status",
    "FLUIDSIM_PT_object_region",
    "FLUIDSIM_PT_object_source",
    "FLUIDSIM_PT_object_lifespan",
    "FLUIDSIM_PT_object_blending",
    "FLUIDSIM_PT_object_collider",
    "FLUIDSIM_PT_object_force",
    "FLUIDSIM_PT_object_field",
    "FLUIDSIM_PT_object_volume",
)


def _certified(name: str):
    """Read one certified release constant from the source tree's version.py."""
    path = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51" / "version.py"
    import ast

    values: dict = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in values:
            value = values[node.value.id]  # e.g. CERTIFIED_X = VALIDATION_HOST_X
        else:
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    if name not in values:
        raise RuntimeError(f"{name} is not a literal constant in {path}")
    return values[name]


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-warp-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class _OperatorProxy:
    def __init__(self, operator_id: str, values: dict[str, object]):
        object.__setattr__(self, "operator_id", operator_id)
        object.__setattr__(self, "values", values)

    def __setattr__(self, name: str, value: object) -> None:
        self.values[name] = value


class RecordingLayout:
    """Small faithful layout recorder for background panel draw validation."""

    def __init__(self, events: list[dict[str, object]] | None = None):
        object.__setattr__(self, "events", events if events is not None else [])
        object.__setattr__(self, "enabled", True)
        object.__setattr__(self, "active", True)
        object.__setattr__(self, "alert", False)
        object.__setattr__(self, "alignment", "EXPAND")
        object.__setattr__(self, "operator_context", "INVOKE_DEFAULT")
        object.__setattr__(self, "use_property_split", False)
        object.__setattr__(self, "use_property_decorate", True)

    def _child(self, kind: str, **values):
        self.events.append({"kind": kind, **values})
        child = RecordingLayout(self.events)
        child.enabled = self.enabled
        child.active = self.active
        return child

    def row(self, *, align=False):
        return self._child("row", align=bool(align))

    def column(self, *, align=False):
        return self._child("column", align=bool(align))

    def box(self):
        return self._child("box")

    def grid_flow(self, **kwargs):
        return self._child("grid_flow", **kwargs)

    def split(self, **kwargs):
        return self._child("split", **kwargs)

    def label(self, *, text="", icon="NONE"):
        self.events.append({"kind": "label", "text": str(text), "icon": str(icon)})

    def prop(self, data, property_name: str, **kwargs):
        if not hasattr(data, property_name):
            raise AssertionError(
                f"draw requested missing property {type(data).__name__}.{property_name}"
            )
        self.events.append(
            {
                "kind": "prop",
                "property": property_name,
                "text": str(kwargs.get("text", "")),
            }
        )

    def operator(self, operator_id: str, **kwargs):
        values: dict[str, object] = {}
        self.events.append(
            {
                "kind": "operator",
                "operator": operator_id,
                "text": str(kwargs.get("text", "")),
                "values": values,
            }
        )
        return _OperatorProxy(operator_id, values)

    def template_list(self, *args, **kwargs):
        self.events.append({"kind": "template_list", "args": [str(value) for value in args]})

    def separator(self, **kwargs):
        self.events.append({"kind": "separator", **kwargs})

    def popover(self, **kwargs):
        self.events.append({"kind": "popover", **kwargs})


class _Reporter:
    def __init__(self):
        self.messages: list[tuple[tuple[str, ...], str]] = []

    def report(self, levels, message):
        self.messages.append((tuple(sorted(str(value) for value in levels)), str(message)))


def _panel(panel_id: str):
    panel = getattr(bpy.types, panel_id, None)
    if panel is None:
        raise AssertionError(f"registered panel missing: {panel_id}")
    return panel


def _panel_metadata(panel_id: str) -> dict[str, object]:
    panel = _panel(panel_id)
    return {
        "space": str(getattr(panel, "bl_space_type", "")),
        "region": str(getattr(panel, "bl_region_type", "")),
        "context": str(getattr(panel, "bl_context", "")),
        "order": int(getattr(panel, "bl_order", 0)),
        "parent": str(getattr(panel, "bl_parent_id", "")),
        "closed": "DEFAULT_CLOSED" in set(getattr(panel, "bl_options", set())),
    }


@contextlib.contextmanager
def _effect_authoring_enabled(module_name: str):
    """Open the D9 developer gate (normally the add-on preference) for a draw."""
    preferences = importlib.import_module(f"{module_name}.preferences")
    gate = preferences.effect_authoring_enabled
    preferences.effect_authoring_enabled = lambda context=None: True
    try:
        yield
    finally:
        preferences.effect_authoring_enabled = gate


def _draw(panel_id: str, context) -> list[dict[str, object]]:
    panel = _panel(panel_id)
    poll = getattr(panel, "poll", None)
    if callable(poll) and not bool(poll(context)):
        raise AssertionError(f"representative poll rejected {panel_id}")
    layout = RecordingLayout()
    panel.draw(SimpleNamespace(layout=layout), context)
    if not layout.events:
        raise AssertionError(f"representative draw emitted no controls: {panel_id}")
    return layout.events


def _assert_metadata() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for panel_id, (order, closed) in EXPECTED_SCENE_ROOTS.items():
        metadata = _panel_metadata(panel_id)
        if (metadata["space"], metadata["region"], metadata["context"]) != (
            "PROPERTIES",
            "WINDOW",
            "scene",
        ):
            raise AssertionError(f"{panel_id} is not a Scene Properties panel: {metadata}")
        if metadata["order"] != order or metadata["closed"] is not closed:
            raise AssertionError(f"{panel_id} order/default state mismatch: {metadata}")
        records[panel_id] = metadata
    for panel_id, order in EXPECTED_OBJECT_ROOTS.items():
        metadata = _panel_metadata(panel_id)
        if (metadata["space"], metadata["region"], metadata["context"]) != (
            "PROPERTIES",
            "WINDOW",
            "object",
        ):
            raise AssertionError(f"{panel_id} is not an Object Properties panel: {metadata}")
        if metadata["order"] != order:
            raise AssertionError(f"{panel_id} order mismatch: {metadata}")
        records[panel_id] = metadata
    for panel_id in EXPECTED_EFFECT_CHILDREN:
        metadata = _panel_metadata(panel_id)
        if metadata["parent"] != "FLUIDSIM_PT_custom_effects" or not metadata["closed"]:
            raise AssertionError(f"{panel_id} is not a closed Custom Effects child: {metadata}")
        records[panel_id] = metadata
    viewport = _panel_metadata("FLUIDSIM_PT_viewport_status")
    if (viewport["space"], viewport["region"]) != ("VIEW_3D", "UI"):
        raise AssertionError(f"compact viewport launcher metadata is incorrect: {viewport}")
    records["FLUIDSIM_PT_viewport_status"] = viewport
    if viewport["parent"]:
        raise AssertionError(f"viewport launcher must be a root strip: {viewport}")
    for removed in REMOVED_PANEL_IDS:
        if hasattr(bpy.types, removed):
            raise AssertionError(f"deleted panel is still registered: {removed}")
    for group, source_ids in PHYSICS_MIRROR_SOURCES.items():
        for source_id in source_ids:
            if hasattr(bpy.types, source_id):
                raise AssertionError(
                    f"Scene Properties still registers the domain panel {source_id}")
            mirror_id = _mirror_id(group, source_id)
            mirror = _panel_metadata(mirror_id)
            if (mirror["space"], mirror["region"], mirror["context"]) != (
                "PROPERTIES",
                "WINDOW",
                "physics",
            ):
                raise AssertionError(
                    f"{mirror_id} is not a Physics Properties panel: {mirror}")
            records[mirror_id] = mirror
    for panel_id, (order, parent, closed) in EXPECTED_STAGE_PANELS.items():
        stage = _panel_metadata(panel_id)
        if (stage["space"], stage["region"]) != ("VIEW_3D", "UI"):
            raise AssertionError(f"{panel_id} is not a 3D View sidebar panel: {stage}")
        if str(getattr(_panel(panel_id), "bl_category", "")) != "FluidSim":
            raise AssertionError(f"{panel_id} is not in the FluidSim sidebar tab")
        if (stage["parent"], stage["closed"]) != (parent, closed) or (
            not parent and stage["order"] != order
        ):
            raise AssertionError(f"{panel_id} order/parent/default state mismatch: {stage}")
        records[panel_id] = stage
    physics_root = _panel_metadata("FLUIDSIM_PT_physics_root")
    if (
        physics_root["space"],
        physics_root["region"],
        physics_root["context"],
    ) != ("PROPERTIES", "WINDOW", "physics"):
        raise AssertionError(
            f"contextual Physics root metadata is incorrect: {physics_root}")
    records["FLUIDSIM_PT_physics_root"] = physics_root
    if not hasattr(bpy.types, "FLUIDSIM_OT_open_scene_properties"):
        raise AssertionError("FLUIDSIM_OT_open_scene_properties is not registered")
    for legacy in (
        "FLUIDSIM_PT_simulation",
        "FLUIDSIM_PT_quality",
        "FLUIDSIM_PT_inputs",
        "FLUIDSIM_PT_physics",
        "FLUIDSIM_PT_snippets",
    ):
        if hasattr(bpy.types, legacy):
            raise AssertionError(f"legacy dense sidebar panel remains registered: {legacy}")
    return records


def _ensure_cube():
    cube = bpy.data.objects.get("FluidSim UI Validation Cube")
    if cube is None:
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        cube = bpy.context.object
        cube.name = "FluidSim UI Validation Cube"
    bpy.ops.object.select_all(action="DESELECT")
    cube.hide_set(False)
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    return cube


def _ensure_domain():
    """Designate the validation domain; domain settings live only on it."""
    scene = bpy.context.scene
    name = "FluidSim UI Validation Domain"
    domain = bpy.data.objects.get(name)
    if domain is None:
        domain = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        scene.collection.objects.link(domain)
    scene.fluidsim.domain_object = domain
    return domain.fluidsim_domain


def _draw_physics_mirrors(module_name: str, group: str, source_ids) -> dict[str, list]:
    """Draw Physics mirrors of ``source_ids`` with their poll's object active."""
    output_publication = importlib.import_module(
        f"{module_name}.blender.output_publication")
    scene = bpy.context.scene
    previous = bpy.context.view_layer.objects.active
    if group == "domain":
        target = scene.fluidsim.domain_object
    else:
        output_publication.ensure_output_shells(scene)
        target = output_publication.output_object(scene, {
            "surface": output_publication.SURFACE_NAME,
            "whitewater": output_publication.WHITEWATER_NAME,
        }[group])
    if target is None:
        raise AssertionError(f"no object to draw the {group} Physics mirrors")
    _activate_object(target)
    try:
        return {
            _mirror_id(group, source_id): _draw(_mirror_id(group, source_id), bpy.context)
            for source_id in source_ids
        }
    finally:
        if previous is not None:
            _activate_object(previous)


def _assert_scene_and_object_draws(module_name: str) -> dict[str, object]:
    scene = bpy.context.scene
    ds = _ensure_domain()
    # Properties panels draw the same in Basic and Advanced; Advanced is set for parity.
    scene.fluidsim.ui_mode = "ADVANCED"
    ds.surface_enabled = True
    ds.whitewater_enabled = True
    ds.surface_tension_enabled = True
    cube = _ensure_cube()
    cube.fluidsim.role = "SOURCE"
    cube.fluidsim.enabled = True
    cube.fluidsim.source_region_kind = "ANALYTIC"
    cube.fluidsim.use_lifespan = True

    draw_ids = (
        "FLUIDSIM_PT_scene_control",
        "FLUIDSIM_PT_output_display",
        "FLUIDSIM_STAGE_PT_bake",
        "FLUIDSIM_PT_object_role",
        "FLUIDSIM_PT_object_region",
        "FLUIDSIM_PT_object_source",
        "FLUIDSIM_PT_object_lifespan",
        "FLUIDSIM_PT_viewport_status",
    )
    draws = {panel_id: _draw(panel_id, bpy.context) for panel_id in draw_ids}
    # Domain controls draw from the Physics mirrors, which poll on the active
    # object's kind: the domain object for the domain group, the generated
    # whitewater output object for the whitewater group.
    domain_mirror = _mirror_id("domain", "FLUIDSIM_PT_scene_domain")
    draws.update(_draw_physics_mirrors(
        module_name, "domain",
        ("FLUIDSIM_PT_scene_domain", "FLUIDSIM_PT_sim_surface_tension")))
    draws.update(_draw_physics_mirrors(
        module_name, "whitewater", ("FLUIDSIM_PT_whitewater_classification",)))

    effects_module = importlib.import_module(f"{module_name}.blender.effects")
    effects_module.add_builtin_effect(scene, "fluidsim.velocity_damping")
    draws["FLUIDSIM_PT_custom_effects"] = _draw("FLUIDSIM_PT_custom_effects", bpy.context)
    draws["FLUIDSIM_PT_effect_packages"] = _draw("FLUIDSIM_PT_effect_packages", bpy.context)
    draws["FLUIDSIM_PT_effect_engineering"] = _draw(
        "FLUIDSIM_PT_effect_engineering", bpy.context)
    # D9: Warp Code is a developer surface, hidden until the add-on preference
    # opts in. Prove the default hides it, then draw it with the gate open.
    warp_code = getattr(bpy.types, "FLUIDSIM_PT_effect_warp_code")
    if warp_code.poll(bpy.context):
        raise AssertionError("Warp Code panel is visible without the developer preference")
    with _effect_authoring_enabled(module_name):
        draws["FLUIDSIM_PT_effect_warp_code"] = _draw(
            "FLUIDSIM_PT_effect_warp_code", bpy.context)

    scene_props = {
        event["property"]
        for event in draws[domain_mirror]
        if event["kind"] == "prop"
    }
    # Resolution Divisions replaced the Voxel Size control (voxel_size is derived).
    if not {"resolution_divisions", "particles_per_voxel", "max_particles"}.issubset(
            scene_props):
        raise AssertionError(f"Domain panel omitted primary controls: {sorted(scene_props)}")
    def _operators(panel_id):
        return {
            event["operator"]
            for event in draws[panel_id]
            if event["kind"] == "operator"
        }

    artist_ops = _operators("FLUIDSIM_PT_custom_effects")
    # 9bbebb0 retired the Preview button: Blender's own Play drives the simulation.
    if not {"screen.animation_play", "fluidsim.apply_changes"}.issubset(artist_ops):
        raise AssertionError(f"Custom Effects artist actions are incomplete: {sorted(artist_ops)}")
    effect_ops = _operators("FLUIDSIM_PT_effect_engineering")
    if not {
        "fluidsim.effect_test_draft",
        "fluidsim.effect_apply_stack",
        "fluidsim.effect_revert_draft",
        "fluidsim.effect_preview_one_frame",
    }.issubset(effect_ops):
        raise AssertionError(f"Effect Engineering workflow is incomplete: {sorted(effect_ops)}")
    return {
        "draw_event_counts": {key: len(value) for key, value in draws.items()},
        "source_role": cube.fluidsim.role,
        "effect_count": len(scene.fluidsim_effects.items),
    }



def _configure_all_authored_scene_states(work_dir: Path) -> None:
    fs = bpy.context.scene.fluidsim
    values = {
        "height_enabled": True,
        "height_back_mode": "PROFILE",
        "sample_type": "FLIP",
        "use_scene_gravity": False,
        "pressure_solver": "PCG",
        "viscosity_enabled": True,
        "viscosity_solver": "PCG",
        "viscosity_from_particles": False,
        "surface_tension_enabled": True,
        "relax_enabled": True,
        "max_speed": 10.0,
        "limit_behavior": "PERCENT",
        "omnidirectional_force_strength": 1.0,
        "dissipation_enabled": True,
        "attraction_enabled": True,
        "surface_enabled": True,
        "surface_resolution_mode": "CUSTOM",
        "surface_filter_iterations": 1,
        "surface_dilate_iterations": 1,
        "surface_erode_iterations": 1,
        "surface_smooth_iterations": 1,
        "whitewater_enabled": True,
        "liquid_view_mode": "SURFACE_PARTICLES",
        "show_whitewater": True,
        "whitewater_output_mode": "BOTH",
        "cache_use_custom_range": True,
        "profiling_enabled": True,
    }
    ds = _ensure_domain()
    for name, value in values.items():
        setattr(ds if name in ds.bl_rna.properties else fs, name, value)
    cache_dir = work_dir / "ui-cache"
    frames_dir = cache_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "frame_000001.npz.validation.tmp").write_bytes(b"uncommitted")
    fs.cache_path = str(cache_dir)


def _assert_all_authored_draws(module_name: str, work_dir: Path) -> dict[str, int]:
    _configure_all_authored_scene_states(work_dir)
    result: dict[str, int] = {}
    with _effect_authoring_enabled(module_name):
        for panel_id in ALL_AUTHORED_SCENE_PANELS:
            events = _draw(panel_id, bpy.context)
            result[panel_id] = len(events)
        # Every domain panel is drawn through its Physics mirror.
        for group, source_ids in PHYSICS_MIRROR_SOURCES.items():
            for panel_id, events in _draw_physics_mirrors(
                module_name, group, source_ids
            ).items():
                result[panel_id] = len(events)
    return result


def _assert_launcher_strip(module_name: str) -> dict[str, object]:
    """The sidebar top is a pinned status/transport strip and nothing else.

    No settings are drawn there (every knob is a stage control), the Live Tuning
    shelf module is gone, and the status note never says "not owned".
    """
    if importlib.util.find_spec(f"{module_name}.blender.ui_view3d") is not None:
        raise AssertionError("the Live Tuning module blender.ui_view3d came back")
    events = _draw("FLUIDSIM_PT_viewport_status", bpy.context)
    props = [e["property"] for e in events if e["kind"] == "prop"]
    if props:
        raise AssertionError(f"launcher strip must not bind properties: {props}")
    operators = {e["operator"] for e in events if e["kind"] == "operator"}
    if "fluidsim.open_scene_properties" in operators:
        raise AssertionError("launcher strip still offers the Scene Properties jump")
    if not operators & {
        "screen.animation_play", "fluidsim.preview_pause",
        "fluidsim.cache_preview_start", "fluidsim.bake_cancel",
    }:
        raise AssertionError(f"launcher strip has no transport action: {operators}")
    jargon = [e["text"] for e in events if e["kind"] == "label" and "not owned" in e["text"]]
    if jargon:
        raise AssertionError(f"launcher status note regressed to jargon: {jargon}")
    return {"operators": sorted(operators), "events": len(events)}


def _assert_header_popover(module_name: str) -> dict[str, object]:
    """The header popover draws the sidebar's own controls; its workspace button works.

    The popover shares ``draw_transport`` / ``draw_quality`` /
    ``draw_simulate_controls`` with the stages, so every live-tunable control must
    appear with a domain, only guided setup without one, and the Open FluidSim
    Workspace operator must build a viewport + sidebar + timeline layout once and
    only switch to it afterwards.
    """
    ui_stages = importlib.import_module(f"{module_name}.blender.ui_stages")
    ui_header = importlib.import_module(f"{module_name}.blender.ui_header")
    metadata = _panel_metadata("FLUIDSIM_PT_header_popover")
    if (metadata["space"], metadata["region"]) != ("VIEW_3D", "HEADER"):
        raise AssertionError(f"header popover metadata is incorrect: {metadata}")
    if not hasattr(bpy.types, "FLUIDSIM_OT_open_workspace"):
        raise AssertionError("FLUIDSIM_OT_open_workspace is not registered")

    fs = bpy.context.scene.fluidsim
    ds = _ensure_domain()
    saved = {
        name: getattr(ds, name)
        for name in ("sample_type", "viscosity_enabled", "viscosity_from_particles",
                     "surface_tension_enabled", "use_scene_gravity")
    }

    def props(events):
        return [e["property"] for e in events if e["kind"] == "prop"]

    def operators(events):
        return {e["operator"] for e in events if e["kind"] == "operator"}

    try:
        # Every conditional control open, so the whole live-tunable set is drawn.
        ds.sample_type = "FLIP"
        ds.viscosity_enabled = True
        ds.viscosity_from_particles = False
        ds.surface_tension_enabled = True
        ds.use_scene_gravity = False
        domain_events = _draw("FLUIDSIM_PT_header_popover", bpy.context)
        domain_props = props(domain_events)
        missing = [
            name for name in ("quality", *ui_stages.SIMULATE_PROPS)
            if name not in domain_props
        ]
        if missing:
            raise AssertionError(f"header popover omitted {missing}: {domain_props}")
        domain_ops = operators(domain_events)
        if not domain_ops & {
            "screen.animation_play", "fluidsim.preview_pause",
            "fluidsim.cache_preview_start", "fluidsim.bake_cancel",
        }:
            raise AssertionError(f"header popover has no transport action: {domain_ops}")
        if "fluidsim.open_workspace" not in domain_ops:
            raise AssertionError(f"header popover has no workspace button: {domain_ops}")
    finally:
        for name, value in saved.items():
            setattr(ds, name, value)

    domain_object = fs.domain_object
    fs.domain_object = None
    try:
        empty_events = _draw("FLUIDSIM_PT_header_popover", bpy.context)
    finally:
        fs.domain_object = domain_object
    if "fluidsim.setup_scene" not in operators(empty_events) or props(empty_events):
        raise AssertionError(
            "header popover without a domain must offer guided setup only: "
            f"{props(empty_events)} {operators(empty_events)}")

    window = bpy.context.window
    if window is None:
        raise AssertionError("no window available to validate the workspace button")
    name = ui_header.WORKSPACE_NAME
    if bpy.data.workspaces.get(name) is not None:
        raise AssertionError(f"validation expected no preexisting {name!r} workspace")
    result = bpy.ops.fluidsim.open_workspace()
    if "FINISHED" not in result:
        raise AssertionError(f"Open FluidSim Workspace failed: {result}")
    workspace = bpy.data.workspaces.get(name)
    if workspace is None:
        raise AssertionError("Open FluidSim Workspace created no workspace")
    try:
        if workspace.object_mode != "OBJECT":
            raise AssertionError(f"workspace mode is {workspace.object_mode}, not OBJECT")
        areas = [area for screen in workspace.screens for area in screen.areas]
        viewports = [area for area in areas if area.type == "VIEW_3D"]
        sidebars = [
            space.show_region_ui
            for area in viewports for space in area.spaces if space.type == "VIEW_3D"
        ]
        timelines = [
            area for area in areas
            if area.type == "DOPESHEET_EDITOR" and area.ui_type == "TIMELINE"
        ]
        if not viewports or not sidebars or not all(sidebars) or not timelines:
            raise AssertionError(
                "FluidSim workspace needs a viewport with its sidebar open and a "
                f"timeline: {[(a.type, a.ui_type) for a in areas]} {sidebars}")
        count = len(bpy.data.workspaces)
        again = bpy.ops.fluidsim.open_workspace()
        if "FINISHED" not in again or len(bpy.data.workspaces) != count:
            raise AssertionError(
                f"second Open FluidSim Workspace must only switch: {again} "
                f"{count} -> {len(bpy.data.workspaces)}")
        summary = {
            "workspace": name,
            "areas": sorted((area.type, area.ui_type) for area in areas),
            "popover_props": domain_props,
            "empty_operators": sorted(operators(empty_events)),
        }
    finally:
        # Leave the validation scene as found (the window switch is deferred, so
        # the workspace is never the active one here). bpy.data.workspaces has no
        # remove(); batch_remove is the only data-API route.
        if window.workspace.name != name:
            bpy.data.batch_remove({workspace})
    return summary


def _assert_workflow_stages(module_name: str) -> dict[str, object]:
    """The staged sidebar draws its pinned controls with and without a domain."""
    ui_stages = importlib.import_module(f"{module_name}.blender.ui_stages")
    fs = bpy.context.scene.fluidsim
    ds = _ensure_domain()
    fs.ui_mode = "ADVANCED"
    # Every conditional control open, so the whole pinned set must be drawn.
    ds.sample_type = "FLIP"
    ds.viscosity_enabled = True
    ds.viscosity_from_particles = False
    ds.surface_tension_enabled = True
    ds.use_scene_gravity = False

    def props(events):
        return [e["property"] for e in events if e["kind"] == "prop"]

    def operators(events):
        return {e["operator"] for e in events if e["kind"] == "operator"}

    setup = _draw("FLUIDSIM_STAGE_PT_setup", bpy.context)
    setup_props = props(setup)
    for name in ("ui_mode", "quality", "product_preset", "domain_object",
                 *ui_stages.SETUP_DOMAIN_PROPS):
        if name not in setup_props:
            raise AssertionError(f"Setup stage omitted {name}: {setup_props}")
    if "fluidsim.apply_product_preset" not in operators(setup):
        raise AssertionError("Setup stage is missing the preset Apply button")

    more = props(_draw("FLUIDSIM_STAGE_PT_setup_advanced", bpy.context))
    if "max_particles" not in more:
        raise AssertionError(f"Setup advanced shelf omitted max_particles: {more}")

    simulate = _draw("FLUIDSIM_STAGE_PT_simulate", bpy.context)
    simulate_props = props(simulate)
    missing = [name for name in ui_stages.SIMULATE_PROPS if name not in simulate_props]
    if missing:
        raise AssertionError(f"Simulate stage omitted {missing}: {simulate_props}")
    if "fluidsim.open_scene_properties" not in operators(simulate):
        raise AssertionError("Simulate stage is missing the Scene Properties jump-back")

    # Look and Bake: open every gate so the whole pinned set must be drawn, then
    # put the user-visible choices back (other checks read the view mode).
    saved = {
        "fs": {name: getattr(fs, name) for name in (
            "liquid_view_mode", "show_whitewater", "cache_use_custom_range",
            "liquid_preview_enabled")},
        "ds": {name: getattr(ds, name) for name in (
            "surface_enabled", "whitewater_enabled")},
    }
    try:
        fs.liquid_view_mode = "SURFACE_PARTICLES"
        fs.show_whitewater = True
        fs.cache_use_custom_range = True
        fs.liquid_preview_enabled = True
        ds.surface_enabled = True
        ds.whitewater_enabled = True

        look_props = props(_draw("FLUIDSIM_STAGE_PT_look", bpy.context))
        missing = [name for name in ui_stages.LOOK_PROPS if name not in look_props]
        if missing:
            raise AssertionError(f"Look stage omitted {missing}: {look_props}")
        materials = props(_draw("FLUIDSIM_STAGE_PT_look_materials", bpy.context))
        missing = [n for n in ui_stages.LOOK_MATERIAL_PROPS if n not in materials]
        if missing:
            raise AssertionError(f"Look materials omitted {missing}: {materials}")

        bake = _draw("FLUIDSIM_STAGE_PT_bake", bpy.context)
        bake_props = props(bake)
        missing = [
            name
            for name in (*ui_stages.BAKE_PROPS, *ui_stages.BAKE_DELEGATED_PROPS)
            if name not in bake_props
        ]
        if missing:
            raise AssertionError(f"Bake stage omitted {missing}: {bake_props}")
        bake_ops = operators(bake)
        for operator in ("fluidsim.cache_preview_start", "fluidsim.export_frame"):
            if operator not in bake_ops:
                raise AssertionError(f"Bake stage is missing {operator}: {bake_ops}")
        if not bake_ops & {"fluidsim.bake_start", "fluidsim.bake_resume",
                           "fluidsim.bake_cancel"}:
            raise AssertionError(f"Bake stage offers no bake action: {bake_ops}")
        storage = props(_draw("FLUIDSIM_STAGE_PT_bake_storage", bpy.context))
        missing = [n for n in ui_stages.BAKE_STORAGE_PROPS if n not in storage]
        if missing:
            raise AssertionError(f"Bake storage shelf omitted {missing}: {storage}")
        # Manage Cache only exists once cache_path holds a FluidSim cache; the
        # authored-draws pass above wrote one, so the shelf must offer its actions.
        manage_ops = operators(_draw("FLUIDSIM_STAGE_PT_bake_manage", bpy.context))
        if not {"fluidsim.cache_integrity_scan", "fluidsim.cache_free"} <= manage_ops:
            raise AssertionError(f"Manage Cache shelf is incomplete: {manage_ops}")
    finally:
        for name, value in saved["fs"].items():
            setattr(fs, name, value)
        for name, value in saved["ds"].items():
            setattr(ds, name, value)

    # Basic mode hides the advanced shelf but keeps the scene-level cache shelves
    # reachable; no domain offers guided setup first.
    fs.ui_mode = "BASIC"
    if _panel("FLUIDSIM_STAGE_PT_setup_advanced").poll(bpy.context):
        raise AssertionError("FLUIDSIM_STAGE_PT_setup_advanced is visible in Basic mode")
    for shelf in ("FLUIDSIM_STAGE_PT_bake_storage", "FLUIDSIM_STAGE_PT_bake_manage"):
        if not _panel(shelf).poll(bpy.context):
            raise AssertionError(f"{shelf} is hidden in Basic mode")
    fs.ui_mode = "ADVANCED"
    domain_object = fs.domain_object
    fs.domain_object = None
    try:
        empty_setup = _draw("FLUIDSIM_STAGE_PT_setup", bpy.context)
        _draw("FLUIDSIM_STAGE_PT_simulate", bpy.context)
        empty_staged = {
            panel_id: _draw(panel_id, bpy.context)
            for panel_id in ("FLUIDSIM_STAGE_PT_look", "FLUIDSIM_STAGE_PT_bake")
        }
        empty_polls = {
            panel_id: bool(_panel(panel_id).poll(bpy.context))
            for panel_id in ("FLUIDSIM_STAGE_PT_look_materials",)
        }
        # An existing cache stays storable and manageable without a domain.
        for shelf in ("FLUIDSIM_STAGE_PT_bake_storage", "FLUIDSIM_STAGE_PT_bake_manage"):
            if not _panel(shelf).poll(bpy.context):
                raise AssertionError(f"{shelf} is hidden without a domain")
    finally:
        fs.domain_object = domain_object
    if "fluidsim.setup_scene" not in operators(empty_setup):
        raise AssertionError("Setup stage offers no guided setup without a domain")
    look_events = empty_staged["FLUIDSIM_STAGE_PT_look"]
    if props(look_events) or operators(look_events):
        raise AssertionError("Look stage drew controls without a domain")
    # Bake offers only the scene-level cache location (nothing can be baked yet).
    bake_events = empty_staged["FLUIDSIM_STAGE_PT_bake"]
    if list(props(bake_events)) != ["cache_path"] or operators(bake_events):
        raise AssertionError(
            "Bake stage must offer only cache_path without a domain: "
            f"{props(bake_events)} {operators(bake_events)}")
    if any(empty_polls.values()):
        raise AssertionError(f"domain-only shelves are visible without one: {empty_polls}")
    return {
        "setup": setup_props,
        "simulate": simulate_props,
        "look": look_props,
        "bake": bake_props,
        "empty_setup_operators": sorted(operators(empty_setup)),
    }


def _activate_object(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _assert_physics_context(module_name: str) -> dict[str, object]:
    ui_physics = importlib.import_module(f"{module_name}.blender.ui_physics")
    output_publication = importlib.import_module(
        f"{module_name}.blender.output_publication")
    scene = bpy.context.scene
    fs = scene.fluidsim
    cube = _ensure_cube()
    cube.fluidsim.role = "NONE"
    fs.domain_object = None
    _activate_object(cube)

    reporter = _Reporter()
    assign = bpy.types.FLUIDSIM_OT_use_active_as_domain
    result = assign.execute(reporter, bpy.context)
    if "FINISHED" not in result or fs.domain_object is not cube:
        raise AssertionError(f"contextual domain assignment failed: {result}")
    if ui_physics.physics_object_kind(bpy.context) != "DOMAIN":
        raise AssertionError("assigned domain is not recognized by exact scene pointer")
    domain_draws = {
        panel_id: len(_draw(panel_id, bpy.context))
        for panel_id in (
            "FLUIDSIM_PT_physics_root",
            "FLUIDSIM_PT_physics_domain_scene_domain",
            "FLUIDSIM_PT_physics_domain_scene_simulation",
        )
    }

    clear = bpy.types.FLUIDSIM_OT_clear_domain_assignment
    result = clear.execute(reporter, bpy.context)
    if "FINISHED" not in result or fs.domain_object is not None:
        raise AssertionError(f"contextual domain removal failed: {result}")

    cube.fluidsim.role = "SOURCE"
    if ui_physics.physics_object_kind(bpy.context) != "INPUT":
        raise AssertionError("canonical input role is not recognized in Physics Properties")
    input_draws = {
        panel_id: len(_draw(panel_id, bpy.context))
        for panel_id in (
            "FLUIDSIM_PT_physics_root",
            "FLUIDSIM_PT_physics_input_object_role",
            "FLUIDSIM_PT_physics_input_object_source",
        )
    }

    shells = output_publication.ensure_output_shells(scene)
    output_panel_ids = {
        output_publication.SURFACE_NAME: "FLUIDSIM_PT_physics_surface_output",
        output_publication.PARTICLES_NAME: "FLUIDSIM_PT_physics_particles_output",
        output_publication.WHITEWATER_NAME: "FLUIDSIM_PT_physics_whitewater_output",
        output_publication.FOAM_NAME: "FLUIDSIM_PT_physics_foam_output",
        output_publication.BUBBLES_NAME: "FLUIDSIM_PT_physics_bubbles_output",
        output_publication.SPRAY_NAME: "FLUIDSIM_PT_physics_spray_output",
    }
    output_draws: dict[str, int] = {}
    for obj in shells:
        _activate_object(obj)
        kind = str(obj.get(output_publication.OUTPUT_TAG, ""))
        if ui_physics.physics_object_kind(bpy.context) == "NONE":
            raise AssertionError(f"tagged output is not recognized: {kind}")
        _draw("FLUIDSIM_PT_physics_root", bpy.context)
        output_draws[kind] = len(_draw(output_panel_ids[kind], bpy.context))

    spray = output_publication.output_object(scene, output_publication.SPRAY_NAME)
    _activate_object(spray)
    removal = _Reporter()
    removal.expected_object = "identity-drift"
    removal.expected_kind = output_publication.SPRAY_NAME
    result = bpy.types.FLUIDSIM_OT_remove_selected_output.execute(
        removal, bpy.context)
    if "CANCELLED" not in result or output_publication.output_object(
        scene, output_publication.SPRAY_NAME
    ) is not spray:
        raise AssertionError("output removal did not fail closed on identity drift")
    removal.expected_object = spray.name
    result = bpy.types.FLUIDSIM_OT_remove_selected_output.execute(
        removal, bpy.context)
    if "FINISHED" not in result or output_publication.output_object(
        scene, output_publication.SPRAY_NAME
    ) is not None:
        raise AssertionError("confirmed owned output removal failed")
    output_publication.ensure_output_shells(scene)
    _activate_object(cube)
    cube.fluidsim.role = "SOURCE"
    return {
        "domain_draws": domain_draws,
        "input_draws": input_draws,
        "output_draws": output_draws,
        "operators": "passed",
    }


def _assert_object_role_matrix() -> dict[str, list[str]]:
    cube = _ensure_cube()
    settings = cube.fluidsim
    settings.enabled = True
    settings.use_lifespan = True
    settings.source_sampling_mode = "MULTIPLIER"
    settings.point_source_representation = "RASTERIZED_REGION"
    settings.blend_mode = "ADD"
    settings.force_type = "DIRECTIONAL"
    settings.field_operation = "VECTOR"
    settings.volume_grid_type = "FOG"
    result: dict[str, list[str]] = {}
    for case_id, role, property_name, value in OBJECT_ROLE_CASES:
        settings.role = role
        if property_name:
            setattr(settings, property_name, value)
        if case_id == "FIELD_VOLUME_GRID":
            settings.field_value_source = "VOLUME_GRID"
            settings.volume_vector_grid = ""
        else:
            settings.field_value_source = "CONSTANT"
        drawn: list[str] = []
        for panel_id in OBJECT_PANEL_IDS:
            panel = _panel(panel_id)
            poll = getattr(panel, "poll", None)
            if callable(poll) and not bool(poll(bpy.context)):
                continue
            _draw(panel_id, bpy.context)
            drawn.append(panel_id)
        if "FLUIDSIM_PT_object_role" not in drawn:
            raise AssertionError(f"role root omitted for {case_id}")
        result[case_id] = drawn
    return result


def _assert_field_payload_coherence() -> dict[str, str]:
    settings = _ensure_cube().fluidsim
    settings.role = "FIELD"
    settings.field_region_kind = "ANALYTIC"
    settings.field_value_source = "VOLUME_GRID"
    if settings.field_region_kind != "VOLUME_GRID":
        raise AssertionError("Volume field payload did not enable Volume Grid coverage")
    settings.field_region_kind = "ANALYTIC"
    if settings.field_value_source != "CONSTANT":
        raise AssertionError("Analytic field coverage did not restore Constant payload")
    return {
        "analytic_payload": settings.field_value_source,
        "analytic_region": settings.field_region_kind,
    }


def _assert_scene_properties_shortcut() -> dict[str, object]:
    area = SimpleNamespace(
        type="PROPERTIES",
        spaces=SimpleNamespace(active=SimpleNamespace(context="OBJECT")),
    )
    context = SimpleNamespace(screen=SimpleNamespace(areas=(area,)))
    reporter = _Reporter()
    operator = bpy.types.FLUIDSIM_OT_open_scene_properties
    result = operator.execute(reporter, context)
    if "FINISHED" not in result or area.spaces.active.context != "SCENE":
        raise AssertionError(f"Scene Properties shortcut failed: {result}, {area.spaces.active.context}")
    return {"result": sorted(result), "messages": reporter.messages}


def _assert_save_reopen(work_dir: Path) -> Path:
    path = work_dir / "properties-ui-validation.blend"
    scene = bpy.context.scene
    scene["fluidsim_properties_ui_validation"] = "passed"
    effect_count = len(scene.fluidsim_effects.items)
    role = bpy.context.active_object.fluidsim.role
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()), check_existing=False)
    if "FINISHED" not in result:
        raise AssertionError(f"save_as_mainfile failed: {result}")
    result = bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
    if "FINISHED" not in result:
        raise AssertionError(f"open_mainfile failed: {result}")
    if bpy.context.scene.get("fluidsim_properties_ui_validation") != "passed":
        raise AssertionError("scene UI marker did not survive save/reopen")
    if len(bpy.context.scene.fluidsim_effects.items) != effect_count:
        raise AssertionError("Custom Effects stack did not survive save/reopen")
    active = bpy.data.objects.get("FluidSim UI Validation Cube")
    if active is None or active.fluidsim.role != role:
        raise AssertionError("Object Properties role did not survive save/reopen")
    return path


def _assert_registration_cycle(extension) -> dict[str, object]:
    extension.unregister()
    for panel_id in (
        "FLUIDSIM_PT_scene_control",
        "FLUIDSIM_PT_object_role",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_viewport_status",
        "FLUIDSIM_PT_header_popover",
        *EXPECTED_STAGE_PANELS,
        "FLUIDSIM_PT_physics_root",
    ):
        if hasattr(bpy.types, panel_id):
            raise AssertionError(f"panel survived unregister: {panel_id}")
    if hasattr(bpy.types, "FLUIDSIM_OT_open_workspace"):
        raise AssertionError("workspace operator survived unregister")
    if hasattr(bpy.types.Scene, "fluidsim") or hasattr(bpy.types.Object, "fluidsim"):
        raise AssertionError("FluidSim RNA survived unregister")
    extension.register()
    for panel_id in (
        "FLUIDSIM_PT_scene_control",
        "FLUIDSIM_PT_object_role",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_viewport_status",
        "FLUIDSIM_PT_header_popover",
        *EXPECTED_STAGE_PANELS,
        "FLUIDSIM_PT_physics_root",
    ):
        _panel(panel_id)
    if not hasattr(bpy.types, "FLUIDSIM_OT_open_workspace"):
        raise AssertionError("workspace operator missing after re-register")
    for removed in REMOVED_PANEL_IDS:
        if hasattr(bpy.types, removed):
            raise AssertionError(f"deleted panel came back after re-register: {removed}")
    return {"unregister": "passed", "reregister": "passed"}


def _assert_warp_root(expected_root: Path) -> dict[str, object]:
    import warp as wp

    actual = Path(wp.__file__).resolve()
    expected = expected_root.resolve()
    if actual != expected and expected not in actual.parents:
        raise AssertionError(f"Warp imported from {actual}, expected beneath {expected}")
    if str(getattr(wp, "__version__", "")) != _certified("CERTIFIED_WARP_VERSION"):
        raise AssertionError(f"unexpected Warp version {getattr(wp, '__version__', None)!r}")
    wp.init()
    return {
        "version": str(wp.__version__),
        "module": str(actual),
        "device": "cpu",
        "cuda_devices": [str(device) for device in wp.get_cuda_devices()],
    }


def main() -> None:
    options = _args()
    options.work_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "failed",
        "blender_version": bpy.app.version_string,
        "module": options.module,
    }
    extension = None
    try:
        expected_blender = tuple(_certified("CERTIFIED_BLENDER_VERSION"))
        if bpy.app.version[:3] != expected_blender and tuple(bpy.app.version[:2]) != (5, 1):
            raise AssertionError(
                f"expected Blender {'.'.join(map(str, expected_blender))}, "
                f"running {bpy.app.version_string}")
        runtime = _assert_warp_root(options.expected_warp_root)
        extension = importlib.import_module(options.module)
        extension.register()
        report["panel_metadata"] = _assert_metadata()
        report["draws"] = _assert_scene_and_object_draws(options.module)
        report["all_authored_draws"] = _assert_all_authored_draws(
            options.module, options.work_dir)
        report["viewport_launcher"] = _assert_launcher_strip(options.module)
        report["workflow_stages"] = _assert_workflow_stages(options.module)
        report["header_popover"] = _assert_header_popover(options.module)
        report["physics_context"] = _assert_physics_context(options.module)
        report["role_matrix"] = _assert_object_role_matrix()
        report["field_payload_coherence"] = _assert_field_payload_coherence()
        report["shortcut"] = _assert_scene_properties_shortcut()
        report["saved_blend"] = str(_assert_save_reopen(options.work_dir))
        report["registration"] = _assert_registration_cycle(extension)
        report["runtime"] = runtime
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(options.report_json, report)
        raise
    else:
        _atomic_json(options.report_json, report)
        print("FLUIDSIM_PROPERTIES_UI_VALIDATION_OK")
    finally:
        if extension is not None:
            try:
                extension.unregister()
            except Exception:  # noqa: BLE001, S110 - preserve the primary gate failure.
                pass


if __name__ == "__main__":
    main()

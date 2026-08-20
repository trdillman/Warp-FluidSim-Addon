"""Validate FluidSim's native Properties workflow in the supplied Blender runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import bpy


EXPECTED_SCENE_ROOTS = {
    "FLUIDSIM_PT_scene_control": (0, False),
    "FLUIDSIM_PT_scene_setup": (10, False),
    "FLUIDSIM_PT_scene_domain": (20, False),
    "FLUIDSIM_PT_scene_height": (30, True),
    "FLUIDSIM_PT_scene_simulation": (40, True),
    "FLUIDSIM_PT_custom_effects": (50, True),
    "FLUIDSIM_PT_scene_surface": (60, True),
    "FLUIDSIM_PT_scene_whitewater": (70, True),
    "FLUIDSIM_PT_scene_outputs": (80, False),
    "FLUIDSIM_PT_scene_cache": (90, True),
    "FLUIDSIM_PT_scene_diagnostics": (100, True),
}
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
    "FLUIDSIM_PT_scene_control",
    "FLUIDSIM_PT_scene_live_timeline",
    "FLUIDSIM_PT_scene_setup",
    "FLUIDSIM_PT_scene_presets",
    "FLUIDSIM_PT_scene_domain",
    "FLUIDSIM_PT_scene_capacity",
    "FLUIDSIM_PT_scene_height",
    "FLUIDSIM_PT_scene_simulation",
    "FLUIDSIM_PT_sim_time",
    "FLUIDSIM_PT_sim_gravity",
    "FLUIDSIM_PT_sim_pressure",
    "FLUIDSIM_PT_sim_viscosity",
    "FLUIDSIM_PT_sim_surface_tension",
    "FLUIDSIM_PT_sim_relax",
    "FLUIDSIM_PT_sim_limits",
    "FLUIDSIM_PT_sim_forces",
    "FLUIDSIM_PT_sim_dissipation",
    "FLUIDSIM_PT_sim_attraction",
    "FLUIDSIM_PT_sim_spacetime",
    "FLUIDSIM_PT_scene_surface",
    "FLUIDSIM_PT_surface_resolution",
    "FLUIDSIM_PT_surface_reconstruct",
    "FLUIDSIM_PT_surface_filters",
    "FLUIDSIM_PT_scene_whitewater",
    "FLUIDSIM_PT_whitewater_emission",
    "FLUIDSIM_PT_whitewater_classification",
    "FLUIDSIM_PT_whitewater_life",
    "FLUIDSIM_PT_whitewater_dynamics",
    "FLUIDSIM_PT_scene_outputs",
    "FLUIDSIM_PT_output_display",
    "FLUIDSIM_PT_output_materials",
    "FLUIDSIM_PT_output_attributes",
    "FLUIDSIM_PT_attrs_surface",
    "FLUIDSIM_PT_attrs_particles",
    "FLUIDSIM_PT_attrs_whitewater",
    "FLUIDSIM_PT_scene_cache",
    "FLUIDSIM_PT_cache_playback",
    "FLUIDSIM_PT_cache_range",
    "FLUIDSIM_PT_cache_storage",
    "FLUIDSIM_PT_cache_bake",
    "FLUIDSIM_PT_cache_manage",
    "FLUIDSIM_PT_scene_diagnostics",
    "FLUIDSIM_PT_diagnostics_profiling",
    "FLUIDSIM_PT_diagnostics_benchmark",
    "FLUIDSIM_PT_diagnostics_report",
    "FLUIDSIM_PT_custom_effects",
    "FLUIDSIM_PT_effect_packages",
    "FLUIDSIM_PT_effect_warp_code",
    "FLUIDSIM_PT_viewport_status",
)

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


def _assert_scene_and_object_draws(module_name: str) -> dict[str, object]:
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.surface_enabled = True
    fs.whitewater_enabled = True
    fs.surface_tension_enabled = True
    cube = _ensure_cube()
    cube.fluidsim.role = "SOURCE"
    cube.fluidsim.enabled = True
    cube.fluidsim.source_region_kind = "ANALYTIC"
    cube.fluidsim.use_lifespan = True

    draw_ids = (
        "FLUIDSIM_PT_scene_control",
        "FLUIDSIM_PT_scene_domain",
        "FLUIDSIM_PT_sim_surface_tension",
        "FLUIDSIM_PT_whitewater_classification",
        "FLUIDSIM_PT_output_display",
        "FLUIDSIM_PT_cache_bake",
        "FLUIDSIM_PT_object_role",
        "FLUIDSIM_PT_object_region",
        "FLUIDSIM_PT_object_source",
        "FLUIDSIM_PT_object_lifespan",
        "FLUIDSIM_PT_viewport_status",
    )
    draws = {panel_id: _draw(panel_id, bpy.context) for panel_id in draw_ids}

    effects_module = importlib.import_module(f"{module_name}.blender.effects")
    effects_module.add_builtin_effect(scene, "fluidsim.velocity_damping")
    draws["FLUIDSIM_PT_custom_effects"] = _draw("FLUIDSIM_PT_custom_effects", bpy.context)
    draws["FLUIDSIM_PT_effect_packages"] = _draw("FLUIDSIM_PT_effect_packages", bpy.context)
    draws["FLUIDSIM_PT_effect_warp_code"] = _draw("FLUIDSIM_PT_effect_warp_code", bpy.context)

    scene_props = {
        event["property"]
        for event in draws["FLUIDSIM_PT_scene_domain"]
        if event["kind"] == "prop"
    }
    if not {"voxel_size", "particles_per_voxel", "max_particles"}.issubset(scene_props):
        raise AssertionError(f"Domain panel omitted primary controls: {sorted(scene_props)}")
    effect_ops = {
        event["operator"]
        for event in draws["FLUIDSIM_PT_custom_effects"]
        if event["kind"] == "operator"
    }
    if not {
        "fluidsim.effect_test_draft",
        "fluidsim.effect_apply_stack",
        "fluidsim.effect_revert_draft",
        "fluidsim.effect_preview_one_frame",
    }.issubset(effect_ops):
        raise AssertionError(f"Custom Effects workflow is incomplete: {sorted(effect_ops)}")
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
        "spacetime_enabled": True,
        "spacetime_variable_projection": True,
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
    for name, value in values.items():
        setattr(fs, name, value)
    cache_dir = work_dir / "ui-cache"
    frames_dir = cache_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "frame_000001.npz.validation.tmp").write_bytes(b"uncommitted")
    fs.cache_path = str(cache_dir)


def _assert_all_authored_draws(work_dir: Path) -> dict[str, int]:
    _configure_all_authored_scene_states(work_dir)
    result: dict[str, int] = {}
    for panel_id in ALL_AUTHORED_SCENE_PANELS:
        events = _draw(panel_id, bpy.context)
        result[panel_id] = len(events)
    return result


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
    operator = getattr(bpy.types, "FLUIDSIM_OT_open_scene_properties")
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
    ):
        if hasattr(bpy.types, panel_id):
            raise AssertionError(f"panel survived unregister: {panel_id}")
    if hasattr(bpy.types.Scene, "fluidsim") or hasattr(bpy.types.Object, "fluidsim"):
        raise AssertionError("FluidSim RNA survived unregister")
    extension.register()
    for panel_id in (
        "FLUIDSIM_PT_scene_control",
        "FLUIDSIM_PT_object_role",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_viewport_status",
    ):
        _panel(panel_id)
    return {"unregister": "passed", "reregister": "passed"}


def _assert_warp_root(expected_root: Path) -> dict[str, object]:
    import warp as wp

    actual = Path(wp.__file__).resolve()
    expected = expected_root.resolve()
    if actual != expected and expected not in actual.parents:
        raise AssertionError(f"Warp imported from {actual}, expected beneath {expected}")
    if str(getattr(wp, "__version__", "")) != "1.16.0":
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
        if bpy.app.version[:3] != (5, 1, 0):
            raise AssertionError(f"expected Blender 5.1.0, running {bpy.app.version_string}")
        runtime = _assert_warp_root(options.expected_warp_root)
        extension = importlib.import_module(options.module)
        extension.register()
        report["panel_metadata"] = _assert_metadata()
        report["draws"] = _assert_scene_and_object_draws(options.module)
        report["all_authored_draws"] = _assert_all_authored_draws(options.work_dir)
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
            except Exception:
                pass


if __name__ == "__main__":
    main()

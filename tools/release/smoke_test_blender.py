"""Installed-package lifecycle, Custom Effects, and external-Warp smoke test."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import bpy
import warp as wp

EXPECTED_WARP_VERSION = "1.16.0"
EXPECTED_BLENDER_VERSION = (5, 1, 0)
EXPECTED_FLUIDSIM_VERSION = (1, 0, 255)
EXPECTED_FLUIDSIM_VERSION_STRING = ".".join(map(str, EXPECTED_FLUIDSIM_VERSION))
EXPECTED_PRESETS = {
    "preview_water",
    "production_water",
    "splash_whitewater",
    "honey",
    "thick_liquid",
    "thin_liquid",
    "water_surface_tension",
    "stylized_high_tension",
}
SUCCESS_MARKER = "FLUIDSIM_SMOKE_OK"


@wp.kernel
def _scale_cpu_kernel(
    values: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    output[index] = values[index] * 2.0


def _require_finished(label: str, result) -> None:
    if "FINISHED" not in result:
        raise AssertionError(f"{label} failed: {result}")


def _installed_module() -> str:
    matches = [
        item.module
        for item in bpy.context.preferences.addons
        if item.module.endswith(".fluidsim_warp") or item.module == "fluidsim_warp"
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one enabled FluidSim module, found {matches}")
    return matches[0]


def _assert_external_warp(module: str) -> tuple[Path, Path]:
    extension = importlib.import_module(module)
    extension_root = Path(extension.__file__).resolve().parent
    repository_value = os.environ.get("FLUIDSIM_CI_REPOSITORY", "")
    user_extensions_value = os.environ.get("BLENDER_USER_EXTENSIONS", "")
    if not repository_value or not user_extensions_value:
        raise AssertionError("release smoke environment is not isolated")
    repository_root = Path(repository_value).resolve()
    user_extensions_root = Path(user_extensions_value).resolve()
    external_root_value = os.environ.get("FLUIDSIM_EXTERNAL_WARP_ROOT", "")
    if not external_root_value:
        raise AssertionError("external Warp validation root was not supplied")
    external_root = Path(external_root_value).resolve()
    warp_path = Path(wp.__file__).resolve()
    if wp.__version__ != EXPECTED_WARP_VERSION:
        raise AssertionError(
            f"expected Warp {EXPECTED_WARP_VERSION}, loaded {wp.__version__} from {warp_path}"
        )
    try:
        extension_root.relative_to(repository_root)
    except ValueError as exc:
        raise AssertionError(
            f"FluidSim loaded outside the isolated installed repository: {extension_root}"
        ) from exc
    try:
        warp_path.relative_to(external_root)
    except ValueError as exc:
        raise AssertionError(
            f"Warp did not load from the external validation runtime: {warp_path}"
        ) from exc
    forbidden = (
        (extension_root, "FluidSim extension"),
        (repository_root, "extension repository"),
        (user_extensions_root, "Blender extension directory"),
    )
    for forbidden_root, label in forbidden:
        try:
            warp_path.relative_to(forbidden_root)
        except ValueError:
            continue
        raise AssertionError(f"Warp was bundled inside the {label}: {warp_path}")
    return extension_root, warp_path


def _assert_packaged_entry_points(extension_root: Path) -> None:
    required = (
        extension_root / "release" / "__init__.py",
        extension_root / "release" / "run_headless_bake.py",
        extension_root / "blender" / "setup_wizard.py",
        extension_root / "blender" / "geometry_nodes.py",
        extension_root / "sim" / "attributes.py",
        extension_root / "sim" / "whitewater_types.py",
        extension_root / "sim" / "surface_tension.py",
        extension_root / "sim" / "surface_tension_math.py",
        extension_root / "sim" / "effects" / "schema.py",
        extension_root / "sim" / "effects" / "compiler.py",
        extension_root / "blender" / "effect_ui.py",
        extension_root / "blender" / "effect_workspace.py",
        extension_root / "blender" / "ui_common.py",
        extension_root / "blender" / "ui_scene.py",
        extension_root / "blender" / "ui_object.py",
        extension_root / "blender" / "ui_physics.py",
        extension_root / "blender" / "ui_view3d.py",
        extension_root / "presets" / "schema.py",
        extension_root / "resources" / "presets" / "preview_water.json",
        extension_root / "resources" / "presets" / "water_surface_tension.json",
        extension_root / "resources" / "presets" / "stylized_high_tension.json",
        extension_root / "resources" / "effects" / "vortex_force.json",
        extension_root / "resources" / "examples" / "05_custom_effects_vortex.blend",
        extension_root / "resources" / "examples" / "06_custom_effects_stack.blend",
        extension_root / "resources" / "examples" / "07_warp_code_authoring.blend",
        extension_root / "resources" / "geometry_nodes" / "fluidsim_render_nodes.blend",
    )
    missing = [str(path.relative_to(extension_root)) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"installed package is missing workflow files: {missing}")
    forbidden = (
        extension_root / "release" / "build_extension.py",
        extension_root / "release" / "smoke_test_blender.py",
        extension_root / "release" / "generate_geometry_nodes.py",
        extension_root / "release" / "validate_phase3_blender.py",
        extension_root / "release" / "validate_phase3_warp.py",
        extension_root / "release" / "validate_phase4_blender.py",
        extension_root / "release" / "validate_phase4_warp.py",
        extension_root / "release" / "validate_phase5_blender.py",
        extension_root / "release" / "validate_phase5_warp.py",
        extension_root / "release" / "validate_properties_ui_blender.py",
        extension_root / "release" / "custom_effect_demo_runtime.py",
        extension_root / "release" / "generate_custom_effect_demos.py",
        extension_root / "release" / "validate_custom_effects_blender.py",
        extension_root / "release" / "validate_custom_effects_warp.py",
        extension_root / "release" / "wheels.lock.json",
    )
    leaked = [str(path.relative_to(extension_root)) for path in forbidden if path.exists()]
    if leaked:
        raise AssertionError(f"development-only release files leaked into package: {leaked}")


def _run_cpu_warp_kernel() -> list[float]:
    wp.init()
    values = wp.array([0.0, 1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
    output = wp.zeros(4, dtype=wp.float32, device="cpu")
    wp.launch(
        _scale_cpu_kernel,
        dim=4,
        inputs=[values],
        outputs=[output],
        device="cpu",
    )
    result = [float(value) for value in output.numpy().tolist()]
    expected = [0.0, 2.0, 4.0, 6.0]
    if result != expected:
        raise AssertionError(f"Warp CPU kernel returned {result}, expected {expected}")
    return result


def _owned_handlers(module: str) -> dict[str, list[object]]:
    result: dict[str, list[object]] = {}
    for name in dir(bpy.app.handlers):
        handler_list = getattr(bpy.app.handlers, name)
        if not isinstance(handler_list, list):
            continue
        owned = [
            function
            for function in handler_list
            if getattr(function, "__module__", "") == module
            or getattr(function, "__module__", "").startswith(module + ".")
        ]
        if owned:
            result[name] = owned
    return result


def _assert_no_duplicate_handlers(module: str) -> None:
    for name, functions in _owned_handlers(module).items():
        identities = [id(function) for function in functions]
        if len(identities) != len(set(identities)):
            raise AssertionError(f"FluidSim duplicated owned handlers in {name}: {functions}")


def _assert_no_owned_handlers(module: str) -> None:
    owned = _owned_handlers(module)
    if owned:
        raise AssertionError(f"FluidSim handlers survived unregister: {owned}")


def _disable_and_verify(module: str) -> None:
    _require_finished("FluidSim disable", bpy.ops.preferences.addon_disable(module=module))
    _assert_no_owned_handlers(module)
    if hasattr(bpy.types.Scene, "fluidsim"):
        raise AssertionError("Scene.fluidsim survived unregister")
    if hasattr(bpy.types.Object, "fluidsim"):
        raise AssertionError("Object.fluidsim survived unregister")


def _enable_and_verify(module: str) -> None:
    _require_finished("FluidSim enable", bpy.ops.preferences.addon_enable(module=module))
    if not hasattr(bpy.types.Scene, "fluidsim"):
        raise AssertionError("Scene.fluidsim was not registered")
    if not hasattr(bpy.types.Object, "fluidsim"):
        raise AssertionError("Object.fluidsim was not registered")
    _assert_no_duplicate_handlers(module)


def _ensure_selected_cube():
    cube = bpy.data.objects.get("Cube")
    if cube is None or cube.type != "MESH":
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        cube = bpy.context.object
        cube.name = "Cube"
    bpy.ops.object.select_all(action="DESELECT")
    cube.hide_set(False)
    cube.hide_viewport = False
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    return cube


def _owned_collection(parent, kind: str):
    matches = [
        collection
        for collection in parent.children
        if collection.get("fluidsim_owned") and collection.get("fluidsim_kind") == kind
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {kind} collection, found {[item.name for item in matches]}")
    return matches[0]


def _domain_settings(module: str, scene):
    scene_bridge = importlib.import_module(f"{module}.blender.scene_bridge")
    domain_fs = scene_bridge.active_domain_settings(scene)
    if domain_fs is None:
        raise AssertionError("installed smoke scene has no designated domain settings")
    return domain_fs


def _assert_phase2_scene_state(module: str) -> dict[str, object]:
    scene = bpy.context.scene
    fs = scene.fluidsim
    domain_fs = _domain_settings(module, scene)
    domain = fs.domain_object
    if domain is None or not domain.get("fluidsim_owned"):
        raise AssertionError("guided setup did not assign an owned domain")
    cube = bpy.data.objects.get("Cube")
    if cube is None or cube.fluidsim.role != "SOURCE":
        raise AssertionError("selected cube did not persist as a liquid source")
    if cube.fluidsim.emission_mode != "INITIAL_FILL":
        raise AssertionError("Quick Liquid did not use Initial Fill")
    if fs.active_preset_id != "preview_water":
        raise AssertionError(f"unexpected active preset {fs.active_preset_id!r}")
    if fs.cache_format != "V2" or fs.checkpoint_interval < 1:
        raise AssertionError("production-bake RNA was not registered")

    root = _owned_collection(scene.collection, "ROOT_COLLECTION")
    inputs = _owned_collection(root, "INPUT_COLLECTION")
    outputs = _owned_collection(root, "OUTPUT_COLLECTION")
    helpers = _owned_collection(root, "HELPER_COLLECTION")
    if cube.name not in inputs.objects:
        raise AssertionError("source is not linked to the owned Inputs collection")
    if domain.name not in helpers.objects:
        raise AssertionError("domain is not linked to the owned Helpers collection")

    publication = importlib.import_module(f"{module}.blender.output_publication")
    output_names = (
        publication.SURFACE_NAME,
        publication.PARTICLES_NAME,
        *publication.WHITEWATER_OUTPUT_NAMES,
    )
    output_objects = [publication.output_object(scene, name) for name in output_names]
    if any(obj is None for obj in output_objects):
        raise AssertionError("guided setup did not create all six output objects")
    if any(obj.name not in outputs.objects for obj in output_objects):
        raise AssertionError("output objects are not linked to the owned Outputs collection")
    if any(len(getattr(obj.data, "materials", ())) != 1 for obj in output_objects):
        raise AssertionError("guided output objects do not each have one default material")

    geometry_nodes = importlib.import_module(f"{module}.blender.geometry_nodes")
    node_state = geometry_nodes.installed_state(scene)
    if set(node_state) != set(geometry_nodes.GROUP_BY_OUTPUT) or any(
        revision != geometry_nodes.ASSET_REVISION
        for revision in node_state.values()
    ):
        raise AssertionError(f"guided setup did not attach current render nodes: {node_state}")
    if any(len(obj.modifiers) != 1 for obj in output_objects):
        raise AssertionError("guided setup did not attach exactly one render modifier per output")
    if (
        not domain_fs.surface_attribute_velocity
        or not domain_fs.surface_attribute_motion
    ):
        raise AssertionError("default surface attributes were not registered")
    if (
        not domain_fs.particles_attribute_age
        or not domain_fs.particles_attribute_lifetime
    ):
        raise AssertionError("default particle attributes were not registered")
    if not domain_fs.whitewater_attribute_type:
        raise AssertionError("mandatory whitewater type attribute was not registered")
    if fs.whitewater_output_mode != "COMBINED":
        raise AssertionError("guided setup did not preserve combined-preview compatibility")

    return {
        "domain": domain.name,
        "source": cube.name,
        "collections": [root.name, inputs.name, outputs.name, helpers.name],
        "outputs": [obj.name for obj in output_objects],
        "materials": [obj.data.materials[0].name for obj in output_objects],
        "active_preset": fs.active_preset_id,
        "cache_format": fs.cache_format,
        "render_nodes": node_state,
        "attribute_defaults": {
            "surface": ["velocity", "motion"],
            "particles": ["age", "lifetime"],
            "whitewater": ["age", "lifetime", "type"],
        },
    }


def _assert_quick_setup_idempotent(module: str) -> dict[str, object]:
    cube = _ensure_selected_cube()
    _require_finished(
        "fluidsim.setup_scene",
        bpy.ops.fluidsim.setup_scene(
            mode="QUICK_LIQUID", preset_id="BUILTIN:preview_water"
        ),
    )
    before_objects = {obj.name for obj in bpy.context.scene.objects}
    before_collections = {collection.name for collection in bpy.data.collections}
    state = _assert_phase2_scene_state(module)

    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    _require_finished(
        "second fluidsim.setup_scene",
        bpy.ops.fluidsim.setup_scene(
            mode="QUICK_LIQUID", preset_id="BUILTIN:preview_water"
        ),
    )
    if before_objects != {obj.name for obj in bpy.context.scene.objects}:
        raise AssertionError("guided setup duplicated objects")
    if before_collections != {collection.name for collection in bpy.data.collections}:
        raise AssertionError("guided setup duplicated collections")
    _assert_phase2_scene_state(module)
    return state


def _assert_preset_library(module: str) -> dict[str, object]:
    library = importlib.import_module(f"{module}.presets.library")
    records = library.built_in_presets()
    ids = {record.preset.id for record in records}
    if ids != EXPECTED_PRESETS:
        raise AssertionError(f"unexpected built-in presets: {sorted(ids)}")
    identifier = f"smoke_user_{os.getpid()}"
    preset = library.capture_scene_preset(
        bpy.context.scene,
        identifier=identifier,
        name="Smoke User Preset",
        description="Installed-package atomic preset round-trip",
        expected_scale="1 Blender unit = 1 metre",
        performance_cost="Test only",
    )
    path = library.save_user_preset(preset)
    try:
        loaded = library.load_preset(f"USER:{identifier}")
        if loaded != preset:
            raise AssertionError("user preset did not round-trip through installed storage")
        if Path(path).read_bytes().endswith(b"\n") is False:
            raise AssertionError("user preset JSON was not normalized")
    finally:
        Path(path).unlink(missing_ok=True)
    return {"builtin_count": len(records), "builtin_ids": sorted(ids)}


def _assert_phase5_surface_tension(module: str) -> dict[str, object]:
    library = importlib.import_module(f"{module}.presets.library")
    solver_settings = importlib.import_module(f"{module}.blender.solver_settings")
    tension = importlib.import_module(f"{module}.sim.surface_tension")
    scene = bpy.context.scene
    physical = library.load_preset("BUILTIN:water_surface_tension")
    result = library.apply_preset(scene, physical)
    domain_fs = _domain_settings(module, scene)
    if result.preset_id != "water_surface_tension" or not result.requires_reset:
        raise AssertionError(f"installed Phase-5 preset result is invalid: {result}")
    if (
        not domain_fs.surface_tension_enabled
        or abs(domain_fs.surface_tension_coefficient - 0.072) > 1.0e-8
    ):
        raise AssertionError("installed Phase-5 physical preset did not apply")
    settings = solver_settings.build_settings(scene)
    if not settings.surface_tension_enabled:
        raise AssertionError("installed scene-to-solver projection lost surface tension")
    stable_dt = tension.capillary_dt(
        settings.surface_tension_density,
        settings.voxel_size,
        settings.surface_tension_coefficient,
        settings.surface_tension_safety,
    )
    frame_dt = settings.time_scale / settings.frame_rate
    required = tension.required_capillary_substeps(frame_dt, stable_dt)
    if stable_dt <= 0.0 or required < 1:
        raise AssertionError("installed capillary stability helper returned invalid data")
    panel_registered = hasattr(bpy.types, "FLUIDSIM_PT_sim_surface_tension")
    if not panel_registered:
        raise AssertionError("installed Phase-5 surface-tension panel is not registered")
    # Restore the default smoke preset so established Phase-2 persistence
    # assertions continue to verify the same authored scene.
    library.apply_preset(scene, library.load_preset("BUILTIN:preview_water"))
    if domain_fs.surface_tension_enabled:
        raise AssertionError(
            "legacy preview preset retained surface tension from the Phase-5 preset")
    return {
        "coefficient": 0.072,
        "capillary_dt": stable_dt,
        "required_substeps": required,
        "panel_registered": panel_registered,
    }



def _assert_properties_ui() -> dict[str, object]:
    expected = {
        "FLUIDSIM_PT_scene_control": ("PROPERTIES", "scene"),
        "FLUIDSIM_PT_scene_domain": ("PROPERTIES", "scene"),
        "FLUIDSIM_PT_sim_surface_tension": ("PROPERTIES", "scene"),
        "FLUIDSIM_PT_object_role": ("PROPERTIES", "object"),
        "FLUIDSIM_PT_custom_effects": ("PROPERTIES", "scene"),
        "FLUIDSIM_PT_effect_packages": ("PROPERTIES", "scene"),
        "FLUIDSIM_PT_effect_warp_code": ("PROPERTIES", "scene"),
        "FLUIDSIM_PT_viewport_status": ("VIEW_3D", ""),
        "FLUIDSIM_VIEW3D_PT_live_tuning": ("VIEW_3D", ""),
        "FLUIDSIM_PT_physics_root": ("PROPERTIES", "physics"),
        "FLUIDSIM_PT_physics_surface_output": ("PROPERTIES", "physics"),
    }
    result = {}
    for panel_id, (space, context) in expected.items():
        panel = getattr(bpy.types, panel_id, None)
        if panel is None:
            raise AssertionError(f"installed Properties workflow panel is missing: {panel_id}")
        actual_space = str(getattr(panel, "bl_space_type", ""))
        actual_context = str(getattr(panel, "bl_context", ""))
        if actual_space != space or (context and actual_context != context):
            raise AssertionError(
                f"installed panel metadata mismatch for {panel_id}: "
                f"{actual_space}/{actual_context}"
            )
        result[panel_id] = {"space": actual_space, "context": actual_context}
    for legacy in (
        "FLUIDSIM_PT_simulation",
        "FLUIDSIM_PT_quality",
        "FLUIDSIM_PT_inputs",
        "FLUIDSIM_PT_physics",
        "FLUIDSIM_PT_snippets",
    ):
        if hasattr(bpy.types, legacy):
            raise AssertionError(f"legacy dense sidebar panel is still registered: {legacy}")
    if not hasattr(bpy.types, "FLUIDSIM_OT_open_scene_properties"):
        raise AssertionError("installed Scene Properties shortcut operator is missing")
    for operator_id in (
        "FLUIDSIM_OT_use_active_as_domain",
        "FLUIDSIM_OT_clear_domain_assignment",
        "FLUIDSIM_OT_remove_selected_output",
    ):
        if not hasattr(bpy.types, operator_id):
            raise AssertionError(
                f"installed contextual workflow operator is missing: {operator_id}")
    return result


def _assert_custom_effects(module: str) -> dict[str, object]:
    if not hasattr(bpy.types, "FLUIDSIM_PT_custom_effects"):
        raise AssertionError("installed Custom Effects artist panel is not registered")
    if not hasattr(bpy.types, "FLUIDSIM_PT_effect_warp_code"):
        raise AssertionError("installed Warp Code workspace panel is not registered")
    builtins = importlib.import_module(f"{module}.sim.effects.builtins")
    definitions = builtins.builtin_effect_definitions()
    if len(definitions) != 8:
        raise AssertionError(f"expected eight installed effects, found {len(definitions)}")
    _require_finished(
        "add Velocity Damping effect",
        bpy.ops.fluidsim.effect_add(effect_id="fluidsim.velocity_damping"),
    )
    _require_finished(
        "test Custom Effects draft",
        bpy.ops.fluidsim.effect_test_draft(device="cpu"),
    )
    _require_finished(
        "apply Custom Effects stack",
        bpy.ops.fluidsim.effect_apply_stack(device="cpu"),
    )
    effects = importlib.import_module(f"{module}.blender.effects")
    signature, inventory = effects.effect_identity(bpy.context.scene)
    item = bpy.context.scene.fluidsim_effects.items[0]
    if len(signature) != 64 or len(inventory) != 1 or item.status != "APPLIED":
        raise AssertionError("installed Custom Effects identity/status is invalid")

    package_module = importlib.import_module(f"{module}.sim.effects.package")
    effect_library = importlib.import_module(f"{module}.blender.effect_library")
    imported_definition = replace(
        definitions[0],
        effect_id="smoke.imported-vortex",
        name="Imported Vortex Smoke",
        built_in=False,
        package_origin="smoke://installed-package",
    )
    with tempfile.TemporaryDirectory(prefix="fluidsim_effect_import_") as temp_dir:
        incoming = Path(temp_dir) / "imported-vortex.fseffect"
        package_module.write_effect_package(incoming, imported_definition)
        _require_finished(
            "import Custom Effects package",
            bpy.ops.fluidsim.effect_import_package(
                filepath=str(incoming), conflict="replace"
            ),
        )
    imported = effect_library.installed_package_for_effect(
        imported_definition.effect_id
    )
    imported_item = bpy.context.scene.fluidsim_effects.items[
        bpy.context.scene.fluidsim_effects.active_index
    ]
    if not imported.path.is_file() or imported_item.enabled:
        raise AssertionError("installed Custom Effects package import did not persist disabled")
    return {
        "builtin_count": len(definitions),
        "builtin_ids": [definition.effect_id for definition in definitions],
        "signature": signature,
        "status": item.status,
        "device": "cpu",
        "imported_package": True,
        "imported_effect_id": imported_definition.effect_id,
    }

def _assert_save_reopen_persistence(module: str) -> Path:
    profile = Path(os.environ["BLENDER_USER_DATAFILES"]).resolve().parent
    path = profile / f"phase5-workflow-{os.getpid()}.blend"
    _require_finished(
        "save Phase 5 workflow",
        bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False),
    )
    _require_finished("reopen Phase 5 workflow", bpy.ops.wm.open_mainfile(filepath=str(path)))
    _assert_phase2_scene_state(module)
    return path


def _append_report(run: dict[str, object]) -> None:
    value = os.environ.get("FLUIDSIM_SMOKE_REPORT", "")
    if not value:
        return
    path = Path(value).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        old_runs = previous.get("runs", []) if isinstance(previous, dict) else []
        if isinstance(old_runs, list):
            runs.extend(item for item in old_runs if isinstance(item, dict))
    runs.append(run)
    payload = {
        "status": "passed" if all(item.get("status") == "passed" for item in runs) else "failed",
        "run_count": len(runs),
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    if bpy.app.version[:3] != EXPECTED_BLENDER_VERSION:
        raise AssertionError(
            f"expected Blender {'.'.join(map(str, EXPECTED_BLENDER_VERSION))}, "
            f"running {bpy.app.version_string}"
        )
    module = _installed_module()
    extension_root, warp_path = _assert_external_warp(module)
    _assert_packaged_entry_points(extension_root)
    extension_version = importlib.import_module(f"{module}.version")
    if extension_version.VERSION != EXPECTED_FLUIDSIM_VERSION:
        raise AssertionError(f"unexpected FluidSim version: {extension_version.VERSION}")

    cpu_kernel_output = _run_cpu_warp_kernel()
    _assert_no_duplicate_handlers(module)
    setup = _assert_quick_setup_idempotent(module)
    presets = _assert_preset_library(module)
    phase5 = _assert_phase5_surface_tension(module)
    properties_ui = _assert_properties_ui()
    custom_effects = _assert_custom_effects(module)
    blend_path = _assert_save_reopen_persistence(module)

    # The operator must report an actionable no-CUDA state rather than fail to execute.
    _require_finished(
        "fluidsim.refresh_runtime_probe",
        bpy.ops.fluidsim.refresh_runtime_probe(),
    )

    _disable_and_verify(module)
    _enable_and_verify(module)
    _assert_phase2_scene_state(module)
    _disable_and_verify(module)
    _enable_and_verify(module)
    _assert_phase2_scene_state(module)
    _require_finished("wm.save_userpref", bpy.ops.wm.save_userpref())

    runtime = importlib.import_module(f"{module}.runtime.warp_runtime")
    probe = runtime.probe_runtime(force=True)
    if probe.warp_version != EXPECTED_WARP_VERSION:
        raise AssertionError(f"runtime probe loaded unexpected Warp: {probe.warp_version}")
    report_module = importlib.import_module(f"{module}.diagnostics.report")
    support_report = report_module.build_report(
        package_name=module,
        blender_version=".".join(str(value) for value in bpy.app.version[:3]),
        selected_device="",
        cache_path=str(Path.home() / "FluidSimCache"),
        scene_signature="phase5-smoke-scene",
        settings_signature="phase5-smoke-settings",
    )
    if support_report["fluidsim"]["version"] != EXPECTED_FLUIDSIM_VERSION_STRING:
        raise AssertionError("support report version drift")
    if support_report["runtime"]["warp_version"] != EXPECTED_WARP_VERSION:
        raise AssertionError("support report Warp version drift")

    _append_report(
        {
            "status": "passed",
            "process_id": os.getpid(),
            "module": module,
            "blender_version": bpy.app.version_string,
            "warp_version": wp.__version__,
            "warp_module": str(warp_path),
            "extension_root": str(extension_root),
            "cpu_kernel_output": cpu_kernel_output,
            "registration_cycles": 2,
            "phase5_setup": setup,
            "phase5_surface_tension": phase5,
            "properties_ui": properties_ui,
            "custom_effects": custom_effects,
            "preset_library": presets,
            "saved_blend": str(blend_path),
            "cuda_available": probe.available,
            "cuda_message": probe.message,
            "cuda_device_count": len(probe.cuda_devices),
            "support_report": support_report,
            "elapsed_seconds": time.time() - started,
        }
    )
    print(SUCCESS_MARKER)


if __name__ == "__main__":
    main()

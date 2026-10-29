from __future__ import annotations

from pathlib import Path
import ast
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"
VALIDATOR = TOOLS_RELEASE / "validate_properties_ui_blender.py"
SMOKE = TOOLS_RELEASE / "smoke_test_blender.py"


def _metadata_validator():
    """Run the real metadata checks without requiring a Blender import."""
    tree = ast.parse(VALIDATOR.read_text(encoding="utf-8"))
    names = {
        "EXPECTED_SCENE_ROOTS", "EXPECTED_OBJECT_ROOTS", "EXPECTED_EFFECT_CHILDREN",
        "EXPECTED_STAGE_PANELS", "PHYSICS_MIRROR_SOURCES", "REMOVED_PANEL_IDS",
        "_mirror_id", "_assert_metadata",
    }
    nodes = [node for node in tree.body if (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ) or (isinstance(node, ast.FunctionDef) and node.name in names)]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(VALIDATOR), "exec"), namespace)
    records = {}

    def add(panel, context="scene", order=0, parent="", closed=False, space="PROPERTIES"):
        records[panel] = dict(space=space, region="WINDOW" if space == "PROPERTIES" else "UI",
                              context=context, order=order, parent=parent, closed=closed)

    for panel, (order, _closed) in namespace["EXPECTED_SCENE_ROOTS"].items():
        add(panel, order=order)
    for panel, order in namespace["EXPECTED_OBJECT_ROOTS"].items():
        add(panel, context="object", order=order)
    for panel in namespace["EXPECTED_EFFECT_CHILDREN"]:
        add(panel, parent="FLUIDSIM_PT_custom_effects")
    for group, sources in namespace["PHYSICS_MIRROR_SOURCES"].items():
        for source in sources:
            if source not in records:
                add(source)
            add(namespace["_mirror_id"](group, source), context="physics")
    for panel, (order, parent, closed) in namespace["EXPECTED_STAGE_PANELS"].items():
        add(panel, order=order, parent=parent, closed=closed, space="VIEW_3D")
    add("FLUIDSIM_PT_viewport_status", space="VIEW_3D")
    add("FLUIDSIM_PT_physics_root", context="physics")
    namespace.update(
        _panel_metadata=records.__getitem__,
        _panel=lambda panel: SimpleNamespace(bl_category="FluidSim"),
        bpy=SimpleNamespace(types=SimpleNamespace(FLUIDSIM_OT_open_scene_properties=object())),
    )
    return namespace["_assert_metadata"], records


def test_release_metadata_accepts_expanded_properties_and_existing_sidebar_defaults():
    validate, records = _metadata_validator()
    assert validate() == records


@pytest.mark.parametrize("panel", [
    "FLUIDSIM_PT_scene_lifecycle", "FLUIDSIM_PT_custom_effects",
    "FLUIDSIM_PT_scene_diagnostics", "FLUIDSIM_PT_effect_packages",
    "FLUIDSIM_PT_effect_warp_code", "FLUIDSIM_PT_object_role",
    "FLUIDSIM_PT_physics_domain_scene_domain", "FLUIDSIM_PT_physics_root",
])
def test_release_metadata_rejects_collapsed_properties(panel):
    validate, records = _metadata_validator()
    records[panel]["closed"] = True
    with pytest.raises(AssertionError, match=panel):
        validate()


@pytest.mark.parametrize("options, expanded", [(None, True), (set(), True), ({"DEFAULT_CLOSED"}, False)])
def test_gui_expanded_check_handles_omitted_panel_options(options, expanded):
    tree = ast.parse((ROOT.parent / "tests/acceptance/blender_gui_acceptance.py").read_text(encoding="utf-8"))
    check = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and node.args and isinstance(node.args[0], ast.Constant)
                 and node.args[0].value == "Lifecycle starts expanded")
    panel = SimpleNamespace() if options is None else SimpleNamespace(bl_options=options)
    namespace = {"bpy": SimpleNamespace(types=SimpleNamespace(FLUIDSIM_PT_scene_lifecycle=panel))}
    assert eval(compile(ast.Expression(check.args[1]), "<gui expanded check>", "eval"), namespace) is expanded


def test_properties_ui_blender_validator_covers_native_surfaces_and_lifecycle():
    source = VALIDATOR.read_text(encoding="utf-8")
    for marker in (
        "EXPECTED_SCENE_ROOTS",
        "EXPECTED_OBJECT_ROOTS",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_effect_packages",
        "FLUIDSIM_PT_effect_warp_code",
        "FLUIDSIM_PT_viewport_status",
        "FLUIDSIM_STAGE_PT_setup",
        "FLUIDSIM_STAGE_PT_simulate",
        "FLUIDSIM_STAGE_PT_look",
        "FLUIDSIM_STAGE_PT_bake",
        "FLUIDSIM_STAGE_PT_bake_storage",
        "FLUIDSIM_STAGE_PT_bake_manage",
        "FLUIDSIM_PT_header_popover",
        "FLUIDSIM_OT_open_workspace",
        "_assert_header_popover",
        "REMOVED_PANEL_IDS",
        "_assert_launcher_strip",
        "EXPECTED_STAGE_PANELS",
        "_assert_workflow_stages",
        "FLUIDSIM_PT_physics_root",
        "FLUIDSIM_OT_use_active_as_domain",
        "FLUIDSIM_OT_remove_selected_output",
        "FLUIDSIM_OT_open_scene_properties",
        "RecordingLayout",
        "save_as_mainfile",
        "open_mainfile",
        "unregister()",
        "register()",
        "_atomic_json",
    ):
        assert marker in source


def test_installed_smoke_requires_the_new_ui_modules_and_panel_inventory():
    source = SMOKE.read_text(encoding="utf-8")
    for path in (
        'extension_root / "blender" / "ui_common.py"',
        'extension_root / "blender" / "ui_scene.py"',
        'extension_root / "blender" / "ui_object.py"',
        'extension_root / "blender" / "ui_physics.py"',
        'extension_root / "blender" / "ui_stages.py"',
        'extension_root / "blender" / "ui_header.py"',
    ):
        assert path in source
    for panel_id in (
        "FLUIDSIM_PT_scene_control",
        "FLUIDSIM_PT_physics_domain_scene_domain",
        "FLUIDSIM_PT_object_role",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_effect_packages",
        "FLUIDSIM_PT_effect_warp_code",
        "FLUIDSIM_PT_viewport_status",
        "FLUIDSIM_STAGE_PT_setup",
        "FLUIDSIM_STAGE_PT_setup_advanced",
        "FLUIDSIM_STAGE_PT_simulate",
        "FLUIDSIM_STAGE_PT_look",
        "FLUIDSIM_STAGE_PT_look_materials",
        "FLUIDSIM_STAGE_PT_bake",
        "FLUIDSIM_STAGE_PT_bake_storage",
        "FLUIDSIM_STAGE_PT_bake_manage",
        "FLUIDSIM_PT_header_popover",
        "FLUIDSIM_OT_open_workspace",
        "FLUIDSIM_PT_physics_root",
        "FLUIDSIM_PT_physics_surface_output",
    ):
        assert panel_id in source
    # The deleted Live Tuning module must not be required of an installed package.
    assert "ui_view3d.py" not in source


def test_properties_validator_is_tools_owned_and_not_packaged():
    manifest = (ROOT / "blender_manifest.toml").read_text(encoding="utf-8")
    assert VALIDATOR.is_file()
    assert "validate_properties_ui_blender.py" not in manifest


def test_properties_validator_exercises_all_authored_panels_and_object_role_matrix():
    source = VALIDATOR.read_text(encoding="utf-8")
    for marker in (
        "ALL_AUTHORED_SCENE_PANELS",
        "OBJECT_ROLE_CASES",
        "SOURCE",
        "SINK",
        "COLLIDER",
        "FORCE",
        "FIELD",
        "VOLUME_GRID",
        "all_authored_draws",
        "viewport_launcher",
        "physics_context",
        "role_matrix",
    ):
        assert marker in source

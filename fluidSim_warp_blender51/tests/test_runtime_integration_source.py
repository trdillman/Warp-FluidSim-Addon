from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _class_methods(path: pathlib.Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: {
            item.name
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def test_external_warp_override_is_explicit_and_has_no_process_scan():
    source = (ROOT / "warp_discovery.py").read_text(encoding="utf-8")
    assert "FLUIDSIM_ALLOW_EXTERNAL_WARP" in source
    assert "def enable_developer_override(" in source
    assert "subprocess" not in source
    assert "site.get" not in source
    assert "PATH" not in source
    assert "ensure_warp_discoverable" not in source


def test_preferences_use_enumerated_runtime_probe():
    source = (ROOT / "preferences.py").read_text(encoding="utf-8")
    assert "EnumProperty" in source
    assert "cuda_device_items" in source
    assert "probe_runtime" in source
    assert 'bl_idname = "fluidsim.refresh_runtime_probe"' in source
    assert 'StringProperty(\n        name="Device"' not in source


def test_gpu_operators_have_runtime_poll_gates():
    methods = {
        **_class_methods(ROOT / "operator_domains" / "session.py"),
        **_class_methods(ROOT / "operator_domains" / "export.py"),
    }
    for class_name in (
        "FLUIDSIM_OT_initialize",
        "FLUIDSIM_OT_toggle_live",
        "FLUIDSIM_OT_export_cache_sequence",
    ):
        assert "poll" in methods[class_name]
    source = (ROOT / "operator_domains" / "common.py").read_text(encoding="utf-8")
    assert "def _runtime_poll(" in source
    assert "def _mutation_poll(" in source
    assert "runtime_status_for_device" in source
    assert "class _FLUIDSIM_OT_session_base" in source
    assert "if not _mutation_poll(cls, context):" in source
    assert "return _runtime_poll(cls)" in source


def test_simulation_and_cache_panels_surface_runtime_state():
    scene_ui = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    launcher = (ROOT / "panels.py").read_text(encoding="utf-8")
    assert "def _runtime_ui_state(" in scene_ui
    assert 'operator("fluidsim.refresh_runtime_probe"' in scene_ui
    assert "runtime_status_for_device" in launcher
    assert "runtime_ready" in scene_ui
    bake = scene_ui.split("class FLUIDSIM_PT_cache_bake", 1)[1].split(
        "class FLUIDSIM_PT_cache_manage", 1
    )[0]
    assert "_runtime_ui_state()" in bake
    assert "actions.enabled = runtime_ready" in bake
    assert "if active:" in bake


def test_support_report_operator_remains_available_when_runtime_is_broken():
    path = ROOT / "operator_domains" / "diagnostics.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    report_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FLUIDSIM_OT_write_report"
    )
    base_names = {base.id for base in report_class.bases if isinstance(base, ast.Name)}
    assert "_FLUIDSIM_OT_session_base" not in base_names
    source = ast.get_source_segment(text, report_class)
    assert "if sess.active" in source

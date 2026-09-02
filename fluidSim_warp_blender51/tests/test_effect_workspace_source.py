from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "blender" / "effect_workspace.py"


def test_warp_code_workspace_uses_text_editor_and_source_mapped_diagnostics():
    source = WORKSPACE.read_text(encoding="utf-8")
    assert 'bl_idname = "fluidsim.effect_open_text_editor"' in source
    assert 'bl_idname = "fluidsim.effect_go_to_error"' in source
    assert 'bl_idname = "fluidsim.effect_show_generated_source"' in source
    assert 'area.type = "TEXT_EDITOR"' in source
    assert "cursor_set" in source
    assert "error_line" in source
    assert "Generated Source" in source


def test_workspace_exposes_registry_permissions_and_profiling():
    source = WORKSPACE.read_text(encoding="utf-8")
    assert "FIELD_SPECS" in source
    assert "Safe Mode" in source
    assert "Expert Access" in source
    for label in (
        "Domain",
        "Stage",
        "Cadence",
        "Calls",
        "Elements",
        "Submission",
        "Warp",
        "Source SHA-256",
        "Last Runtime Error",
    ):
        assert label in source


def test_workspace_does_not_reintroduce_inline_line_editor():
    source = WORKSPACE.read_text(encoding="utf-8")
    assert "snippet_edit_line" not in source
    assert "snippet_line_add" not in source
    assert "snippet_line_delete" not in source


def test_text_workspace_prefers_an_existing_text_editor(monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace

    fake_bpy = ModuleType("bpy")
    fake_bpy.types = SimpleNamespace(Operator=object, Panel=object)
    fake_props = ModuleType("bpy.props")
    fake_props.IntProperty = lambda **_kwargs: None
    fake_props.BoolProperty = lambda **_kwargs: None
    fake_bpy.props = fake_props
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_props)

    effects_module = ModuleType("fluidsim_testpkg.blender.effects")
    enums_module = ModuleType("fluidsim_testpkg.sim.effects.enums")
    enums_module.TrustState = SimpleNamespace(
        BUILT_IN_TRUSTED=SimpleNamespace(value="BUILT_IN_TRUSTED"),
        LOCAL_TRUSTED=SimpleNamespace(value="LOCAL_TRUSTED"),
    )
    fields_module = ModuleType("fluidsim_testpkg.sim.effects.fields")
    fields_module.FIELD_SPECS = {}
    source_module = ModuleType("fluidsim_testpkg.sim.effects.source")
    source_module.generate_kernel_source = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, effects_module.__name__, effects_module)
    monkeypatch.setitem(sys.modules, enums_module.__name__, enums_module)
    monkeypatch.setitem(sys.modules, fields_module.__name__, fields_module)
    monkeypatch.setitem(sys.modules, source_module.__name__, source_module)

    module_name = "fluidsim_testpkg.blender.effect_workspace_contract"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "blender" / "effect_workspace.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    current = SimpleNamespace(type="PROPERTIES")
    existing_text = SimpleNamespace(type="TEXT_EDITOR")
    context = SimpleNamespace(
        area=current,
        screen=SimpleNamespace(areas=[current, existing_text]),
    )
    assert module._text_area(context) is existing_text

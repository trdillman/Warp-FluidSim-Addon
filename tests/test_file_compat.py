"""2.0 recognizes .blend files saved before FluidSim 1.1 and says so.

Before 1.1 the domain settings lived on the scene group; 1.1 moved them onto
the domain object and stamped ``schema_version = 1``. 2.0 reads only the
domain object, so a file that skipped 1.1 must be flagged, not silently opened
with default settings.
"""

from types import SimpleNamespace

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def _scene(stored):
    return SimpleNamespace(
        name="Scene",
        bl_system_properties_get=lambda: None if stored is None else {"fluidsim": stored})


def _file_compat(monkeypatch):
    _install_bpy_stub(monkeypatch)
    return _reload_module("fluidSim_warp_blender51.blender.file_compat")


def test_pre_bridge_file_is_flagged(monkeypatch):
    file_compat = _file_compat(monkeypatch)
    assert file_compat.is_pre_bridge_scene(_scene({"voxel_size": 0.037, "status": "Idle"}))
    assert file_compat.is_pre_bridge_scene(
        _scene({"voxel_size": 0.037, "schema_version": 0}))


def test_files_saved_by_1_1_or_later_are_not_flagged(monkeypatch):
    file_compat = _file_compat(monkeypatch)
    # 1.1 moved the values and stamped 1; the stale scene copy stays behind.
    assert not file_compat.is_pre_bridge_scene(
        _scene({"voxel_size": 0.037, "schema_version": 1}))
    # A file authored in 1.1 or 2.0 stores only scene-level settings.
    assert not file_compat.is_pre_bridge_scene(_scene({"status": "Idle", "live": True}))
    assert not file_compat.is_pre_bridge_scene(_scene({}))
    assert not file_compat.is_pre_bridge_scene(_scene(None))
    assert not file_compat.is_pre_bridge_scene(SimpleNamespace(name="no system props"))


def test_detection_keys_are_exactly_the_domain_only_settings(monkeypatch):
    file_compat = _file_compat(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    domain_only = file_compat._domain_only_fields()
    assert "voxel_size" in domain_only
    assert "domain_object" not in domain_only
    assert domain_only == (set(properties.FLUIDSIM_PG_domain.__annotations__)
                           - set(properties.FLUIDSIM_PG_scene.__annotations__))


def test_load_handler_registers_once_and_unregisters(monkeypatch):
    file_compat = _file_compat(monkeypatch)
    load_post = []
    monkeypatch.setattr(file_compat, "bpy", SimpleNamespace(
        app=SimpleNamespace(handlers=SimpleNamespace(load_post=load_post))))
    file_compat.register_handlers()
    file_compat.register_handlers()
    assert load_post.count(file_compat._on_load_post) == 1
    file_compat.unregister_handlers()
    assert file_compat._on_load_post not in load_post


def test_setup_and_physics_panels_draw_the_warning():
    from pathlib import Path

    blender = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51" / "blender"
    for name in ("ui_scene.py", "ui_physics.py"):
        assert "file_compat.draw_pre_bridge_warning(layout" in (
            blender / name).read_text(encoding="utf-8"), name

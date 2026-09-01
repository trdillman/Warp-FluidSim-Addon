"""Phase 3 domain-settings migration: schema field, pointer, pure functions, handler."""

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def test_settings_group_has_schema_version_and_domain_pointer(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    assert "schema_version" in properties.FLUIDSIM_PG_settings.__annotations__
    # register() must attach the per-object domain pointer.
    properties.register()
    assert hasattr(properties.bpy.types.Object, "fluidsim_domain")

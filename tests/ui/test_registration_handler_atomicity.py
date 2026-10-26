"""Real handler owners keep prior callbacks on partial enable failures."""
from types import SimpleNamespace

import pytest
from test_file_compat import _file_compat, _sources_module


@pytest.mark.parametrize("prior", [False, True])
def test_source_migration_failure_rolls_back_only_new_handler(monkeypatch, prior):
    module = _sources_module(monkeypatch)
    callbacks = [module._on_load_post] if prior else []
    monkeypatch.setattr(module, "bpy", SimpleNamespace(
        app=SimpleNamespace(handlers=SimpleNamespace(load_post=callbacks)),
        context=SimpleNamespace(scene=None)))
    error = ValueError("injected source schema migration failure")
    def fail(_scene):
        raise error
    monkeypatch.setattr(module, "migrate_source_schema", fail)
    with pytest.raises(ValueError) as raised:
        module.register_handlers()
    assert raised.value is error
    assert callbacks == ([module._on_load_post] if prior else [])


def test_compat_second_handler_failure_keeps_existing_first_handler(monkeypatch):
    module = _file_compat(monkeypatch)
    error = RuntimeError("injected second compat handler failure")
    class Refusing(list):
        def append(self, _callback):
            raise error
    load = [module._on_load_post]
    monkeypatch.setattr(module, "bpy", SimpleNamespace(
        app=SimpleNamespace(handlers=SimpleNamespace(
            load_post=load, blend_import_post=Refusing()))))
    with pytest.raises(RuntimeError) as raised:
        module.register_handlers()
    assert raised.value is error
    assert load == [module._on_load_post]


def test_compat_followup_failure_unwinds_new_handlers(monkeypatch):
    module = _file_compat(monkeypatch)
    handlers = SimpleNamespace(**{name: [] for name, _callback in module._COMPAT_HANDLERS})
    monkeypatch.setattr(module, "bpy", SimpleNamespace(
        app=SimpleNamespace(handlers=handlers)))
    error = ValueError("injected compat migration failure")
    def fail():
        raise error
    monkeypatch.setattr(module, "migrate_domain_resolution", fail)
    with pytest.raises(ValueError) as raised:
        module.register_handlers()
    assert raised.value is error
    assert all(not getattr(handlers, name) for name, _callback in module._COMPAT_HANDLERS)


@pytest.mark.parametrize("prior", [False, True])
def test_liquid_preview_failure_preserves_prior_resources(monkeypatch, prior):
    from test_liquid_preview import _install_handler_bpy, _scene
    module, bpy, draw_handlers, keymaps = _install_handler_bpy(monkeypatch, [_scene(True)])
    module._FAILURE = "prior diagnostic"
    if prior:
        module._register_keymap()
        module._add_draw_handler()
        bpy.app.handlers.load_post.append(module._on_load_post)
    prior_items = list(module._KEYMAP_ITEMS)
    prior_draws = list(draw_handlers)
    error = ValueError("injected preview sync failure after draw setup")
    def fail():
        module._add_draw_handler()
        raise error
    monkeypatch.setattr(module, "sync_registration", fail)
    with pytest.raises(ValueError) as raised:
        module.register()
    assert raised.value is error
    assert module._FAILURE == "prior diagnostic"
    assert module._KEYMAP_ITEMS == prior_items
    assert draw_handlers == prior_draws
    assert bpy.app.handlers.load_post == ([module._on_load_post] if prior else [])
    assert sum(len(keymap.keymap_items) for keymap in keymaps) == len(prior_items)
    module.unregister()



def test_overlay_registration_preserves_existing_handle_on_repeat(monkeypatch):
    import importlib
    module = importlib.import_module("fluidSim_warp_blender51.blender.viewport_overlay")
    prior = object()
    monkeypatch.setattr(module, "_OVERLAY_HANDLE", prior)
    def fail(*_args):
        raise RuntimeError("must not replace existing overlay")
    monkeypatch.setattr(module, "bpy", SimpleNamespace(types=SimpleNamespace(
        SpaceView3D=SimpleNamespace(draw_handler_add=fail, draw_handler_remove=fail))))
    module.register_overlay()
    assert module._OVERLAY_HANDLE is prior


def test_overlay_initial_failure_leaves_no_handle(monkeypatch):
    import importlib
    module = importlib.import_module("fluidSim_warp_blender51.blender.viewport_overlay")
    monkeypatch.setattr(module, "_OVERLAY_HANDLE", None)
    error = ValueError("injected overlay creation failure")
    def fail(*_args):
        raise error
    monkeypatch.setattr(module, "bpy", SimpleNamespace(types=SimpleNamespace(
        SpaceView3D=SimpleNamespace(draw_handler_add=fail))))
    with pytest.raises(ValueError) as raised:
        module.register_overlay()
    assert raised.value is error
    assert module._OVERLAY_HANDLE is None

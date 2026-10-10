"""Append routing and handler lifecycle; real RNA coverage lives in acceptance."""
from types import SimpleNamespace

from test_file_compat import _file_compat


def test_append_handler_is_registered_once_and_removed(monkeypatch):
    module = _file_compat(monkeypatch)
    handlers = SimpleNamespace(load_post=[], blend_import_post=[])
    monkeypatch.setattr(module, "bpy", SimpleNamespace(app=SimpleNamespace(handlers=handlers)))
    module.register_handlers()
    module.register_handlers()
    assert len(handlers.blend_import_post) == 1, "Append migration is not registered"
    callback = handlers.blend_import_post[0]
    module.unregister_handlers()
    assert callback not in handlers.blend_import_post
    assert not handlers.load_post


def test_append_only_routes_new_local_objects_and_scenes(monkeypatch):
    module = _file_compat(monkeypatch)
    handlers = SimpleNamespace(load_post=[], blend_import_post=[])
    monkeypatch.setattr(module, "bpy", SimpleNamespace(app=SimpleNamespace(handlers=handlers)))
    calls = []
    monkeypatch.setattr(module, "migrate_domain_resolution", lambda objects=None, scenes=None: calls.append((objects, scenes)))
    monkeypatch.setattr(module, "migrate_removed_particle_color_mode", lambda scenes=None: None)
    module.register_handlers()
    calls.clear()
    assert len(handlers.blend_import_post) == 1, "Append migration is not registered"
    scene = SimpleNamespace(view_layers=[], library=None)
    new = SimpleNamespace(fluidsim_domain=object(), library=None, users_scene=(scene,))
    reused = SimpleNamespace(fluidsim_domain=object(), library=None, users_scene=(scene,))
    linked = SimpleNamespace(fluidsim_domain=object(), library=object(), users_scene=(scene,))
    context = SimpleNamespace(options=set(), import_items=[
        SimpleNamespace(id=new, append_action="MAKE_LOCAL"),
        SimpleNamespace(id=reused, append_action="REUSE_LOCAL"),
        SimpleNamespace(id=linked, append_action="KEEP_LINKED"),
        SimpleNamespace(id=None, append_action="UNSET"),
    ])
    handlers.blend_import_post[0](context)
    assert len(calls) == 1
    assert list(calls[0][0]) == [new]
    assert list(calls[0][1]) == [scene]
    calls.clear()
    context.options = {"LINK"}
    handlers.blend_import_post[0](context)
    assert calls == []

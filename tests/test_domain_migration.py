"""Phase 3 domain-settings migration: schema field, pointer, pure functions, handler."""

from types import SimpleNamespace

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def test_settings_group_has_schema_version_and_domain_pointer(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    assert "schema_version" in properties.FLUIDSIM_PG_settings.__annotations__
    # register() must attach the per-object domain pointer.
    properties.register()
    assert hasattr(properties.bpy.types.Object, "fluidsim_domain")


def test_migrated_field_names_skips_identity_and_guard_but_keeps_materials(monkeypatch):
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")

    class _Cls:
        __annotations__ = {
            "voxel_size": None, "surface_material": None,
            "domain_object": None, "schema_version": None,
        }
    names = migration.migrated_field_names(_Cls)
    assert "voxel_size" in names and "surface_material" in names   # real settings migrate
    assert "domain_object" not in names                            # identity pointer skipped
    assert "schema_version" not in names                           # guard skipped


def test_copy_domain_settings_tolerates_a_read_only_field(monkeypatch):
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")

    class _Target:  # 'derived' rejects assignment, like a read-only RNA prop
        def __setattr__(self, name, value):
            if name == "derived":
                raise AttributeError("read-only")
            object.__setattr__(self, name, value)
    src = SimpleNamespace(voxel_size=0.05, derived=9, surface_material="M")
    tgt = _Target()
    migration.copy_domain_settings(src, tgt, ("voxel_size", "derived", "surface_material"))
    assert tgt.voxel_size == 0.05 and tgt.surface_material == "M"   # good fields still copied
    assert not hasattr(tgt, "derived")                             # bad field skipped, no raise


def test_migration_moves_settings_once_and_never_re_triggers(monkeypatch):
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")
    monkeypatch.setattr(migration, "migrated_field_names",
                        lambda _cls: ("voxel_size", "max_particles"))

    target = SimpleNamespace(fluidsim_domain=SimpleNamespace(schema_version=0))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        schema_version=0, voxel_size=0.05, max_particles=2_000_000))

    assert migration.migrate_domain_settings_schema(scene, target) is True
    assert (target.fluidsim_domain.voxel_size,
            target.fluidsim_domain.max_particles) == (0.05, 2_000_000)
    assert target.fluidsim_domain.schema_version == 1
    assert scene.fluidsim.schema_version == 1        # guard stamped LAST

    # second run on the same scene is a no-op
    assert migration.migrate_domain_settings_schema(scene, target) is False
    # guard is on schema_version, NOT on field emptiness: zeroing moved fields must not re-trigger
    scene.fluidsim.voxel_size = 0.0
    assert migration.migrate_domain_settings_schema(scene, target) is False
    # a None domain object never stamps-and-drops
    virgin = SimpleNamespace(fluidsim=SimpleNamespace(schema_version=0, voxel_size=0.1))
    assert migration.migrate_domain_settings_schema(virgin, None) is False
    assert virgin.fluidsim.schema_version == 0       # untouched → data preserved


def _install_handler_stub(monkeypatch):
    fake_bpy = _install_bpy_stub(monkeypatch)
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(load_post=[], persistent=lambda fn: fn))
    return fake_bpy


def test_load_post_migrates_every_scene_and_registration_is_idempotent(monkeypatch):
    fake_bpy = _install_handler_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")
    monkeypatch.setattr(migration, "migrated_field_names",
                        lambda _cls: ("voxel_size",))

    # Two pre-migration scenes; each resolves to its own prepared domain object.
    def _scene(vx):
        obj = SimpleNamespace(fluidsim_domain=SimpleNamespace(schema_version=0))
        scene = SimpleNamespace(
            fluidsim=SimpleNamespace(schema_version=0, voxel_size=vx),
            _domain=obj)
        return scene
    scenes = [_scene(0.05), _scene(0.2)]
    fake_bpy.data = SimpleNamespace(scenes=scenes)
    monkeypatch.setattr(migration, "_resolve_or_create_domain",
                        lambda scene: scene._domain)

    # register_handlers() is idempotent: two calls leave exactly one handler.
    migration.register_handlers()
    migration.register_handlers()
    load_post = fake_bpy.app.handlers.load_post
    assert load_post.count(migration._on_load_post) == 1

    # Firing the handler migrates every scene onto its domain object.
    migration._on_load_post(None)
    for scene in scenes:
        assert scene._domain.fluidsim_domain.voxel_size == scene.fluidsim.voxel_size
        assert scene._domain.fluidsim_domain.schema_version == 1
        assert scene.fluidsim.schema_version == 1

    # unregister_handlers() removes it.
    migration.unregister_handlers()
    assert migration._on_load_post not in load_post

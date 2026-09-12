"""Domain-settings migration: the split groups, pure functions, and the load handler."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def test_split_groups_overlap_only_on_schema_and_register_domain_pointer(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    scene_fields = set(properties.FLUIDSIM_PG_scene.__annotations__)
    domain_fields = set(properties.FLUIDSIM_PG_domain.__annotations__)
    assert scene_fields & domain_fields == {"schema_version"}
    assert {"schema_version", "domain_object"} <= scene_fields
    assert "domain_object" not in domain_fields
    # The legacy union is exactly the pre-split field set, used only by migration.
    legacy_fields = set(properties.FLUIDSIM_PG_legacy_settings.__annotations__)
    assert legacy_fields == scene_fields | domain_fields
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


def test_migration_moves_the_domain_group_fields(monkeypatch):
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")
    properties = _reload_module("fluidSim_warp_blender51.properties")

    names = set(migration.migrated_field_names(migration._domain_settings_cls()))
    assert names == set(properties.FLUIDSIM_PG_domain.__annotations__) - {"schema_version"}
    # Scene-owned settings stay where they are.
    assert "quality" not in names and "ui_mode" not in names


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


def test_default_domain_geometry_spans_the_fallback_bounds(monkeypatch):
    # Regression: an auto-created domain must reproduce the bounds an unset
    # domain_object used to yield, or domain_bounds collapses to a degenerate
    # AABB and solver initialization fails after a reopen.
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")
    solver_settings = _reload_module("fluidSim_warp_blender51.blender.solver_settings")

    verts, faces = migration._default_domain_geometry()
    assert len(verts) == 8 and len(faces) == 6            # a closed box
    lo = tuple(min(v[axis] for v in verts) for axis in range(3))
    hi = tuple(max(v[axis] for v in verts) for axis in range(3))
    assert lo == solver_settings.DEFAULT_DOMAIN_MIN
    assert hi == solver_settings.DEFAULT_DOMAIN_MAX
    # Non-degenerate on every axis — the invariant the solver enforces.
    assert all(hi[axis] > lo[axis] for axis in range(3))


def test_migration_moves_settings_once_and_never_re_triggers(monkeypatch):
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")
    monkeypatch.setattr(migration, "migrated_field_names",
                        lambda _cls: ("voxel_size", "max_particles"))

    stamp_order = []

    class _StampedGroup(SimpleNamespace):
        def __init__(self, label, **values):
            object.__setattr__(self, "_label", label)
            super().__init__(**values)

        def __setattr__(self, name, value):
            if name == "schema_version":
                stamp_order.append(self._label)
            super().__setattr__(name, value)

    target = SimpleNamespace(
        fluidsim_domain=_StampedGroup("target", schema_version=0))
    scene = SimpleNamespace(fluidsim=_StampedGroup(
        "source",
        schema_version=0, voxel_size=0.05, max_particles=2_000_000))
    stamp_order.clear()

    assert migration.migrate_domain_settings_schema(scene, target) is True
    assert (target.fluidsim_domain.voxel_size,
            target.fluidsim_domain.max_particles) == (0.05, 2_000_000)
    assert target.fluidsim_domain.schema_version == 1
    assert scene.fluidsim.schema_version == 1
    assert stamp_order == ["target", "source"]       # source guard stamped LAST

    # second run on the same scene is a no-op
    assert migration.migrate_domain_settings_schema(scene, target) is False
    # guard is on schema_version, NOT on field emptiness: zeroing moved fields must not re-trigger
    scene.fluidsim.voxel_size = 0.0
    assert migration.migrate_domain_settings_schema(scene, target) is False
    # a None domain object never stamps-and-drops
    virgin = SimpleNamespace(fluidsim=SimpleNamespace(schema_version=0, voxel_size=0.1))
    assert migration.migrate_domain_settings_schema(virgin, None) is False
    assert virgin.fluidsim.schema_version == 0       # untouched → data preserved


def test_migration_refuses_a_scene_group_that_hides_the_legacy_fields(monkeypatch):
    # Outside legacy_scene_settings() the narrow scene group declares none of the
    # domain fields. Copying from it would stamp the guard having moved nothing,
    # and the file's settings would be stranded for good.
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")
    monkeypatch.setattr(migration, "migrated_field_names",
                        lambda _cls: ("voxel_size", "max_particles"))

    target = SimpleNamespace(fluidsim_domain=SimpleNamespace(voxel_size=0.02))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(schema_version=0, voxel_size=0.05))
    with pytest.raises(RuntimeError, match="max_particles"):
        migration.migrate_domain_settings_schema(scene, target)
    assert scene.fluidsim.schema_version == 0          # still pending for the next load
    assert target.fluidsim_domain.voxel_size == 0.02   # nothing half-copied


class _RnaGroup(SimpleNamespace):
    """Group double that, like an RNA group, knows which fields hold a stored value."""

    def __init__(self, authored, **values):
        super().__init__(**values)
        self._authored = frozenset(authored)

    def is_property_set(self, name):
        return name in self._authored


def test_migration_copies_only_authored_legacy_fields(monkeypatch):
    # A scene created at runtime starts at schema_version 0 with every domain
    # field unset. Its RNA defaults must not overwrite what the domain object
    # already holds the next time the file is opened.
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")
    monkeypatch.setattr(migration, "migrated_field_names",
                        lambda _cls: ("voxel_size", "max_particles"))

    target = SimpleNamespace(fluidsim_domain=SimpleNamespace(voxel_size=0.02, max_particles=9))
    scene = SimpleNamespace(fluidsim=_RnaGroup(
        {"max_particles"}, schema_version=0, voxel_size=0.1, max_particles=500))

    assert migration.migrate_domain_settings_schema(scene, target) is True
    assert target.fluidsim_domain.voxel_size == 0.02     # unset default left alone
    assert target.fluidsim_domain.max_particles == 500   # authored value moved
    assert scene.fluidsim.schema_version == 1


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

    events = []

    @contextmanager
    def _legacy_scope():
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(migration, "_legacy_scene_settings", _legacy_scope)

    # Two pre-migration scenes; each resolves to its own prepared domain object.
    def _scene(vx):
        obj = SimpleNamespace(fluidsim_domain=SimpleNamespace())
        return SimpleNamespace(
            fluidsim=SimpleNamespace(schema_version=0, voxel_size=vx),
            _domain=obj)
    scenes = [_scene(0.05), _scene(0.2)]
    fake_bpy.data = SimpleNamespace(scenes=scenes)

    def _resolve(scene):
        events.append(scene)
        return scene._domain
    monkeypatch.setattr(migration, "_resolve_or_create_domain", _resolve)

    # register_handlers() is idempotent: two calls leave exactly one handler.
    migration.register_handlers()
    migration.register_handlers()
    load_post = fake_bpy.app.handlers.load_post
    assert load_post.count(migration._on_load_post) == 1

    # Firing the handler migrates every scene onto its domain object, all
    # inside one legacy rebind of the scene pointer.
    migration._on_load_post(None)
    assert events[0] == "enter" and events[-1] == "exit"
    assert events[1:-1] == scenes
    for scene in scenes:
        assert scene._domain.fluidsim_domain.voxel_size == scene.fluidsim.voxel_size
        assert scene._domain.fluidsim_domain.schema_version == 1
        assert scene.fluidsim.schema_version == 1

    # With nothing left to migrate the pointer is never rebound.
    events.clear()
    migration._on_load_post(None)
    assert events == []

    # unregister_handlers() removes it.
    migration.unregister_handlers()
    assert migration._on_load_post not in load_post

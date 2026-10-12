"""2.0 recognizes .blend files saved before FluidSim 1.1 and says so.

Before 1.1 the domain settings lived on the scene group; 1.1 moved them onto
the domain object and stamped ``schema_version = 1``. 2.0 reads only the
domain object, so a file that skipped 1.1 must be flagged, not silently opened
with default settings.
"""

from types import SimpleNamespace

import pytest

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def _scene(stored, effects=None):
    groups = {"fluidsim": stored}
    if effects is not None:
        groups["fluidsim_effects"] = effects
    return SimpleNamespace(
        name="Scene",
        bl_system_properties_get=lambda: None if stored is None else groups)


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


def test_snippets_1_1_never_copied_are_flagged(monkeypatch):
    file_compat = _file_compat(monkeypatch)
    snippets = [{"kernel_name": "legacy"}]
    unmigrated = _scene({"schema_version": 1, "snippets": snippets})
    assert file_compat.has_unmigrated_snippets(unmigrated)
    assert file_compat.is_pre_bridge_scene(unmigrated)
    assert file_compat.has_unmigrated_snippets(
        _scene({"schema_version": 1, "snippets": snippets}, {"migration_complete": 0}))


def test_snippets_1_1_copied_or_no_snippets_are_not_flagged(monkeypatch):
    file_compat = _file_compat(monkeypatch)
    migrated = _scene({"schema_version": 1, "snippets": [{"kernel_name": "legacy"}]},
                      {"migration_complete": 1})
    assert not file_compat.has_unmigrated_snippets(migrated)
    assert not file_compat.is_pre_bridge_scene(migrated)
    assert not file_compat.has_unmigrated_snippets(_scene({"snippets": []}))
    assert not file_compat.has_unmigrated_snippets(_scene({"status": "Idle"}))
    assert not file_compat.has_unmigrated_snippets(_scene(None))
    assert not file_compat.has_unmigrated_snippets(SimpleNamespace(name="no system props"))


def test_admission_refuses_unmigrated_snippets(monkeypatch):
    import pytest

    file_compat = _file_compat(monkeypatch)
    effects = _reload_module("fluidSim_warp_blender51.blender.effects")
    scene = _scene({"schema_version": 1, "snippets": [{"kernel_name": "legacy"}]})
    with pytest.raises(RuntimeError, match="Custom Warp Snippets were not carried over"):
        effects.build_effect_stack_for_admission(scene)
    assert "re-save this file in FluidSim 1.1" in file_compat.UNMIGRATED_SNIPPETS_MESSAGE


def test_detection_keys_are_exactly_the_domain_only_settings(monkeypatch):
    file_compat = _file_compat(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    domain_only = file_compat._domain_only_fields()
    assert "voxel_size" in domain_only
    assert "domain_object" not in domain_only
    assert domain_only == (set(properties.FLUIDSIM_PG_domain.__annotations__)
                           - set(properties.FLUIDSIM_PG_scene.__annotations__))


def test_writing_a_setting_to_the_wrong_group_raises(monkeypatch):
    """A 1.x-style ``scene.fluidsim.voxel_size = x`` must not silently no-op.

    Blender accepts any Python attribute on a PropertyGroup instance, so
    without the guard the write reads back as if it worked and never reaches
    the solver (validate_phase3_blender.py did exactly this).
    """
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    scene = properties.FLUIDSIM_PG_scene()
    domain = properties.FLUIDSIM_PG_domain()
    for name in ("voxel_size", "surface_enabled", "surface_attribute_velocity"):
        with pytest.raises(AttributeError, match=f"'{name}'.*domain_object.fluidsim_domain"):
            setattr(scene, name, 1)
    with pytest.raises(AttributeError, match="'show_surface'.*scene.fluidsim"):
        domain.show_surface = True
    scene.show_surface = True
    domain.voxel_size = 0.1
    assert (scene.show_surface, domain.voxel_size) == (True, 0.1)


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

    blender = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51" / "blender"
    for name in ("ui_scene.py", "ui_physics.py"):
        assert "file_compat.draw_pre_bridge_warning(layout" in (
            blender / name).read_text(encoding="utf-8"), name


def _particle_mode_scene(stored_index):
    seen = {}

    class _Group:
        def __setattr__(self, name, value):
            seen[name] = value

    scene = SimpleNamespace(
        name="Scene", fluidsim=_Group(),
        bl_system_properties_get=lambda: {"fluidsim": {"particle_attribute_mode": stored_index}})
    return scene, seen


def test_removed_time_offset_particle_mode_falls_back_to_none(monkeypatch):
    file_compat = _file_compat(monkeypatch)
    from fluidSim_warp_blender51.properties import PARTICLE_ATTR_ITEMS

    assert "TIME_OFFSET" not in [item[0] for item in PARTICLE_ATTR_ITEMS]
    stale, stale_seen = _particle_mode_scene(len(PARTICLE_ATTR_ITEMS))
    valid, valid_seen = _particle_mode_scene(len(PARTICLE_ATTR_ITEMS) - 1)
    assert file_compat.migrate_removed_particle_color_mode([stale, valid]) == 1
    assert stale_seen == {"particle_attribute_mode": "NONE"}
    assert valid_seen == {}


def _source_group(role, **overrides):
    values = dict(
        role=role, enabled=True, source_ui_schema_version=4,
        source_mode="PARTICLES", source_region_kind="ANALYTIC",
        sink_region_kind="ANALYTIC", field_region_kind="ANALYTIC",
        mesh_evaluation_mode="FIRST_SIM_FRAME",
        source_sampling_mode="INHERIT", source_sampling_multiplier=1.0)
    values.update(overrides)
    return SimpleNamespace(fluidsim=SimpleNamespace(**values), library=None)


def _sources_module(monkeypatch):
    _install_bpy_stub(monkeypatch)
    return _reload_module("fluidSim_warp_blender51.blender.sources")


def test_saved_native_sources_keep_animating_after_the_first_frame_fix(monkeypatch):
    """B47: pre-fix point/volume/field sources stored First Sim Frame but animated.

    First Sim Frame now really freezes those inputs, so a file saved before the
    fix is switched once to Every Sim Frame, which is what it did in practice.
    Mesh sources always froze and keep their mode; new sources keep theirs.
    """
    sources = _sources_module(monkeypatch)
    points = _source_group("SOURCE", source_region_kind="POINTS")
    volume = _source_group("SOURCE", source_region_kind="VOLUME_GRID")
    volume_sink = _source_group("SINK", sink_region_kind="VOLUME_GRID")
    field = _source_group("FIELD", source_ui_schema_version=0)
    legacy_field = _source_group(
        "SOURCE", source_ui_schema_version=2, source_mode="FIELD",
        source_sampling_mode="INHERIT")
    mesh = _source_group("SOURCE", source_region_kind="MESH_SOLID")
    every = _source_group("SOURCE", source_region_kind="POINTS",
                          mesh_evaluation_mode="EVERY_SIM_FRAME")
    new = _source_group("SOURCE", source_region_kind="POINTS",
                        source_ui_schema_version=sources.CURRENT_SOURCE_UI_SCHEMA)
    untagged = _source_group("NONE", source_region_kind="POINTS")
    scene = SimpleNamespace(objects=[
        points, volume, volume_sink, field, legacy_field, mesh, every, new,
        untagged])

    assert sources.migrate_source_schema(scene) == 5

    for obj in (points, volume, volume_sink, field, legacy_field, every):
        assert obj.fluidsim.mesh_evaluation_mode == "EVERY_SIM_FRAME"
    assert mesh.fluidsim.mesh_evaluation_mode == "FIRST_SIM_FRAME"
    assert new.fluidsim.mesh_evaluation_mode == "FIRST_SIM_FRAME"
    assert untagged.fluidsim.mesh_evaluation_mode == "FIRST_SIM_FRAME"
    for obj in (points, volume, volume_sink, field, legacy_field, mesh, every, new):
        assert obj.fluidsim.source_ui_schema_version == sources.CURRENT_SOURCE_UI_SCHEMA
    assert untagged.fluidsim.source_ui_schema_version == 4
    # The legacy field still went through the sampling migration first.
    assert legacy_field.fluidsim.source_sampling_mode == "LEGACY_ABSOLUTE"

    # Stamped: a user's later choice of First Sim Frame survives reloads.
    points.fluidsim.mesh_evaluation_mode = "FIRST_SIM_FRAME"
    assert sources.migrate_source_schema(scene) == 0
    assert points.fluidsim.mesh_evaluation_mode == "FIRST_SIM_FRAME"


def test_cadence_migration_preserves_pre_region_schema_semantics(monkeypatch):
    """Below schema 3 the runtime ignored region kinds; stamping must not revive them."""
    sources = _sources_module(monkeypatch)
    sink = _source_group("SINK", source_ui_schema_version=2,
                         sink_region_kind="VOLUME_GRID")
    mesh_sink = _source_group("SINK", source_ui_schema_version=0,
                              sink_region_kind="ANALYTIC", geometry_mode="MESH",
                              sink_region_mode="SURFACE")
    scene = SimpleNamespace(objects=[sink, mesh_sink])

    assert sources.migrate_source_schema(scene) == 0
    assert sink.fluidsim.sink_region_kind == "ANALYTIC"
    assert mesh_sink.fluidsim.sink_region_kind == "MESH_SHELL"
    for obj in (sink, mesh_sink):
        assert obj.fluidsim.mesh_evaluation_mode == "FIRST_SIM_FRAME"
        assert obj.fluidsim.source_ui_schema_version == sources.CURRENT_SOURCE_UI_SCHEMA


def test_source_schema_migration_runs_on_file_load(monkeypatch):
    sources = _sources_module(monkeypatch)
    points = _source_group("SOURCE", source_region_kind="POINTS")
    monkeypatch.setattr(sources.bpy, "data", SimpleNamespace(objects=[points]),
                        raising=False)

    sources._on_load_post(None)

    assert points.fluidsim.mesh_evaluation_mode == "EVERY_SIM_FRAME"
    assert points.fluidsim.source_ui_schema_version == sources.CURRENT_SOURCE_UI_SCHEMA


@pytest.mark.parametrize("tag", ["tag_source", "tag_sink", "tag_field"])
def test_newly_tagged_inputs_are_stamped_current(monkeypatch, tag):
    sources = _sources_module(monkeypatch)
    obj = _source_group("NONE", source_ui_schema_version=0)
    getattr(sources, tag)(obj)
    assert obj.fluidsim.source_ui_schema_version == sources.CURRENT_SOURCE_UI_SCHEMA
    assert obj.fluidsim.mesh_evaluation_mode == "FIRST_SIM_FRAME"

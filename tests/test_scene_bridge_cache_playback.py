"""Regression tests for cache playback frame-change routing."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeCollection:
    """Minimal bpy Collection stand-in: named, holds objects, iterable-linkable."""

    def __init__(self, name):
        self.name = name
        self.objects = []


class _FakeChildren(list):
    """Minimal bpy_prop_collection stand-in supporting iteration + .link()."""

    def link(self, child):
        if child not in self:
            self.append(child)


class _FakeCollectionsStore:
    """Minimal bpy.data.collections stand-in: name-keyed get/new."""

    def __init__(self):
        self._by_name = {}

    def get(self, name):
        return self._by_name.get(name)

    def new(self, name):
        coll = _FakeCollection(name)
        self._by_name[name] = coll
        return coll


def _fake_scene_collection():
    """scene.collection stand-in with a linkable, iterable .children."""
    return SimpleNamespace(children=_FakeChildren())


def _install_scene_bridge_stubs(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(evaluated_depsgraph_get=lambda: object())
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(frame_change_post=[])
    )

    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    sys.modules.pop("fluidSim_warp_blender51.blender.playback", None)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    scene_bridge = importlib.import_module(
        "fluidSim_warp_blender51.blender.scene_bridge")
    scene_bridge.playback = importlib.import_module(
        "fluidSim_warp_blender51.blender.playback")
    return scene_bridge


def _scene(frame=7, *, playback=True):
    return SimpleNamespace(
        frame_current=frame,
        frame_start=1,
        frame_end=10,
        objects=(),
        fluidsim=SimpleNamespace(
            live=True,
            cache_playback_enabled=playback,
            cache_use_custom_range=False,
            cache_frame_start=1,
            cache_frame_end=10,
            cache_path="//cache",
            status="",
            stat_particles=0,
            stat_voxels=0,
            stat_whitewater=0,
            stat_whitewater_required=0,
            stat_whitewater_admitted=0,
            stat_whitewater_dropped=0,
            stat_whitewater_policy="",
            stat_whitewater_status="",
            stat_step_ms=0.0,
            stat_substeps=0,
        ),
    )


def _install_cache_playback_modules(monkeypatch, fake_cache, fake_viewport):
    import fluidSim_warp_blender51.blender as blender_pkg
    import fluidSim_warp_blender51.io as io_pkg

    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.output_publication", fake_viewport)
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.io.cache", fake_cache)
    monkeypatch.setattr(blender_pkg, "output_publication", fake_viewport, raising=False)
    monkeypatch.setattr(io_pkg, "cache", fake_cache, raising=False)


def test_phase3_deep_modules_own_playback_and_production_bake(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    bridge_source = inspect.getsource(scene_bridge)
    bake_source = inspect.getsource(cache_bake.bake)
    transaction_source = inspect.getsource(
        production_bake.ProductionBakeBackend.step_and_commit)
    assert "def handle_frame_change" not in bridge_source
    assert "def export_cache_sequence" not in bridge_source
    assert scene_bridge.playback._on_frame_change.__module__.endswith(".playback")
    assert scene_bridge.playback._on_playback_pre.__module__.endswith(".playback")
    assert "output_publication" not in bake_source
    assert "_on_frame_change" not in bake_source
    assert "run_synchronous" in bake_source
    assert "publish_outputs=False" in transaction_source


def _authoritative_payload(particle_count=2, whitewater_count=0):
    payload = {
        "positions": np.ones((particle_count, 3), np.float32),
        "velocities": np.ones((particle_count, 3), np.float32),
        "particle_age": np.ones(particle_count, np.float32),
        "particle_life": np.ones(particle_count, np.float32),
        "particle_temperature": np.ones(particle_count, np.float32),
        "particle_viscosity": np.ones(particle_count, np.float32),
        "particle_density": np.ones(particle_count, np.float32),
        "particle_time_offset": np.ones(particle_count, np.float32),
        "whitewater": np.ones((whitewater_count, 3), np.float32),
        "whitewater_velocities": np.ones((whitewater_count, 3), np.float32),
        "whitewater_age": np.ones(whitewater_count, np.float32),
        "whitewater_life": np.ones(whitewater_count, np.float32),
    }
    return payload


def test_authoritative_payload_counts_validate_schema_alignment_and_finiteness():
    from fluidSim_warp_blender51.io import cache

    payload = _authoritative_payload(particle_count=2, whitewater_count=1)
    counts = cache.authoritative_payload_counts(payload)
    assert set(counts["particles"].values()) == {2}
    assert set(counts["whitewater"].values()) == {1}

    payload["particle_age"] = payload["particle_age"][:1]
    with pytest.raises(ValueError, match="attribute lengths do not match"):
        cache.authoritative_payload_counts(payload)

    payload = _authoritative_payload(particle_count=2, whitewater_count=1)
    payload["whitewater_life"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite fields"):
        cache.authoritative_payload_counts(payload)


def test_preview_decimated_payload_counts_fail_solver_count_contract():
    from fluidSim_warp_blender51.io import cache

    solver_particle_count = 8
    full = _authoritative_payload(
        particle_count=solver_particle_count, whitewater_count=2)
    decimated = {
        name: (value[::2] if value.ndim else value)
        for name, value in full.items()
    }
    counts = cache.authoritative_payload_counts(decimated)

    assert set(counts["particles"].values()) == {4}
    assert set(counts["particles"].values()) != {solver_particle_count}


MAX_LOOKUP_SAMPLES = 120_000  # mirrors output_publication._nearest_values


def _import_output_publication(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    sys.modules.pop("fluidSim_warp_blender51.blender.output_publication", None)
    return importlib.import_module(
        "fluidSim_warp_blender51.blender.output_publication")


def _nearest_values_reference(points, sample_points, sample_values, cell_size):
    """Brute-force oracle for the neighbourhood-nearest-value contract."""
    points = np.asarray(points, dtype=np.float32)
    sample_points = np.asarray(sample_points, dtype=np.float32)
    sample_values = np.asarray(sample_values, dtype=np.float32)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(sample_points) == 0:
        return np.zeros(len(points), dtype=np.float32)

    if len(sample_points) > MAX_LOOKUP_SAMPLES:
        step = int(np.ceil(len(sample_points) / MAX_LOOKUP_SAMPLES))
        sample_points = sample_points[::step]
        sample_values = sample_values[::step]

    cell_size = max(float(cell_size), 1.0e-5)
    sample_cells = np.floor(sample_points / cell_size).astype(np.int64)
    point_cells = np.floor(points / cell_size).astype(np.int64)

    out = np.full(len(points), float(np.nanmean(sample_values)), dtype=np.float32)
    for index in range(len(points)):
        adjacent = np.all(
            np.abs(sample_cells - point_cells[index]) <= 1, axis=1)
        if not adjacent.any():
            continue
        candidates = np.flatnonzero(adjacent)
        delta = sample_points[candidates] - points[index]
        out[index] = sample_values[candidates[
            np.argmin(np.einsum("ij,ij->i", delta, delta))]]
    return out


def test_nearest_values_picks_nearest_sample(monkeypatch):
    output_publication = _import_output_publication(monkeypatch)
    points = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float32)
    samples = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    values = np.array([2.0, 7.0], dtype=np.float32)

    np.testing.assert_array_equal(
        output_publication._nearest_values(points, samples, values, 0.5),
        [2.0, 7.0])


@pytest.mark.parametrize("cell_size", [0.05, 0.25, 1.0])
def test_nearest_values_matches_bruteforce_reference(monkeypatch, cell_size):
    output_publication = _import_output_publication(monkeypatch)
    rng = np.random.default_rng(20260809)
    samples = rng.uniform(-2.0, 2.0, size=(300, 3)).astype(np.float32)
    values = rng.uniform(-10.0, 10.0, size=300).astype(np.float32)
    points = rng.uniform(-2.5, 2.5, size=(400, 3)).astype(np.float32)

    np.testing.assert_allclose(
        output_publication._nearest_values(points, samples, values, cell_size),
        _nearest_values_reference(points, samples, values, cell_size),
        rtol=0, atol=0)


def test_nearest_values_falls_back_to_mean_outside_neighbourhood(monkeypatch):
    output_publication = _import_output_publication(monkeypatch)
    samples = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    values = np.array([2.0, 8.0], dtype=np.float32)
    points = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)

    np.testing.assert_allclose(
        output_publication._nearest_values(points, samples, values, 0.25),
        [5.0])


def test_nearest_values_decimates_above_sample_cap(monkeypatch):
    output_publication = _import_output_publication(monkeypatch)
    count = MAX_LOOKUP_SAMPLES + 1
    samples = np.zeros((count, 3), dtype=np.float32)
    samples[:, 0] = np.arange(count, dtype=np.float32) * 0.1
    values = np.arange(count, dtype=np.float32)
    points = samples[np.array([0, 1, 2, 3, 4], dtype=np.intp)] + np.array(
        [0.01, 0.0, 0.0], dtype=np.float32)

    result = output_publication._nearest_values(
        points, samples, values, 0.25)

    np.testing.assert_allclose(
        result, _nearest_values_reference(points, samples, values, 0.25))
    assert result[0] == 0.0 and result[2] == 2.0 and result[4] == 4.0
    assert 1.0 not in set(result.tolist())
    assert 3.0 not in set(result.tolist())


def test_frame_change_cache_playback_uses_cached_snapshot_without_solver_step(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    calls = []

    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    fake_viewport.update_outputs_from_cache_frame = (
        lambda scene_arg, data: calls.append((scene_arg.frame_current, int(data["frame"])))
    )
    fake_cache = types.ModuleType("fluidSim_warp_blender51.io.cache")
    fake_cache.read_frame = lambda base, frame: {
        "frame": np.array(frame, dtype=np.int32),
        "positions": np.zeros((2, 3), dtype=np.float32),
        "whitewater": np.zeros((1, 3), dtype=np.float32),
        "whitewater_required": np.array(6, dtype=np.int64),
        "whitewater_admitted": np.array(5, dtype=np.int64),
        "whitewater_dropped": np.array(1, dtype=np.int64),
        "whitewater_policy": np.array("CLAMP"),
        "whitewater_status": np.array("DROPPED"),
    }
    fake_cache.resolve_frame_range = lambda *args, **kwargs: (1, 10)
    fake_cache.read_sequence_metadata = lambda base: None
    _install_cache_playback_modules(monkeypatch, fake_cache, fake_viewport)

    solver = SimpleNamespace(step_frame=lambda: (_ for _ in ()).throw(
        AssertionError("live solver should not step during cache playback")))
    scene_bridge.session().solver = solver
    scene_bridge.session().last_frame = -1

    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    assert calls == [(7, 7)]
    assert scene.fluidsim.stat_particles == 2
    assert scene.fluidsim.stat_whitewater == 1
    assert scene.fluidsim.stat_whitewater_required == 6
    assert scene.fluidsim.stat_whitewater_admitted == 5
    assert scene.fluidsim.stat_whitewater_dropped == 1
    assert scene.fluidsim.stat_whitewater_policy == "CLAMP"
    assert scene.fluidsim.stat_whitewater_status == "DROPPED"
    assert scene.fluidsim.status == "Cache frame 7: 2 particles"


def test_live_property_cache_playback_does_not_initialize_solver(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(playback=True)

    monkeypatch.setattr(scene_bridge, "initialize_session", lambda _scene: (
        (_ for _ in ()).throw(AssertionError("cache playback must not initialize solver"))
    ))

    scene_bridge.release_session()
    scene_bridge.playback.set_live_enabled(scene, True)

    assert not scene_bridge.session().active
    assert scene.fluidsim.status == "Cache playback enabled - press play"


def test_enabling_cache_playback_releases_live_solver(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(playback=False)
    scene.fluidsim.live = True
    scene_bridge.session().solver = object()
    scene_bridge.session().scene_identity = id(scene)

    scene.fluidsim.cache_playback_enabled = True
    scene_bridge.playback.set_cache_playback_enabled(scene, True)

    assert not scene_bridge.session().active
    assert scene.fluidsim.status == "Cache playback enabled - press play"


def test_active_production_bake_rejects_live_and_cache_playback(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import production_bake

    scene = _scene(playback=False)
    scene.fluidsim.live = True
    scene.fluidsim.cache_playback_enabled = True
    sess = scene_bridge.session()
    owned_solver = object()
    sess.solver = owned_solver
    controller = SimpleNamespace(status=production_bake.BakeStatus.RUNNING)
    production_bake.register_active_bake(scene, controller)
    try:
        assert scene_bridge.playback.set_live_enabled(scene, True) is False
        assert scene.fluidsim.live is False
        assert sess.solver is owned_solver
        assert "Production Bake" in scene.fluidsim.status

        assert scene_bridge.playback.set_cache_playback_enabled(scene, True) is False
        assert scene.fluidsim.cache_playback_enabled is False
        assert sess.solver is owned_solver
        assert "Production Bake" in scene.fluidsim.status
    finally:
        production_bake.clear_active_bake(scene, controller)


def test_cache_identity_adapter_preserves_pre_move_signatures(monkeypatch):
    """The Blender adapter must preserve cache identities across extraction."""
    assert importlib.util.find_spec(
        "fluidSim_warp_blender51.blender.cache_identity") is not None, (
            "cache identity adapter has not been extracted")
    from fluidSim_warp_blender51.blender import cache_identity, solver_settings

    class Prop:
        def __init__(self, prop_type, *, is_array=False):
            self.type = prop_type
            self.is_array = is_array

    class Group:
        role = "SOURCE"
        emit_rate = 2.5
        source_status_message = "ignored"
        bl_rna = SimpleNamespace(properties={
            "rna_type": Prop("POINTER"),
            "role": Prop("ENUM"),
            "emit_rate": Prop("FLOAT"),
            "source_status_message": Prop("STRING"),
        })

    class Vertices:
        def __init__(self):
            self.values = np.array(
                [[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        def __len__(self):
            return len(self.values)

        def __bool__(self):
            return bool(len(self.values))

        def foreach_get(self, _name, output):
            output[:] = self.values.reshape(-1)

    class SourceObject:
        name = "Source"
        name_full = "Source"
        fluidsim = Group()
        matrix_world = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        library = None
        type = "MESH"
        data = SimpleNamespace(vertices=Vertices(), polygons=(object(),))

        def get(self, key, default=""):
            return "source-fixed" if key == "fluidsim_persistent_id" else default

        def evaluated_get(self, _depsgraph):
            return self

    scene = SimpleNamespace(objects=(SourceObject(),))
    settings = SimpleNamespace(
        device="cuda:0", flip_ratio=0.95, voxel_size=0.1)
    monkeypatch.setattr(
        solver_settings, "build_settings", lambda _scene: settings)

    assert cache_identity.scene_key(scene) == (
        "f66de3ed6fb1e4f4b669b86d393fabe0")
    assert cache_identity.source_signature(scene, object()) == (
        "a096363f31671213e0e1b0720033059b3e3cae8d1257c81cbd554ad5498eb217")
    assert cache_identity.signature_bundle(scene, object()) == {
        "scene_key": (
            "ad40ef588d1f46c74f920d984d7d7028aa1367e8b59b40715f8e18280d539365"),
        "settings_signature": (
            "f820c4f570ef4d5a7fbf9698e7046a528fe94ec06e1b8ced58bb6f9bea94d398"),
        "source_signature": (
            "a096363f31671213e0e1b0720033059b3e3cae8d1257c81cbd554ad5498eb217"),
    }


def test_cache_playback_refuses_a_sequence_baked_from_other_settings(
        monkeypatch, tmp_path):
    """A cache is only an answer to the scene it was baked from.

    ``format_version`` alone cannot catch this: the file is perfectly well
    formed, it just answers a different question. Playback must refuse rather
    than draw a stale result as if it were current.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import cache_identity
    from fluidSim_warp_blender51.io import cache

    base = str(tmp_path)
    scene = _scene()
    scene.fluidsim.cache_path = base

    voxel_size = [0.1]
    from fluidSim_warp_blender51.blender import solver_settings
    monkeypatch.setattr(
        solver_settings, "build_settings",
        lambda _scene: SimpleNamespace(device="cuda:0", voxel_size=voxel_size[0]))

    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    drawn = []
    fake_viewport.update_outputs_from_cache_frame = (
        lambda _scene, data: drawn.append(int(data["frame"])))
    _install_cache_playback_modules(monkeypatch, cache, fake_viewport)

    baked = cache_identity.scene_key(scene)
    cache.write_sequence_metadata(base, 1, 10, scene_key_value=baked)
    cache._savez_fast_compressed(cache._frame_path(base, 7), {
        "format_version": np.array(cache.CACHE_FORMAT_VERSION, dtype=np.int32),
        "scene_key": np.array(baked),
        "frame": np.array(7, dtype=np.int32),
        "positions": np.zeros((2, 3), dtype=np.float32),
        "whitewater": np.zeros((1, 3), dtype=np.float32),
    })

    scene_bridge.playback.play_cache_frame(scene_bridge.session(), scene, 7)
    assert drawn == [7]

    voxel_size[0] = 0.2
    with pytest.raises(ValueError, match="different scene settings"):
        scene_bridge.playback.play_cache_frame(scene_bridge.session(), scene, 7)
    assert drawn == [7]


def test_frame_change_cache_playback_reports_missing_frame_without_fallback(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()

    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    fake_viewport.update_outputs_from_cache_frame = (
        lambda *_args: (_ for _ in ()).throw(AssertionError("no snapshot to draw"))
    )
    fake_cache = types.ModuleType("fluidSim_warp_blender51.io.cache")
    fake_cache.read_frame = lambda base, frame: None
    fake_cache.resolve_frame_range = lambda *args, **kwargs: (1, 10)
    _install_cache_playback_modules(monkeypatch, fake_cache, fake_viewport)

    solver = SimpleNamespace(step_frame=lambda: (_ for _ in ()).throw(
        AssertionError("live solver fallback is not allowed")))
    scene_bridge.session().solver = solver
    scene_bridge.session().last_frame = -1

    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    assert scene.fluidsim.status.startswith("Cache playback error:")
    assert "missing or corrupt" in scene_bridge.current_error(scene)


def test_cache_frame_output_matches_live_display_toggles_and_scales(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    import bpy

    bpy.data = SimpleNamespace(
        objects=SimpleNamespace(get=lambda _name: None),
        meshes=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
        pointclouds=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
        collections=_FakeCollectionsStore(),
    )
    fake_materials = types.ModuleType("fluidSim_warp_blender51.blender.materials")
    fake_materials.ensure_surface_material = lambda *_args: None
    fake_materials.ensure_particle_material = lambda *_args: None
    fake_materials.ensure_whitewater_material = lambda *_args: None
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.blender.materials", fake_materials)
    import fluidSim_warp_blender51.blender as blender_pkg
    monkeypatch.setattr(blender_pkg, "materials", fake_materials, raising=False)
    sys.modules.pop("fluidSim_warp_blender51.blender.output_publication", None)
    output_publication = importlib.import_module("fluidSim_warp_blender51.blender.output_publication")

    calls = []
    monkeypatch.setattr(output_publication, "_supports_pointcloud", lambda: True)
    monkeypatch.setattr(
        output_publication,
        "_ensure_object",
        lambda _scene, name, _factory, *_a, **_k: SimpleNamespace(name=name, data=SimpleNamespace()),
    )
    monkeypatch.setattr(
        output_publication,
        "output_object",
        lambda _scene, name: None,
    )
    monkeypatch.setattr(
        output_publication,
        "_swap_mesh",
        lambda obj, *_args, **_kwargs: calls.append(("mesh", obj.name)),
    )
    monkeypatch.setattr(
        output_publication,
        "_swap_points",
        lambda obj, _pos, radius, _colors=None, _attributes=None, **_kwargs: calls.append(
            ("points", obj.name, radius)
        ),
    )
    monkeypatch.setattr(
        output_publication,
        "_clear_points",
        lambda _scene, name: calls.append(("clear_points", name)),
    )

    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            show_surface=True,
            surface_enabled=False,
            show_particles=True,
            particle_display_fraction=1.0,
            particle_display_scale=2.0,
            particle_attribute_mode="NONE",
            surface_attribute_mode="NONE",
            show_whitewater=True,
            whitewater_enabled=False,
            whitewater_display_scale=3.0,
            whitewater_attribute_mode="NONE",
            voxel_size=0.2,
        ),
        collection=_fake_scene_collection(),
    )
    frame_data = {
        "voxel_size": np.array(0.2, dtype=np.float32),
        "positions": np.ones((2, 3), dtype=np.float32),
        "velocities": np.zeros((2, 3), dtype=np.float32),
        "whitewater": np.ones((1, 3), dtype=np.float32),
        "surface_verts": np.ones((3, 3), dtype=np.float32),
        "surface_tris": np.array([[0, 1, 2]], dtype=np.int32),
    }

    output_publication.update_outputs_from_cache_frame(scene, frame_data)

    assert ("mesh", output_publication.SURFACE_NAME) not in calls
    assert ("clear_points", output_publication.WHITEWATER_NAME) in calls
    assert any(
        call[0] == "points"
        and call[1] == output_publication.PARTICLES_NAME
        and abs(call[2] - 0.1) < 1e-6
        for call in calls
    )
    assert all(call[:2] != ("points", output_publication.WHITEWATER_NAME) for call in calls)


def test_live_and_cached_outputs_share_rendering_policy(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    import bpy

    bpy.data = SimpleNamespace(
        objects=SimpleNamespace(get=lambda _name: None),
        meshes=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
        pointclouds=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
        collections=_FakeCollectionsStore(),
    )
    fake_materials = types.ModuleType("fluidSim_warp_blender51.blender.materials")
    calls = []
    fake_materials.ensure_surface_material = (
        lambda _scene, obj: calls.append(("material", "surface", obj.name)))
    fake_materials.ensure_particle_material = (
        lambda _scene, obj: calls.append(("material", "particles", obj.name)))
    fake_materials.ensure_whitewater_material = (
        lambda _scene, obj: calls.append(("material", "whitewater", obj.name)))
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.materials", fake_materials
    )
    import fluidSim_warp_blender51.blender as blender_pkg
    monkeypatch.setattr(blender_pkg, "materials", fake_materials, raising=False)
    sys.modules.pop("fluidSim_warp_blender51.blender.output_publication", None)
    output_publication = importlib.import_module("fluidSim_warp_blender51.blender.output_publication")

    def values(array):
        if array is None:
            return None
        return np.asarray(array).round(6).tolist()

    monkeypatch.setattr(output_publication, "_supports_pointcloud", lambda: True)
    monkeypatch.setattr(
        output_publication,
        "_ensure_object",
        lambda _scene, name, _factory, *_a, **_k: SimpleNamespace(name=name, data=SimpleNamespace()),
    )
    monkeypatch.setattr(
        output_publication,
        "output_object",
        lambda _scene, name: None,
    )
    monkeypatch.setattr(
        output_publication,
        "_swap_mesh",
        lambda obj, verts, tris, colors=None, attributes=None: calls.append(
            ("mesh", obj.name, values(verts), values(tris), values(colors))
        ),
    )
    monkeypatch.setattr(
        output_publication,
        "_swap_points",
        lambda obj, positions, radius, colors=None, attributes=None, *,
        reuse_datablock=True: calls.append(
            ("points", obj.name, values(positions), round(radius, 6),
             values(colors), reuse_datablock)
        ),
    )
    monkeypatch.setattr(
        output_publication, "_clear_points",
        lambda _scene, name: calls.append(("clear", name))
    )

    fs = SimpleNamespace(
        show_surface=True,
        surface_enabled=False,
        surface_attribute_mode="NONE",
        show_particles=True,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        particle_attribute_mode="NONE",
        show_whitewater=False,
        whitewater_enabled=False,
        whitewater_display_scale=1.0,
        whitewater_attribute_mode="NONE",
        voxel_size=0.2,
    )
    scene = SimpleNamespace(fluidsim=fs, collection=_fake_scene_collection())
    positions = np.ones((2, 3), dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    verts = np.ones((3, 3), dtype=np.float32)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    solver = SimpleNamespace(
        settings=SimpleNamespace(
            surface_enabled=True,
            whitewater_enabled=False,
            voxel_size=0.2,
        ),
        surface_mesh=lambda: (verts, tris),
        particle_positions=lambda: positions,
        particle_velocities=lambda: velocities,
        whitewater_positions=lambda: np.zeros((0, 3), dtype=np.float32),
    )
    session = SimpleNamespace(active=True, solver=solver)
    frame_data = {
        "voxel_size": np.array(0.2, dtype=np.float32),
        "positions": positions,
        "velocities": velocities,
        "surface_verts": verts,
        "surface_tris": tris,
    }

    output_publication.update_outputs(scene, session)
    live_calls = list(calls)
    calls.clear()
    output_publication.update_outputs_from_cache_frame(scene, frame_data)

    assert live_calls == calls

    calls.clear()
    fs.surface_enabled = True
    fs.surface_attribute_mode = "HEIGHT"
    fs.particle_display_fraction = 0.5
    fs.particle_attribute_mode = "HEIGHT"
    fs.show_whitewater = True
    fs.whitewater_enabled = True
    fs.whitewater_attribute_mode = "HEIGHT"
    solver.settings.surface_enabled = True
    solver.settings.whitewater_enabled = True
    solver.whitewater_positions = lambda: np.array(
        ((0.0, 0.0, 0.5), (0.0, 0.0, 1.5)), dtype=np.float32)
    frame_data.pop("velocities")
    frame_data["whitewater"] = solver.whitewater_positions()

    output_publication.update_outputs(scene, session)
    live_calls = list(calls)
    calls.clear()
    output_publication.update_outputs_from_cache_frame(scene, frame_data)

    assert live_calls == calls

    calls.clear()
    solver.whitewater_positions = lambda: np.zeros((0, 3), dtype=np.float32)
    frame_data["whitewater"] = solver.whitewater_positions()

    output_publication.update_outputs(scene, session)
    live_calls = list(calls)
    calls.clear()
    output_publication.update_outputs_from_cache_frame(scene, frame_data)

    assert live_calls == calls


def test_ensure_outputs_allocates_ids_without_writing_geometry(monkeypatch):
    """``initialize()`` must create the output IDs, not the first live frame.

    ``_ensure_object`` allocates objects and links a collection -- a depsgraph
    relations change. Leaving that to ``_render_outputs`` means it happens
    inside ``frame_change_post``, against a depsgraph still being evaluated.
    """
    _install_scene_bridge_stubs(monkeypatch)
    import bpy

    bpy.data = SimpleNamespace(
        objects=SimpleNamespace(get=lambda _name: None),
        meshes=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
        pointclouds=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
        collections=_FakeCollectionsStore(),
    )
    fake_materials = types.ModuleType("fluidSim_warp_blender51.blender.materials")
    calls = []
    for kind in ("surface", "particle", "whitewater"):
        setattr(
            fake_materials, f"ensure_{kind}_material",
            lambda _scene, obj, kind=kind: calls.append(("material", kind)))
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.materials", fake_materials
    )
    import fluidSim_warp_blender51.blender as blender_pkg
    monkeypatch.setattr(blender_pkg, "materials", fake_materials, raising=False)
    sys.modules.pop("fluidSim_warp_blender51.blender.output_publication", None)
    output_publication = importlib.import_module("fluidSim_warp_blender51.blender.output_publication")

    monkeypatch.setattr(output_publication, "_supports_pointcloud", lambda: True)
    monkeypatch.setattr(
        output_publication, "_ensure_object",
        lambda _scene, name, _factory, *_a, **_k: (
            calls.append(("object", name)) or SimpleNamespace(name=name)),
    )
    for name in ("_swap_mesh", "_swap_points", "_swap_vertex_cloud",
                 "_clear_points"):
        monkeypatch.setattr(
            output_publication, name, _no_geometry_write(name))

    fs = SimpleNamespace(
        show_surface=True, surface_enabled=True,
        show_particles=True,
        show_whitewater=True, whitewater_enabled=True,
    )
    scene = SimpleNamespace(fluidsim=fs, collection=_fake_scene_collection())

    output_publication.prepare_outputs(scene)

    assert calls == [
        ("object", output_publication.SURFACE_NAME), ("material", "surface"),
        ("object", output_publication.PARTICLES_NAME), ("material", "particle"),
        ("object", output_publication.WHITEWATER_NAME), ("material", "whitewater"),
    ]

    # Disabled outputs stay unallocated -- toggling one on later goes through
    # request_scene_update, which runs outside the handler.
    calls.clear()
    fs.surface_enabled = False
    fs.show_whitewater = False
    output_publication.prepare_outputs(scene)

    assert calls == [("object", output_publication.PARTICLES_NAME),
                     ("material", "particle")]


def _no_geometry_write(name):
    def _fail(*_args, **_kwargs):
        raise AssertionError(f"ensure_outputs must not call {name}")
    return _fail


def test_padded_pointclouds_hold_capacity_across_a_drifting_count(monkeypatch):
    """A realloc detaches the cloud, so it must stay rare.

    An attached cloud can never be resized -- Blender 5.1 faults natively --
    so the only way to avoid a datablock swap per frame is to keep spare
    capacity and only realloc when the count leaves the band.
    """
    _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import output_publication

    class _AttributeMap(dict):
        def __iter__(self):
            return iter(self.values())

    def cloud(capacity, *, has_color=False, users=1, schema=True):
        attributes = _AttributeMap()
        if schema:
            attributes.update({
                "position": SimpleNamespace(
                    name="position", data_type="FLOAT_VECTOR", domain="POINT"),
                "radius": SimpleNamespace(
                    name="radius", data_type="FLOAT", domain="POINT"),
            })
        if has_color:
            attributes[output_publication.COLOR_ATTR] = SimpleNamespace(
                name=output_publication.COLOR_ATTR,
                data_type="FLOAT_COLOR",
                domain="POINT",
            )
        return SimpleNamespace(
            users=users,
            resize=lambda _count: None,
            points=[None] * capacity,
            attributes=attributes,
        )

    cap = output_publication._point_capacity_for(100_000)
    assert cap > 100_000

    # In-band counts reuse the attached cloud, including a count that changed.
    assert not output_publication._needs_realloc(cloud(cap), 100_000, False)
    assert not output_publication._needs_realloc(cloud(cap), 99_000, False)
    assert not output_publication._needs_realloc(cloud(cap), cap, False)
    # ...and the reuse band is wide enough that ordinary FLIP drift stays in it.
    assert not output_publication._needs_realloc(cloud(cap), int(cap * 0.55), False)

    # Structural changes force a fresh, detached cloud.
    assert output_publication._needs_realloc(cloud(cap), cap + 1, False)          # grow
    assert output_publication._needs_realloc(cloud(cap), cap // 8, False)         # shrink
    assert output_publication._needs_realloc(cloud(cap), 100_000, True)           # +color
    assert output_publication._needs_realloc(
        cloud(cap, has_color=True), 100_000, False)                     # -color
    assert output_publication._needs_realloc(cloud(cap, users=2), 100_000, False)
    assert output_publication._needs_realloc(cloud(cap, schema=False), 100_000, False)
    assert output_publication._needs_realloc(None, 100_000, False)

    # An empty frame must not realloc every time just for sitting under the
    # floor -- whitewater is empty for the whole start of a sim.
    floor = output_publication._point_capacity_for(0)
    assert not output_publication._needs_realloc(cloud(floor), 0, False)

    # Shrinking settles: the capacity chosen for n is inside n's own band.
    for count in (0, 1, 5_000, 250_000):
        assert not output_publication._needs_realloc(
            cloud(output_publication._point_capacity_for(count)), count, False)


def test_release_session_clears_all_runtime_state_and_error(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    sess = scene_bridge.session()
    sess.solver = object()
    sess.scene_identity = id(scene)
    sess.profiler = object()
    sess.start_frame = 9
    sess.last_frame = 12
    sess.frame_stats = [{"frame": 12}]
    sess.scene_inputs._collider_prev = {"Collider": (object(), 12)}
    sess.scene_inputs._collider_rotation_prev = {"Collider": (object(), 12)}
    sess.scene_inputs._collider_mesh_cache = {"Collider": object()}
    sess.scene_inputs.collider_mesh_cache_stats = {"rebuilds": 3, "reuses": 4}
    scene_bridge.set_error(scene, "boom")

    scene_bridge.release_session(scene)
    scene_bridge.release_session(scene)

    assert not sess.active
    assert sess.profiler is None
    assert sess.start_frame == 1
    assert sess.last_frame == -1
    assert sess.frame_stats == []
    assert sess.scene_inputs._collider_prev == {}
    assert sess.scene_inputs._collider_rotation_prev == {}
    assert sess.scene_inputs._collider_mesh_cache == {}
    assert sess.collider_mesh_cache_stats == {"rebuilds": 0, "reuses": 0}
    assert scene_bridge.current_error(scene) == ""


def test_release_session_no_argument_preserves_current_scene_compatibility(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(playback=False)
    scene_bridge.bpy.context.scene = scene
    sess = scene_bridge.session()
    sess.solver = object()
    sess.scene_identity = id(scene)

    assert scene_bridge.release_session() is True
    assert not sess.active


def test_release_session_closes_solver_before_dropping_runtime(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    calls = []

    class Solver:
        device = "cuda:0"

        def release(self):
            calls.append("solver")

    sess = scene_bridge.session()
    sess.solver = Solver()
    sess.scene_identity = id(scene)

    scene_bridge.release_session(scene)

    assert calls == ["solver"]


def test_release_session_drops_solver_before_cuda_pool_trim(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    from fluidSim_warp_blender51.sim import memory_safety

    class Solver:
        device = "cuda:0"

        def release(self):
            pass

    solver = Solver()
    solver_ref = weakref.ref(solver)
    scene_bridge.session().solver = solver
    scene_bridge.session().scene_identity = id(scene)
    del solver

    observed = []
    monkeypatch.setattr(
        memory_safety,
        "release_cuda_memory",
        lambda _device: observed.append(solver_ref() is None),
    )

    scene_bridge.release_session(scene)

    assert observed == [True]


def test_initialize_failure_releases_partial_session(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    scene.frame_start = 1
    scene.fluidsim.profiling_enabled = False
    scene.fluidsim.snippets = []
    fake_solver_module = types.ModuleType("fluidSim_warp_blender51.sim.solver")
    fake_solver_module.FluidSolver = lambda *_args, **_kwargs: object()
    fake_profiler_module = types.ModuleType(
        "fluidSim_warp_blender51.diagnostics.profiler")
    fake_profiler_module.Profiler = object
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.sim.solver", fake_solver_module)
    monkeypatch.setitem(
        sys.modules,
        "fluidSim_warp_blender51.diagnostics.profiler",
        fake_profiler_module,
    )
    from fluidSim_warp_blender51.blender import simulation_admission

    sess = scene_bridge.session()
    capacity_plan = SimpleNamespace(
        blockers=(), estimated_bytes=0, budget_bytes=1, suggestions=())
    snapshot = SimpleNamespace(
        configured_device="cuda:0", settings=SimpleNamespace(device="cuda:0"),
        profiling_enabled=False, capacity_plan=capacity_plan,
        scene_identity=1, scene_key="key",
        frame_start=1, release_transients=lambda: None)
    monkeypatch.setattr(
        simulation_admission.SimulationAdmission, "_preflight",
        lambda *_args, **_kwargs: (
            snapshot, {"safe": True, "estimated_bytes": 0}, None))
    monkeypatch.setattr(
        simulation_admission.SimulationAdmission, "_configure_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("sync failed")))

    with pytest.raises(RuntimeError, match="sync failed"):
        sess.initialize(scene)

    assert not sess.active
    assert sess.profiler is None
    assert sess.frame_stats == []
    assert sess.scene_inputs._collider_prev == {}
    assert sess.scene_inputs._collider_mesh_cache == {}


def test_constructor_failure_collects_unassigned_solver_cycle(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    scene.frame_start = 1
    scene.fluidsim.profiling_enabled = False
    scene.fluidsim.snippets = []
    refs = []

    class BrokenSolver:
        def __init__(self, *_args, **_kwargs):
            self.surface = SimpleNamespace(solver=self)
            refs.append(weakref.ref(self))
            raise RuntimeError("constructor failed")

    fake_solver_module = types.ModuleType("fluidSim_warp_blender51.sim.solver")
    fake_solver_module.FluidSolver = BrokenSolver
    fake_profiler_module = types.ModuleType(
        "fluidSim_warp_blender51.diagnostics.profiler")
    fake_profiler_module.Profiler = object
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.sim.solver", fake_solver_module)
    monkeypatch.setitem(
        sys.modules,
        "fluidSim_warp_blender51.diagnostics.profiler",
        fake_profiler_module,
    )
    from fluidSim_warp_blender51.blender import simulation_admission

    sess = scene_bridge.session()
    capacity_plan = SimpleNamespace(
        blockers=(), estimated_bytes=0, budget_bytes=1, suggestions=())
    snapshot = SimpleNamespace(
        configured_device="cpu", settings=SimpleNamespace(device="cpu"),
        profiling_enabled=False, capacity_plan=capacity_plan,
        scene_identity=1, scene_key="key",
        frame_start=1, release_transients=lambda: None)
    monkeypatch.setattr(
        simulation_admission.SimulationAdmission, "_preflight",
        lambda *_args, **_kwargs: (
            snapshot, {"safe": True, "estimated_bytes": 0}, None))

    with pytest.raises(RuntimeError, match="constructor failed"):
        sess.initialize(scene)

    assert refs[0]() is None
    assert not scene_bridge.session().active


def test_session_step_retains_only_latest_thousand_frame_stats(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    scene.fluidsim.cache_enabled = False
    sess = scene_bridge.session()
    stats = SimpleNamespace(
        active_particles=3,
        active_voxels=4,
        whitewater_particles=5,
        step_time=0.01,
        substeps=2,
        cfl_substeps_clamped=False,
        as_dict=lambda: {"frame": 1001},
    )
    sess.solver = SimpleNamespace(frame=1001, step_frame=lambda: stats)
    sess.scene_identity = id(scene)
    sess.frame_stats = [{"frame": index} for index in range(1, 1001)]
    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    fake_viewport.update_outputs = lambda *_args: None
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.output_publication", fake_viewport)
    monkeypatch.setattr(blender_pkg, "output_publication", fake_viewport, raising=False)

    scene_bridge.step_frame(scene, sess)

    assert len(sess.frame_stats) == 1000
    assert sess.frame_stats[0] == {"frame": 2}
    assert sess.frame_stats[-1] == {"frame": 1001}


def test_live_step_never_writes_legacy_cache(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=7, playback=False)
    scene.fluidsim.cache_enabled = True
    sess = scene_bridge.session()
    stats = SimpleNamespace(
        active_particles=3,
        active_voxels=4,
        whitewater_particles=0,
        step_time=0.01,
        substeps=1,
        cfl_substeps_clamped=False,
        as_dict=lambda: {"frame": 7},
    )
    sess.solver = SimpleNamespace(
        frame=7,
        settings=SimpleNamespace(),
        step_frame=lambda: stats,
    )
    sess.scene_identity = id(scene)
    writes = []
    from fluidSim_warp_blender51.io import cache

    monkeypatch.setattr(
        cache, "write_frame",
        lambda *_args, **_kwargs: writes.append(scene.frame_current))
    monkeypatch.setattr(cache, "settings_record", lambda _settings: {})

    scene_bridge.step_frame(
        scene, sess, timeline_frame=scene.frame_current, publish_outputs=False)

    assert writes == []


def test_frame_change_rewind_delegates_to_session_reset(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=3, playback=False)
    sess = scene_bridge.session()
    sess.solver = object()
    sess.scene_identity = id(scene)
    sess.start_frame = 1
    sess.last_frame = 8
    calls = []
    monkeypatch.setattr(
        sess, "reset",
        lambda scene_arg, **kwargs: calls.append((scene_arg, kwargs)))
    depsgraph = object()

    scene_bridge.playback._on_frame_change(scene, depsgraph=depsgraph)

    assert calls == [(scene, {"depsgraph": depsgraph})]
    assert sess.last_frame == 3


def test_frame_change_catchup_publishes_only_final_timeline_frame(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=4, playback=False)
    sess = scene_bridge.session()
    sess.solver = SimpleNamespace(
        frame=1,
        settings=SimpleNamespace(flip_ratio=0.9, cfl=1.0),
    )
    sess.scene_identity = id(scene)
    sess.start_frame = 1
    sess.last_frame = 1
    calls = []
    monkeypatch.setattr(sess, "sync_scene", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sess,
        "step",
        lambda _scene, **kwargs: calls.append((
            kwargs["timeline_frame"], kwargs["publish_outputs"])),
    )

    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    assert calls == [(2, False), (3, False), (4, True)]


def test_frame_change_reconciles_animated_hot_settings_before_step(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    scene.fluidsim.flip_ratio = 0.3
    scene.fluidsim.cfl = 2.5
    scene.fluidsim.gravity = (0.0, 0.0, -2.0)
    scene.fluidsim.use_scene_gravity = False
    scene.use_gravity = False
    scene.gravity = (0.0, 0.0, -9.81)
    sess = scene_bridge.session()
    sess.solver = SimpleNamespace(
        frame=1,
        settings=SimpleNamespace(flip_ratio=0.9, cfl=1.0),
    )
    sess.scene_identity = id(scene)
    sess.start_frame = 1
    sess.last_frame = 1
    observed = []
    monkeypatch.setattr(sess, "sync_scene", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sess, "step", lambda *_args, **_kwargs: observed.append(
        (sess.solver.settings.flip_ratio, sess.solver.settings.cfl)))

    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    assert observed == [(0.3, 2.5)]


def test_frame_change_contains_bad_live_enum_and_keeps_playback_solver(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    scene.fluidsim.sample_type = "BAD_ENUM"
    sess = scene_bridge.session()
    solver = SimpleNamespace(
        frame=1,
        settings=SimpleNamespace(sample_type="previous"),
    )
    sess.solver = solver
    sess.scene_identity = id(scene)
    sess.start_frame = 1
    sess.last_frame = 1
    stepped = []
    monkeypatch.setattr(sess, "sync_scene", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sess, "step", lambda *_args, **_kwargs: stepped.append(True))

    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    assert stepped == [True]
    assert scene.fluidsim.live is True
    assert sess.solver is solver
    assert solver.settings.sample_type == "previous"
    assert "BAD_ENUM" in sess.current_error(scene)
    assert scene.fluidsim.status.startswith("Live settings error:")


def test_playback_manual_and_production_share_post_evaluation_reconcile_seam(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=7, playback=False)
    scene.fluidsim.flip_ratio = 0.9
    scene.fluidsim.cfl = 1.0
    scene.fluidsim.gravity = (0.0, 0.0, -9.81)
    scene.fluidsim.use_scene_gravity = False
    scene.use_gravity = False
    scene.gravity = (0.0, 0.0, -9.81)
    scene.fluidsim.cache_enabled = False
    values = {
        2: (0.2, 1.2, (0.0, 0.0, -2.0)),
        3: (0.3, 1.3, (0.0, 0.0, -3.0)),
        4: (0.4, 1.4, (0.0, 0.0, -4.0)),
    }

    def frame_set(frame):
        scene.frame_current = frame
        if frame in values:
            (scene.fluidsim.flip_ratio, scene.fluidsim.cfl,
             scene.fluidsim.gravity) = values[frame]

    scene.frame_set = frame_set
    sess = scene_bridge.session()
    sess.solver = SimpleNamespace(
        frame=0, settings=SimpleNamespace(flip_ratio=0.9, cfl=1.0))
    sess.scene_identity = id(scene)
    monkeypatch.setattr(sess, "initialize", lambda _scene: sess)
    monkeypatch.setattr(sess, "sync_scene", lambda *_args, **_kwargs: None)
    observed = []
    monkeypatch.setattr(sess, "step", lambda *_args, **_kwargs: observed.append((
        sess.solver.settings.flip_ratio,
        sess.solver.settings.cfl,
        sess.solver.settings.gravity)))
    for frame in range(2, 5):
        frame_set(frame)
        scene_bridge.step_frame(
            scene,
            sess,
            write_cache=False,
            refresh_pending=True,
            timeline_frame=frame,
            publish_outputs=False,
        )

    expected = [values[2], values[3], values[4]]
    assert observed == expected
    assert scene.frame_current == 4

    observed.clear()
    sess.solver.settings.flip_ratio = 0.9
    sess.solver.settings.cfl = 1.0
    for frame in range(2, 5):
        frame_set(frame)
        scene_bridge.step_frame(scene, sess)
    assert observed == expected

    observed.clear()
    scene.fluidsim.live = True
    sess.start_frame = 1
    sess.last_frame = 1
    sess.solver.settings.flip_ratio = 0.9
    sess.solver.settings.cfl = 1.0
    for frame in range(2, 5):
        frame_set(frame)
        scene_bridge.playback._on_frame_change(scene, depsgraph=object())
    assert observed == expected


def test_frame_change_reinitializes_saved_live_scene_without_step_button(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    sess = scene_bridge.session()
    calls = []
    depsgraphs = []

    def initialize(scene_arg, **kwargs):
        depsgraphs.append(kwargs.get("depsgraph"))
        calls.append(("initialize", scene_arg.frame_current))
        sess.solver = SimpleNamespace(frame=0)
        sess.start_frame = scene_arg.frame_start
        return sess

    monkeypatch.setattr(sess, "initialize", initialize)
    monkeypatch.setattr(
        sess, "sync_scene", lambda *_args: calls.append(("sync", scene.frame_current)))
    monkeypatch.setattr(
        sess, "step",
        lambda _scene, **kwargs: calls.append(
            ("step", kwargs.get("timeline_frame", scene.frame_current))))
    depsgraph = object()

    scene_bridge.playback._on_frame_change(scene, depsgraph=depsgraph)

    assert calls == [("initialize", 2), ("sync", 2), ("step", 2)]
    assert depsgraphs == [depsgraph]


def _count_initialize(monkeypatch, sess):
    """Record initialize() calls and mark the session active, as the real one does."""
    calls = []

    def initialize(scene_arg):
        calls.append(scene_arg)
        sess.solver = SimpleNamespace(frame=0)
        sess.scene_identity = id(scene_arg)
        sess.start_frame = scene_arg.frame_start
        return sess

    monkeypatch.setattr(sess, "initialize", initialize)
    return calls


def test_playback_pre_initializes_before_any_frame_change(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    sess = scene_bridge.session()
    calls = _count_initialize(monkeypatch, sess)

    scene_bridge.playback._on_playback_pre(scene)

    assert len(calls) == 1
    assert sess.active


def test_playback_pre_is_idempotent_once_the_session_is_active(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    sess = scene_bridge.session()
    calls = _count_initialize(monkeypatch, sess)

    scene_bridge.playback._on_playback_pre(scene)
    scene_bridge.playback._on_playback_pre(scene)
    scene_bridge.playback.ensure_session_before_frames(scene, "render")

    assert len(calls) == 1


def test_playback_pre_replaces_session_owned_by_another_scene(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    other_scene = _scene(frame=2, playback=False)
    sess = scene_bridge.session()
    sess.solver = SimpleNamespace(frame=0)
    sess.scene_identity = id(other_scene)
    calls = _count_initialize(monkeypatch, sess)

    scene_bridge.playback.ensure_session_before_frames(scene, "playback")

    assert calls == [scene]


def test_playback_pre_does_not_allocate_when_not_live(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    scene.fluidsim.live = False
    sess = scene_bridge.session()
    calls = _count_initialize(monkeypatch, sess)

    scene_bridge.playback._on_playback_pre(scene)
    scene_bridge.playback.ensure_session_before_frames(scene, "render")

    # Pressing play on a non-live scene must not build a GPU solver.
    assert calls == []


def test_playback_pre_does_not_allocate_for_cache_playback(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=True)
    sess = scene_bridge.session()
    calls = _count_initialize(monkeypatch, sess)

    scene_bridge.playback._on_playback_pre(scene)

    assert calls == []


def test_playback_pre_reports_init_failure_without_breaking_playback(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    sess = scene_bridge.session()

    def initialize(_scene_arg):
        raise ValueError(
            "world box (19.5, 19.5, 19.5) .. (20.5, 20.5, 20.5) lies entirely "
            "outside the simulation domain")

    monkeypatch.setattr(sess, "initialize", initialize)

    scene_bridge.playback._on_playback_pre(scene)

    assert "world box (19.5, 19.5, 19.5)" in scene_bridge.current_error(scene)
    assert sess.solver is None
    assert "Solver initialization error" in scene.fluidsim.status
    # Flipping live off here would fire the property callback outside the frame
    # handler, where _IN_FRAME_HANDLER cannot suppress it.
    assert scene.fluidsim.live is True


def test_frame_change_after_playback_pre_does_not_reinitialize(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    sess = scene_bridge.session()
    calls = []

    def initialize(scene_arg):
        calls.append(("initialize", scene_arg.frame_current))
        sess.solver = SimpleNamespace(frame=0)
        sess.scene_identity = id(scene_arg)
        sess.start_frame = scene_arg.frame_start
        return sess

    monkeypatch.setattr(sess, "initialize", initialize)
    monkeypatch.setattr(
        sess, "sync_scene", lambda *_args: calls.append(("sync", scene.frame_current)))
    monkeypatch.setattr(
        sess, "step",
        lambda _scene, **kwargs: calls.append(
            ("step", kwargs.get("timeline_frame", scene.frame_current))))

    scene_bridge.playback._on_playback_pre(scene)
    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    # The inverse of the lazy-init test above: the solver already exists, so the
    # handler never reaches initialize() with the depsgraph mid-evaluation.
    assert calls == [("initialize", 2), ("sync", 2), ("step", 2)]


def _record_solver_work(monkeypatch, scene, sess):
    """Log sync/step/reset so a frame change's effect on the solver is visible."""
    calls = []
    sess.solver = SimpleNamespace(frame=3)
    sess.scene_identity = id(scene)
    sess.start_frame = scene.frame_start
    monkeypatch.setattr(
        sess, "sync_scene", lambda *_args, **_kw: calls.append("sync"))
    monkeypatch.setattr(
        sess, "step",
        lambda _scene, **kwargs: calls.append(
            ("step", kwargs.get("timeline_frame", scene.frame_current))))
    monkeypatch.setattr(sess, "reset", lambda _scene: calls.append("reset"))
    return calls


def test_render_frame_restore_does_not_touch_the_solver(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=5, playback=False)
    sess = scene_bridge.session()
    calls = _record_solver_work(monkeypatch, scene, sess)
    sess.last_frame = 4

    # Blender jumps back to the pre-render frame when the job ends; that step
    # wrote a cache frame the user never simulated.
    scene_bridge.playback.arm_render_frame_restore(5)
    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    assert calls == []
    assert sess.last_frame == 5


def test_render_frame_restore_is_one_shot(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=5, playback=False)
    sess = scene_bridge.session()
    calls = _record_solver_work(monkeypatch, scene, sess)
    sess.last_frame = 4

    scene_bridge.playback.arm_render_frame_restore(5)
    scene_bridge.playback._on_frame_change(scene, depsgraph=object())
    scene.frame_current = 6
    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    # The frame after the restore is a real user frame again.
    assert calls == ["sync", ("step", 6)]


def test_render_frame_restore_does_not_swallow_a_different_frame(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=6, playback=False)
    sess = scene_bridge.session()
    calls = _record_solver_work(monkeypatch, scene, sess)
    sess.last_frame = 5

    # Render ended at frame 5, but the user scrubbed elsewhere first: that is a
    # real gesture and must still simulate.
    scene_bridge.playback.arm_render_frame_restore(5)
    scene_bridge.playback._on_frame_change(scene, depsgraph=object())

    assert calls == ["sync", ("step", 6)]


def test_production_bake_failure_runs_backend_cleanup_and_preserves_error():
    from fluidSim_warp_blender51.blender.production_bake import (
        BakeController,
        BakeStatus,
    )

    calls = []

    class Backend:
        base_dir = "//cache"

        def initialize(self, _scene, _start, _end, **_kwargs):
            calls.append("initialize")

        def step_and_commit(self, _scene, _frame):
            raise RuntimeError("step failed")

        def fail(self, _scene, error):
            calls.append(("fail", str(error)))

    controller = BakeController(backend=Backend())
    scene = SimpleNamespace()
    controller.start(scene, 2, 4)
    controller.tick(scene)
    controller.tick(scene)

    assert controller.status is BakeStatus.FAILED
    assert "step failed" in controller.error
    assert calls == ["initialize", ("fail", "step failed")]


def test_production_bake_registry_is_scene_owned_and_compare_cleared():
    from fluidSim_warp_blender51.blender import production_bake

    first_scene = SimpleNamespace()
    second_scene = SimpleNamespace()
    first = SimpleNamespace(status=production_bake.BakeStatus.RUNNING)
    second = SimpleNamespace(status=production_bake.BakeStatus.CANCELLING)

    production_bake.register_active_bake(first_scene, first)
    production_bake.register_active_bake(second_scene, second)
    try:
        assert production_bake.active_bake(first_scene) is first
        assert production_bake.active_bake(second_scene) is second
        assert production_bake.bake_is_active(first_scene)
        production_bake.clear_active_bake(first_scene, second)
        assert production_bake.active_bake(first_scene) is first
        production_bake.clear_active_bake(first_scene, first)
        assert production_bake.active_bake(first_scene) is None
        assert production_bake.active_bake(second_scene) is second
    finally:
        production_bake.clear_active_bake(first_scene)
        production_bake.clear_active_bake(second_scene)


def test_activity_snapshot_prioritizes_active_production_bake(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import production_bake

    scene = _scene(playback=True)
    scene.fluidsim.bake_status = "IDLE"
    scene.fluidsim.bake_progress = 0.0
    scene.fluidsim.bake_message = ""
    controller = SimpleNamespace(
        status=production_bake.BakeStatus.RUNNING,
        progress=0.5,
        message="Committed frame 4",
    )
    production_bake.register_active_bake(scene, controller)
    try:
        snapshot = scene_bridge.playback.activity_snapshot(
            scene, is_animation_playing=True)
        assert snapshot.state == "BAKING"
        assert snapshot.label == "Baking"
        assert snapshot.detail == "Committed frame 4"
        assert snapshot.busy is True
        assert snapshot.progress == 0.5
    finally:
        production_bake.clear_active_bake(scene, controller)


def test_activity_snapshot_distinguishes_live_and_cache_playback(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(playback=False)
    scene.fluidsim.bake_status = "IDLE"
    scene.fluidsim.bake_progress = 0.0
    scene.fluidsim.bake_message = ""
    scene.fluidsim.live = True

    ready = scene_bridge.playback.activity_snapshot(
        scene, is_animation_playing=False)
    playing = scene_bridge.playback.activity_snapshot(
        scene, is_animation_playing=True)
    scene.fluidsim.cache_playback_enabled = True
    cached = scene_bridge.playback.activity_snapshot(
        scene, is_animation_playing=True)

    assert (ready.state, ready.label) == ("LIVE_READY", "Live Ready")
    assert (playing.state, playing.label) == ("LIVE_PLAYING", "Playing Live")
    assert (cached.state, cached.label) == ("CACHE_PLAYBACK", "Playing Cache")


def test_activity_snapshot_shows_terminal_bake_state_when_idle(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(playback=False)
    scene.fluidsim.live = False
    scene.fluidsim.bake_status = "COMPLETE"
    scene.fluidsim.bake_progress = 1.0
    scene.fluidsim.bake_message = "Production bake complete"

    snapshot = scene_bridge.playback.activity_snapshot(scene)

    assert snapshot.state == "COMPLETE"
    assert snapshot.label == "Bake Complete"
    assert snapshot.detail == "Production bake complete"
    assert snapshot.progress == 1.0

from types import SimpleNamespace

import numpy as np

from fluidSim_warp_blender51.sim.frame_output import HostFrame


class _Array:
    def __init__(self, value, calls, name):
        self.value = np.asarray(value)
        self.calls = calls
        self.name = name

    def __getitem__(self, item):
        return _Array(self.value[item], self.calls, self.name)

    def numpy(self):
        self.calls[self.name] = self.calls.get(self.name, 0) + 1
        return self.value


def _solver(calls):
    attrs = {name: _Array([1, 2], calls, name) for name in
             ("age", "life", "temperature", "viscosity", "density",
              "time_offset")}
    particles = SimpleNamespace(
        host_count=2,
        refresh_host_count=lambda: 2,
        position=_Array([[1, 2, 3], [4, 5, 6]], calls, "positions"),
        velocity=_Array([[.1, .2, .3], [.4, .5, .6]], calls, "velocities"),
        time_offset=_Array([0, .5], calls, "time_offset"),
        attr=lambda name: attrs[name],
    )
    whitewater = SimpleNamespace(
        host_count=2,
        refresh_host_count=lambda: 2,
        position=_Array([[1, 2, 3], [4, 5, 6]], calls, "whitewater"),
        velocity=_Array([[.1, .2, .3], [.4, .5, .6]], calls, "whitewater_velocity"),
        age=_Array([1, 2], calls, "whitewater_age"),
        life=_Array([3, 4], calls, "whitewater_life"),
        type=_Array([0, 1], calls, "whitewater_type"),
    )
    solver = SimpleNamespace(frame=4, particles=particles, whitewater=whitewater)
    solver.surface_mesh = lambda: calls.__setitem__(
        "surface", calls.get("surface", 0) + 1) or (
            np.zeros((1, 3), np.float32), np.zeros((1, 3), np.int32))
    return solver


def test_host_frame_reads_each_requested_output_once_and_nothing_eagerly():
    from fluidSim_warp_blender51.sim.frame_output import (
        host_output_operation_counts,
    )

    calls = {}
    host_output_operation_counts(reset=True)
    frame = HostFrame(_solver(calls))

    frame.positions
    frame.positions
    frame.velocities
    frame.surface_mesh
    frame.surface_mesh
    frame.particle_attr("age")
    frame.particle_attr("age")

    assert calls == {"positions": 1, "velocities": 1, "surface": 1, "age": 1}
    assert host_output_operation_counts() == {
        "host_frame_constructions": 1,
        "device_readbacks": 3,
        "surface_extractions": 1,
    }


def test_new_host_frame_rereads_requested_outputs():
    calls = {}
    solver = _solver(calls)
    HostFrame(solver).positions
    solver.frame += 1
    HostFrame(solver).positions
    assert calls == {"positions": 2}


def test_disabled_attribute_inventory_performs_no_host_readback():
    calls = {}
    frame = HostFrame(_solver(calls))

    assert frame.output_attributes(SimpleNamespace(), "particles") == {}
    assert calls == {}


def test_each_attribute_refreshes_stale_host_count_when_requested_first():
    calls = {}
    solver = _solver(calls)
    solver.particles.host_count = 0
    solver.whitewater.host_count = 0
    refreshes = {"particles": 0, "whitewater": 0}
    solver.particles.refresh_host_count = lambda: (
        refreshes.__setitem__("particles", refreshes["particles"] + 1) or 2)
    solver.whitewater.refresh_host_count = lambda: (
        refreshes.__setitem__("whitewater", refreshes["whitewater"] + 1) or 2)

    frame = HostFrame(solver)

    assert len(frame.velocities) == 2
    assert len(frame.particle_attr("age")) == 2
    assert len(frame.whitewater_attr("age")) == 2
    assert refreshes == {"particles": 1, "whitewater": 1}


def test_host_frame_owns_live_surface_publication_cadence():
    solver = _solver({})

    assert HostFrame(solver).should_publish_surface(4)
    solver.frame = 5
    assert not HostFrame(solver).should_publish_surface(4)
    solver.frame = 8
    assert HostFrame(solver).should_publish_surface(4)
    assert HostFrame(solver).should_publish_surface(1)


def test_solver_reset_invalidates_frame_zero_host_snapshot():
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    calls = {}
    solver = FluidSolver.__new__(FluidSolver)
    solver.frame = 0
    solver._host_frame_cache = None
    solver.settings = SimpleNamespace(
        spacetime_seed=0,
        spacetime_reference_mass=0.0,
        spacetime_enabled=False,
    )
    solver.particles = SimpleNamespace(
        clear=lambda: None, host_count=1, refresh_host_count=lambda: 1,
        position=_Array([[1, 2, 3]], calls, "positions"),
        optional_channels=(),
    )
    solver.whitewater = SimpleNamespace(clear=lambda: None)
    solver.churn = SimpleNamespace(zero=lambda: None)
    solver.grid = SimpleNamespace(
        active_voxel_count=0, begin_topology=lambda: None,
        finalize_topology=lambda: None)
    solver.surface = SimpleNamespace(reset=lambda: None)

    solver.host_frame().positions
    solver.reset()
    solver.host_frame().positions

    assert solver.frame == 0
    assert calls["positions"] == 2


def test_viewport_frame_module_does_not_require_blender_runtime(monkeypatch):
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def import_without_bpy(name, *args, **kwargs):
        if name == "bpy":
            raise ModuleNotFoundError("blocked by host-Python boundary test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_bpy)
    sys.modules.pop("fluidSim_warp_blender51.blender.output_publication", None)
    output_publication = importlib.import_module("fluidSim_warp_blender51.blender.output_publication")

    assert output_publication.prepare_live_payload.__name__ == "prepare_live_payload"


def test_live_payload_excludes_unselected_production_attributes():
    from fluidSim_warp_blender51.blender import output_publication

    class LiveHost:
        solver = SimpleNamespace(settings=SimpleNamespace(voxel_size=0.1))
        whitewater = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        surface_mesh = (
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
        )

        def positions_strided(self, _step):
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        def output_attributes(self, *_args, **_kwargs):
            raise AssertionError(
                "live viewport adaptation must not capture production attributes")

    fs = SimpleNamespace(
        show_particles=True,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        particle_attribute_mode="NONE",
        show_whitewater=True,
        whitewater_enabled=True,
        whitewater_display_scale=1.0,
        whitewater_attribute_mode="NONE",
        show_surface=True,
        surface_enabled=True,
        surface_attribute_mode="NONE",
    )

    payload = output_publication.prepare_live_payload(
        SimpleNamespace(fluidsim=fs), LiveHost())

    assert payload.surface_attributes == {}
    assert payload.particle_attributes == {}
    assert payload.whitewater_attributes == {}


def test_viewport_cache_and_export_share_one_frame(tmp_path):
    import sys
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.io import cache, export

    calls = {}
    solver = _solver(calls)
    solver.settings = SimpleNamespace(
        surface_enabled=True, voxel_size=.1, surface_resolution_mode=0,
        surface_resolution_multiplier=1., surface_update_cadence=1,
        surface_smooth_iterations=0)
    solver.surface = SimpleNamespace(metadata=lambda: {"voxel_size": .1})
    frames = {}
    solver.host_frame = lambda: frames.setdefault(solver.frame, HostFrame(solver))
    fs = SimpleNamespace(
        show_particles=True, particle_attribute_mode="SPEED",
        show_whitewater=False, whitewater_enabled=False,
        whitewater_attribute_mode="NONE", show_surface=True,
        surface_enabled=True, surface_attribute_mode="NONE")

    output_publication.prepare_live_payload(
        SimpleNamespace(fluidsim=fs), solver.host_frame())
    payload = cache._frame_payload(solver)
    export.export_frame(solver, str(tmp_path / "frame"))

    assert set(payload) == {
        "format_version", "scene_key", "settings_signature", "positions", "velocities", "whitewater",

        "surface_verts", "surface_tris", "frame", "voxel_size",
        "surface_voxel_size", "surface_resolution_multiplier",
        "surface_resolution_mode", "surface_update_cadence",
        "surface_smooth_iterations", "particle_age", "particle_life",
        "particle_temperature", "particle_viscosity", "particle_density",
        "particle_time_offset", "whitewater_velocities", "whitewater_age",
        "whitewater_life", "whitewater_required", "whitewater_available",
        "whitewater_admitted", "whitewater_dropped", "whitewater_policy",
        "whitewater_status", "whitewater_type", "attribute_schema_version",
        "attribute_inventory_json", "surface_attribute_transfer_method"}
    assert calls["positions"] == calls["velocities"] == calls["surface"] == 1
    for name in ("age", "life", "temperature", "viscosity", "density",
                 "time_offset", "whitewater", "whitewater_velocity",
                 "whitewater_age", "whitewater_life", "whitewater_type"):
        assert calls[name] == 1
    assert calls.get("unused", 0) == 0
    sys.modules.pop("fluidSim_warp_blender51.blender.output_publication", None)
    sys.modules.pop("mathutils.kdtree", None)


def test_live_update_acquires_one_host_frame_then_publishes_owned_payload(monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication

    calls = []
    host = SimpleNamespace(
        should_publish_surface=lambda cadence: calls.append(
            ("cadence", cadence)) or True)
    solver = SimpleNamespace(
        host_frame=lambda: calls.append("host_frame") or host)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        live_surface_publish_cadence=3))
    session = SimpleNamespace(active=True, solver=solver)
    payload = object()

    monkeypatch.setattr(output_publication, "_require_bpy", lambda: None)
    existing_surface = SimpleNamespace(
        type="MESH", data=SimpleNamespace(vertices=[1]))
    monkeypatch.setattr(
        output_publication, "output_object",
        lambda *_args: existing_surface)
    monkeypatch.setattr(
        output_publication, "prepare_live_payload",
        lambda scene_arg, host_arg, publish_surface: (
            calls.append(("live_prepare", scene_arg, host_arg, publish_surface))
            or payload))
    monkeypatch.setattr(
        output_publication, "publish_payload",
        lambda scene_arg, payload_arg: calls.append(
            ("publish", scene_arg, payload_arg)))

    output_publication.update_outputs(scene, session)

    assert calls == [
        "host_frame", ("cadence", 3),
        ("live_prepare", scene, host, True),
        ("publish", scene, payload),
    ]


def test_cache_update_prepares_cache_payload_without_live_readback(monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication

    calls = []
    scene = object()
    frame_data = {"frame": 4}
    payload = object()
    monkeypatch.setattr(output_publication, "_require_bpy", lambda: None)
    monkeypatch.setattr(
        output_publication, "prepare_live_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache publication entered live readback")))
    monkeypatch.setattr(
        output_publication, "prepare_cache_payload",
        lambda scene_arg, data_arg: calls.append(
            ("cache_prepare", scene_arg, data_arg)) or payload)
    monkeypatch.setattr(
        output_publication, "publish_payload",
        lambda scene_arg, payload_arg: calls.append(
            ("publish", scene_arg, payload_arg)))

    output_publication.update_outputs_from_cache_frame(scene, frame_data)

    assert calls == [
        ("cache_prepare", scene, frame_data),
        ("publish", scene, payload),
    ]


def test_publisher_only_path_poisons_actual_solver_warp_and_blender_entrypoints(
        monkeypatch):
    import warp as wp
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.sim import frame_output, solver, surface

    def forbidden(name):
        def fail(*_args, **_kwargs):
            raise AssertionError(f"publisher called forbidden entry point: {name}")
        return fail

    # Poison the real imported/module-global boundaries, not attributes added
    # to a duck-typed payload. A new publisher dependency fails at its source.
    for name in ("host_frame", "particle_positions", "particle_velocities",
                 "surface_mesh"):
        monkeypatch.setattr(solver.FluidSolver, name, forbidden(name))
    monkeypatch.setattr(frame_output.HostFrame, "__init__", forbidden("HostFrame"))
    monkeypatch.setattr(surface.SurfacePipeline, "mesh", forbidden("surface.mesh"))
    monkeypatch.setattr(
        surface.SurfaceExtractor, "extract", forbidden("surface.extract"))
    monkeypatch.setattr(wp.array, "numpy", forbidden("warp.array.numpy"))
    monkeypatch.setattr(wp, "synchronize", forbidden("warp.synchronize"))
    monkeypatch.setattr(
        wp, "synchronize_device", forbidden("warp.synchronize_device"))

    poisoned_ids = SimpleNamespace(
        new=forbidden("Blender ID creation"),
        remove=forbidden("Blender ID removal"), get=lambda _name: None)
    poisoned_links = SimpleNamespace(
        link=forbidden("Blender relation link"),
        unlink=forbidden("Blender relation unlink"))
    fake_bpy = SimpleNamespace(
        data=SimpleNamespace(
            collections=poisoned_ids, objects=poisoned_ids,
            meshes=poisoned_ids, pointclouds=poisoned_ids,
            materials=poisoned_ids),
        context=SimpleNamespace(
            evaluated_depsgraph_get=forbidden("depsgraph acquisition"),
            view_layer=SimpleNamespace(update=forbidden("depsgraph update"))),
        relation_entrypoints=poisoned_links,
    )
    monkeypatch.setattr(output_publication, "bpy", fake_bpy)
    monkeypatch.setattr(output_publication, "_require_bpy", lambda: fake_bpy)

    payload = output_publication.PublicationPayload(
        voxel_size=.1, publish_surface=True,
        surface_visible=False, surface_attribute_mode="NONE",
        surface_verts=np.zeros((0, 3), np.float32),
        surface_tris=np.zeros((0, 3), np.int32), surface_attributes={},
        particles_visible=False, particle_attribute_mode="NONE",
        particle_radius=.01, particles=np.zeros((0, 3), np.float32),
        particle_attributes={}, whitewater_visible=False,
        whitewater_output_mode="COMBINED",
        whitewater_attribute_mode="NONE", whitewater_radius=.01,
        whitewater=np.zeros((0, 3), np.float32), whitewater_attributes={})
    calls = []
    monkeypatch.setattr(
        output_publication, "output_object",
        lambda _scene, name: calls.append(("lookup", name)))
    monkeypatch.setattr(
        output_publication, "_clear_points",
        lambda _scene, name: calls.append(("clear", name)))
    for name in ("_swap_mesh", "_swap_points", "_swap_vertex_cloud",
                 "_replace_datablock", "_link_relation"):
        monkeypatch.setattr(output_publication, name, forbidden(name))

    output_publication.publish_payload(object(), payload)

    assert calls == [
        ("lookup", output_publication.SURFACE_NAME),
        ("clear", output_publication.PARTICLES_NAME),
        ("clear", output_publication.WHITEWATER_NAME),
        ("lookup", output_publication.WHITEWATER_NAME),
        ("clear", output_publication.FOAM_NAME),
        ("lookup", output_publication.FOAM_NAME),
        ("clear", output_publication.BUBBLES_NAME),
        ("lookup", output_publication.BUBBLES_NAME),
        ("clear", output_publication.SPRAY_NAME),
        ("lookup", output_publication.SPRAY_NAME),
    ]


def test_output_cost_counters_separate_all_four_phases(monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication

    fs = SimpleNamespace(
        show_surface=False, surface_enabled=False, surface_attribute_mode="NONE",
        show_particles=False, particle_attribute_mode="NONE",
        show_whitewater=False, whitewater_enabled=False,
        whitewater_attribute_mode="NONE", voxel_size=.1)
    scene = SimpleNamespace(fluidsim=fs)
    host = SimpleNamespace(solver=SimpleNamespace(
        settings=SimpleNamespace(voxel_size=.1)))

    output_publication.output_cost_counts(reset=True)
    live = output_publication.prepare_live_payload(scene, host)
    cache = output_publication.prepare_cache_payload(scene, {})
    monkeypatch.setattr(output_publication, "_require_bpy", lambda: None)
    monkeypatch.setattr(output_publication, "output_object", lambda *_args: None)
    monkeypatch.setattr(output_publication, "_clear_points", lambda *_args: None)
    output_publication.prepare_outputs(scene)
    output_publication.publish_payload(scene, live)

    counts = output_publication.output_cost_counts()
    assert {phase: values["invocations"] for phase, values in counts.items()} == {
        "live_adaptation": 1,
        "cache_adaptation": 1,
        "output_preparation": 1,
        "output_publication": 1,
    }
    assert counts["live_adaptation"]["host_frame_acquisitions"] == 0
    assert counts["live_adaptation"]["device_readbacks"] == 0
    assert counts["cache_adaptation"]["device_readbacks"] == 0
    assert counts["output_preparation"]["blender_id_creations"] == 0
    assert counts["output_publication"]["blender_id_creations"] == 0
    assert cache.publish_surface is True

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
    )
    solver = SimpleNamespace(frame=4, particles=particles, whitewater=whitewater)
    solver.surface_mesh = lambda: calls.__setitem__(
        "surface", calls.get("surface", 0) + 1) or (
            np.zeros((1, 3), np.float32), np.zeros((1, 3), np.int32))
    return solver


def test_host_frame_reads_each_requested_output_once_and_nothing_eagerly():
    calls = {}
    frame = HostFrame(_solver(calls))

    frame.positions
    frame.positions
    frame.velocities
    frame.surface_mesh
    frame.surface_mesh
    frame.particle_attr("age")
    frame.particle_attr("age")

    assert calls == {"positions": 1, "velocities": 1, "surface": 1, "age": 1}


def test_new_host_frame_rereads_requested_outputs():
    calls = {}
    solver = _solver(calls)
    HostFrame(solver).positions
    solver.frame += 1
    HostFrame(solver).positions
    assert calls == {"positions": 2}


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
    solver.particles = SimpleNamespace(
        clear=lambda: None, host_count=1, refresh_host_count=lambda: 1,
        position=_Array([[1, 2, 3]], calls, "positions"))
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
    sys.modules.pop("fluidSim_warp_blender51.blender.viewport", None)
    viewport = importlib.import_module("fluidSim_warp_blender51.blender.viewport")

    assert viewport._solver_frame.__name__ == "_solver_frame"


def test_viewport_cache_and_export_share_one_frame(tmp_path):
    import sys
    from fluidSim_warp_blender51.blender import viewport
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

    viewport._solver_frame(SimpleNamespace(fluidsim=fs), solver)
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
        "whitewater_life"}
    assert calls["positions"] == calls["velocities"] == calls["surface"] == 1
    for name in ("age", "life", "temperature", "viscosity", "density",
                 "time_offset", "whitewater", "whitewater_velocity",
                 "whitewater_age", "whitewater_life"):
        assert calls[name] == 1
    assert calls.get("unused", 0) == 0
    sys.modules.pop("fluidSim_warp_blender51.blender.viewport", None)
    sys.modules.pop("mathutils.kdtree", None)

"""Device-side display decimation and lazy Warp discovery contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _Array:
    """Records the row count each ``.numpy()`` actually pulls to the host."""

    def __init__(self, value, log, name):
        self.value = np.asarray(value)
        self.log = log
        self.name = name

    def __getitem__(self, item):
        return _Array(self.value[item], self.log, self.name)

    def numpy(self):
        self.log.append((self.name, len(self.value)))
        return self.value


def _solver(log, count):
    positions = np.column_stack(
        [np.arange(count, dtype=np.float32)] * 3)
    velocities = positions * 0.5
    attrs = {name: _Array(np.arange(count, dtype=np.float32), log, name)
             for name in ("age", "life", "temperature", "viscosity", "density",
                          "time_offset")}
    particles = SimpleNamespace(
        host_count=count,
        refresh_host_count=lambda: count,
        position=_Array(positions, log, "positions"),
        velocity=_Array(velocities, log, "velocities"),
        attr=lambda name: attrs[name],
    )
    whitewater = SimpleNamespace(
        host_count=0,
        refresh_host_count=lambda: 0,
        position=_Array(np.zeros((0, 3), np.float32), log, "whitewater"),
        velocity=_Array(np.zeros((0, 3), np.float32), log, "whitewater_velocity"),
        age=_Array(np.zeros(0, np.float32), log, "whitewater_age"),
        life=_Array(np.zeros(0, np.float32), log, "whitewater_life"),
    )
    solver = SimpleNamespace(frame=4, particles=particles, whitewater=whitewater)
    solver.settings = SimpleNamespace(
        surface_enabled=True, voxel_size=.1, surface_resolution_mode=0,
        surface_resolution_multiplier=1., surface_update_cadence=1,
        surface_smooth_iterations=0)
    solver.surface = SimpleNamespace(metadata=lambda: {"voxel_size": .1})
    solver.surface_mesh = lambda: (
        np.zeros((1, 3), np.float32), np.zeros((1, 3), np.int32))
    return solver


def test_display_fraction_strides_the_readback_but_not_the_cache_frame():
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.io import cache
    from fluidSim_warp_blender51.sim.frame_output import HostFrame

    count = 64
    log = []
    solver = _solver(log, count)
    frames = {}
    solver.host_frame = lambda: frames.setdefault(solver.frame, HostFrame(solver))
    fs = SimpleNamespace(
        show_particles=True, particle_attribute_mode="SPEED",
        particle_display_fraction=0.25,
        show_whitewater=False, whitewater_enabled=False,
        whitewater_attribute_mode="NONE", show_surface=True,
        surface_enabled=True, surface_attribute_mode="NONE")

    def reads():
        return [n for name, n in log if name == "positions"]

    frame = output_publication.prepare_live_payload(
        SimpleNamespace(fluidsim=fs), solver.host_frame())

    # Half one: the viewport read decimated before crossing the bus.
    assert len(frame.particles) == 16
    assert reads() == [16]
    assert set(frame.particle_attributes) == {"speed"}
    assert len(frame.particle_attributes["speed"]) == 16
    assert np.array_equal(frame.particles[:, 0], np.arange(0, 64, 4))

    payload = cache._frame_payload(solver)

    # Half two: the full-fidelity consumer in the same frame still gets
    # every particle; a shared cache key would have handed it the 16.
    assert len(payload["positions"]) == count
    assert reads() == [16, count]


def test_warp_discovery_never_mutates_path_without_explicit_opt_in(
        monkeypatch, tmp_path):
    from fluidSim_warp_blender51 import warp_discovery

    modules = tmp_path / "addons" / "modules"
    modules.mkdir(parents=True)
    monkeypatch.delenv("FLUIDSIM_ALLOW_EXTERNAL_WARP", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert warp_discovery.enable_developer_override(str(modules)) is False
    assert str(modules) not in sys.path

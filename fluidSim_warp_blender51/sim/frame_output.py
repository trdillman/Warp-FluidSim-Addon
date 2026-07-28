"""Lazy, per-solver-frame GPU-to-host output snapshot."""

from __future__ import annotations

import numpy as np


class HostFrame:
    """Materialize each requested solver output at most once."""

    def __init__(self, solver):
        self.solver = solver
        self.frame = int(getattr(solver, "frame", 0))
        self._values = {}

    def _get(self, key, load):
        if key not in self._values:
            self._values[key] = load()
        return self._values[key]

    def invalidate_surface(self):
        self._values.pop("surface_mesh", None)

    def should_publish_surface(self, cadence):
        cadence = max(1, int(cadence))
        return self.frame <= 1 or cadence <= 1 or self.frame % cadence == 0

    @property
    def positions(self):
        return self.positions_strided(1)

    def positions_strided(self, stride=1):
        """Read back every `stride`-th live particle, decimating on device.

        Keyed per stride so a decimated viewport read is never handed to the
        cache/export consumers, which need every particle from the same frame.
        """
        stride = max(1, int(stride))

        def load():
            if not hasattr(self.solver, "particles"):
                return self.solver.particle_positions()[::stride]
            n = self.solver.particles.refresh_host_count()
            return (self.solver.particles.position[:n:stride].numpy() if n else
                    np.zeros((0, 3), dtype=np.float32))
        return self._get(("positions", stride), load)

    @property
    def velocities(self):
        def load():
            if not hasattr(self.solver, "particles"):
                return self.solver.particle_velocities()
            n = self.solver.particles.host_count
            return (self.solver.particles.velocity[:n].numpy() if n else
                    np.zeros((0, 3), dtype=np.float32))
        return self._get("velocities", load)

    @property
    def whitewater(self):
        def load():
            if not hasattr(self.solver, "whitewater"):
                return self.solver.whitewater_positions()
            n = self.solver.whitewater.refresh_host_count()
            return (self.solver.whitewater.position[:n].numpy() if n else
                    np.zeros((0, 3), dtype=np.float32))
        return self._get("whitewater", load)

    @property
    def surface_mesh(self):
        return self._get("surface_mesh", self.solver.surface_mesh)

    def particle_attr(self, name):
        def load():
            try:
                n = self.solver.particles.host_count
                # Warp 1.12.1 can fail converting sliced array views to NumPy.
                return self.solver.particles.attr(name).numpy()[:n]
            except AttributeError:
                return np.zeros(0, dtype=np.float32)
        return self._get(("particle", name), load)

    def whitewater_attr(self, name):
        def load():
            try:
                n = self.solver.whitewater.host_count
                # Warp 1.12.1 cannot convert sliced vec3 views to NumPy.
                return getattr(self.solver.whitewater, name).numpy()[:n]
            except AttributeError:
                return np.zeros(0, dtype=np.float32)
        return self._get(("whitewater", name), load)

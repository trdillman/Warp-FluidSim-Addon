"""Lazy, per-solver-frame GPU-to-host output snapshot."""

from __future__ import annotations

from collections import Counter

import numpy as np


_HOST_OUTPUT_OPERATIONS = Counter()


def host_output_operation_counts(*, reset=False):
    """Count actual HostFrame construction/readback/extraction operations."""
    counts = {
        name: int(_HOST_OUTPUT_OPERATIONS[name])
        for name in (
            "host_frame_constructions", "device_readbacks",
            "surface_extractions",
        )
    }
    if reset:
        _HOST_OUTPUT_OPERATIONS.clear()
    return counts


class HostFrame:
    """Materialize each requested solver output at most once."""

    def __init__(self, solver):
        _HOST_OUTPUT_OPERATIONS["host_frame_constructions"] += 1
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
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return self.solver.particle_positions()[::stride]
            n = self.solver.particles.refresh_host_count()
            if n:
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return self.solver.particles.position[:n:stride].numpy()
            return np.zeros((0, 3), dtype=np.float32)
        return self._get(("positions", stride), load)

    @property
    def velocities(self):
        def load():
            if not hasattr(self.solver, "particles"):
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return self.solver.particle_velocities()
            n = self.solver.particles.host_count
            if n:
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return self.solver.particles.velocity[:n].numpy()
            return np.zeros((0, 3), dtype=np.float32)
        return self._get("velocities", load)

    @property
    def whitewater(self):
        def load():
            if not hasattr(self.solver, "whitewater"):
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return self.solver.whitewater_positions()
            n = self.solver.whitewater.refresh_host_count()
            if n:
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return self.solver.whitewater.position[:n].numpy()
            return np.zeros((0, 3), dtype=np.float32)
        return self._get("whitewater", load)

    @property
    def surface_mesh(self):
        def load():
            _HOST_OUTPUT_OPERATIONS["surface_extractions"] += 1
            return self.solver.surface_mesh()
        return self._get("surface_mesh", load)

    def particle_attr(self, name):
        def load():
            try:
                n = self.solver.particles.host_count
                # Warp 1.12.1 can fail converting sliced array views to NumPy.
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return self.solver.particles.attr(name).numpy()[:n]
            except AttributeError:
                return np.zeros(0, dtype=np.float32)
        return self._get(("particle", name), load)

    def whitewater_attr(self, name):
        def load():
            try:
                n = self.solver.whitewater.host_count
                # Warp 1.12.1 cannot convert sliced vec3 views to NumPy.
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return getattr(self.solver.whitewater, name).numpy()[:n]
            except AttributeError:
                return np.zeros(0, dtype=np.float32)
        return self._get(("whitewater", name), load)

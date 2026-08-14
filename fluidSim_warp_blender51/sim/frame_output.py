"""Lazy, per-solver-frame GPU-to-host output snapshot."""

from __future__ import annotations

from collections import Counter

import numpy as np
import warp as wp

from .attributes import (
    ATTRIBUTE_SPECS,
    enabled_attribute_keys,
    motion_from_velocity,
    nearest_sample_indices,
    speed_from_velocity,
    validate_attribute_values,
)
from .fields import sample_mac_velocity, world_to_voxel

wp.set_module_options({"enable_backward": False})

_HOST_OUTPUT_OPERATIONS = Counter()


@wp.kernel
def k_sample_particle_vorticity(
    positions: wp.array(dtype=wp.vec3),
    count: int,
    origin: wp.vec3,
    voxel_size: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    out_vorticity: wp.array(dtype=wp.vec3),
):
    """Sample curl of the projected MAC velocity at liquid particles."""
    tid = wp.tid()
    if tid >= count:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    ex = wp.vec3(1.0, 0.0, 0.0)
    ey = wp.vec3(0.0, 1.0, 0.0)
    ez = wp.vec3(0.0, 0.0, 1.0)
    vx0 = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, vp - ex)
    vx1 = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, vp + ex)
    vy0 = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, vp - ey)
    vy1 = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, vp + ey)
    vz0 = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, vp - ez)
    vz1 = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, vp + ez)
    inv_two_h = 0.5 / voxel_size
    out_vorticity[tid] = wp.vec3(
        (vy1[2] - vy0[2] - (vz1[1] - vz0[1])) * inv_two_h,
        (vz1[0] - vz0[0] - (vx1[2] - vx0[2])) * inv_two_h,
        (vx1[1] - vx0[1] - (vy1[0] - vy0[0])) * inv_two_h,
    )


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
        for key in tuple(self._values):
            if isinstance(key, tuple) and key and key[0] == "surface_attributes":
                self._values.pop(key, None)

    def should_publish_surface(self, cadence):
        cadence = max(1, int(cadence))
        return self.frame <= 1 or cadence <= 1 or self.frame % cadence == 0

    @property
    def positions(self):
        return self.positions_strided(1)

    def positions_strided(self, stride=1):
        """Read back every `stride`-th live particle, decimating on device."""
        stride = max(1, int(stride))

        def load():
            if not hasattr(self.solver, "particles"):
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.solver.particle_positions()[::stride], dtype=np.float32)
            n = self.solver.particles.refresh_host_count()
            if n:
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.solver.particles.position[:n:stride].numpy(), dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32)
        return self._get(("positions", stride), load)

    @property
    def velocities(self):
        def load():
            if not hasattr(self.solver, "particles"):
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.solver.particle_velocities(), dtype=np.float32)
            n = self.solver.particles.refresh_host_count()
            if n:
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.solver.particles.velocity.numpy()[:n], dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32)
        return self._get("velocities", load)

    @property
    def particle_vorticity(self):
        def load():
            particles = getattr(self.solver, "particles", None)
            grid = getattr(self.solver, "grid", None)
            vel = getattr(self.solver, "vel", None)
            if particles is None or grid is None or vel is None:
                return np.zeros_like(self.velocities)
            n = particles.refresh_host_count()
            if not n:
                return np.zeros((0, 3), dtype=np.float32)
            out = wp.empty(n, dtype=wp.vec3, device=self.solver.device)
            wp.launch(
                k_sample_particle_vorticity,
                dim=n,
                inputs=[
                    particles.position, n, grid.origin,
                    float(self.solver.settings.voxel_size),
                    vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww,
                    grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                ],
                outputs=[out],
                device=self.solver.device,
            )
            _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
            return np.ascontiguousarray(out.numpy(), dtype=np.float32)
        return self._get("particle_vorticity", load)

    @property
    def whitewater(self):
        def load():
            if not hasattr(self.solver, "whitewater"):
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.solver.whitewater_positions(), dtype=np.float32)
            n = self.solver.whitewater.refresh_host_count()
            if n:
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.solver.whitewater.position.numpy()[:n], dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32)
        return self._get("whitewater", load)

    @property
    def surface_mesh(self):
        def load():
            _HOST_OUTPUT_OPERATIONS["surface_extractions"] += 1
            verts, tris = self.solver.surface_mesh()
            return (np.ascontiguousarray(verts, dtype=np.float32),
                    np.ascontiguousarray(tris, dtype=np.int32))
        return self._get("surface_mesh", load)

    def particle_attr(self, name):
        def load():
            try:
                n = self.solver.particles.refresh_host_count()
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.solver.particles.attr(name).numpy()[:n])
            except (AttributeError, KeyError):
                return np.zeros(0, dtype=np.float32)
        return self._get(("particle", name), load)

    def whitewater_attr(self, name):
        def load():
            try:
                n = self.solver.whitewater.refresh_host_count()
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    getattr(self.solver.whitewater, name).numpy()[:n])
            except AttributeError:
                return np.zeros(0, dtype=np.float32)
        return self._get(("whitewater", name), load)

    def particle_attribute(self, key: str, *, stride: int = 1):
        """Return one authoritative liquid-particle output channel."""
        stride = max(1, int(stride))
        key = str(key)
        n = len(self.positions)
        if key == "velocity":
            values = self.velocities
        elif key == "speed":
            values = speed_from_velocity(self.velocities)
        elif key == "vorticity":
            values = self.particle_vorticity
        elif key == "age":
            values = self.particle_attr("age")
        elif key == "lifetime":
            values = self.particle_attr("life")
        elif key in {"color", "source_id", "uid", "viscosity", "density"}:
            values = self.particle_attr(key)
            if key in {"color", "source_id", "uid"} and len(values) != n:
                raise RuntimeError(
                    f"enabled particle attribute {key!r} is not allocated")
        elif key == "motion":
            values = motion_from_velocity(
                self.velocities,
                frame_rate=float(self.solver.settings.frame_rate),
                time_scale=float(self.solver.settings.time_scale),
            )
        else:
            raise ValueError(f"unsupported particle attribute {key!r}")
        return np.ascontiguousarray(values[::stride])

    def whitewater_attribute(self, key: str):
        key = str(key)
        if key == "velocity":
            values = self.whitewater_attr("velocity")
        elif key == "speed":
            values = speed_from_velocity(self.whitewater_attr("velocity"))
        elif key == "age":
            values = self.whitewater_attr("age")
        elif key == "lifetime":
            values = self.whitewater_attr("life")
        else:
            raise ValueError(f"unsupported whitewater attribute {key!r}")
        return np.ascontiguousarray(values)

    def output_attributes(
        self,
        settings,
        output_kind: str,
        *,
        positions=None,
        stride: int = 1,
    ) -> dict[str, np.ndarray]:
        """Return every enabled channel for one output with strict alignment."""
        output_kind = str(output_kind).lower()
        keys = enabled_attribute_keys(settings, output_kind)
        if not keys:
            return {}
        if output_kind == "particles":
            count = len(self.positions_strided(stride))
            values = {
                key: self.particle_attribute(key, stride=stride)
                for key in keys
            }
        elif output_kind == "whitewater":
            count = len(self.whitewater)
            values = {key: self.whitewater_attribute(key) for key in keys}
        elif output_kind == "surface":
            targets = (self.surface_mesh[0] if positions is None
                       else np.asarray(positions, dtype=np.float32).reshape((-1, 3)))
            count = len(targets)
            source_positions = self.positions
            if len(source_positions):
                nearest = nearest_sample_indices(
                    targets,
                    source_positions,
                    cell_size=max(
                        float(self.solver.settings.voxel_size) * 2.0,
                        1.0e-5,
                    ),
                )
                values = {
                    key: np.ascontiguousarray(
                        self.particle_attribute(key)[nearest]
                    )
                    for key in keys
                }
            else:
                values = {
                    key: np.zeros(
                        (count,) if ATTRIBUTE_SPECS[key].components == 1
                        else (count, ATTRIBUTE_SPECS[key].components),
                        dtype=ATTRIBUTE_SPECS[key].numpy_dtype,
                    )
                    for key in keys
                }
        else:
            raise ValueError(f"unknown output kind {output_kind!r}")
        return {
            key: validate_attribute_values(ATTRIBUTE_SPECS[key], value, count)
            for key, value in values.items()
        }

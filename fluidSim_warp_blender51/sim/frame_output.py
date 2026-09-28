"""Lazy, per-solver-frame GPU-to-host output snapshot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
from .surface_attribute_transfer import SurfaceAttributeTransfer

wp.set_module_options({"enable_backward": False})

_HOST_OUTPUT_OPERATIONS = Counter()


def _live_numpy(device_array, live_count, stride=1):
    """Transfer only the live device prefix to NumPy, decimated on device."""
    if live_count == 0:
        dtype = device_array.dtype
        scalar_dtype = getattr(dtype, "_wp_scalar_type_", dtype)
        if getattr(scalar_dtype, "__module__", "").startswith("warp."):
            numpy_dtype = wp.dtype_to_numpy(scalar_dtype)
        else:
            numpy_dtype = np.dtype(scalar_dtype)
        array_shape = tuple(int(axis) for axis in device_array.shape)
        element_shape = tuple(int(axis) for axis in getattr(dtype, "_shape_", ()))
        return np.empty((0, *array_shape[1:], *element_shape), dtype=numpy_dtype)
    return device_array[:live_count:max(1, int(stride))].numpy()


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


@wp.kernel
def k_sample_surface_velocity(
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
    out_velocity: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= count:
        return
    voxel_position = world_to_voxel(
        positions[tid], origin, voxel_size)
    out_velocity[tid] = sample_mac_velocity(
        u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim,
        voxel_position,
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
        self._surface_transfer = getattr(
            solver, "_surface_attribute_transfer", None
        )
        if self._surface_transfer is not None:
            self._surface_transfer.invalidate()

    def _get(self, key, load):
        if key not in self._values:
            self._values[key] = load()
        return self._values[key]

    def invalidate_surface(self):
        self._values.pop("surface_mesh", None)
        self._values.pop("surface_velocity", None)
        for key in tuple(self._values):
            if (
                isinstance(key, tuple)
                and key
                and key[0] in {
                    "surface_attributes",
                    "surface_attribute_targets_device",
                }
            ):
                self._values.pop(key, None)
        if self._surface_transfer is not None:
            self._surface_transfer.invalidate()

    def surface_attribute_transfer_counts(self):
        """Expose opt-in GPU transfer launches/readbacks to the profiler."""
        if self._surface_transfer is None:
            return {
                "sample_gathers": 0,
                "hash_grid_builds": 0,
                "local_lookups": 0,
                "fallback_launches": 0,
                "gathers": 0,
                "readbacks": 0,
            }
        return self._surface_transfer.operation_counts()

    def _live_count(self, owner_name):
        key = (owner_name, "count")
        return self._get(
            key,
            lambda: getattr(self.solver, owner_name).refresh_host_count(),
        )

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
            n = self._live_count("particles")
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
            n = self._live_count("particles")
            if n:
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    _live_numpy(self.solver.particles.velocity, n),
                    dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32)
        return self._get("velocities", load)

    @property
    def particle_vorticity(self):
        def load():
            particles = getattr(self.solver, "particles", None)
            if particles is not None and isinstance(
                getattr(particles, "position", None), wp.array
            ):
                n = self._live_count("particles")
                if not n:
                    return np.zeros((0, 3), dtype=np.float32)
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.particle_vorticity_device[:n].numpy(),
                    dtype=np.float32,
                )
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
    def particle_vorticity_device(self):
        """Return the cached live-particle vorticity device buffer."""
        def load():
            particles = getattr(self.solver, "particles", None)
            if particles is None or not isinstance(
                getattr(particles, "position", None), wp.array
            ):
                raise RuntimeError("device particle vorticity is unavailable")
            n = self._live_count("particles")
            grid = getattr(self.solver, "grid", None)
            vel = getattr(self.solver, "vel", None)
            if not n or grid is None or vel is None:
                return wp.zeros(n, dtype=wp.vec3, device=self.solver.device)
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
            return out

        return self._get("particle_vorticity_device", load)

    @property
    def whitewater(self):
        return self.whitewater_strided(1)

    def whitewater_strided(self, stride=1):
        """Read back every `stride`-th live whitewater point, decimating on device."""
        stride = max(1, int(stride))

        def load():
            if not hasattr(self.solver, "whitewater"):
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    self.solver.whitewater_positions()[::stride], dtype=np.float32)
            n = self._live_count("whitewater")
            if n:
                _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    _live_numpy(self.solver.whitewater.position, n, stride=stride),
                    dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32)
        return self._get(("whitewater", stride), load)

    @property
    def surface_mesh(self):
        def load():
            _HOST_OUTPUT_OPERATIONS["surface_extractions"] += 1
            verts, tris = self.solver.surface_mesh()
            return (np.ascontiguousarray(verts, dtype=np.float32),
                    np.ascontiguousarray(tris, dtype=np.int32))
        return self._get("surface_mesh", load)

    def surface_velocity(self, positions):
        targets = np.asarray(
            positions, dtype=np.float32).reshape((-1, 3))

        def load():
            if not len(targets):
                return np.zeros((0, 3), dtype=np.float32)
            grid = getattr(self.solver, "grid", None)
            velocity = getattr(self.solver, "vel", None)
            device = getattr(self.solver, "device", None)
            if grid is None or velocity is None or device is None:
                nearest = nearest_sample_indices(
                    targets,
                    self.positions,
                    cell_size=max(
                        float(self.solver.settings.voxel_size) * 2.0,
                        1.0e-5,
                    ),
                )
                return np.ascontiguousarray(self.velocities[nearest])
            device_targets = wp.array(
                targets, dtype=wp.vec3, device=device)
            out = wp.empty(len(targets), dtype=wp.vec3, device=device)
            wp.launch(
                k_sample_surface_velocity,
                dim=len(targets),
                inputs=[
                    device_targets, len(targets), grid.origin,
                    float(self.solver.settings.voxel_size),
                    velocity.u, velocity.v, velocity.w,
                    velocity.wu, velocity.wv, velocity.ww,
                    grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                ],
                outputs=[out],
                device=device,
            )
            _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
            return np.ascontiguousarray(out.numpy(), dtype=np.float32)

        return self._get("surface_velocity", load)

    def particle_attr(self, name):
        def load():
            try:
                n = self._live_count("particles")
                if n:
                    _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    _live_numpy(self.solver.particles.attr(name), n))
            except (AttributeError, KeyError):
                return np.zeros(0, dtype=np.float32)
        return self._get(("particle", name), load)

    def whitewater_attr(self, name):
        def load():
            try:
                n = self._live_count("whitewater")
                if n:
                    _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                return np.ascontiguousarray(
                    _live_numpy(getattr(self.solver.whitewater, name), n))
            except AttributeError:
                if name == "type":
                    from .whitewater_types import LEGACY_UNCLASSIFIED

                    return np.full(
                        len(self.whitewater), LEGACY_UNCLASSIFIED,
                        dtype=ATTRIBUTE_SPECS["type"].numpy_dtype)
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

    def particle_attribute_device(self, key: str):
        """Return one nearest-transfer particle channel without host readback."""
        particles = getattr(self.solver, "particles", None)
        if particles is None or not isinstance(
            getattr(particles, "position", None), wp.array
        ):
            raise RuntimeError("device particle attributes are unavailable")
        key = str(key)
        if key == "vorticity":
            return self.particle_vorticity_device
        attr_name = "life" if key == "lifetime" else key
        if key not in {
            "age", "lifetime", "color", "source_id", "viscosity", "density"
        }:
            raise ValueError(f"unsupported device particle attribute {key!r}")
        try:
            return particles.attr(attr_name)
        except KeyError as exc:
            if key in {"color", "source_id"}:
                raise RuntimeError(
                    f"enabled particle attribute {key!r} is not allocated"
                ) from exc
            raise

    def _surface_targets_device(self, source, targets):
        array_interface = targets.__array_interface__
        key = (
            "surface_attribute_targets_device",
            id(source),
            tuple(targets.shape),
            int(array_interface["data"][0]),
        )
        cached = self._values.get(key)
        if cached is not None and cached[0] is source:
            return cached[1]
        device_targets = wp.array(
            targets, dtype=wp.vec3, device=self.solver.device
        )
        # Retain the source object so Python cannot recycle its identity while
        # this HostFrame cache entry is live.
        self._values[key] = (source, device_targets)
        return device_targets

    def _gpu_surface_transfer(self):
        particles = getattr(self.solver, "particles", None)
        if (
            particles is None
            or not isinstance(getattr(particles, "position", None), wp.array)
            or getattr(self.solver, "device", None) is None
        ):
            return None
        if self._surface_transfer is None:
            self._surface_transfer = SurfaceAttributeTransfer(
                device=self.solver.device
            )
            self.solver._surface_attribute_transfer = self._surface_transfer
        return self._surface_transfer

    def whitewater_attribute(self, key: str, *, stride: int = 1):
        stride = max(1, int(stride))
        key = str(key)
        if key == "velocity":
            values = self.whitewater_attr("velocity")
        elif key == "speed":
            values = speed_from_velocity(self.whitewater_attr("velocity"))
        elif key == "age":
            values = self.whitewater_attr("age")
        elif key == "lifetime":
            values = self.whitewater_attr("life")
        elif key == "type":
            values = self.whitewater_attr("type")
        else:
            raise ValueError(f"unsupported whitewater attribute {key!r}")
        return np.ascontiguousarray(values[::stride])

    def _surface_output_attributes(
        self,
        keys,
        *,
        positions=None,
    ) -> dict[str, np.ndarray]:
        """Return enabled surface channels with nearest-transfer caching."""
        target_source = self.surface_mesh[0] if positions is None else positions
        targets = np.asarray(
            target_source, dtype=np.float32
        ).reshape((-1, 3))
        count = len(targets)
        velocity_keys = frozenset({"velocity", "speed", "motion"})
        nearest_keys = [key for key in keys if key not in velocity_keys]
        surface_cache_key = None
        if nearest_keys:
            target_pointer = int(targets.__array_interface__["data"][0])
            surface_cache_key = (
                "surface_attributes",
                tuple(keys),
                id(target_source),
                tuple(targets.shape),
                target_pointer,
                float(getattr(self.solver.settings, "frame_rate", 0.0)),
                float(getattr(self.solver.settings, "time_scale", 0.0)),
            )
            cached = self._values.get(surface_cache_key)
            if cached is not None and cached[0] is target_source:
                return cached[1]
        surface_velocity = (
            self.surface_velocity(targets)
            if velocity_keys.intersection(keys) else None)
        nearest = None
        source_positions = None
        gpu_transfer = None
        if count and nearest_keys:
            gpu_transfer = self._gpu_surface_transfer()
            if gpu_transfer is not None:
                if not np.all(np.isfinite(targets)):
                    raise ValueError(
                        "nearest-transfer positions must be finite"
                    )
                live_count = self._live_count("particles")
                if live_count:
                    gpu_transfer.prepare(
                        self.solver.particles,
                        live_count,
                        max(
                            float(self.solver.settings.voxel_size) * 2.0,
                            1.0e-5,
                        ),
                    )
                    gpu_transfer.nearest_indices(
                        self._surface_targets_device(
                            target_source, targets
                        ),
                        count,
                    )
                else:
                    gpu_transfer = None
        values = {}
        for key in keys:
            if key == "velocity":
                value = surface_velocity
            elif key == "speed":
                value = speed_from_velocity(surface_velocity)
            elif key == "motion":
                value = motion_from_velocity(
                    surface_velocity,
                    frame_rate=float(self.solver.settings.frame_rate),
                    time_scale=float(self.solver.settings.time_scale),
                )
            else:
                if gpu_transfer is not None:
                    gpu_transfer.gather(
                        key,
                        self.particle_attribute_device(key),
                        count,
                    )
                    value = gpu_transfer.readback(key, count)
                    _HOST_OUTPUT_OPERATIONS["device_readbacks"] += 1
                elif source_positions is None:
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
                if gpu_transfer is None:
                    if nearest is None:
                        spec = ATTRIBUTE_SPECS[key]
                        value = np.zeros(
                            (count,) if spec.components == 1
                            else (count, spec.components),
                            dtype=spec.numpy_dtype,
                        )
                    else:
                        value = self.particle_attribute(key)[nearest]
            values[key] = np.ascontiguousarray(value)
        result = {
            key: validate_attribute_values(ATTRIBUTE_SPECS[key], value, count)
            for key, value in values.items()
        }
        if surface_cache_key is not None:
            self._values[surface_cache_key] = (target_source, result)
        return result

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
            count = len(self.whitewater_strided(stride))
            values = {
                key: self.whitewater_attribute(key, stride=stride)
                for key in keys
            }
        elif output_kind == "surface":
            return self._surface_output_attributes(keys, positions=positions)
        else:
            raise ValueError(f"unknown output kind {output_kind!r}")
        result = {
            key: validate_attribute_values(ATTRIBUTE_SPECS[key], value, count)
            for key, value in values.items()
        }
        return result


@dataclass(frozen=True)
class CacheFramePayload:
    """Fully materialized simulation output accepted by cache persistence."""

    frame: int
    settings_signature: str
    positions: np.ndarray
    velocities: np.ndarray
    whitewater: np.ndarray
    surface_verts: np.ndarray
    surface_tris: np.ndarray
    voxel_size: float
    surface_voxel_size: float
    surface_resolution_multiplier: float
    surface_resolution_mode: int
    surface_update_cadence: int
    surface_smooth_iterations: int
    particle_attributes: Mapping[str, np.ndarray]
    whitewater_attributes: Mapping[str, np.ndarray]
    output_attributes: Mapping[str, Mapping[str, np.ndarray]]
    attribute_inventory: Mapping[str, tuple[str, ...]]
    whitewater_required: int
    whitewater_available: int
    whitewater_admitted: int
    whitewater_dropped: int
    whitewater_policy: str
    whitewater_status: str

    @classmethod
    def capture(cls, solver, *, settings_signature: str):
        """Materialize one solver frame at the simulation/output boundary."""
        host = solver.host_frame() if hasattr(solver, "host_frame") else HostFrame(solver)
        settings = solver.settings
        empty_verts = np.zeros((0, 3), np.float32)
        empty_tris = np.zeros((0, 3), np.int32)
        verts, tris = (
            host.surface_mesh
            if bool(getattr(settings, "surface_enabled", False))
            else (empty_verts, empty_tris)
        )
        surface_metadata = (
            solver.surface.metadata()
            if hasattr(solver, "surface")
            else {"voxel_size": getattr(settings, "voxel_size", 0.05)}
        )
        output_attributes = {
            output_kind: MappingProxyType(host.output_attributes(
                settings, output_kind, positions=positions
            ))
            for output_kind, positions in (
                ("surface", verts),
                ("particles", None),
                ("whitewater", None),
            )
        }
        stats = getattr(solver, "stats", None)
        return cls(
            frame=int(getattr(solver, "frame", 0)),
            settings_signature=str(settings_signature),
            positions=host.positions,
            velocities=host.velocities,
            whitewater=host.whitewater,
            surface_verts=verts,
            surface_tris=tris,
            voxel_size=float(getattr(settings, "voxel_size", 0.05)),
            surface_voxel_size=float(surface_metadata["voxel_size"]),
            surface_resolution_multiplier=float(getattr(
                settings, "surface_resolution_multiplier", 2.0)),
            surface_resolution_mode=int(getattr(
                settings, "surface_resolution_mode", 0)),
            surface_update_cadence=int(getattr(
                settings, "surface_update_cadence", 1)),
            surface_smooth_iterations=int(getattr(
                settings, "surface_smooth_iterations", 1)),
            particle_attributes=MappingProxyType({
                name: host.particle_attr(name)
                for name in (
                    "age", "life", "temperature", "viscosity", "density",
                    "time_offset",
                )
            }),
            whitewater_attributes=MappingProxyType({
                name: host.whitewater_attr(name)
                for name in ("velocity", "age", "life", "type")
            }),
            output_attributes=MappingProxyType(output_attributes),
            attribute_inventory=MappingProxyType({
                output_kind: enabled_attribute_keys(settings, output_kind)
                for output_kind in ("particles", "whitewater", "surface")
            }),
            whitewater_required=int(getattr(stats, "whitewater_required", 0)),
            whitewater_available=int(getattr(stats, "whitewater_available", 0)),
            whitewater_admitted=int(getattr(stats, "whitewater_admitted", 0)),
            whitewater_dropped=int(getattr(stats, "whitewater_dropped", 0)),
            whitewater_policy=str(getattr(
                stats, "whitewater_policy", "FIXED_CAPACITY_DETERMINISTIC_DROPS")),
            whitewater_status=str(getattr(stats, "whitewater_status", "OK")),
        )

"""GPU-resident mixing of the existing persistent particle color channel."""

from __future__ import annotations

import math

import warp as wp

from .fields import voxel_flat_index, voxel_floor, world_to_voxel


_EPSILON = 1.0e-12


@wp.kernel
def k_accumulate_particle_color(
    positions: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec4),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    weight: wp.array(dtype=float),
    color_mass: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    q = world_to_voxel(positions[tid], origin, voxel_size) - wp.vec3(0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    color = colors[tid]
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                flat = voxel_flat_index(
                    leaf_table, voxels_dim, leaves_dim,
                    wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz),
                )
                if flat >= 0:
                    sample_weight = (
                        wp.where(dx == 1, f[0], 1.0 - f[0])
                        * wp.where(dy == 1, f[1], 1.0 - f[1])
                        * wp.where(dz == 1, f[2], 1.0 - f[2])
                    )
                    wp.atomic_add(weight, flat, sample_weight)
                    wp.atomic_add(color_mass, flat, sample_weight * color)


@wp.kernel
def k_blend_particle_color(
    positions: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec4),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    weight: wp.array(dtype=float),
    color_mass: wp.array(dtype=wp.vec4),
    blend: float,
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    q = world_to_voxel(positions[tid], origin, voxel_size) - wp.vec3(0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    total_weight = float(0.0)
    total_color = wp.vec4(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                flat = voxel_flat_index(
                    leaf_table, voxels_dim, leaves_dim,
                    wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz),
                )
                if flat >= 0 and weight[flat] > _EPSILON:
                    sample_weight = (
                        wp.where(dx == 1, f[0], 1.0 - f[0])
                        * wp.where(dy == 1, f[1], 1.0 - f[1])
                        * wp.where(dz == 1, f[2], 1.0 - f[2])
                    )
                    total_weight += sample_weight
                    total_color += sample_weight * (color_mass[flat] / weight[flat])
    if total_weight > _EPSILON:
        local = total_color / total_weight
        current = colors[tid]
        mixed = current + blend * (local - current)
        colors[tid] = wp.vec4(
            wp.clamp(mixed[0], 0.0, 1.0),
            wp.clamp(mixed[1], 0.0, 1.0),
            wp.clamp(mixed[2], 0.0, 1.0),
            wp.clamp(mixed[3], 0.0, 1.0),
        )


class ColorMixTransport:
    """Two-launch local color exchange using the solver's sparse leaf table."""

    def __init__(self, grid, device):
        self.weight = wp.zeros(grid.voxel_capacity, dtype=float, device=device)
        self.color_mass = wp.zeros(
            grid.voxel_capacity, dtype=wp.vec4, device=device)
        self.device = device

    def mix(self, particles, grid, dt: float, rate: float, active_voxels: int):
        dt = max(0.0, float(dt))
        rate = max(0.0, float(rate))
        if dt == 0.0 or rate == 0.0 or particles.host_count <= 0:
            return
        blend = max(0.0, min(1.0, -math.expm1(-rate * dt)))
        if blend == 0.0:
            return
        voxel_count = max(1, min(int(active_voxels), len(self.weight)))
        self.weight[:voxel_count].zero_()
        self.color_mass[:voxel_count].zero_()
        particle_count = max(1, min(particles.capacity, particles.host_count))
        common = [
            particles.position,
            particles.attr("color"),
            particles.count,
            grid.leaf_table,
            grid.voxels_dim,
            grid.leaves_dim,
            grid.origin,
            grid.voxel_size,
            self.weight,
            self.color_mass,
        ]
        wp.launch(
            k_accumulate_particle_color,
            dim=particle_count,
            inputs=common,
            device=self.device,
        )
        wp.launch(
            k_blend_particle_color,
            dim=particle_count,
            inputs=[*common, blend],
            device=self.device,
        )

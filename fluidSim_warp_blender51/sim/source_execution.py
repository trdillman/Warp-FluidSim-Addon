"""Compiled source execution phases used by the transaction facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

from . import constants as C
from .attributes import normalize_color, stable_source_id
from .enums import (
    SourceEmissionMode,
    SourceEvaluationCadence,
    SourceLifecycle,
    SourceOperation,
    SourceShapeForceType,
    SourceShapeSourceMode,
    SourceShapeType,
)
from .source_contracts import (
    DirectInjectionPacket,
    InjectionResult,
    NormalizedEmissionPlan,
    SourceApplyResult,
    SourceShape,
    SourceShapeDesc,
)
from .source_kernels import (
    k_add_emission_demand,
    k_admit_emission_batch,
    k_apply_compiled_source,
    k_apply_compiled_staggered_source,
    k_assign_emitted_uids,
    k_count_compiled_source_particles,
    k_count_mesh_source_particles,
    k_count_source_shapes_particles,
    k_emit_compiled_source_particles,
    k_mesh_source_field,
    k_mesh_source_particles,
    k_reserve_owned_point_slots,
    k_reserve_scanned_slots,
    k_source_height,
    k_source_owned_points,
    k_source_resolve_type,
    k_source_shape_forces,
    k_source_shapes,
    k_source_shapes_particle_demand,
    k_source_shapes_particles,
)


@dataclass(frozen=True)
class SourceRuntime:
    """Concrete simulation dependencies required by source execution."""

    device: str
    grid: object
    particles: object
    settings: object
    voxel_type: object
    collision_sdf: object
    collision_vel: object
    vel: object
    temperature: object
    viscosity_field: object
    density: object
    voxel_counts: object
    pressure_solver: object
    active_voxel_count: Callable[[], int]
    current_frame: Callable[[], int]
    ensure_particle_capacity: Callable[[int], object] | None
    invalidate_host_cache: Callable[[], None]
    invalidate_particle_surface: Callable[[], None]

    @property
    def _nvox(self) -> int:
        return max(1, int(self.active_voxel_count()))

    @property
    def frame(self) -> int:
        return int(self.current_frame())


class CompiledSourceExecution:
    def apply_compiled(self, batch, dt, *, transaction_ready=False, finalize=True):
        """Apply compiled operations in fixed precedence/semantic order."""
        from .source_regions import (
            liquid_coverage_emission_enabled,
            normalized_liquid_target,
        )

        if not transaction_ready:
            self._begin_emission_transaction(batch, include_legacy=False)
        owner, grid = self.runtime, self.runtime.grid
        if self._compiled_empty_scalar is None:
            self._compiled_empty_scalar = wp.empty(0, dtype=float, device=self.device)
            self._compiled_empty_vector = wp.empty(0, dtype=wp.vec3, device=self.device)
        empty_scalar = self._compiled_empty_scalar
        empty_vector = self._compiled_empty_vector
        pressure = owner.pressure_solver.pressure
        particles = owner.particles
        order = {
            SourceOperation.COLLISION: 0,
            SourceOperation.SINK: 1,
            SourceOperation.ACTIVATION: 2,
            SourceOperation.LIQUID: 3,
            SourceOperation.PRESSURE: 4,
            SourceOperation.SCALAR_FIELD: 5,
            SourceOperation.VECTOR_FIELD: 6,
            SourceOperation.STAGGERED_VELOCITY: 7,
            SourceOperation.FORCE: 8,
        }
        for compiled in sorted(batch.sources, key=lambda value: order[value.plan.operation]):
            plan, region = compiled.plan, compiled.region
            if any(item.severity == "ERROR" for item in region.diagnostics):
                continue
            stable_id = str(plan.stable_id)
            if (
                plan.lifecycle in {SourceLifecycle.ONCE, SourceLifecycle.DIRECT_APPEND}
                and stable_id in self._initial_fired
            ):
                continue
            payload = plan.payload
            if plan.operation == SourceOperation.LIQUID:
                if plan.lifecycle == SourceLifecycle.DIRECT_APPEND:
                    if region.points is None or not len(region.points.positions):
                        continue
                elif not liquid_coverage_emission_enabled(plan):
                    continue
            # Liquid coverage is materialized by the particle kernel and the
            # immediate re-index/P2G path. Avoid a redundant field-stamp launch;
            # particle indexing truthfully owns Fluid/Active classification.
            if plan.operation in {
                SourceOperation.COLLISION,
                SourceOperation.SINK,
                SourceOperation.ACTIVATION,
            }:
                wp.launch(
                    k_apply_compiled_source,
                    dim=max(1, region.leaf_count * C.LEAF_VOLUME),
                    inputs=[
                        int(plan.operation),
                        region.device_coverage,
                        (
                            region.device_scalar
                            if region.device_scalar is not None
                            else empty_scalar
                        ),
                        (
                            region.device_vector
                            if region.device_vector is not None
                            else empty_vector
                        ),
                        region.device_leaf_coords,
                        region.leaf_count,
                        grid.leaf_table,
                        grid.voxels_dim,
                        grid.leaves_dim,
                        float(
                            payload.pressure
                            if plan.operation == SourceOperation.PRESSURE
                            else payload.scalar
                        ),
                        wp.vec3(
                            *(
                                payload.vector
                                if plan.operation
                                in {
                                    SourceOperation.VECTOR_FIELD,
                                    SourceOperation.STAGGERED_VELOCITY,
                                }
                                else payload.velocity
                            )
                        ),
                        int(plan.blend.mode),
                        float(plan.blend.amount),
                        float(dt),
                    ],
                    outputs=[
                        owner.voxel_type.data,
                        pressure,
                        owner.temperature.data,
                        owner.viscosity_field.data,
                        owner.density.data,
                        owner.vel.u,
                        owner.vel.v,
                        owner.vel.w,
                    ],
                    device=self.device,
                )
            if (
                plan.operation == SourceOperation.LIQUID
                and region.points is not None
                and plan.lifecycle == SourceLifecycle.DIRECT_APPEND
            ):
                point_values = region.points
                count = len(point_values.positions)
                wp.launch(
                    k_reserve_owned_point_slots,
                    dim=1,
                    inputs=[count, particles.capacity, self._emission_state],
                    outputs=[
                        particles.count,
                        self._point_slot_base,
                        self._point_slot_limit,
                    ],
                    device=self.device,
                )
                wp.launch(
                    k_source_owned_points,
                    dim=max(1, count),
                    inputs=[
                        point_values.device_positions,
                        (
                            point_values.device_velocities
                            if point_values.device_velocities is not None
                            else empty_vector
                        ),
                        (
                            point_values.device_temperature
                            if point_values.device_temperature is not None
                            else empty_scalar
                        ),
                        (
                            point_values.device_viscosity
                            if point_values.device_viscosity is not None
                            else empty_scalar
                        ),
                        (
                            point_values.device_density
                            if point_values.device_density is not None
                            else empty_scalar
                        ),
                        (
                            point_values.device_lifespan
                            if point_values.device_lifespan is not None
                            else empty_scalar
                        ),
                        count,
                        1 if point_values.device_velocities is not None else 0,
                        1 if point_values.device_temperature is not None else 0,
                        1 if point_values.device_viscosity is not None else 0,
                        1 if point_values.device_density is not None else 0,
                        1 if point_values.device_lifespan is not None else 0,
                        wp.vec3(*payload.velocity),
                        payload.temperature,
                        payload.viscosity,
                        payload.density,
                        wp.vec4(*normalize_color(payload.color)),
                        int(payload.source_id or stable_source_id(stable_id)),
                        1 if particles.has_optional_channel("color") else 0,
                        1 if particles.has_optional_channel("source_id") else 0,
                        payload.lifespan,
                        1 if payload.use_lifespan else 0,
                        particles.capacity,
                        self._point_slot_base,
                        self._point_slot_limit,
                    ],
                    outputs=[
                        particles.position,
                        particles.velocity,
                        particles.attr("age"),
                        particles.attr("life"),
                        particles.attr("temperature"),
                        particles.attr("viscosity"),
                        particles.attr("density"),
                        particles.optional_array("color"),
                        particles.optional_array("source_id"),
                        particles.time_offset,
                        particles.alive,
                        self._compiled_result,
                        self._emission_state,
                    ],
                    device=self.device,
                )
            elif plan.operation == SourceOperation.LIQUID:
                if not region.leaf_count:
                    continue
                wp.launch(
                    k_emit_compiled_source_particles,
                    dim=region.leaf_count * C.LEAF_VOLUME,
                    inputs=[
                        region.device_coverage,
                        region.device_leaf_coords,
                        region.leaf_count,
                        grid.leaf_table,
                        grid.voxels_dim,
                        grid.leaves_dim,
                        grid.origin,
                        self.runtime.settings.voxel_size,
                        normalized_liquid_target(plan),
                        wp.vec3(*payload.velocity),
                        payload.temperature,
                        payload.viscosity,
                        payload.density,
                        wp.vec4(*normalize_color(payload.color)),
                        int(payload.source_id or stable_source_id(stable_id)),
                        1 if particles.has_optional_channel("color") else 0,
                        1 if particles.has_optional_channel("source_id") else 0,
                        payload.lifespan,
                        1 if payload.use_lifespan else 0,
                        particles.capacity,
                        owner.voxel_counts.data,
                    ],
                    outputs=[
                        particles.position,
                        particles.velocity,
                        particles.attr("age"),
                        particles.attr("life"),
                        particles.attr("temperature"),
                        particles.attr("viscosity"),
                        particles.attr("density"),
                        particles.optional_array("color"),
                        particles.optional_array("source_id"),
                        particles.time_offset,
                        particles.alive,
                        particles.count,
                        self._compiled_result,
                        self._emission_state,
                    ],
                    device=self.device,
                )
            if plan.lifecycle == SourceLifecycle.ONCE and plan.operation in {
                SourceOperation.COLLISION,
                SourceOperation.SINK,
                SourceOperation.ACTIVATION,
            }:
                self._initial_fired.add(stable_id)
        if finalize:
            return self._finish_emission_transaction()
        return False

    def apply_post_p2g_fields(self, dt):
        """Apply scalar/vector/staggered fields after P2G so they persist."""
        batch = self._compiled_batch
        self.last_post_p2g_apply_count = 0
        if batch is None:
            return 0
        owner, grid = self.runtime, self.runtime.grid
        empty_scalar = self._compiled_empty_scalar
        empty_vector = self._compiled_empty_vector
        operations = {
            SourceOperation.SCALAR_FIELD,
            SourceOperation.VECTOR_FIELD,
            SourceOperation.STAGGERED_VELOCITY,
        }
        for compiled in batch.sources:
            plan, region = compiled.plan, compiled.region
            if plan.operation not in operations:
                continue
            stable_id = str(plan.stable_id)
            if plan.lifecycle == SourceLifecycle.ONCE and stable_id in self._initial_fired:
                continue
            if any(item.severity == "ERROR" for item in region.diagnostics):
                continue
            payload = plan.payload
            if plan.operation == SourceOperation.STAGGERED_VELOCITY:
                has_faces = all(
                    value is not None
                    for value in (
                        region.device_staggered_u,
                        region.device_staggered_v,
                        region.device_staggered_w,
                    )
                )
                wp.launch(
                    k_apply_compiled_staggered_source,
                    dim=max(1, region.leaf_count * C.LEAF_VOLUME),
                    inputs=[
                        region.device_coverage,
                        (region.device_staggered_u if has_faces else empty_scalar),
                        (region.device_staggered_v if has_faces else empty_scalar),
                        (region.device_staggered_w if has_faces else empty_scalar),
                        region.device_leaf_coords,
                        region.leaf_count,
                        grid.leaf_table,
                        grid.voxels_dim,
                        grid.leaves_dim,
                        wp.vec3(*payload.vector),
                        1 if has_faces else 0,
                        int(plan.blend.mode),
                        float(plan.blend.amount),
                        float(dt),
                    ],
                    outputs=[owner.vel.u, owner.vel.v, owner.vel.w],
                    device=self.device,
                )
            else:
                wp.launch(
                    k_apply_compiled_source,
                    dim=max(1, region.leaf_count * C.LEAF_VOLUME),
                    inputs=[
                        int(plan.operation),
                        region.device_coverage,
                        (
                            region.device_scalar
                            if region.device_scalar is not None
                            else empty_scalar
                        ),
                        (
                            region.device_vector
                            if region.device_vector is not None
                            else empty_vector
                        ),
                        region.device_leaf_coords,
                        region.leaf_count,
                        grid.leaf_table,
                        grid.voxels_dim,
                        grid.leaves_dim,
                        float(payload.scalar),
                        wp.vec3(*payload.vector),
                        int(plan.blend.mode),
                        float(plan.blend.amount),
                        float(dt),
                    ],
                    outputs=[
                        owner.voxel_type.data,
                        owner.pressure_solver.pressure,
                        owner.temperature.data,
                        owner.viscosity_field.data,
                        owner.density.data,
                        owner.vel.u,
                        owner.vel.v,
                        owner.vel.w,
                    ],
                    device=self.device,
                )
            self.last_post_p2g_apply_count += int(region.covered_voxel_count)
            if plan.lifecycle == SourceLifecycle.ONCE:
                self._initial_fired.add(stable_id)
        return self.last_post_p2g_apply_count

    def apply_pressure_sources(self, dt):
        """Blend FIELD pressure after the solve and immediately before project."""
        batch = self._compiled_batch
        self.last_pressure_apply_count = 0
        if batch is None:
            return 0
        owner, grid = self.runtime, self.runtime.grid
        empty_scalar = self._compiled_empty_scalar
        empty_vector = self._compiled_empty_vector
        for compiled in batch.sources:
            plan, region = compiled.plan, compiled.region
            if plan.operation != SourceOperation.PRESSURE:
                continue
            stable_id = str(plan.stable_id)
            if plan.lifecycle == SourceLifecycle.ONCE and stable_id in self._initial_fired:
                continue
            if any(item.severity == "ERROR" for item in region.diagnostics):
                continue
            payload = plan.payload
            wp.launch(
                k_apply_compiled_source,
                dim=max(1, region.leaf_count * C.LEAF_VOLUME),
                inputs=[
                    int(SourceOperation.PRESSURE),
                    region.device_coverage,
                    (region.device_scalar if region.device_scalar is not None else empty_scalar),
                    empty_vector,
                    region.device_leaf_coords,
                    region.leaf_count,
                    grid.leaf_table,
                    grid.voxels_dim,
                    grid.leaves_dim,
                    float(payload.pressure),
                    wp.vec3(0.0, 0.0, 0.0),
                    int(plan.blend.mode),
                    float(plan.blend.amount),
                    float(dt),
                ],
                outputs=[
                    owner.voxel_type.data,
                    owner.pressure_solver.pressure,
                    owner.temperature.data,
                    owner.viscosity_field.data,
                    owner.density.data,
                    owner.vel.u,
                    owner.vel.v,
                    owner.vel.w,
                ],
                device=self.device,
            )
            self.last_pressure_apply_count += int(region.covered_voxel_count)
            if plan.lifecycle == SourceLifecycle.ONCE:
                self._initial_fired.add(stable_id)
        return self.last_pressure_apply_count

    def _bake_struct(self, source):
        """Build the GPU struct with initial-fill armed state applied.

        Continuous sources are always armed (emit every substep). An initial-
        fill source is armed only until it has fired once this run.
        """
        struct = source.to_struct()
        if (
            int(getattr(source, "emission_mode", SourceEmissionMode.CONTINUOUS_INFLOW))
            == SourceEmissionMode.INITIAL_FILL
            and getattr(source, "source_id", "") in self._initial_fired
        ):
            struct.armed = 0
        return struct

    def _shapes(self):
        if self._shape_array is None:
            structs = [self._bake_struct(source) for source in self.analytic_sources]
            if not structs:
                structs = [SourceShapeDesc(enabled=False).to_struct()]
            self._shape_array = wp.array(structs, dtype=SourceShape, device=self.device)
        return self._shape_array

    def apply_forces(self, dt):
        """Apply analytic FORCE-role shapes in the solver forces phase."""
        analytic_sources = getattr(self, "analytic_sources", self.sources)
        if not any(
            source.enabled
            and source.stype == SourceShapeType.SOURCE
            and source.mode == SourceShapeSourceMode.FIELD
            and source.force_type != SourceShapeForceType.OFF
            for source in analytic_sources
        ):
            return
        owner = self.runtime
        grid = owner.grid
        wp.launch(
            k_source_shape_forces,
            dim=owner._nvox,
            inputs=[
                self._shapes(),
                len(analytic_sources),
                grid.leaf_coords,
                grid.num_leaves,
                grid.origin,
                owner.settings.voxel_size,
                owner.vel.u,
                owner.vel.v,
                owner.vel.w,
                owner.vel.wu,
                owner.vel.wv,
                owner.vel.ww,
                dt,
            ],
            device=self.device,
        )

    @staticmethod
    def _legacy_source_armed(source, fired):
        return not (
            int(getattr(source, "emission_mode", SourceEmissionMode.CONTINUOUS_INFLOW))
            == SourceEmissionMode.INITIAL_FILL
            and getattr(source, "source_id", "") in fired
        )

    @staticmethod
    def _legacy_liquid_target(source):
        # Legacy kernels historically truncate emit_rate; preserve that
        # behavior while making the zero predicate explicit and shared.
        return max(0, min(C.MAX_PARTICLES_PER_VOXEL,
                          int(float(getattr(source, "emit_rate", 0.0)))))

    @classmethod
    def legacy_descriptor_requires_topology(cls, source):
        if not getattr(source, "enabled", False):
            return False
        if (
            getattr(source, "stype", SourceShapeType.SOURCE) != SourceShapeType.SOURCE
            or getattr(source, "mode", SourceShapeSourceMode.PARTICLES)
            != SourceShapeSourceMode.PARTICLES
        ):
            return True
        return cls._legacy_liquid_target(source) > 0

    def _prepare_legacy_emission(self):
        """Freeze legacy emitters and their deterministic seed for one batch."""
        settings = self.runtime.settings
        analytic = tuple(getattr(self, "analytic_sources", self.sources))
        meshes = tuple(getattr(self, "mesh_sources", ()))
        height = self.height_source
        height_armed = self._legacy_source_armed(height, self._initial_fired)
        emit_height = bool(
            height.enabled
            and height.emit_particles
            and height_armed
            and self._legacy_liquid_target(height) > 0
        )
        emit_shapes = any(
            source.enabled
            and source.stype == SourceShapeType.SOURCE
            and source.mode == SourceShapeSourceMode.PARTICLES
            and self._legacy_liquid_target(source) > 0
            and self._legacy_source_armed(source, self._initial_fired)
            for source in analytic
        )
        emit_meshes = tuple(
            source
            for source in meshes
            if source.enabled
            and source.mesh is not None
            and source.stype == SourceShapeType.SOURCE
            and source.mode == SourceShapeSourceMode.PARTICLES
            and self._legacy_liquid_target(source) > 0
            and self._legacy_source_armed(source, self._initial_fired)
        )
        self._transaction_height_armed = height_armed
        self._transaction_emit_seed = (self._emit_seed * 1664525 + 1013904223) & 0x7FFFFFFF
        self._transaction_mesh_sources = emit_meshes
        self._transaction_shapes = None
        self._transaction_shape_count = 0

        if emit_shapes or emit_height:
            emit_sources = list(analytic)
            if emit_height:
                domain_min, domain_max = settings.domain_min, settings.domain_max
                center_z = (domain_min[2] + height.base_height) * 0.5
                half_z = max((height.base_height - domain_min[2]) * 0.5, 0.0)
                emit_sources.append(
                    SourceShapeDesc(
                        enabled=True,
                        shape=2,
                        stype=0,
                        mode=1,
                        position=(
                            (domain_min[0] + domain_max[0]) * 0.5,
                            (domain_min[1] + domain_max[1]) * 0.5,
                            center_z,
                        ),
                        half_extents=(
                            (domain_max[0] - domain_min[0]) * 0.5,
                            (domain_max[1] - domain_min[1]) * 0.5,
                            half_z,
                        ),
                        velocity=tuple(height.velocity),
                        emit_rate=height.emit_rate,
                        emission_mode=int(
                            getattr(height, "emission_mode", SourceEmissionMode.CONTINUOUS_INFLOW)
                        ),
                        source_id=getattr(height, "source_id", ""),
                        color=normalize_color(getattr(height, "color", (0.1, 0.45, 1.0, 1.0))),
                        source_numeric_id=int(getattr(height, "source_numeric_id", 0)),
                    )
                )
            key = tuple(repr(source) for source in emit_sources)
            if self._emit_shape_array is None or self._emit_shape_key != key:
                self._emit_shape_array = wp.array(
                    [self._bake_struct(source) for source in emit_sources],
                    dtype=SourceShape,
                    device=self.device,
                )
                self._emit_shape_key = key
            self._transaction_shapes = self._emit_shape_array
            self._transaction_shape_count = len(emit_sources)

        pending = set()
        if (
            emit_height
            and int(getattr(height, "emission_mode", SourceEmissionMode.CONTINUOUS_INFLOW))
            == SourceEmissionMode.INITIAL_FILL
            and getattr(height, "source_id", "")
        ):
            pending.add(height.source_id)
        for source in tuple(analytic) + emit_meshes:
            if (
                source.enabled
                and source.stype == SourceShapeType.SOURCE
                and source.mode == SourceShapeSourceMode.PARTICLES
                and self._legacy_liquid_target(source) > 0
                and int(getattr(source, "emission_mode", SourceEmissionMode.CONTINUOUS_INFLOW))
                == SourceEmissionMode.INITIAL_FILL
                and getattr(source, "source_id", "")
                and self._legacy_source_armed(source, self._initial_fired)
            ):
                pending.add(source.source_id)
        return bool(emit_shapes or emit_height or emit_meshes), pending

    def _normalize_emission_plan(self, batch, *, include_legacy):
        """Freeze every armed emitter before aggregate capacity admission."""
        from .source_regions import liquid_coverage_emission_enabled

        compiled_candidates = []
        pending_fired = set()
        if batch is not None:
            for compiled in batch.sources:
                plan, region = compiled.plan, compiled.region
                if plan.operation != SourceOperation.LIQUID or any(
                    item.severity == "ERROR" for item in region.diagnostics
                ):
                    continue
                stable_id = str(plan.stable_id)
                if (
                    plan.lifecycle in {SourceLifecycle.ONCE, SourceLifecycle.DIRECT_APPEND}
                    and stable_id in self._initial_fired
                ):
                    continue
                if plan.lifecycle == SourceLifecycle.DIRECT_APPEND:
                    if region.points is None or not len(region.points.positions):
                        continue
                elif not liquid_coverage_emission_enabled(plan):
                    continue
                compiled_candidates.append(compiled)
                if plan.lifecycle in {SourceLifecycle.ONCE, SourceLifecycle.DIRECT_APPEND}:
                    pending_fired.add(stable_id)

        legacy_attempted = False
        if include_legacy:
            legacy_attempted, pending = self._prepare_legacy_emission()
            pending_fired |= pending
        return NormalizedEmissionPlan(
            compiled_sources=tuple(compiled_candidates),
            legacy_attempted=legacy_attempted,
            pending_fired=frozenset(pending_fired),
        )

    def _begin_emission_transaction(self, batch=None, *, include_legacy=False):
        """Count a normalized source plan and enqueue one device admission gate."""
        from .source_regions import normalized_liquid_target

        owner, grid, particles = self.runtime, self.runtime.grid, self.runtime.particles
        emission_plan = self._normalize_emission_plan(batch, include_legacy=include_legacy)
        compiled_candidates = emission_plan.compiled_sources
        self._emission_pending_fired = set(emission_plan.pending_fired)
        self._emission_legacy_attempted = emission_plan.legacy_attempted

        attempted = emission_plan.attempted
        self._emission_initial_count = int(particles.host_count)
        self._emission_uid_base = int(particles.next_uid)
        self._emission_available = max(0, int(particles.capacity) - self._emission_initial_count)
        self._emission_transaction_active = attempted
        if not attempted:
            self.last_apply_result = SourceApplyResult(
                emitted=0,
                capacity_limited=False,
                required=0,
                available=self._emission_available,
                admitted=True,
            )
            return False

        self._emission_state.zero_()
        self._compiled_result.zero_()
        wp.copy(self._emission_scratch_counts, owner.voxel_counts.data, count=int(owner._nvox))

        for compiled in compiled_candidates:
            plan, region = compiled.plan, compiled.region
            if plan.lifecycle == SourceLifecycle.DIRECT_APPEND and region.points is not None:
                count = int(len(region.points.positions))
                if count:
                    wp.launch(
                        k_add_emission_demand,
                        dim=1,
                        inputs=[count, self._emission_state],
                        device=self.device,
                    )
            elif region.leaf_count:
                wp.launch(
                    k_count_compiled_source_particles,
                    dim=region.leaf_count * C.LEAF_VOLUME,
                    inputs=[
                        region.device_coverage,
                        region.device_leaf_coords,
                        region.leaf_count,
                        grid.leaf_table,
                        grid.voxels_dim,
                        grid.leaves_dim,
                        normalized_liquid_target(plan),
                        self._emission_scratch_counts,
                        self._emission_state,
                    ],
                    device=self.device,
                )

        if self._transaction_shape_count:
            wp.launch(
                k_count_source_shapes_particles,
                dim=owner._nvox,
                inputs=[
                    self._transaction_shapes,
                    self._transaction_shape_count,
                    grid.leaf_coords,
                    grid.num_leaves,
                    grid.origin,
                    owner.settings.voxel_size,
                    self._emission_scratch_counts,
                    self._emission_state,
                ],
                device=self.device,
            )
        for mesh_index, source in enumerate(self._transaction_mesh_sources):
            wp.launch(
                k_count_mesh_source_particles,
                dim=owner._nvox,
                inputs=[
                    source.mesh.id,
                    self._bake_struct(source),
                    source.query_radius(),
                    grid.leaf_coords,
                    grid.num_leaves,
                    grid.origin,
                    owner.settings.voxel_size,
                    self._emission_scratch_counts,
                    self._transaction_emit_seed + mesh_index + 1,
                    self._emission_state,
                ],
                device=self.device,
            )

        wp.launch(
            k_admit_emission_batch,
            dim=1,
            inputs=[particles.count, particles.capacity, self._emission_state],
            device=self.device,
        )
        return True

    def _shape_emission_scan_buffers(self, nvox):
        """Per-voxel demand and exclusive-offset views for shape emission."""
        buffers = getattr(self, "_shape_emit_scan", None)
        if buffers is None or buffers[0].shape[0] < nvox:
            size = max(1, int(nvox))
            buffers = (
                wp.zeros(size, dtype=wp.int32, device=self.device),
                wp.zeros(size, dtype=wp.int32, device=self.device),
            )
            self._shape_emit_scan = buffers
        return buffers[0][:nvox], buffers[1][:nvox]

    def _request_particle_growth(self, required_total):
        grow = getattr(self.runtime, "ensure_particle_capacity", None)
        if callable(grow):
            return grow(required_total)
        from .particle_capacity import ParticleGrowthResult

        current = int(getattr(self.runtime.particles, "capacity", 0))
        return ParticleGrowthResult(
            "DISABLED",
            current,
            current,
            int(required_total),
            current,
            reason="owner does not expose transactional particle growth",
        )

    def _finish_emission_transaction(self):
        if not self._emission_transaction_active:
            return False
        values = np.asarray(self._emission_state.numpy(), dtype=np.int64)
        required, admitted, emitted, invariant = map(int, values[:4])
        self._emission_available = int(values[4])
        self._emission_initial_count = max(
            0, int(self.runtime.particles.capacity) - self._emission_available
        )
        self._emission_transaction_active = False
        if invariant:
            self.last_apply_result = SourceApplyResult(
                emitted=emitted,
                capacity_limited=True,
                required=required,
                available=self._emission_available,
                admitted=False,
                invariant_violation=True,
            )
            raise RuntimeError(
                "source emission exceeded capacity after admission; "
                "the solver session must be discarded"
            )
        if admitted:
            if self._emission_legacy_attempted:
                self._emit_seed = self._transaction_emit_seed
            if self._emission_pending_fired:
                self._initial_fired |= self._emission_pending_fired
                self._shape_array = None
                self._emit_shape_array = None
                self._emit_shape_key = None
            if emitted:
                particles = self.runtime.particles
                if particles.has_optional_channel("uid"):
                    wp.launch(
                        k_assign_emitted_uids,
                        dim=emitted,
                        inputs=[
                            self._emission_initial_count,
                            emitted,
                            self._emission_uid_base,
                            particles.int_attrs["uid"],
                        ],
                        device=self.device,
                    )
                    particles.next_uid = self._emission_uid_base + emitted
                particles.refresh_host_count()
                self.runtime.invalidate_host_cache()
            self.last_apply_result = SourceApplyResult(
                emitted=emitted,
                capacity_limited=False,
                required=required,
                available=self._emission_available,
                admitted=True,
            )
            return bool(emitted)

        # Guarded commit kernels have made no particle/lifecycle/RNG changes.
        # Grow only after that fact is known; the prepared source remains armed
        # and retries on the next source phase without stopping the solver.
        total_required = self._emission_initial_count + required
        growth = self._request_particle_growth(total_required)
        policy = str(
            getattr(self.runtime.settings, "particle_growth_failure_policy", "DEFER")
        ).upper()
        pause_requested = not growth.grew and policy == "PAUSE"
        self.last_apply_result = SourceApplyResult(
            emitted=0,
            capacity_limited=True,
            required=required,
            available=self._emission_available,
            admitted=False,
            deferred=True,
            pause_requested=pause_requested,
            growth_status=growth.status,
            growth_previous=growth.previous,
            growth_current=growth.current,
            growth_target=growth.target,
            growth_reason=growth.reason,
        )
        return False

    def apply(self, dt):
        """Apply one all-or-nothing particle-emission transaction."""
        batch = None
        if getattr(self, "plans", ()):
            dynamic = any(
                plan.cadence == SourceEvaluationCadence.EVERY_SIM_FRAME for plan in self.plans
            )
            batch = getattr(self, "_compiled_batch", None)
            if batch is None or dynamic:
                batch = self.compile(frame=int(getattr(self.runtime, "frame", 0)), substep=0)

        include_legacy = bool(self.sources or self.height_source.enabled)
        self._begin_emission_transaction(batch, include_legacy=include_legacy)
        if batch is not None:
            self.apply_compiled(batch, dt, transaction_ready=True, finalize=False)
        if not include_legacy:
            return self._finish_emission_transaction()

        owner = self.runtime
        grid = owner.grid
        settings = owner.settings
        analytic_sources = tuple(getattr(self, "analytic_sources", self.sources))
        mesh_sources = tuple(getattr(self, "mesh_sources", ()))
        shape_count = len(analytic_sources)
        shapes = self._shapes()

        if shape_count:
            wp.launch(
                k_source_shapes,
                dim=owner._nvox,
                inputs=[
                    shapes,
                    shape_count,
                    grid.leaf_coords,
                    grid.num_leaves,
                    grid.origin,
                    settings.voxel_size,
                    dt,
                ],
                outputs=[
                    owner.voxel_type.data,
                    owner.collision_sdf.data,
                    owner.collision_vel,
                    owner.vel.u,
                    owner.vel.v,
                    owner.vel.w,
                    owner.temperature.data,
                    owner.viscosity_field.data,
                ],
                device=self.device,
            )

        for source in mesh_sources:
            if not source.enabled or source.mesh is None:
                continue
            wp.launch(
                k_mesh_source_field,
                dim=owner._nvox,
                inputs=[
                    source.mesh.id,
                    source.to_struct(),
                    source.query_radius(),
                    grid.leaf_coords,
                    grid.num_leaves,
                    grid.origin,
                    settings.voxel_size,
                ],
                outputs=[
                    owner.voxel_type.data,
                    owner.vel.u,
                    owner.vel.v,
                    owner.vel.w,
                    owner.temperature.data,
                    owner.viscosity_field.data,
                ],
                device=self.device,
            )

        height = self.height_source
        if height.enabled and self._transaction_height_armed:
            profile_res = wp.vec2i(
                int(self._height_profile_res[0]),
                int(self._height_profile_res[1]),
            )
            wp.launch(
                k_source_height,
                dim=owner._nvox,
                inputs=[
                    grid.leaf_coords,
                    grid.num_leaves,
                    grid.origin,
                    settings.voxel_size,
                    height.base_height,
                    int(height.back_mode),
                    self._height_profile,
                    profile_res,
                    wp.vec3(*settings.domain_min),
                    wp.vec3(*settings.domain_max),
                    wp.vec3(*height.velocity),
                    1 if (height.emit_particles and self._legacy_liquid_target(height) > 0) else 0,
                    self._emission_state,
                ],
                outputs=[owner.voxel_type.data, owner.vel.u, owner.vel.v, owner.vel.w],
                device=self.device,
            )

        wp.launch(
            k_source_resolve_type,
            dim=owner._nvox,
            inputs=[owner.voxel_type.data, grid.num_leaves],
            device=self.device,
        )

        particles = owner.particles
        if self._transaction_shape_count:
            nvox = int(owner._nvox)
            demand, offsets = self._shape_emission_scan_buffers(nvox)
            wp.launch(
                k_source_shapes_particle_demand,
                dim=nvox,
                inputs=[
                    self._transaction_shapes,
                    self._transaction_shape_count,
                    grid.leaf_coords,
                    grid.num_leaves,
                    grid.origin,
                    settings.voxel_size,
                    owner.voxel_counts.data,
                    self._emission_state,
                ],
                outputs=[demand],
                device=self.device,
            )
            wp.utils.array_scan(demand, offsets, inclusive=False)
            wp.launch(
                k_reserve_scanned_slots,
                dim=1,
                inputs=[
                    demand,
                    offsets,
                    nvox,
                    particles.capacity,
                    self._emission_state,
                ],
                outputs=[
                    particles.count,
                    self._point_slot_base,
                    self._point_slot_limit,
                ],
                device=self.device,
            )
            wp.launch(
                k_source_shapes_particles,
                dim=nvox,
                inputs=[
                    self._transaction_shapes,
                    self._transaction_shape_count,
                    grid.leaf_coords,
                    grid.num_leaves,
                    grid.origin,
                    settings.voxel_size,
                    owner.voxel_counts.data,
                    self._transaction_emit_seed,
                    1 if particles.has_optional_channel("color") else 0,
                    1 if particles.has_optional_channel("source_id") else 0,
                    offsets,
                    self._point_slot_base,
                    self._point_slot_limit,
                ],
                outputs=[
                    particles.position,
                    particles.velocity,
                    particles.attr("age"),
                    particles.attr("life"),
                    particles.attr("temperature"),
                    particles.attr("viscosity"),
                    particles.attr("density"),
                    particles.optional_array("color"),
                    particles.optional_array("source_id"),
                    particles.time_offset,
                    particles.alive,
                    self._emission_state,
                ],
                device=self.device,
            )
        for mesh_index, source in enumerate(self._transaction_mesh_sources):
            wp.launch(
                k_mesh_source_particles,
                dim=owner._nvox,
                inputs=[
                    source.mesh.id,
                    self._bake_struct(source),
                    source.query_radius(),
                    grid.leaf_coords,
                    grid.num_leaves,
                    grid.origin,
                    settings.voxel_size,
                    owner.voxel_counts.data,
                    self._transaction_emit_seed + mesh_index + 1,
                    1 if particles.has_optional_channel("color") else 0,
                    1 if particles.has_optional_channel("source_id") else 0,
                    particles.capacity,
                ],
                outputs=[
                    particles.position,
                    particles.velocity,
                    particles.attr("age"),
                    particles.attr("life"),
                    particles.attr("temperature"),
                    particles.attr("viscosity"),
                    particles.attr("density"),
                    particles.optional_array("color"),
                    particles.optional_array("source_id"),
                    particles.time_offset,
                    particles.alive,
                    particles.count,
                    self._emission_state,
                ],
                device=self.device,
            )
        return self._finish_emission_transaction()

    def _upload_direct_packets(self, packets):
        if self._compiled_empty_scalar is None:
            self._compiled_empty_scalar = wp.empty(0, dtype=float, device=self.device)
            self._compiled_empty_vector = wp.empty(0, dtype=wp.vec3, device=self.device)
        uploaded = []
        for packet in packets:
            positions = wp.array(packet.positions, dtype=wp.vec3, device=self.device)
            velocities = (
                wp.array(packet.velocities, dtype=wp.vec3, device=self.device)
                if packet.velocities is not None
                else self._compiled_empty_vector
            )
            uploaded.append((packet, positions, velocities))
        return tuple(uploaded)

    def _direct_packet_attempt(self, uploaded, required):
        """Run one device-authoritative admit/commit attempt for all packets."""
        particles = self.runtime.particles
        initial_count = int(particles.refresh_host_count())
        uid_base = int(particles.next_uid)
        self._emission_state.zero_()
        self._compiled_result.zero_()
        wp.launch(
            k_add_emission_demand,
            dim=1,
            inputs=[required, self._emission_state],
            device=self.device,
        )
        wp.launch(
            k_admit_emission_batch,
            dim=1,
            inputs=[particles.count, particles.capacity, self._emission_state],
            device=self.device,
        )
        empty_scalar = self._compiled_empty_scalar
        for packet, positions, velocities in uploaded:
            count = len(packet.positions)
            wp.launch(
                k_reserve_owned_point_slots,
                dim=1,
                inputs=[count, particles.capacity, self._emission_state],
                outputs=[
                    particles.count,
                    self._point_slot_base,
                    self._point_slot_limit,
                ],
                device=self.device,
            )
            wp.launch(
                k_source_owned_points,
                dim=max(1, count),
                inputs=[
                    positions,
                    velocities,
                    empty_scalar,
                    empty_scalar,
                    empty_scalar,
                    empty_scalar,
                    count,
                    1 if packet.velocities is not None else 0,
                    0,
                    0,
                    0,
                    0,
                    wp.vec3(*packet.velocity),
                    float(packet.temperature),
                    float(packet.viscosity),
                    float(packet.density),
                    wp.vec4(*packet.color),
                    int(packet.source_id),
                    1 if particles.has_optional_channel("color") else 0,
                    1 if particles.has_optional_channel("source_id") else 0,
                    float(packet.life),
                    1 if packet.use_lifespan else 0,
                    particles.capacity,
                    self._point_slot_base,
                    self._point_slot_limit,
                ],
                outputs=[
                    particles.position,
                    particles.velocity,
                    particles.attr("age"),
                    particles.attr("life"),
                    particles.attr("temperature"),
                    particles.attr("viscosity"),
                    particles.attr("density"),
                    particles.optional_array("color"),
                    particles.optional_array("source_id"),
                    particles.time_offset,
                    particles.alive,
                    self._compiled_result,
                    self._emission_state,
                ],
                device=self.device,
            )

        values = np.asarray(self._emission_state.numpy(), dtype=np.int64)
        demand, admitted, emitted, invariant, available = map(int, values[:5])
        initial = max(0, int(particles.capacity) - available)
        if initial != initial_count:
            raise RuntimeError("direct particle transaction initial count drifted")
        if invariant or (admitted and emitted != demand):
            raise RuntimeError("direct particle transaction violated its admitted count")
        if admitted:
            if emitted and particles.has_optional_channel("uid"):
                wp.launch(
                    k_assign_emitted_uids,
                    dim=emitted,
                    inputs=[initial, emitted, uid_base, particles.int_attrs["uid"]],
                    device=self.device,
                )
                particles.next_uid = uid_base + emitted
            total = particles.refresh_host_count() if emitted else initial
            if emitted:
                self.runtime.invalidate_host_cache()
                self.runtime.invalidate_particle_surface()
            return InjectionResult(True, emitted, demand, available, False, int(total))
        return InjectionResult(False, 0, demand, available, True, initial)

    def inject_particle_packets_exact(self, packets):
        """Append every packet or none and return exact device-derived truth."""
        normalized = tuple(
            packet
            if isinstance(packet, DirectInjectionPacket)
            else DirectInjectionPacket(
                positions=packet.positions,
                velocity=tuple(packet.velocity),
                velocities=getattr(packet, "velocities", None),
                temperature=float(packet.temperature),
                viscosity=float(packet.viscosity),
                density=float(packet.density),
                color=tuple(getattr(packet, "color", (0.1, 0.45, 1.0, 1.0))),
                source_id=int(getattr(packet, "source_id", 0)),
                life=float(packet.life),
                use_lifespan=bool(packet.use_lifespan),
            )
            for packet in packets
            if len(packet.positions)
        )
        required = sum(len(packet.positions) for packet in normalized)
        if not required:
            particles = self.runtime.particles
            result = InjectionResult(
                True,
                0,
                0,
                max(0, int(particles.capacity) - int(particles.host_count)),
                False,
                int(particles.host_count),
            )
            self.last_injection_result = result
            self.last_apply_result = SourceApplyResult(
                emitted=0,
                capacity_limited=False,
                required=0,
                available=result.available,
                admitted=True,
                total=result.total,
            )
            return result

        uploaded = self._upload_direct_packets(normalized)
        result = self._direct_packet_attempt(uploaded, required)
        growth = None
        if not result.admitted:
            growth = self._request_particle_growth(result.total + required)
            if growth.grew:
                result = self._direct_packet_attempt(uploaded, required)

        self.last_injection_result = result
        self.last_apply_result = SourceApplyResult(
            emitted=result.emitted,
            capacity_limited=result.capacity_limited,
            required=result.required,
            available=result.available,
            admitted=result.admitted,
            total=result.total,
        )
        if not result.admitted:
            policy = str(
                getattr(
                    self.runtime.settings,
                    "particle_growth_failure_policy",
                    "DEFER",
                )
            ).upper()
            self.last_apply_result = SourceApplyResult(
                emitted=result.emitted,
                capacity_limited=result.capacity_limited,
                required=result.required,
                available=result.available,
                admitted=result.admitted,
                total=result.total,
                deferred=True,
                pause_requested=policy == "PAUSE",
                growth_status=growth.status,
                growth_previous=growth.previous,
                growth_current=growth.current,
                growth_target=growth.target,
                growth_reason=growth.reason,
            )
        return result

    def inject_particles_exact(
        self,
        positions,
        velocity=(0.0, 0.0, 0.0),
        velocities=None,
        temperature=0.0,
        viscosity=0.0,
        density=1.0,
        color=(0.1, 0.45, 1.0, 1.0),
        source_id=0,
        life=0.0,
        use_lifespan=False,
    ):
        """Compatibility alias for the structured direct-injection API."""
        return self.inject_particles(
            positions,
            velocity=velocity,
            velocities=velocities,
            temperature=temperature,
            viscosity=viscosity,
            density=density,
            color=color,
            source_id=source_id,
            life=life,
            use_lifespan=use_lifespan,
        )

    def inject_particles(
        self,
        positions,
        velocity=(0.0, 0.0, 0.0),
        velocities=None,
        temperature=0.0,
        viscosity=0.0,
        density=1.0,
        color=(0.1, 0.45, 1.0, 1.0),
        source_id=0,
        life=0.0,
        use_lifespan=False,
    ):
        """Append one particle packet and return exact admission truth.

        Supply either one constant ``velocity`` or per-particle ``velocities``.
        Providing both is ambiguous and rejected before device work begins.
        """
        constant_velocity = tuple(float(component) for component in velocity)
        if velocities is not None and constant_velocity != (0.0, 0.0, 0.0):
            raise ValueError(
                "direct injection accepts either velocity or velocities, not both"
            )
        packet = DirectInjectionPacket(
            positions=positions,
            velocity=constant_velocity,
            velocities=velocities,
            temperature=float(temperature),
            viscosity=float(viscosity),
            density=float(density),
            color=tuple(color),
            source_id=int(source_id),
            life=float(life),
            use_lifespan=bool(use_lifespan),
        )
        return self.inject_particle_packets_exact((packet,))

    def inject_particles_legacy_total(
        self,
        positions,
        velocity=(0.0, 0.0, 0.0),
        velocities=None,
        temperature=0.0,
        viscosity=0.0,
        density=1.0,
        color=(0.1, 0.45, 1.0, 1.0),
        source_id=0,
        life=0.0,
        use_lifespan=False,
    ):
        """Return the legacy lossy count: final total, or zero for refusal/empty.

        Prefer :meth:`inject_particles`; this mapping cannot distinguish an
        admitted empty request from a refused transaction. When both velocity
        forms are supplied, this compatibility adapter preserves the former
        precedence rule by discarding the constant velocity.
        """
        result = self.inject_particles(
            positions,
            velocity=(0.0, 0.0, 0.0) if velocities is not None else velocity,
            velocities=velocities,
            temperature=temperature,
            viscosity=viscosity,
            density=density,
            color=color,
            source_id=source_id,
            life=life,
            use_lifespan=use_lifespan,
        )
        return result.total if result.admitted and result.emitted else 0

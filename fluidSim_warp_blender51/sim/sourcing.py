"""Transaction-facing source facade with legacy compatibility exports."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import warp as wp

from . import constants as C
from .enums import (
    BlendMode,
    MeshSinkRegionMode,
    MeshSourceRegion,
    SourceEmissionMode,
    SourceEvaluationCadence,
    SourceLifecycle,
    SourceOperation,
    SourcePayloadKind,
    SourceShapeForceType,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceShapeType,
)
from .source_contracts import (
    DirectInjectionPacket,
    HeightSourceDesc,
    InjectionResult,
    MeshSource,
    MeshSourceDesc,
    NormalizedEmissionPlan,
    SourceApplyResult,
    SourceBatch,
    SourceShape,
    SourceShapeDesc,
)
from .source_execution import CompiledSourceExecution, SourceRuntime
from .source_kernels import (
    _watertight_mesh_contains,
    blend_scalar,
    blend_vec,
    k_add_emission_demand,
    k_admit_emission_batch,
    k_apply_compiled_source,
    k_apply_compiled_staggered_source,
    k_assign_emitted_uids,
    k_core_source_float,
    k_core_source_float_particle_attribute,
    k_core_source_staggered_vector,
    k_core_source_vector,
    k_core_source_vector_particle_attribute,
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
    k_source_particle_integer_constant,
    k_source_particle_scalar_constant,
    k_source_particle_scalar_field_attribute,
    k_source_particle_vector_field_attribute,
    k_source_particles,
    k_source_resolve_type,
    k_source_shape_forces,
    k_source_shapes,
    k_source_shapes_particle_demand,
    k_source_shapes_particles,
    k_source_volume_float,
    k_source_volume_mask_fog,
    k_source_volume_mask_sdf,
    k_source_volume_type_fog,
    k_source_volume_type_fog_with_temperature,
    k_source_volume_type_sdf,
    k_source_volume_type_sdf_with_temperature,
    k_source_volume_vector,
    k_source_volume_vector_component,
    mesh_source_force_velocity,
    shape_force_acceleration,
    shape_force_velocity,
    shape_local_sdf,
    shape_point_velocity,
    shape_sdf_world,
)

wp.set_module_options({"enable_backward": False})

__all__ = [
    "BlendMode",
    "MeshSinkRegionMode",
    "MeshSourceRegion",
    "SourceEmissionMode",
    "SourceEvaluationCadence",
    "SourceLifecycle",
    "SourceOperation",
    "SourcePayloadKind",
    "SourceShapeForceType",
    "SourceShapeShape",
    "SourceShapeSourceMode",
    "SourceShapeType",
    "SourceShape",
    "MeshSource",
    "SourceShapeDesc",
    "MeshSourceDesc",
    "HeightSourceDesc",
    "NormalizedEmissionPlan",
    "SourceApplyResult",
    "SourceBatch",
    "DirectInjectionPacket",
    "InjectionResult",
    "SourcePipeline",
    "blend_scalar",
    "blend_vec",
    "shape_local_sdf",
    "shape_sdf_world",
    "shape_point_velocity",
    "shape_force_velocity",
    "shape_force_acceleration",
    "mesh_source_force_velocity",
    "k_source_shape_forces",
    "k_source_shapes",
    "k_source_shapes_particles",
    "k_source_shapes_particle_demand",
    "k_reserve_scanned_slots",
    "_watertight_mesh_contains",
    "k_mesh_source_field",
    "k_mesh_source_particles",
    "k_source_height",
    "k_source_resolve_type",
    "k_source_particles",
    "k_reserve_owned_point_slots",
    "k_source_owned_points",
    "k_source_particle_scalar_field_attribute",
    "k_source_particle_vector_field_attribute",
    "k_source_particle_scalar_constant",
    "k_source_particle_integer_constant",
    "k_core_source_float",
    "k_core_source_vector",
    "k_core_source_staggered_vector",
    "k_core_source_float_particle_attribute",
    "k_core_source_vector_particle_attribute",
    "k_source_volume_float",
    "k_source_volume_vector",
    "k_source_volume_vector_component",
    "k_source_volume_type_fog",
    "k_source_volume_type_fog_with_temperature",
    "k_source_volume_mask_fog",
    "k_source_volume_type_sdf",
    "k_source_volume_type_sdf_with_temperature",
    "k_source_volume_mask_sdf",
    "k_apply_compiled_source",
    "k_apply_compiled_staggered_source",
    "k_emit_compiled_source_particles",
    "k_assign_emitted_uids",
    "k_add_emission_demand",
    "k_admit_emission_batch",
    "k_count_compiled_source_particles",
    "k_count_source_shapes_particles",
    "k_count_mesh_source_particles",
]


class SourcePipeline(CompiledSourceExecution):
    """Own source state and the complete source-launch transaction."""

    def __init__(
        self,
        *,
        device,
        grid,
        particles,
        settings,
        active_voxel_count,
        current_frame,
        ensure_particle_capacity,
        invalidate_host_cache,
        invalidate_particle_surface,
        voxel_type=None,
        collision_sdf=None,
        collision_velocity=None,
        velocity=None,
        temperature=None,
        viscosity_field=None,
        density=None,
        voxel_counts=None,
        pressure_solver=None,
    ):
        self.runtime = SourceRuntime(
            device=device,
            grid=grid,
            particles=particles,
            settings=settings,
            voxel_type=voxel_type,
            collision_sdf=collision_sdf,
            collision_vel=collision_velocity,
            vel=velocity,
            temperature=temperature,
            viscosity_field=viscosity_field,
            density=density,
            voxel_counts=voxel_counts,
            pressure_solver=pressure_solver,
            active_voxel_count=active_voxel_count,
            current_frame=current_frame,
            ensure_particle_capacity=ensure_particle_capacity,
            invalidate_host_cache=invalidate_host_cache,
            invalidate_particle_surface=invalidate_particle_surface,
        )
        self.device = device
        self.sources: list[SourceShapeDesc | MeshSourceDesc] = []
        self.analytic_sources: list[SourceShapeDesc] = []
        self.mesh_sources: list[MeshSourceDesc] = []
        self.height_source = HeightSourceDesc()
        self._shape_array = None
        self._emit_shape_array = None
        self._emit_shape_key = None
        self._height_profile = wp.zeros((2, 2), dtype=float, device=self.device)
        self._height_profile_res = (2, 2)
        self._emit_seed = 12345
        self._compiled_result = wp.zeros(2, dtype=wp.int32, device=self.device)
        # [required, admitted, emitted, invariant-overflow, available]. Count
        # passes and guarded commit kernels remain on-device until one
        # aggregate result is read after the whole source batch.
        self._emission_state = wp.zeros(5, dtype=wp.int32, device=self.device)
        # Contiguous slot block reserved for the owned-point packet currently
        # being appended, so placement is source-order rather than
        # thread-arrival order. See k_reserve_owned_point_slots.
        self._point_slot_base = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._point_slot_limit = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._emission_scratch_counts = wp.zeros(
            max(1, int(getattr(grid, "voxel_capacity", 0) or active_voxel_count())),
            dtype=wp.int32,
            device=self.device,
        )
        self._emission_transaction_active = False
        self._emission_pending_fired: set[str] = set()
        self._emission_initial_count = 0
        self._emission_available = 0
        self._emission_legacy_attempted = False
        self._transaction_shapes = None
        self._transaction_shape_count = 0
        self._transaction_mesh_sources = ()
        self._transaction_height_armed = False
        self._compiled_empty_scalar = None
        self._compiled_empty_vector = None
        self.last_apply_result = SourceApplyResult(
            emitted=0,
            capacity_limited=False,
            required=0,
            available=int(particles.capacity),
            admitted=True,
        )
        self.last_injection_result = None
        self.last_pressure_apply_count = 0
        self.last_post_p2g_apply_count = 0
        # Per-source initial-fill armed state, keyed by the stable descriptor
        # ``source_id`` (Blender object name). Persists across per-frame
        # ``set_source_batch`` calls; cleared by ``reset`` (Initialize/Reset).
        self._initial_fired: set[str] = set()
        from .source_regions import SourceCompiler

        self.compiler = SourceCompiler()
        self.batch = SourceBatch()
        self.plans = ()
        self._compiled_batch = None
        self._session_generation = 0
        self.last_diagnostics = ()

    def reset(self):
        """Re-arm sources and release compiler-owned device/cache state."""
        self._initial_fired = set()
        self._shape_array = None
        self._emit_shape_array = None
        self._emit_shape_key = None
        self._session_generation = getattr(self, "_session_generation", 0) + 1
        compiler = getattr(self, "compiler", None)
        if compiler is not None:
            compiler.release()
        self._compiled_batch = None
        self.last_diagnostics = ()
        self.last_injection_result = None
        self.last_pressure_apply_count = 0
        self.last_post_p2g_apply_count = 0
        self._emission_transaction_active = False
        self._emission_pending_fired = set()
        particles = getattr(getattr(self, "runtime", None), "particles", None)
        self.last_apply_result = SourceApplyResult(
            emitted=0,
            capacity_limited=False,
            required=0,
            available=int(getattr(particles, "capacity", 0)),
            admitted=True,
        )

    def update_available_capacity(self, available: int) -> None:
        """Refresh capacity truth without exposing mutable result internals."""
        self.last_apply_result = replace(self.last_apply_result, available=max(0, int(available)))

    def replace_particle_storage(self, particles) -> None:
        """Rebind source execution after the solver atomically swaps its pool.

        The compiled batch holds region data only, so it survives the swap:
        growth happens mid-substep, and dropping it skipped the compiled field
        and pressure stages for the rest of that substep.
        """
        self.runtime = replace(self.runtime, particles=particles)
        self.update_available_capacity(
            int(particles.capacity) - int(particles.host_count)
        )

    def export_checkpoint_state(self):
        """Return deterministic source lifecycle/RNG state at a frame boundary."""
        if self._emission_transaction_active:
            raise RuntimeError("cannot checkpoint while a source emission transaction is active")
        return {
            "schema_version": 1,
            "emit_seed": int(self._emit_seed),
            "initial_fired": sorted(str(value) for value in self._initial_fired),
            "session_generation": int(self._session_generation),
        }

    def import_checkpoint_state(self, state):
        """Restore one-shot source identity and RNG without changing descriptors."""
        if not isinstance(state, dict) or set(state) != {
            "schema_version",
            "emit_seed",
            "initial_fired",
            "session_generation",
        }:
            raise ValueError("source checkpoint state has an invalid schema")
        if int(state["schema_version"]) != 1:
            raise ValueError("unsupported source checkpoint schema")
        fired = state["initial_fired"]
        if not isinstance(fired, list) or not all(
            isinstance(value, str) and value for value in fired
        ):
            raise ValueError("source initial_fired must be a list of IDs")
        if len(fired) != len(set(fired)):
            raise ValueError("source initial_fired contains duplicate IDs")
        emit_seed = int(state["emit_seed"])
        generation = int(state["session_generation"])
        if emit_seed < 0 or generation < 0:
            raise ValueError("source checkpoint integers must be non-negative")

        self._emit_seed = emit_seed
        self._initial_fired = set(fired)
        self._session_generation = generation
        self._emission_transaction_active = False
        self._emission_pending_fired = set()
        self._shape_array = None
        self._emit_shape_array = None
        self._emit_shape_key = None
        compiler = getattr(self, "compiler", None)
        if compiler is not None:
            compiler.release()
        self._compiled_batch = None
        self.last_diagnostics = ()

    def release(self):
        """Release all compiler-owned GPU values on solver teardown/failure."""
        self.reset()
        self.runtime = None

    def set_source_batch(self, batch):
        """Install both runtime representations without compatibility copies."""
        from .source_regions import SourcePlan

        if not isinstance(batch, SourceBatch):
            raise TypeError("set_source_batch accepts a SourceBatch")
        descriptors = tuple(batch.legacy_descriptors)
        plans = tuple(batch.native_plans)
        if not all(isinstance(plan, SourcePlan) for plan in plans):
            raise TypeError("native_plans accepts SourcePlan values only")
        if not all(isinstance(value, (SourceShapeDesc, MeshSourceDesc)) for value in descriptors):
            raise TypeError("legacy_descriptors accepts source descriptors only")
        self.batch = SourceBatch(descriptors, plans)
        self.sources = list(descriptors)
        self.analytic_sources = [
            source for source in descriptors if isinstance(source, SourceShapeDesc)
        ]
        self.mesh_sources = [source for source in descriptors if isinstance(source, MeshSourceDesc)]
        self.plans = plans
        self._compiled_batch = None
        self._shape_array = None
        self._emit_shape_array = None
        self._emit_shape_key = None

    def required_leaf_boxes(self):
        """World-space plan bounds used by solver topology marking."""
        for plan in self.plans:
            bounds = getattr(plan.region, "bounds", None)
            if bounds is not None and plan.activation.enabled:
                yield bounds()

    def compile(self, frame=0, substep=0, *, upload=True):
        """Compile current plans for the owner's sparse-grid context."""
        from .source_regions import CompileContext

        owner, settings = self.runtime, self.runtime.settings
        context = CompileContext(
            frame=int(frame),
            substep=int(substep),
            session_generation=self._session_generation,
            device=self.device,
            origin=tuple(float(v) for v in owner.grid.origin),
            voxel_size=float(settings.voxel_size),
            domain_min=tuple(float(v) for v in settings.domain_min),
            domain_max=tuple(float(v) for v in settings.domain_max),
            leaf_capacity=int(getattr(owner.grid, "capacity", C.DEFAULT_LEAF_CAPACITY)),
            particle_capacity=int(owner.particles.capacity),
            upload=upload,
        )
        self._compiled_batch = self.compiler.compile(self.plans, context)
        self.last_diagnostics = self._compiled_batch.diagnostics
        return self._compiled_batch

    def set_height_source(self, desc):
        desc = HeightSourceDesc() if desc is None else desc
        self.height_source = desc
        self._emit_shape_array = None
        self._emit_shape_key = None
        if desc.profile is not None:
            profile = np.ascontiguousarray(desc.profile, dtype=np.float32)
            self._height_profile = wp.array(profile, dtype=float, device=self.device)
            self._height_profile_res = profile.shape
        else:
            self._height_profile_res = (2, 2)

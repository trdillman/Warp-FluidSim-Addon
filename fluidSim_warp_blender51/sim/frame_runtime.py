"""Frame scheduling policy and host-visible runtime statistics.

This module is Warp-independent.  The solver facade supplies the measured
fluid/collider speeds, then executes the selected schedule on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .surface_tension import capillary_dt, required_capillary_substeps


@dataclass(frozen=True)
class SubstepPolicy:
    min_substeps: int
    max_substeps: int
    cfl: float
    voxel_size: float
    gravity_magnitude: float
    surface_tension_enabled: bool
    surface_tension_density: float
    surface_tension_coefficient: float
    surface_tension_safety: float

    @classmethod
    def from_settings(cls, settings: Any) -> SubstepPolicy:
        gravity = tuple(float(value) for value in settings.gravity)
        min_substeps = max(1, int(settings.min_substeps))
        return cls(
            min_substeps=min_substeps,
            max_substeps=max(min_substeps, int(settings.max_substeps)),
            cfl=float(settings.cfl),
            voxel_size=float(settings.voxel_size),
            gravity_magnitude=math.sqrt(sum(value * value for value in gravity)),
            surface_tension_enabled=bool(getattr(
                settings, "surface_tension_enabled", False)),
            surface_tension_density=float(getattr(
                settings, "surface_tension_density", 1000.0)),
            surface_tension_coefficient=float(getattr(
                settings, "surface_tension_coefficient", 0.0)),
            surface_tension_safety=float(getattr(
                settings, "surface_tension_safety", 0.5)),
        )

    @property
    def adaptive(self) -> bool:
        return self.min_substeps < self.max_substeps


@dataclass(frozen=True)
class SubstepSchedule:
    substeps: int
    cfl_clamped: bool
    capillary_dt: float
    capillary_substeps_required: int
    capillary_substeps_clamped: bool
    limiter: str

    def diagnostics(self) -> dict[str, float | int | bool | str]:
        return {
            "capillary_dt": self.capillary_dt,
            "capillary_substeps_required": self.capillary_substeps_required,
            "capillary_substeps_clamped": self.capillary_substeps_clamped,
            "substep_limiter": self.limiter,
        }


def select_substeps(
        policy: SubstepPolicy,
        frame_dt: float,
        *,
        has_particles: bool,
        fluid_max_speed: float = 0.0,
        collider_max_speed: float = 0.0,
) -> SubstepSchedule:
    """Select a bounded frame schedule from explicit host-side evidence."""
    cfl_required = 1
    capillary_required = 1
    capillary_limit = float("inf")
    if (policy.surface_tension_enabled
            and policy.surface_tension_coefficient > 0.0):
        capillary_limit = capillary_dt(
            policy.surface_tension_density,
            policy.voxel_size,
            policy.surface_tension_coefficient,
            policy.surface_tension_safety,
        )
        capillary_required = required_capillary_substeps(
            frame_dt, capillary_limit)

    if policy.adaptive and has_particles:
        vmax = max(
            max(0.0, fluid_max_speed) + policy.gravity_magnitude * frame_dt,
            max(0.0, collider_max_speed),
        )
        if vmax >= 1.0e-6:
            dt_cfl = policy.cfl * policy.voxel_size / vmax
            cfl_required = max(
                1, int(math.ceil(frame_dt / max(dt_cfl, 1.0e-6))))

    required = max(
        policy.min_substeps, cfl_required, capillary_required)
    selected = min(policy.max_substeps, required)
    if required <= policy.min_substeps:
        limiter = "MINIMUM"
    elif cfl_required == capillary_required == required:
        limiter = "CFL_AND_CAPILLARY"
    elif capillary_required == required:
        limiter = "CAPILLARY"
    else:
        limiter = "CFL"
    return SubstepSchedule(
        substeps=max(policy.min_substeps, selected),
        cfl_clamped=cfl_required > policy.max_substeps,
        capillary_dt=capillary_limit,
        capillary_substeps_required=capillary_required,
        capillary_substeps_clamped=(
            capillary_required > policy.max_substeps),
        limiter=limiter,
    )


@dataclass
class FrameStats:
    """Per-frame statistics for the profiler/UI (host mirror)."""

    step_time: float = 0.0
    active_voxels: int = 0
    initialized_voxels: int = 0
    total_voxels: int = 0
    active_particles: int = 0
    total_particles: int = 0
    whitewater_particles: int = 0
    whitewater_foam: int = 0
    whitewater_bubbles: int = 0
    whitewater_spray: int = 0
    whitewater_invalid_types: int = 0
    whitewater_required: int = 0
    whitewater_available: int = 0
    whitewater_admitted: int = 0
    whitewater_dropped: int = 0
    whitewater_policy: str = "FIXED_CAPACITY_DETERMINISTIC_DROPS"
    whitewater_status: str = "OK"
    whitewater_message: str = ""
    surface_voxel_size: float = 0.0
    surface_active_voxels: int = 0
    surface_active_leaves: int = 0
    substeps: int = 0
    cfl_substeps_clamped: bool = False
    capillary_dt: float = float("inf")
    capillary_substeps_required: int = 1
    capillary_substeps_clamped: bool = False
    substep_limiter: str = "MINIMUM"
    surface_tension_active_voxels: int = 0
    surface_tension_max_curvature: float = 0.0
    surface_tension_max_acceleration: float = 0.0
    surface_tension_temporary_bytes: int = 0
    surface_tension_submission_seconds: float = 0.0
    surface_tension_status: str = "DISABLED"
    surface_tension_message: str = ""
    custom_effects: list[dict[str, object]] = field(default_factory=list)
    pressure_iterations: int = 0
    pressure_residual: float = 0.0
    #: Worst of the three per-axis viscosity solves, by severity. These
    #: mirror the pressure fields above; without them a solve that
    #: exhausts its iteration budget every frame reports nothing at all.
    viscosity_status: str = "DISABLED"
    viscosity_converged: bool = True
    viscosity_iterations: int = 0
    viscosity_residual: float = 0.0
    #: Set once a CUDA-graph PCG path has fallen back to the slower
    #: host-driven loop for the rest of the run (REVIEW-016).
    native_solver_message: str = ""
    timers: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        vps = self.active_voxels / self.step_time if self.step_time > 0 else 0.0
        pps = self.active_particles / self.step_time if self.step_time > 0 else 0.0
        return {
            "stepTime": self.step_time,
            "activeVoxels": self.active_voxels,
            "initializedVoxels": self.initialized_voxels,
            "totalVoxels": self.total_voxels,
            "activeParticles": self.active_particles,
            "totalParticles": self.total_particles,
            "whitewaterParticles": self.whitewater_particles,
            "whitewaterFoam": self.whitewater_foam,
            "whitewaterBubbles": self.whitewater_bubbles,
            "whitewaterSpray": self.whitewater_spray,
            "whitewaterInvalidTypes": self.whitewater_invalid_types,
            "whitewaterRequired": self.whitewater_required,
            "whitewaterAvailable": self.whitewater_available,
            "whitewaterAdmitted": self.whitewater_admitted,
            "whitewaterDropped": self.whitewater_dropped,
            "whitewaterPolicy": self.whitewater_policy,
            "whitewaterStatus": self.whitewater_status,
            "whitewaterMessage": self.whitewater_message,
            "surfaceVoxelSize": self.surface_voxel_size,
            "surfaceActiveVoxels": self.surface_active_voxels,
            "surfaceActiveLeaves": self.surface_active_leaves,
            "substeps": self.substeps,
            "cflSubstepsClamped": self.cfl_substeps_clamped,
            "capillaryDt": (
                self.capillary_dt if math.isfinite(self.capillary_dt) else 0.0),
            "capillarySubstepsRequired": self.capillary_substeps_required,
            "capillarySubstepsClamped": self.capillary_substeps_clamped,
            "substepLimiter": self.substep_limiter,
            "surfaceTensionActiveVoxels": self.surface_tension_active_voxels,
            "surfaceTensionMaxCurvature": self.surface_tension_max_curvature,
            "surfaceTensionMaxAcceleration": self.surface_tension_max_acceleration,
            "surfaceTensionTemporaryBytes": self.surface_tension_temporary_bytes,
            "surfaceTensionSubmissionSeconds": self.surface_tension_submission_seconds,
            "surfaceTensionStatus": self.surface_tension_status,
            "surfaceTensionMessage": self.surface_tension_message,
            "customEffects": [dict(item) for item in self.custom_effects],
            "pressureIterations": self.pressure_iterations,
            "pressureResidual": self.pressure_residual,
            "viscosityStatus": self.viscosity_status,
            "viscosityConverged": self.viscosity_converged,
            "viscosityIterations": self.viscosity_iterations,
            "viscosityResidual": self.viscosity_residual,
            "nativeSolverMessage": self.native_solver_message,
            "voxelsPerSecond": vps,
            "voxelsPerFrame": self.active_voxels,
            "particlesPerSecond": pps,
            "particlesPerFrame": self.active_particles,
            "usedParticleAllocation": (
                self.active_particles / max(self.total_particles, 1)),
            "usedVoxelAllocation": (
                self.active_voxels / max(self.initialized_voxels, 1)),
            "activeInitializedVoxelRatio": (
                self.active_voxels / max(self.initialized_voxels, 1)),
            "particleInitializedVoxelRatio": (
                self.active_particles / max(self.initialized_voxels, 1)),
            "particleVoxelFillRatio": (
                self.active_particles / max(self.active_voxels, 1)),
            "timers": dict(self.timers),
        }

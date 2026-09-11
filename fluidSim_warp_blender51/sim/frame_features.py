"""Custom feature-phase orchestration for one solver frame."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .effects.enums import EffectStage
from .effects.runtime import EffectRuntimeManager


class FrameFeaturePhase(Enum):
    BEFORE_EMISSION = "before_emission"
    BEFORE_SOURCING = "before_sourcing"
    AFTER_PARTICLE_TRANSFER = "after_particle_transfer"
    BEFORE_PRESSURE = "before_pressure"
    AFTER_PARTICLE_UPDATE = "after_particle_update"


_PHASE_HOOKS = {
    FrameFeaturePhase.BEFORE_EMISSION: EffectStage.BEFORE_EMISSION,
    FrameFeaturePhase.BEFORE_SOURCING: None,
    FrameFeaturePhase.AFTER_PARTICLE_TRANSFER: EffectStage.AFTER_PARTICLE_TRANSFER,
    FrameFeaturePhase.BEFORE_PRESSURE: EffectStage.BEFORE_PRESSURE,
    FrameFeaturePhase.AFTER_PARTICLE_UPDATE: EffectStage.AFTER_PARTICLE_UPDATE,
}


class FrameFeatures:
    """Own the admitted v2 effect stack and its ordered frame hooks."""

    def __init__(self, *, device: str):
        self.effects = EffectRuntimeManager(device=device)

    def begin_frame(self) -> None:
        self.effects.begin_frame()

    @property
    def rebuild_topology_each_substep(self) -> bool:
        return self.effects.requirements.rebuild_topology_each_substep

    def swap_candidate(self, candidate: Any) -> Any:
        return self.effects.swap_candidate(candidate)

    def update_runtime_bindings(self, instances: Any) -> bool:
        return self.effects.update_runtime_bindings(instances)

    def run(
            self,
            phase: FrameFeaturePhase,
            solver: Any,
            time_context: dict[str, Any],
    ) -> None:
        """Run the admitted v2 hooks assigned to one semantic phase."""
        effect_stage = _PHASE_HOOKS[phase]
        if effect_stage is None:
            return
        requirements = self.effects.requirements_for_stage(effect_stage)
        if requirements.surface_sdf:
            solver.surface.ensure_solver_sdf_current()
        self.effects.run(effect_stage, solver, time_context)
        if self.effects.runtime_errors:
            latest = self.effects.runtime_errors[-1]
            raise RuntimeError(
                "Custom Effect runtime failure "
                f"[{latest.effect_id}/{latest.kernel_id}]: {latest.message}"
            )

    def frame_stats_snapshot(self) -> list[dict[str, object]]:
        return self.effects.frame_stats_snapshot()

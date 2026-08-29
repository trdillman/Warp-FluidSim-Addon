"""Custom feature-phase orchestration for one solver frame."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .custom import SnippetManager
from .effects.enums import EffectStage
from .effects.runtime import EffectRuntimeManager
from .enums import CustomKernelOrder


class FrameFeaturePhase(Enum):
    BEFORE_EMISSION = "before_emission"
    BEFORE_SOURCING = "before_sourcing"
    AFTER_PARTICLE_TRANSFER = "after_particle_transfer"
    BEFORE_PRESSURE = "before_pressure"
    AFTER_PARTICLE_UPDATE = "after_particle_update"


_PHASE_HOOKS = {
    FrameFeaturePhase.BEFORE_EMISSION: (
        CustomKernelOrder.FIRST_STEP, EffectStage.BEFORE_EMISSION),
    FrameFeaturePhase.BEFORE_SOURCING: (
        CustomKernelOrder.BEFORE_SOURCING, None),
    FrameFeaturePhase.AFTER_PARTICLE_TRANSFER: (
        CustomKernelOrder.BEFORE_FORCES,
        EffectStage.AFTER_PARTICLE_TRANSFER,
    ),
    FrameFeaturePhase.BEFORE_PRESSURE: (
        CustomKernelOrder.BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION,
        EffectStage.BEFORE_PRESSURE,
    ),
    FrameFeaturePhase.AFTER_PARTICLE_UPDATE: (
        CustomKernelOrder.FINAL_STEP, EffectStage.AFTER_PARTICLE_UPDATE),
}


class FrameFeatures:
    """Own legacy snippets, v2 effects, and their ordered frame hooks."""

    def __init__(self, *, device: str):
        self.snippets = SnippetManager()
        self.effects = EffectRuntimeManager(device=device)

    def begin_frame(self) -> None:
        self.effects.begin_frame()

    @property
    def rebuild_topology_each_substep(self) -> bool:
        return (
            self.snippets.has_any()
            or self.effects.requirements.rebuild_topology_each_substep
        )

    def set_legacy_snippets(self, snippets: Any) -> Any:
        self.snippets.set_snippets(snippets)
        return self.snippets.errors

    def swap_candidate(self, candidate: Any) -> Any:
        return self.effects.swap_candidate(candidate)

    def run(
            self,
            phase: FrameFeaturePhase,
            solver: Any,
            time_context: dict[str, Any],
    ) -> None:
        """Run the legacy and v2 hooks assigned to one semantic phase."""
        snippet_order, effect_stage = _PHASE_HOOKS[phase]
        if snippet_order is not None:
            self.snippets.run(snippet_order, solver, time_context)
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

from __future__ import annotations

from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    ParameterKind,
)
from fluidsim_testpkg.sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)
from fluidsim_testpkg.sim.effects.source import generate_kernel_source


def _instance():
    definition = EffectDefinitionV2(
        effect_id="test.source",
        version="1.0.0",
        name="Source",
        summary="Source",
        description="Source",
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        parameters=(
            ParameterDefinitionV2("strength", "Strength", ParameterKind.FLOAT, 2.0),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="particle_kernel",
                label="Particle Kernel",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="value = strength\nvalue = value * 2.0",
                parameter_keys=("strength",),
            ),
        ),
    )
    return EffectInstanceV2("source-1", definition)


def test_generated_source_is_content_addressed_and_maps_user_lines():
    generated = generate_kernel_source(_instance(), 0)
    assert generated.module_name.startswith("fluidsim_effect_")
    assert len(generated.source_digest) == 64
    assert "@wp.kernel" in generated.source
    assert "strength: float" in generated.source
    assert generated.original_line_for_generated(generated.user_line_start) == 1
    assert generated.original_line_for_generated(generated.user_line_start + 1) == 2


def test_equivalent_generation_is_deterministic():
    first = generate_kernel_source(_instance(), 0)
    second = generate_kernel_source(_instance(), 0)
    assert first.source == second.source
    assert first.module_name == second.module_name
    assert first.kernel_name == second.kernel_name

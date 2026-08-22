from __future__ import annotations

from fluidsim_testpkg.sim import custom
from fluidsim_testpkg.sim.effects.enums import EffectDomain, EffectStage, ParameterKind
from fluidsim_testpkg.sim.enums import (
    CustomKernelArgumentMode,
    CustomKernelConstantType,
    CustomKernelOrder,
    CustomKernelRunOver,
)


def test_legacy_particle_snippet_converts_to_one_kernel_v2_effect():
    snippet = custom.CustomSnippet(
        enabled=True,
        kernel_name="Legacy Vortex",
        order=CustomKernelOrder.BEFORE_FORCES,
        run_over=CustomKernelRunOver.PARTICLES,
        code="particle_velocity[particle_index] = velocity * strength",
        arguments=[
            custom.CustomArgument(
                mode=CustomKernelArgumentMode.FIELD,
                field="particle_velocity",
            ),
            custom.CustomArgument(
                mode=CustomKernelArgumentMode.CONSTANT,
                constant_type=CustomKernelConstantType.FLOAT,
                constant_name="strength",
                constant_value=2.5,
            ),
        ],
    )
    instance, warnings = custom.legacy_snippet_to_effect(snippet, instance_id="legacy-1")
    assert warnings == ()
    assert instance.definition.kernels[0].domain is EffectDomain.PARTICLES
    assert instance.definition.kernels[0].stage is EffectStage.AFTER_PARTICLE_TRANSFER
    assert instance.definition.parameters[0].kind is ParameterKind.FLOAT
    assert instance.parameter_values["strength"] == 2.5
    assert instance.safe_mode is False


def test_legacy_voxel_tile_maps_to_active_voxels_with_warning():
    snippet = custom.CustomSnippet(
        kernel_name="Legacy Tile",
        order=CustomKernelOrder.FINAL_STEP,
        run_over=CustomKernelRunOver.VOXEL_TILE,
        code="temperature[flat_voxel_index] = 0.0",
        arguments=[custom.CustomArgument(field="temperature")],
    )
    instance, warnings = custom.legacy_snippet_to_effect(snippet, instance_id="legacy-2")
    assert instance.definition.kernels[0].domain is EffectDomain.ACTIVE_VOXELS
    assert any("VOXEL_TILE" in warning for warning in warnings)

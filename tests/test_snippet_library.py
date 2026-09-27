"""Unit tests for the legacy snippet records that migrate 1.x scenes.

Bpy-free: the PropertyGroup adapter, record defaults, and the conversion to a
Custom Effects v2 instance.
"""

import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fluidSim_warp_blender51.sim import custom as lib  # noqa: E402


def test_field_permission_and_kernel_type_roundtrip_through_adapter():
    from fluidSim_warp_blender51.blender.snippets import snippet_from_pg
    from fluidSim_warp_blender51.sim.enums import (
        CustomKernelFieldPermission, CustomKernelType,
    )

    argument_pg = SimpleNamespace(
        mode="FIELD", field="temperature", field_permission="READ",
        constant_type="FLOAT", constant_name="value", constant_value=0.0)
    snippet_pg = SimpleNamespace(
        enabled=True, kernel_name="cooling", snippet_type="FUNCTION",
        order="FINAL_STEP", run_over="VOXEL_LINEAR",
        text_block=SimpleNamespace(
            as_string=lambda: "x = temperature[flat_voxel_index]"),
        arguments=[argument_pg], frame_start=0, frame_end=0, frame_step=1)

    snippet = snippet_from_pg(snippet_pg)
    assert snippet.snippet_type is CustomKernelType.FUNCTION
    assert snippet.arguments[0].field_permission is CustomKernelFieldPermission.READ


def test_legacy_runtime_objects_default_to_read_write_kernel():
    from fluidSim_warp_blender51.sim.enums import (
        CustomKernelFieldPermission, CustomKernelType,
    )

    argument = lib.CustomArgument()
    snippet = lib.CustomSnippet()

    assert argument.field_permission is CustomKernelFieldPermission.READ_WRITE
    assert snippet.snippet_type is CustomKernelType.KERNEL


def test_function_snippet_is_rejected_by_effect_conversion():
    from fluidSim_warp_blender51.sim.enums import CustomKernelType

    snippet = lib.CustomSnippet(
        kernel_name="helper", snippet_type=CustomKernelType.FUNCTION,
        code="x = velocity[0]",
    )
    with pytest.raises(ValueError, match="FUNCTION snippets are unsupported"):
        lib.legacy_snippet_to_effect(snippet)


def test_legacy_effect_conversion_preserves_field_permission():
    from fluidSim_warp_blender51.sim.effects.enums import FieldAccess
    from fluidSim_warp_blender51.sim.enums import (
        CustomKernelArgumentMode, CustomKernelFieldPermission,
    )

    snippet = lib.CustomSnippet(
        kernel_name="permission_probe",
        code="value = temperature[flat_voxel_index]",
        arguments=[lib.CustomArgument(
            mode=CustomKernelArgumentMode.FIELD,
            field="temperature",
            field_permission=CustomKernelFieldPermission.READ,
        )],
    )

    effect, _warnings = lib.legacy_snippet_to_effect(snippet)

    assert effect.definition.kernels[0].fields[0].access is FieldAccess.READ

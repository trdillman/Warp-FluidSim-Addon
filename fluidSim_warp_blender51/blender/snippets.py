"""Blender property-group adapter for the custom snippet lifecycle.

The bpy-free validation, compilation, persistence, and runtime implementation
remains in ``sim.custom``. This module converts Blender-authored values only.
"""

from __future__ import annotations


def snippet_from_pg(pg):
    """Convert one snippet property group into a sim CustomSnippet."""
    from ..sim.custom import CustomArgument, CustomSnippet
    from ..sim.enums import (
        CustomKernelArgumentMode,
        CustomKernelConstantType,
        CustomKernelFieldPermission,
        CustomKernelOrder,
        CustomKernelRunOver,
        CustomKernelType,
    )

    code = pg.text_block.as_string() if pg.text_block else ""
    args = []
    for apg in pg.arguments:
        args.append(CustomArgument(
            mode=CustomKernelArgumentMode[apg.mode],
            field=apg.field,
            field_permission=CustomKernelFieldPermission[
                getattr(apg, "field_permission", "READ_WRITE")],
            constant_type=CustomKernelConstantType[apg.constant_type],
            constant_name=apg.constant_name,
            constant_value=apg.constant_value,
        ))
    return CustomSnippet(
        enabled=pg.enabled,
        kernel_name=pg.kernel_name,
        snippet_type=CustomKernelType[
            getattr(pg, "snippet_type", "KERNEL")],
        order=CustomKernelOrder[pg.order],
        run_over=CustomKernelRunOver[pg.run_over],
        code=code,
        arguments=args,
        frame_start=pg.frame_start,
        frame_end=pg.frame_end,
        frame_step=pg.frame_step,
    )

"""Narrow saved-file compatibility forwards for retired snippet operators."""

from __future__ import annotations

import bpy


class FLUIDSIM_OT_snippet_apply(bpy.types.Operator):
    """Forward the retired apply ID to the Custom Effects v2 transaction."""

    bl_idname = "fluidsim.snippet_apply"
    bl_label = "Apply Custom Effects"

    def execute(self, _context):
        result = bpy.ops.fluidsim.effect_apply_stack()
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}

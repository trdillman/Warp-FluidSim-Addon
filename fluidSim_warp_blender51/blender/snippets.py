"""Blender property-group adapter for the custom snippet lifecycle.

The bpy-free validation, compilation, persistence, and runtime implementation
remains in ``sim.custom``. This module converts Blender-authored values only.
"""

from __future__ import annotations

def snippet_argument_names(pg):
    """Names a snippet's declared arguments expose to its body."""
    names = []
    for apg in pg.arguments:
        names.append(apg.field if apg.mode == "FIELD" else apg.constant_name)
    return names


def snippet_field_permissions(pg):
    """Map each authored field argument to its declared effect permission."""
    return {
        apg.field: getattr(apg, "field_permission", "READ_WRITE")
        for apg in pg.arguments
        if apg.mode == "FIELD"
    }


def validate_snippet_pg(pg):
    """Statically validate one snippet property group and mirror the result.

    Updates ``is_valid`` / ``validated_hash`` / ``validation_report`` /
    ``last_error`` on the property group and returns the list of issues.
    """
    from ..sim import custom

    code = pg.text_block.as_string() if pg.text_block else ""
    if getattr(pg, "snippet_type", "KERNEL") != "KERNEL":
        issues = [custom.Issue(
            "ERROR", 0, custom.FUNCTION_UNSUPPORTED_MESSAGE)]
    else:
        issues = custom.validate_snippet(
            code, pg.run_over, snippet_argument_names(pg),
            snippet_field_permissions(pg))
    errors = [i for i in issues if i.level == "ERROR"]
    pg.is_valid = not errors
    pg.validated_hash = custom.code_hash(code)
    pg.validation_report = "\n".join(str(i) for i in issues)[:1024]
    if errors:
        pg.last_error = str(errors[0])[:256]
    elif pg.last_error.startswith("ERROR:"):
        pg.last_error = ""
    return issues


def build_snippets(scene):
    """Convert snippet property groups into sim CustomSnippet objects.

    This is the single choke point through which snippets reach the solver
    (both solver init and Apply). Enabled snippets are statically validated
    here; snippets with errors are gated out so only valid code can run in
    the scene.
    """
    result = []
    for pg in scene.fluidsim.snippets:
        if pg.enabled:
            validate_snippet_pg(pg)
            if not pg.is_valid:
                continue
        result.append(snippet_from_pg(pg))
    return result


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

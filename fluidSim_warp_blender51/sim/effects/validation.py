"""Static validation for Custom Effects v2 Warp kernel bodies."""

from __future__ import annotations

import ast
import difflib
import textwrap
from dataclasses import dataclass

from .enums import EffectDomain, FieldAccess
from .fields import FIELD_SPECS, validate_field_binding
from .schema import EffectInstanceV2, EffectIssue, FieldBindingV2, KernelDefinitionV2


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    issues: tuple[EffectIssue, ...]


_FORBIDDEN_NODES = {
    ast.Import: ("IMPORT_FORBIDDEN", "Import statements are not allowed in effect kernels."),
    ast.ImportFrom: ("IMPORT_FORBIDDEN", "Import statements are not allowed in effect kernels."),
    ast.FunctionDef: ("FUNCTION_FORBIDDEN", "Nested function definitions are not supported."),
    ast.AsyncFunctionDef: ("FUNCTION_FORBIDDEN", "Async functions are not supported."),
    ast.ClassDef: ("CLASS_FORBIDDEN", "Class definitions are not supported."),
    ast.Lambda: ("LAMBDA_FORBIDDEN", "Lambda expressions are not supported."),
    ast.Global: ("GLOBAL_FORBIDDEN", "Global declarations are not allowed."),
    ast.Nonlocal: ("NONLOCAL_FORBIDDEN", "Nonlocal declarations are not allowed."),
    ast.Try: ("TRY_FORBIDDEN", "try/except is not supported in Warp kernels."),
    ast.With: ("WITH_FORBIDDEN", "with blocks are not supported in Warp kernels."),
    ast.Raise: ("RAISE_FORBIDDEN", "raise is not supported in Warp kernels."),
    ast.Assert: ("ASSERT_FORBIDDEN", "assert is not supported in production effect kernels."),
    ast.Delete: ("DELETE_FORBIDDEN", "del is not supported in Warp kernels."),
    ast.Yield: ("YIELD_FORBIDDEN", "yield is not supported in Warp kernels."),
    ast.YieldFrom: ("YIELD_FORBIDDEN", "yield is not supported in Warp kernels."),
    ast.Await: ("AWAIT_FORBIDDEN", "await is not supported in Warp kernels."),
    ast.ListComp: ("COMPREHENSION_FORBIDDEN", "Comprehensions are not supported in Warp kernels."),
    ast.SetComp: ("COMPREHENSION_FORBIDDEN", "Comprehensions are not supported in Warp kernels."),
    ast.DictComp: ("COMPREHENSION_FORBIDDEN", "Comprehensions are not supported in Warp kernels."),
    ast.GeneratorExp: ("COMPREHENSION_FORBIDDEN", "Comprehensions are not supported in Warp kernels."),
    ast.While: ("UNBOUNDED_LOOP", "while loops are not allowed; use a small constant range loop."),
}

_ALLOWED_GLOBALS = {
    "wp", "C", "float", "int", "bool", "abs", "min", "max", "range",
    "True", "False",
}
_TIME_CONTEXT = {
    "frame", "time", "real_time_step", "solver_time_step", "time_scale", "substeps",
}
_PARTICLE_CONTEXT = {"particle_index", "position", "velocity", "voxel_size"}
_VOXEL_CONTEXT = {
    "flat_voxel_index", "leaf_index", "local_x", "local_y", "local_z",
    "global_x", "global_y", "global_z", "world_position", "voxel_size",
}
# Module-level names the generated wrapper imports (source.py). Its generated
# arguments and prelude locals (_positions, _count, _tid, _ijk, ...) all start
# with an underscore and are covered by _is_generated_name.
_GENERATED_MODULE_NAMES = {"voxel_coord_of_flat", "voxel_center_world"}
_RESERVED_ARGUMENTS = (
    _TIME_CONTEXT | _PARTICLE_CONTEXT | _VOXEL_CONTEXT | _ALLOWED_GLOBALS
    | _GENERATED_MODULE_NAMES
)
_GENERATED_NAME_MESSAGE = (
    "Names starting with '_' are reserved for generated kernel code."
)


def _is_generated_name(name: str) -> bool:
    """Whether ``name`` is in the generated-code namespace (bare ``_`` is not)."""
    return len(name) > 1 and name.startswith("_")


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()

    def add_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                add_target(item)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                add_target(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For)):
            add_target(node.target)
    return names


def _line_issue(code: str, message: str, node: ast.AST, kernel: KernelDefinitionV2, *, hint: str = "") -> EffectIssue:
    return EffectIssue(
        code=code,
        message=message,
        kernel_id=kernel.kernel_id,
        line=int(getattr(node, "lineno", 0) or 0),
        column=int(getattr(node, "col_offset", 0) or 0) + 1 if getattr(node, "col_offset", None) is not None else 0,
        hint=hint,
    )


def _same_index(node: ast.AST, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _append_forbidden_node_issue(
    node: ast.AST,
    kernel: KernelDefinitionV2,
    issues: list[EffectIssue],
) -> None:
    for node_type, (code, message) in _FORBIDDEN_NODES.items():
        if isinstance(node, node_type):
            issues.append(_line_issue(code, message, node, kernel))
            break


def _append_loop_issue(
    node: ast.AST,
    kernel: KernelDefinitionV2,
    issues: list[EffectIssue],
) -> None:
    if not isinstance(node, ast.For):
        return
    call = node.iter
    bounded = (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and 1 <= len(call.args) <= 3
        and all(isinstance(arg, ast.Constant) and isinstance(arg.value, int) for arg in call.args)
    )
    if not bounded:
        issues.append(_line_issue(
            "UNBOUNDED_LOOP",
            "Effect loops must use range() with constant integer bounds.",
            node,
            kernel,
        ))
        return

    values = [int(arg.value) for arg in call.args]
    if len(values) == 1:
        start, stop, step = 0, values[0], 1
    elif len(values) == 2:
        start, stop, step = values[0], values[1], 1
    else:
        start, stop, step = values
    iterations = len(range(start, stop, step)) if step else 10_000
    if iterations > 64:
        issues.append(_line_issue(
            "LOOP_LIMIT_EXCEEDED",
            f"Effect loops are limited to 64 iterations; this loop has {iterations}.",
            node,
            kernel,
        ))


def _append_attribute_issues(
    node: ast.AST,
    kernel: KernelDefinitionV2,
    issues: list[EffectIssue],
) -> None:
    if not isinstance(node, ast.Attribute):
        return
    allowed_module = isinstance(node.value, ast.Name) and node.value.id in {"wp", "C"}
    if not allowed_module:
        issues.append(_line_issue(
            "ATTRIBUTE_ACCESS_FORBIDDEN",
            "Attribute access is allowed only on wp.* and C.*.",
            node,
            kernel,
        ))
    elif isinstance(node.value, ast.Name) and node.value.id == "wp":
        try:
            import warp as _wp
        except ModuleNotFoundError:
            _wp = None
        if _wp is not None and not hasattr(_wp, node.attr):
            issues.append(_line_issue(
                "UNKNOWN_WARP_SYMBOL",
                f"Warp has no public symbol wp.{node.attr}.",
                node,
                kernel,
                hint="Choose a function listed in the Warp Code API browser.",
            ))


def _append_name_issues(
    node: ast.AST,
    kernel: KernelDefinitionV2,
    issues: list[EffectIssue],
    *,
    allowed: set[str],
    assigned: set[str],
    known: list[str],
) -> None:
    if not isinstance(node, ast.Name):
        return
    if _is_generated_name(node.id):
        issues.append(_line_issue(
            "RESERVED_NAME",
            f"{node.id!r}: {_GENERATED_NAME_MESSAGE}",
            node,
            kernel,
            hint="Rename it without the leading underscore.",
        ))
        return
    if not isinstance(node.ctx, ast.Load):
        return
    if node.id not in allowed and node.id not in assigned:
        close = difflib.get_close_matches(node.id, known, n=1, cutoff=0.6)
        hint = f"Did you mean {close[0]!r}?" if close else ""
        issues.append(_line_issue(
            "UNKNOWN_NAME",
            f"Unknown name {node.id!r}.",
            node,
            kernel,
            hint=hint,
        ))


_EXPERT_ACCESS = {FieldAccess.ATOMIC_SCATTER, FieldAccess.UNSAFE_ARBITRARY}
# Names whose rebinding would defeat a check: the current-element index the
# same-element rule compares against, and the modules/builtins it trusts.
_CURRENT_INDEX_NAMES = {"particle_index", "flat_voxel_index"}
_PROTECTED_NAMES = _CURRENT_INDEX_NAMES | _ALLOWED_GLOBALS | _GENERATED_MODULE_NAMES


def _atomic_call_field(node: ast.AST, binding_map: dict[str, FieldBindingV2]):
    """Return (call, field Name) when ``node`` is ``wp.atomic_*(field, ...)``."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "wp"
        and node.func.attr.startswith("atomic_")
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in binding_map
    ):
        return None
    return node, node.args[0]


def _subscript_root(node: ast.AST) -> ast.AST:
    while isinstance(node, ast.Subscript):
        node = node.value
    return node


def _append_element_access_issues(
    node: ast.AST,
    index: ast.AST | None,
    binding: FieldBindingV2,
    kernel: KernelDefinitionV2,
    issues: list[EffectIssue],
    *,
    safe_mode: bool,
    is_write: bool,
) -> None:
    spec = FIELD_SPECS[binding.field]
    if safe_mode and binding.access not in _EXPERT_ACCESS:
        if index is None or not _same_index(index, spec.current_index):
            issues.append(_line_issue(
                "NON_CURRENT_ELEMENT_ACCESS",
                f"Safe mode requires {binding.field}[{spec.current_index}].",
                node,
                kernel,
                hint="Use the current element index or switch to reviewed Expert access.",
            ))
    if is_write and binding.access is FieldAccess.READ:
        issues.append(_line_issue(
            "WRITE_REQUIRES_PERMISSION",
            f"Field {binding.field!r} is declared READ-only.",
            node,
            kernel,
        ))


def _append_field_access_issues(
    node: ast.AST,
    kernel: KernelDefinitionV2,
    issues: list[EffectIssue],
    *,
    safe_mode: bool,
    binding_map: dict[str, FieldBindingV2],
) -> None:
    atomic = _atomic_call_field(node, binding_map)
    if atomic is not None:
        call, field_name = atomic
        _append_element_access_issues(
            call, call.args[1] if len(call.args) > 1 else None,
            binding_map[field_name.id], kernel, issues,
            safe_mode=safe_mode, is_write=True,
        )
        return
    if not isinstance(node, ast.Subscript):
        return
    if isinstance(node.value, ast.Name):
        binding = binding_map.get(node.value.id)
        if binding is None:
            return
        _append_element_access_issues(
            node, node.slice, binding, kernel, issues,
            safe_mode=safe_mode, is_write=isinstance(node.ctx, ast.Store),
        )
        return
    # field[i][c] = x stores into field[i]; the inner subscript is a Load
    # node, so check the write against the field at the chain's root.
    root = _subscript_root(node)
    if isinstance(node.ctx, ast.Store) and isinstance(root, ast.Name):
        binding = binding_map.get(root.id)
        if binding is not None and binding.access is FieldAccess.READ:
            issues.append(_line_issue(
                "WRITE_REQUIRES_PERMISSION",
                f"Field {binding.field!r} is declared READ-only.",
                node,
                kernel,
            ))


def _direct_field_uses(tree: ast.AST, binding_map: dict[str, FieldBindingV2]) -> set[int]:
    """ids of field Name nodes used as ``field[...]`` or ``wp.atomic_*(field, ...)``."""
    direct: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            direct.add(id(node.value))
        atomic = _atomic_call_field(node, binding_map)
        if atomic is not None:
            direct.add(id(atomic[1]))
    return direct


def _append_binding_name_issues(
    node: ast.AST,
    kernel: KernelDefinitionV2,
    issues: list[EffectIssue],
    *,
    binding_map: dict[str, FieldBindingV2],
    direct_uses: set[int],
) -> None:
    """Reject aliases of bound fields and rebinding of names checks rely on."""
    if not isinstance(node, ast.Name):
        return
    if isinstance(node.ctx, ast.Store):
        if node.id in _PROTECTED_NAMES or node.id in binding_map:
            issues.append(_line_issue(
                "RESERVED_NAME_ASSIGNMENT",
                f"{node.id!r} cannot be reassigned in an effect kernel.",
                node,
                kernel,
                hint="Use a new local variable name.",
            ))
        return
    if node.id in binding_map and id(node) not in direct_uses:
        issues.append(_line_issue(
            "FIELD_ALIAS_FORBIDDEN",
            f"Field {node.id!r} may only be used as {node.id}[index] "
            "or as the array of a wp.atomic_* call.",
            node,
            kernel,
            hint="Index the field directly instead of assigning or passing the array.",
        ))


def _validate_kernel(instance: EffectInstanceV2, kernel: KernelDefinitionV2) -> list[EffectIssue]:
    issues: list[EffectIssue] = []
    safe_mode = bool(instance.safe_mode and kernel.safe_mode)
    for binding in kernel.fields:
        issues.extend(validate_field_binding(binding, kernel, safe_mode=safe_mode))

    parameter_map = instance.definition.parameter_map()
    for key in kernel.parameter_keys:
        if _is_generated_name(key):
            issues.append(EffectIssue(
                "RESERVED_PARAMETER_NAME",
                f"Parameter {key!r}: {_GENERATED_NAME_MESSAGE}",
                effect_id=instance.definition.effect_id,
                kernel_id=kernel.kernel_id,
                hint="Rename the parameter without the leading underscore.",
            ))
        elif key in _RESERVED_ARGUMENTS or key in FIELD_SPECS:
            issues.append(EffectIssue(
                "RESERVED_PARAMETER_NAME",
                f"Parameter {key!r} conflicts with an effect context, field, "
                "or generated kernel name.",
                effect_id=instance.definition.effect_id,
                kernel_id=kernel.kernel_id,
            ))
        if key not in parameter_map:
            issues.append(EffectIssue(
                "UNKNOWN_PARAMETER",
                f"Kernel references unknown parameter {key!r}.",
                effect_id=instance.definition.effect_id,
                kernel_id=kernel.kernel_id,
            ))

    body = textwrap.dedent(kernel.source).strip()
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        issues.append(EffectIssue(
            "SYNTAX_ERROR",
            exc.msg,
            effect_id=instance.definition.effect_id,
            kernel_id=kernel.kernel_id,
            line=int(exc.lineno or 0),
            column=int(exc.offset or 0),
        ))
        return issues

    allowed = set(_ALLOWED_GLOBALS) | set(_TIME_CONTEXT)
    allowed.update(_PARTICLE_CONTEXT if kernel.domain is EffectDomain.PARTICLES else _VOXEL_CONTEXT)
    allowed.update(binding.field for binding in kernel.fields)
    allowed.update(kernel.parameter_keys)
    assigned = _assigned_names(tree)
    known = sorted(allowed | set(FIELD_SPECS))
    binding_map = {binding.field: binding for binding in kernel.fields}
    direct_uses = _direct_field_uses(tree, binding_map)

    for node in ast.walk(tree):
        _append_forbidden_node_issue(node, kernel, issues)
        _append_loop_issue(node, kernel, issues)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            if safe_mode:
                issues.append(_line_issue(
                    "PRINT_FORBIDDEN",
                    "print() is disabled in Safe mode and production bakes.",
                    node,
                    kernel,
                    hint="Use diagnostics or explicitly enable Expert mode for local debugging.",
                ))

        _append_attribute_issues(node, kernel, issues)
        _append_name_issues(
            node,
            kernel,
            issues,
            allowed=allowed,
            assigned=assigned,
            known=known,
        )
        _append_field_access_issues(
            node,
            kernel,
            issues,
            safe_mode=safe_mode,
            binding_map=binding_map,
        )
        _append_binding_name_issues(
            node,
            kernel,
            issues,
            binding_map=binding_map,
            direct_uses=direct_uses,
        )

    return issues


def validate_effect(instance: EffectInstanceV2, *, safe_mode: bool | None = None) -> ValidationResult:
    if safe_mode is not None and bool(safe_mode) != instance.safe_mode:
        from dataclasses import replace
        instance = replace(instance, safe_mode=bool(safe_mode))
    issues: list[EffectIssue] = []
    for kernel in instance.definition.kernels:
        issues.extend(_validate_kernel(instance, kernel))
    enriched = tuple(
        EffectIssue(
            code=issue.code,
            message=issue.message,
            severity=issue.severity,
            effect_id=issue.effect_id or instance.definition.effect_id,
            instance_id=issue.instance_id or instance.instance_id,
            kernel_id=issue.kernel_id,
            line=issue.line,
            column=issue.column,
            hint=issue.hint,
        )
        for issue in issues
    )
    return ValidationResult(
        valid=not any(issue.severity == "ERROR" for issue in enriched),
        issues=enriched,
    )

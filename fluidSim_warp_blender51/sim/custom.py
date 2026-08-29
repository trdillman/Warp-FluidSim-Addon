"""Bpy-free custom Warp snippet lifecycle.

This Module owns snippet validation, conservative source repair, presets,
shortcuts, persistence, runtime descriptions, compilation, and launch.
Blender-facing code imports this Module through adapters; no bpy dependency
crosses the Seam.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import textwrap
from dataclasses import dataclass, field as dc_field

from . import constants as C
from .enums import (
    CustomKernelFieldPermission,
    CustomKernelOrder,
    CustomKernelRunOver,
    CustomKernelArgumentMode,
    CustomKernelConstantType,
    CustomKernelType,
)


# ---------------------------------------------------------------------------
# Known snippet context (kept in sync with runtime templates below)
# ---------------------------------------------------------------------------

# Mirrors FIELD_ACCESSORS keys used by snippet declarations. Grid fields run
# over voxels; particle_* fields run over particles.
KNOWN_FIELDS = (
    "velocity_u", "velocity_v", "velocity_w",
    "weight_u", "weight_v", "weight_w",
    "temperature", "viscosity", "density", "pressure", "divergence",
    "surface_sdf", "collision_sdf", "churn",
    "voxel_type", "voxel_counts",
    "particle_position", "particle_velocity",
    "particle_age", "particle_life", "particle_temperature",
    "particle_viscosity", "particle_density",
)

TIME_CONTEXT = ("frame", "time", "real_time_step", "solver_time_step",
                "time_scale", "substeps")

VOXEL_CONTEXT = ("flat_voxel_index", "leaf_index",
                 "local_x", "local_y", "local_z",
                 "global_x", "global_y", "global_z",
                 "world_position", "voxel_size")

PARTICLE_CONTEXT = ("particle_index", "position", "velocity", "voxel_size")

# Names always readable inside a snippet body.
_ALLOWED_BUILTINS = ("float", "int", "bool", "abs", "min", "max", "range",
                     "print", "True", "False")

_ALLOWED_MODULES = ("wp", "C")


def context_names(run_over: str):
    """All readable names for a run-over ('PARTICLES' or VOXEL_*)."""
    base = PARTICLE_CONTEXT if run_over == "PARTICLES" else VOXEL_CONTEXT
    return set(base) | set(TIME_CONTEXT) | set(_ALLOWED_BUILTINS) | set(_ALLOWED_MODULES)


def code_hash(code: str) -> str:
    """Stable hash used to detect edits after validation."""
    return hashlib.sha1(code.encode("utf-8", "replace")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    level: str      # "ERROR" or "WARNING"
    line: int       # 1-based line in the snippet body (0 = whole snippet)
    message: str

    def __str__(self):
        where = f"L{self.line}: " if self.line else ""
        return f"{self.level}: {where}{self.message}"


_FORBIDDEN_NODES = {
    ast.Import: "import statements are not allowed in snippets",
    ast.ImportFrom: "import statements are not allowed in snippets",
    ast.FunctionDef: "function definitions are not supported in Warp kernel bodies",
    ast.AsyncFunctionDef: "async functions are not supported",
    ast.ClassDef: "class definitions are not supported",
    ast.Lambda: "lambda is not supported in Warp kernels",
    ast.Global: "global is not allowed",
    ast.Nonlocal: "nonlocal is not allowed",
    ast.Try: "try/except is not supported in Warp kernels",
    ast.With: "with blocks are not supported in Warp kernels",
    ast.Raise: "raise is not supported in Warp kernels",
    ast.Assert: "assert is not supported in Warp kernels (use wp.expect_eq in tests)",
    ast.Delete: "del is not supported",
    ast.Yield: "yield is not supported",
    ast.YieldFrom: "yield is not supported",
    ast.Await: "await is not supported",
    ast.ListComp: "comprehensions are not supported in Warp kernels",
    ast.SetComp: "comprehensions are not supported in Warp kernels",
    ast.DictComp: "comprehensions are not supported in Warp kernels",
    ast.GeneratorExp: "generator expressions are not supported in Warp kernels",
    ast.Dict: "dict literals are not supported in Warp kernels",
    ast.Set: "set literals are not supported in Warp kernels",
    ast.List: "list literals are not supported in Warp kernels (use wp.vec3(...))",
    ast.Slice: "slicing is not supported in Warp kernels",
    ast.JoinedStr: "f-strings are not supported in Warp kernels",
    ast.Starred: "star expressions are not supported",
}


def _add_target_names(tgt, names):
    """Collect names bound by an assignment target.

    Only plain names (and tuple unpacking of plain names) introduce new
    bindings; ``field[i] = ...`` stores into an existing array, so the base
    name must already be known and is NOT treated as assigned.
    """
    if isinstance(tgt, ast.Name):
        names.add(tgt.id)
    elif isinstance(tgt, (ast.Tuple, ast.List)):
        for elt in tgt.elts:
            _add_target_names(elt, names)


def _collect_assigned_names(tree: ast.AST):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                _add_target_names(tgt, names)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
            _add_target_names(node.target, names)
    return names


def _first_assignment_positions(tree: ast.AST):
    """Return the source position after the first binding of each local."""
    positions = {}
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
            targets = [node.target]
        names = set()
        for target in targets:
            _add_target_names(target, names)
        if not names:
            continue
        position = (
            getattr(node, "end_lineno", node.lineno),
            getattr(node, "end_col_offset", node.col_offset),
        )
        for name in names:
            positions[name] = min(positions.get(name, position), position)
    return positions


def _suggest(name: str, candidates):
    close = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return f" (did you mean '{close[0]}'?)" if close else ""


_MODELED_ATOMICS = {
    "atomic_add", "atomic_sub", "atomic_min", "atomic_max",
    "atomic_and", "atomic_or", "atomic_xor", "atomic_exch", "atomic_cas",
}

FUNCTION_UNSUPPORTED_MESSAGE = (
    "FUNCTION snippets are unsupported; helper linkage is undefined"
)


def _permission_name(permission) -> str:
    if isinstance(permission, CustomKernelFieldPermission):
        return permission.name
    return str(permission).upper()


def _field_permission_issues(tree: ast.AST, field_permissions) -> list:
    """Conservatively enforce effects on permission-bearing field arguments."""
    if not field_permissions:
        return []

    permissions = {}
    issues = []
    for name, permission in dict(field_permissions).items():
        permission_name = _permission_name(permission)
        if permission_name not in {"READ", "WRITE", "READ_WRITE"}:
            issues.append(Issue(
                "ERROR", 0,
                f"field '{name}' has unknown permission '{permission_name}'"))
        else:
            permissions[str(name)] = permission_name
    if not permissions:
        return issues

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def add(node, message):
        issues.append(Issue("ERROR", getattr(node, "lineno", 0), message))

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            field_name = node.value.id
            permission = permissions.get(field_name)
            if permission is None:
                continue
            parent = parents.get(node)
            read_modify_write = (
                isinstance(parent, ast.AugAssign) and parent.target is node)
            if read_modify_write and permission != "READ_WRITE":
                add(node, f"{permission} field '{field_name}' forbids read-modify-write")
            elif isinstance(node.ctx, ast.Load) and permission == "WRITE":
                add(node, f"WRITE field '{field_name}' forbids loads")
            elif isinstance(node.ctx, (ast.Store, ast.Del)) and permission == "READ":
                add(node, f"READ field '{field_name}' forbids stores")

        elif isinstance(node, ast.Call):
            atomic_name = None
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "wp"
                    and node.func.attr in _MODELED_ATOMICS):
                atomic_name = node.func.attr
            if atomic_name:
                if not node.args or not isinstance(node.args[0], ast.Name):
                    add(node, f"wp.{atomic_name} requires a direct bound field argument")
                    continue
                field_name = node.args[0].id
                permission = permissions.get(field_name)
                if permission is None:
                    add(node, f"wp.{atomic_name} target '{field_name}' is not a permission-bearing field")
                elif permission != "READ_WRITE":
                    add(node, f"{permission} field '{field_name}' forbids atomic read-modify-write")
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Name) and arg.id in permissions:
                        add(arg, f"field '{arg.id}' cannot alias an atomic operand")
                continue
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in permissions:
                    add(arg, f"field '{arg.id}' cannot be passed to an unmodelled call")

        elif isinstance(node, ast.Name) and node.id in permissions:
            parent = parents.get(node)
            if isinstance(parent, ast.Subscript) and parent.value is node:
                continue
            if (isinstance(parent, ast.Call) and parent.args
                    and parent.args[0] is node
                    and isinstance(parent.func, ast.Attribute)
                    and isinstance(parent.func.value, ast.Name)
                    and parent.func.value.id == "wp"
                    and parent.func.attr in _MODELED_ATOMICS):
                continue
            if isinstance(parent, ast.Call) and node in parent.args:
                continue
            add(node, f"field '{node.id}' cannot be aliased or rebound")

    return issues


def validate_snippet(code: str, run_over: str = "VOXEL_LINEAR",
                     arg_names=(), field_permissions=None) -> list:
    """Statically validate a snippet body. Returns a list of Issues.

    A snippet is runnable only when no ERROR-level issues remain. This is a
    pre-flight check; Warp codegen remains the final authority at compile
    time (fluidsim.snippet_test / snippet_apply run that stage).
    """
    issues = []
    body = textwrap.dedent(code or "").strip()
    if not body:
        issues.append(Issue("ERROR", 0, "snippet is empty"))
        return issues

    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        issues.append(Issue("ERROR", exc.lineno or 0,
                            f"syntax error: {exc.msg}"))
        return issues

    issues.extend(_field_permission_issues(tree, field_permissions))

    allowed = context_names(run_over) | set(arg_names)
    assigned = _collect_assigned_names(tree)
    first_assignments = _first_assignment_positions(tree)
    known_all = sorted(allowed | set(KNOWN_FIELDS))

    for node in ast.walk(tree):
        for node_type, message in _FORBIDDEN_NODES.items():
            if isinstance(node, node_type):
                issues.append(Issue("ERROR", getattr(node, "lineno", 0), message))
                break
        if isinstance(node, ast.Attribute):
            base = node.value
            if not (isinstance(base, ast.Name) and base.id in _ALLOWED_MODULES):
                if node.attr in ("x", "y", "z", "w"):
                    issues.append(Issue(
                        "ERROR", node.lineno,
                        f"attribute access '.{node.attr}' is not supported; "
                        f"index vectors instead (e.g. v[0]) - Auto Fix can do this"))
                else:
                    issues.append(Issue(
                        "ERROR", node.lineno,
                        f"attribute access '.{node.attr}' is only allowed on "
                        f"wp.* and C.*"))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if (node.id not in allowed
                    and (node.id not in assigned
                         or (node.lineno, node.col_offset)
                         < first_assignments[node.id])):
                extra = ""
                if node.id in KNOWN_FIELDS:
                    extra = (" - it is a known solver field; declare it as a "
                             "FIELD argument on this snippet first")
                else:
                    extra = _suggest(node.id, known_all)
                issues.append(Issue(
                    "ERROR", node.lineno, f"unknown name '{node.id}'{extra}"))
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            position = (
                getattr(node, "end_lineno", node.lineno),
                getattr(node, "end_col_offset", node.col_offset),
            )
            if name not in allowed and first_assignments.get(name) == position:
                issues.append(Issue(
                    "ERROR", node.lineno,
                    f"unknown name '{name}' before augmented assignment"))
        elif isinstance(node, ast.Call) and node.keywords:
            issues.append(Issue(
                "WARNING", node.lineno,
                "keyword arguments are not supported by most Warp builtins"))

    # Particle copies are local; writing them has no effect on the sim.
    if run_over == "PARTICLES":
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) \
                    and node.id in ("position", "velocity"):
                issues.append(Issue(
                    "WARNING", node.lineno,
                    f"'{node.id}' is a per-thread copy; bind the "
                    f"'particle_{node.id}' field argument and write "
                    f"particle_{node.id}[particle_index] to affect the sim"))
    return issues


def is_valid(code: str, run_over: str = "VOXEL_LINEAR", arg_names=(),
             field_permissions=None) -> bool:
    return not any(i.level == "ERROR"
                   for i in validate_snippet(
                       code, run_over, arg_names, field_permissions))


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------

_SMART_CHARS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ",
}

# `1.0f` / `2f` style C float suffixes.
_FLOAT_SUFFIX_RE = re.compile(r"\b(\d+(?:\.\d*)?)[fF]\b")
# `.x/.y/.z/.w` on names other than wp/C -> component indexing.
_COMPONENT_RE = re.compile(r"\b(?!wp\b|C\b)([A-Za-z_][A-Za-z0-9_]*)\.([xyzw])\b")
_COMPONENT_MAP = {"x": "[0]", "y": "[1]", "z": "[2]", "w": "[3]"}
# math.func -> wp.func
_MATH_RE = re.compile(r"\bmath\.")
# np.func -> wp.func
_NUMPY_RE = re.compile(r"\bnp\.")


def autofix_snippet(code: str):
    """Conservatively repair common snippet mistakes.

    Returns ``(fixed_code, [descriptions of applied fixes])``. Only textual,
    meaning-preserving repairs are attempted (OpenCL/C habits, smart quotes,
    tabs, vector component attributes ...). The result still needs to pass
    validation before it can run.
    """
    fixes = []
    text = (code or "").replace("\r\n", "\n").replace("\r", "\n")

    for bad, good in _SMART_CHARS.items():
        if bad in text:
            text = text.replace(bad, good)
            fixes.append("replaced typographic characters with ASCII")
    if "\t" in text:
        text = text.replace("\t", "    ")
        fixes.append("converted tabs to 4 spaces")

    lines = text.split("\n")
    out_lines = []
    for raw in lines:
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("//"):
            indent = line[: len(line) - len(stripped)]
            line = indent + "#" + stripped[2:]
            if "converted // comments to #" not in fixes:
                fixes.append("converted // comments to #")
        else:
            code_part, sep, comment = line.partition("#")
            new_code = code_part
            if new_code.rstrip().endswith(";"):
                new_code = new_code.rstrip()[:-1]
                if "removed trailing semicolons" not in fixes:
                    fixes.append("removed trailing semicolons")
            for pat, repl, desc in (
                (r"&&", " and ", "replaced && with 'and'"),
                (r"\|\|", " or ", "replaced || with 'or'"),
                (r"\btrue\b", "True", "capitalized boolean literals"),
                (r"\bfalse\b", "False", "capitalized boolean literals"),
            ):
                new_code, n = re.subn(pat, repl, new_code)
                if n and desc not in fixes:
                    fixes.append(desc)
            new_code, n = _FLOAT_SUFFIX_RE.subn(r"\1", new_code)
            if n and "removed C float suffixes (1.0f -> 1.0)" not in fixes:
                fixes.append("removed C float suffixes (1.0f -> 1.0)")
            new_code, n = _MATH_RE.subn("wp.", new_code)
            if n and "rewrote math.* to wp.*" not in fixes:
                fixes.append("rewrote math.* to wp.*")
            new_code, n = _NUMPY_RE.subn("wp.", new_code)
            if n and "rewrote np.* to wp.*" not in fixes:
                fixes.append("rewrote np.* to wp.*")
            new_code, n = _COMPONENT_RE.subn(
                lambda m: m.group(1) + _COMPONENT_MAP[m.group(2)], new_code)
            if n and "rewrote vector .x/.y/.z to [0]/[1]/[2]" not in fixes:
                fixes.append("rewrote vector .x/.y/.z to [0]/[1]/[2]")
            # collapse doubled spaces introduced by and/or substitution
            new_code = re.sub(r"(\S) {2,}", r"\1 ", new_code)
            line = new_code + (sep + comment if sep else "")
        out_lines.append(line)

    fixed = "\n".join(out_lines)
    dedented = textwrap.dedent(fixed)
    if dedented != fixed:
        fixed = dedented
        fixes.append("normalized indentation")
    if not fixed.strip():
        fixed = "pass"
        fixes.append("inserted 'pass' for empty snippet")
    return fixed, fixes


# ---------------------------------------------------------------------------
# Example presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnippetPreset:
    preset_id: str
    label: str
    category: str          # Impressive / Creative / Learning / Utility
    description: str
    order: str             # CustomKernelOrder member name
    run_over: str          # CustomKernelRunOver member name
    code: str
    arguments: tuple = ()  # dicts: FIELD {mode,field} / CONSTANT {mode,name,type,value}
    frame_start: int = 0   # 0 = from the first frame
    frame_end: int = 0     # 0 = never stops
    frame_step: int = 1    # run every Nth frame
    snippet_type: str = "KERNEL"


def _f(name, field_permission="READ_WRITE"):
    return {
        "mode": "FIELD",
        "field": name,
        "field_permission": field_permission,
    }


def _c(name, value, ctype="FLOAT"):
    return {"mode": "CONSTANT", "name": name, "type": ctype, "value": value}


EXAMPLE_PRESETS = (
    # -- Impressive ---------------------------------------------------------
    SnippetPreset(
        preset_id="vortex_tornado",
        label="Vortex Tornado",
        category="Impressive",
        description=("Spins the liquid around a vertical axis with suction and "
                     "updraft - a self-organising tornado funnel"),
        order="BEFORE_FORCES",
        run_over="PARTICLES",
        arguments=(_f("particle_velocity"),
                   _c("strength", 14.0), _c("center_x", 0.0), _c("center_y", 0.0)),
        code=(
            "# Tornado: tangential swirl + inward suction + core updraft.\n"
            "to_axis = wp.vec3(position[0] - center_x, position[1] - center_y, 0.0)\n"
            "r = wp.length(to_axis) + 1.0e-5\n"
            "radial = to_axis / r\n"
            "tangent = wp.vec3(-radial[1], radial[0], 0.0)\n"
            "falloff = 1.0 / (1.0 + r * r)\n"
            "swirl = tangent * (strength * falloff)\n"
            "suction = radial * (-0.35 * strength * falloff)\n"
            "updraft = wp.vec3(0.0, 0.0, 0.45 * strength * falloff)\n"
            "particle_velocity[particle_index] = velocity + "
            "(swirl + suction + updraft) * solver_time_step\n"
        ),
    ),
    SnippetPreset(
        preset_id="curl_noise_turbulence",
        label="Curl Noise Turbulence",
        category="Impressive",
        description=("Divergence-free curl noise churns the liquid into "
                     "animated, billowing turbulence"),
        order="BEFORE_FORCES",
        run_over="PARTICLES",
        arguments=(_f("particle_velocity"),
                   _c("strength", 6.0), _c("frequency", 1.5),
                   _c("drift", 0.35), _c("seed", 42.0, "INT")),
        code=(
            "# Curl noise advection: swirling, volume-preserving turbulence.\n"
            "state = wp.rand_init(seed)\n"
            "p = position * frequency + wp.vec3(0.0, 0.0, time * drift)\n"
            "turb = wp.curlnoise(state, p)\n"
            "particle_velocity[particle_index] = velocity + "
            "turb * (strength * solver_time_step)\n"
        ),
    ),
    # -- Creative -----------------------------------------------------------
    SnippetPreset(
        preset_id="moses_split",
        label="Moses Split",
        category="Creative",
        description=("Parts the liquid along a plane like the Red Sea - "
                     "push strength fades with distance from the divide"),
        order="BEFORE_FORCES",
        run_over="PARTICLES",
        arguments=(_f("particle_velocity"),
                   _c("plane_x", 0.0), _c("width", 0.8), _c("strength", 18.0)),
        code=(
            "# Part the sea: push particles away from the x = plane_x wall.\n"
            "d = position[0] - plane_x\n"
            "side = wp.sign(d)\n"
            "inside = wp.max(0.0, 1.0 - wp.abs(d) / width)\n"
            "push = wp.vec3(side * strength * inside, 0.0, 0.0)\n"
            "particle_velocity[particle_index] = velocity + push * solver_time_step\n"
        ),
    ),
    SnippetPreset(
        preset_id="heartbeat_pulse",
        label="Heartbeat Pulse",
        category="Creative",
        description=("The whole body of liquid throbs outward from a centre "
                     "point on a rhythmic sine beat"),
        order="BEFORE_FORCES",
        run_over="PARTICLES",
        arguments=(_f("particle_velocity"),
                   _c("beats_per_second", 1.2), _c("strength", 9.0),
                   _c("center_x", 0.0), _c("center_y", 0.0)),
        code=(
            "# Radial pulse: only the positive half of the sine pushes.\n"
            "beat = wp.sin(time * 6.2831853 * beats_per_second)\n"
            "pulse = wp.max(0.0, beat)\n"
            "outward = wp.vec3(position[0] - center_x, position[1] - center_y, 0.0)\n"
            "r = wp.length(outward) + 1.0e-5\n"
            "push = (outward / r) * (strength * pulse / (1.0 + r))\n"
            "particle_velocity[particle_index] = velocity + push * solver_time_step\n"
        ),
    ),
    # -- Learning -----------------------------------------------------------
    SnippetPreset(
        preset_id="learn_fields_cooling",
        label="Learn: Fields & Indexing",
        category="Learning",
        description=("Commented walkthrough of voxel snippets: reading and "
                     "writing a bound grid field with flat_voxel_index"),
        order="FINAL_STEP",
        run_over="VOXEL_LINEAR",
        arguments=(_f("temperature"), _c("cool_rate", 0.8)),
        code=(
            "# LEARNING EXAMPLE - grid fields and voxel indexing.\n"
            "#\n"
            "# This body runs once per active fluid voxel, every substep.\n"
            "# Collision voxels are skipped automatically before you run.\n"
            "#\n"
            "# Useful context variables (all read-only):\n"
            "#   flat_voxel_index          - index into bound grid fields\n"
            "#   global_x/global_y/global_z - integer voxel coordinates\n"
            "#   world_position            - voxel centre in world space (wp.vec3)\n"
            "#   time / solver_time_step   - simulation clock\n"
            "#\n"
            "# 'temperature' works below because it is declared as a FIELD\n"
            "# argument on this snippet (see the Arguments list).\n"
            "t = temperature[flat_voxel_index]\n"
            "# Exponential decay is frame-rate independent:\n"
            "t = t * wp.exp(-cool_rate * solver_time_step)\n"
            "temperature[flat_voxel_index] = t\n"
        ),
    ),
    SnippetPreset(
        preset_id="learn_particles_speed_limit",
        label="Learn: Particles & Velocity",
        category="Learning",
        description=("Commented walkthrough of particle snippets: reading "
                     "velocity, branching, and writing back safely"),
        order="FINAL_STEP",
        run_over="PARTICLES",
        arguments=(_f("particle_velocity"), _c("max_speed", 4.0)),
        code=(
            "# LEARNING EXAMPLE - particle snippets.\n"
            "#\n"
            "# This body runs once per live particle, every substep.\n"
            "#   particle_index - this particle's index into bound arrays\n"
            "#   position       - world position (per-thread copy, wp.vec3)\n"
            "#   velocity       - velocity (per-thread copy, wp.vec3)\n"
            "#\n"
            "# IMPORTANT: 'position'/'velocity' are copies. To change the\n"
            "# simulation, bind a particle field argument and write to it,\n"
            "# as done with 'particle_velocity' here.\n"
            "speed = wp.length(velocity)\n"
            "if speed > max_speed:\n"
            "    particle_velocity[particle_index] = velocity * (max_speed / speed)\n"
        ),
    ),
    # -- Utility / fun ------------------------------------------------------
    SnippetPreset(
        preset_id="whirlpool_drain",
        label="Whirlpool Drain",
        category="Utility",
        description=("Bathtub drain: tangential swirl plus a strong downward "
                     "pull near the centre"),
        order="BEFORE_FORCES",
        run_over="PARTICLES",
        arguments=(_f("particle_velocity"),
                   _c("strength", 10.0), _c("radius", 1.5),
                   _c("center_x", 0.0), _c("center_y", 0.0)),
        code=(
            "# Whirlpool: swirl everywhere, drain-pull inside 'radius'.\n"
            "to_axis = wp.vec3(position[0] - center_x, position[1] - center_y, 0.0)\n"
            "r = wp.length(to_axis) + 1.0e-5\n"
            "radial = to_axis / r\n"
            "tangent = wp.vec3(-radial[1], radial[0], 0.0)\n"
            "core = wp.max(0.0, 1.0 - r / radius)\n"
            "swirl = tangent * (strength / (1.0 + r))\n"
            "sink = wp.vec3(0.0, 0.0, -strength * core)\n"
            "pull = radial * (-0.5 * strength * core)\n"
            "particle_velocity[particle_index] = velocity + "
            "(swirl + sink + pull) * solver_time_step\n"
        ),
    ),
    SnippetPreset(
        preset_id="wave_maker",
        label="Wave Maker",
        category="Utility",
        description=("Oscillating horizontal push that builds standing waves "
                     "and sloshing - great with open water scenes"),
        order="BEFORE_FORCES",
        run_over="PARTICLES",
        arguments=(_f("particle_velocity"),
                   _c("period", 2.5), _c("strength", 3.0)),
        code=(
            "# Sinusoidal gust along +X/-X. Layer with Curl Noise for chop.\n"
            "phase = time * 6.2831853 / period\n"
            "gust = wp.sin(phase) * strength\n"
            "particle_velocity[particle_index] = velocity + "
            "wp.vec3(gust, 0.0, 0.0) * solver_time_step\n"
        ),
    ),
    SnippetPreset(
        preset_id="thermal_buoyancy",
        label="Thermal Buoyancy",
        category="Utility",
        description=("Hot particles rise: lifts particles by their "
                     "temperature attribute (emit heat from a source)"),
        order="BEFORE_FORCES",
        run_over="PARTICLES",
        arguments=(_f("particle_velocity"), _f("particle_temperature"),
                   _c("buoyancy", 4.0)),
        code=(
            "# Boussinesq-style lift from the particle temperature attribute.\n"
            "heat = particle_temperature[particle_index]\n"
            "lift = wp.vec3(0.0, 0.0, buoyancy * heat)\n"
            "particle_velocity[particle_index] = velocity + lift * solver_time_step\n"
        ),
    ),
    SnippetPreset(
        preset_id="age_dye",
        label="Age Dye (Flow History)",
        category="Utility",
        description=("Writes particle age into the temperature attribute so "
                     "'Temperature' viewport coloring reveals flow history"),
        order="FINAL_STEP",
        run_over="PARTICLES",
        arguments=(_f("particle_temperature"), _f("particle_age"),
                   _c("dye_scale", 1.0)),
        code=(
            "# Dye by age: set Output > Particle Attribute to 'Temperature'\n"
            "# to see where old vs freshly emitted liquid travels.\n"
            "particle_temperature[particle_index] = "
            "particle_age[particle_index] * dye_scale\n"
        ),
    ),
)


def preset_by_id(preset_id: str):
    for preset in EXAMPLE_PRESETS:
        if preset.preset_id == preset_id:
            return preset
    return None


# ---------------------------------------------------------------------------
# User preset persistence (JSON files, one per preset)
# ---------------------------------------------------------------------------


def _normalized_preset_arguments(arguments):
    normalized = []
    for argument in arguments:
        item = dict(argument)
        if item.get("mode") == "FIELD":
            item.setdefault("field_permission", "READ_WRITE")
        normalized.append(item)
    return normalized


def preset_to_dict(label, description, order, run_over, code, arguments,
                   frame_start=0, frame_end=0, frame_step=1,
                   snippet_type="KERNEL"):
    return {
        "format": "fluidsim_warp_snippet/1",
        "label": label,
        "description": description,
        "snippet_type": snippet_type,
        "order": order,
        "run_over": run_over,
        "code": code,
        "arguments": _normalized_preset_arguments(arguments),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_step": int(frame_step),
    }


def _safe_filename(label: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "snippet"
    return base + ".json"


def save_user_preset(directory: str, data: dict) -> str:
    """Write a user preset JSON into ``directory``; returns the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, _safe_filename(data.get("label", "snippet")))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def load_user_presets(directory: str) -> list:
    """Load user preset dicts from ``directory`` (sorted by label)."""
    presets = []
    if not directory or not os.path.isdir(directory):
        return presets
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or "code" not in data:
                continue
            data.setdefault("label", os.path.splitext(name)[0])
            data.setdefault("snippet_type", "KERNEL")
            data.setdefault("order", "BEFORE_FORCES")
            data.setdefault("run_over", "PARTICLES")
            data.setdefault("description", "User snippet")
            data.setdefault("arguments", [])
            data["arguments"] = _normalized_preset_arguments(data["arguments"])
            data["_path"] = path
            presets.append(data)
        except (OSError, TypeError, ValueError):
            continue
    return presets


# ---------------------------------------------------------------------------
# Shortcuts (Houdini-wrangle-style quick insertions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shortcut:
    shortcut_id: str
    label: str
    category: str            # Context / Time / Math / Constants
    code: str                 # line(s) inserted into the snippet
    run_over: str = "ANY"    # ANY / PARTICLES / VOXEL
    description: str = ""


SHORTCUTS = (
    # Context ---------------------------------------------------------------
    Shortcut("ctx_world_pos", "World Position", "Context",
             "pos = world_position", "VOXEL",
             "Voxel centre in world space (wp.vec3)"),
    Shortcut("ctx_voxel_coord", "Voxel Coordinate", "Context",
             "cell = wp.vec3(float(global_x), float(global_y), float(global_z))",
             "VOXEL", "Integer voxel coordinate as a vector"),
    Shortcut("ctx_particle_speed", "Speed", "Context",
             "speed = wp.length(velocity)", "PARTICLES",
             "Velocity magnitude of this particle"),
    Shortcut("ctx_flow_dir", "Flow Direction", "Context",
             "flow_dir = velocity / (wp.length(velocity) + 1.0e-6)",
             "PARTICLES", "Normalized velocity direction"),
    # Time --------------------------------------------------------------------
    Shortcut("time_dt", "Substep dt", "Time",
             "dt = solver_time_step", "ANY",
             "Seconds advanced by this substep (scale forces by this)"),
    Shortcut("time_seconds", "Sim Time", "Time",
             "t = time", "ANY", "Simulation time in seconds"),
    Shortcut("time_sine", "Sine Wave", "Time",
             "wave = wp.sin(time * 6.2831853 * 1.0)  # 1.0 = Hz", "ANY",
             "Oscillator driven by simulation time"),
    Shortcut("time_frame", "Frame Number", "Time",
             "f = float(frame)", "ANY", "Current frame as a float"),
    # Math ----------------------------------------------------------------------
    Shortcut("math_lerp", "Lerp", "Math",
             "mixed = wp.lerp(0.0, 1.0, 0.5)  # a, b, t", "ANY",
             "Linear interpolation"),
    Shortcut("math_clamp", "Clamp", "Math",
             "limited = wp.clamp(0.0, -1.0, 1.0)  # x, lo, hi", "ANY",
             "Clamp a value into a range"),
    Shortcut("math_noise", "Noise", "Math",
             "state = wp.rand_init(42)\n"
             "n = wp.noise(state, position * 2.0 + wp.vec3(0.0, 0.0, time))",
             "PARTICLES", "Scalar gradient noise (-1..1)"),
    Shortcut("math_curl", "Curl Noise", "Math",
             "state = wp.rand_init(42)\n"
             "turb = wp.curlnoise(state, position * 1.5)",
             "PARTICLES", "Divergence-free vector noise"),
    # Constants -------------------------------------------------------------------
    Shortcut("const_leaf", "Leaf Constants", "Constants",
             "vol = C.LEAF_VOLUME  # 512 voxels per leaf (8x8x8)", "VOXEL",
             "Sparse-grid leaf constants from sim.constants"),
    Shortcut("const_tau", "Tau (2*pi)", "Constants",
             "TAU = 6.2831853", "ANY", "Full circle in radians"),
    Shortcut("const_gravity", "Gravity", "Constants",
             "GRAVITY = 9.81", "ANY", "Standard gravity magnitude"),
)


def shortcuts_for(run_over: str):
    """Shortcuts applicable to a run-over member name."""
    domain = "PARTICLES" if run_over == "PARTICLES" else "VOXEL"
    return [s for s in SHORTCUTS if s.run_over in ("ANY", domain)]


def shortcut_by_id(shortcut_id: str):
    for s in SHORTCUTS:
        if s.shortcut_id == shortcut_id:
            return s
    return None



# ---------------------------------------------------------------------------
# Runtime compilation and launch
# ---------------------------------------------------------------------------

# Field name -> (dtype string, accessor on solver)
FIELD_ACCESSORS = {
    "velocity_u": ("float", lambda s: s.vel.u),
    "velocity_v": ("float", lambda s: s.vel.v),
    "velocity_w": ("float", lambda s: s.vel.w),
    "weight_u": ("float", lambda s: s.vel.wu),
    "weight_v": ("float", lambda s: s.vel.wv),
    "weight_w": ("float", lambda s: s.vel.ww),
    "temperature": ("float", lambda s: s.temperature.data),
    "viscosity": ("float", lambda s: s.viscosity_field.data),
    "density": ("float", lambda s: s.density.data),
    "pressure": ("float", lambda s: s.pressure_solver.pressure),
    "divergence": ("float", lambda s: s.pressure_solver.divergence),
    "surface_sdf": ("float", lambda s: s.surface_sdf.data),
    "collision_sdf": ("float", lambda s: s.collision_sdf.data),
    "churn": ("float", lambda s: s.churn.data),
    "voxel_type": ("int", lambda s: s.voxel_type.data),
    "voxel_counts": ("int", lambda s: s.voxel_counts.data),
    # Particle arrays (Particles run-over).
    "particle_position": ("vec3", lambda s: s.particles.position),
    "particle_velocity": ("vec3", lambda s: s.particles.velocity),
    "particle_age": ("float", lambda s: s.particles.attr("age")),
    "particle_life": ("float", lambda s: s.particles.attr("life")),
    "particle_temperature": ("float", lambda s: s.particles.attr("temperature")),
    "particle_viscosity": ("float", lambda s: s.particles.attr("viscosity")),
    "particle_density": ("float", lambda s: s.particles.attr("density")),
    "particle_time_offset": ("float", lambda s: s.particles.time_offset),
}

_DTYPE_DECL = {"float": "wp.array(dtype=float)",
               "int": "wp.array(dtype=wp.int32)",
               "vec3": "wp.array(dtype=wp.vec3)"}


@dataclass
class CustomArgument:
    mode: CustomKernelArgumentMode = CustomKernelArgumentMode.FIELD
    field: str = "temperature"
    field_permission: CustomKernelFieldPermission = CustomKernelFieldPermission.READ_WRITE
    constant_type: CustomKernelConstantType = CustomKernelConstantType.FLOAT
    constant_name: str = "value"
    constant_value: float = 0.0


@dataclass
class CustomSnippet:
    enabled: bool = True
    kernel_name: str = "user_kernel"
    snippet_type: CustomKernelType = CustomKernelType.KERNEL
    order: CustomKernelOrder = CustomKernelOrder.FINAL_STEP
    run_over: CustomKernelRunOver = CustomKernelRunOver.VOXEL_LINEAR
    code: str = ""
    arguments: list = dc_field(default_factory=list)  # list[CustomArgument]
    # Layering window: restrict when this snippet fires so multiple snippets
    # can be combined at different times (0 = open ended, step = every Nth).
    frame_start: int = 0
    frame_end: int = 0
    frame_step: int = 1

    def active_on_frame(self, frame: int) -> bool:
        if self.frame_start and frame < self.frame_start:
            return False
        if self.frame_end and frame > self.frame_end:
            return False
        step = max(1, int(self.frame_step))
        if step > 1:
            base = self.frame_start if self.frame_start else 0
            if (frame - base) % step != 0:
                return False
        return True

    def argument_names(self):
        names = []
        for arg in self.arguments:
            if arg.mode == CustomKernelArgumentMode.FIELD:
                names.append(arg.field)
            else:
                names.append(arg.constant_name)
        return names

    def field_permissions(self):
        return {
            arg.field: arg.field_permission
            for arg in self.arguments
            if arg.mode == CustomKernelArgumentMode.FIELD
        }

    def signature_params(self):
        """Returns (param_decl_list, launch_binder) for declared arguments."""
        decls = []
        binders = []
        for arg in self.arguments:
            if arg.mode == CustomKernelArgumentMode.FIELD:
                if arg.field not in FIELD_ACCESSORS:
                    raise ValueError(f"Unknown custom-kernel field '{arg.field}'")
                dtype, accessor = FIELD_ACCESSORS[arg.field]
                decls.append(f"{arg.field}: {_DTYPE_DECL[dtype]}")
                binders.append(("field", accessor))
            else:
                pyt = "float" if arg.constant_type == CustomKernelConstantType.FLOAT else "int"
                decls.append(f"{arg.constant_name}: {pyt}")
                value = arg.constant_value
                binders.append(("constant",
                                (float(value) if pyt == "float" else int(round(value)))))
        return decls, binders


def addon_root_package(package: str) -> str:
    """Return the installed add-on root for a package ending in ``.sim``."""
    suffix = ".sim"
    if not package.endswith(suffix):
        raise ValueError(f"expected a .sim package, got {package!r}")
    return package[:-len(suffix)]


_ADDON_ROOT = addon_root_package(__package__)

_VOXEL_IMPORT_PREAMBLE = (
    "import warp as wp\n"
    'wp.set_module_options({{"enable_backward": False}})\n'
    f"from {_ADDON_ROOT}.sim import constants as C\n"
    f"from {_ADDON_ROOT}.sim.fields import voxel_coord_of_flat, voxel_center_world\n"
)

_VOXEL_TEMPLATE_BODY = """

@wp.kernel
def {name}(
    _leaf_coords: wp.array(dtype=wp.vec3i),
    _num_leaves: wp.array(dtype=wp.int32),
    _voxel_type: wp.array(dtype=wp.int32),
    _origin: wp.vec3,
    voxel_size: float,
    frame: int,
    time: float,
    real_time_step: float,
    solver_time_step: float,
    time_scale: float,
    substeps: int,
    {extra_params}
):
    _tid = wp.tid()
    if _tid >= _num_leaves[0] * C.LEAF_VOLUME:
        return
    if (_voxel_type[_tid] & C.VOXEL_COLLISION) != 0:
        return
    flat_voxel_index = _tid
    leaf_index = _tid // C.LEAF_VOLUME
    _ijk = voxel_coord_of_flat(_leaf_coords, _tid)
    global_x = _ijk[0]
    global_y = _ijk[1]
    global_z = _ijk[2]
    local_x = global_x & C.LEAF_MASK
    local_y = global_y & C.LEAF_MASK
    local_z = global_z & C.LEAF_MASK
    world_position = voxel_center_world(_ijk, _origin, voxel_size)
{body}
"""

_VOXEL_TEMPLATE = _VOXEL_IMPORT_PREAMBLE + _VOXEL_TEMPLATE_BODY

_PARTICLE_IMPORT_PREAMBLE = (
    "import warp as wp\n"
    'wp.set_module_options({{"enable_backward": False}})\n'
    f"from {_ADDON_ROOT}.sim import constants as C\n"
)

_PARTICLE_TEMPLATE_BODY = """

@wp.kernel
def {name}(
    _positions: wp.array(dtype=wp.vec3),
    _velocities: wp.array(dtype=wp.vec3),
    _alive: wp.array(dtype=wp.int32),
    _count: wp.array(dtype=wp.int32),
    voxel_size: float,
    frame: int,
    time: float,
    real_time_step: float,
    solver_time_step: float,
    time_scale: float,
    substeps: int,
    {extra_params}
):
    _tid = wp.tid()
    if _tid >= _count[0]:
        return
    if _alive[_tid] == 0:
        return
    particle_index = _tid
    position = _positions[_tid]
    velocity = _velocities[_tid]
{body}
"""

_PARTICLE_TEMPLATE = _PARTICLE_IMPORT_PREAMBLE + _PARTICLE_TEMPLATE_BODY


class CompiledSnippet:
    """A compiled user snippet ready to launch against a solver."""

    def __init__(self, snippet: CustomSnippet):
        self.snippet = snippet
        self.error = None
        self.kernel = None
        self._binders = []
        try:
            self._compile()
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.error = f"{type(exc).__name__}: {exc}"

    def _compile(self):
        sn = self.snippet
        if sn.snippet_type != CustomKernelType.KERNEL:
            raise ValueError(FUNCTION_UNSUPPORTED_MESSAGE)
        decls, binders = sn.signature_params()
        self._binders = binders
        extra = ",\n    ".join(decls) if decls else "_unused_pad: float"
        if not decls:
            self._binders = [("constant", 0.0)]
        body = textwrap.indent(textwrap.dedent(sn.code).strip() or "pass", "    ")
        name = "custom_" + "".join(
            ch if ch.isalnum() or ch == "_" else "_" for ch in sn.kernel_name
        )
        if sn.run_over == CustomKernelRunOver.PARTICLES:
            src = _PARTICLE_TEMPLATE.format(name=name, extra_params=extra, body=body)
        else:
            # VoxelTile and VoxelLinear share the flat launch; VoxelTile
            # additionally exposes leaf-local indices, which the template
            # always provides.
            src = _VOXEL_TEMPLATE.format(name=name, extra_params=extra, body=body)
        # Warp requires kernels to live in real source files (it re-reads the
        # file for codegen), so write the generated module to a snippet cache
        # directory and import it. Content-hashed names give deduplication
        # and stable Warp kernel caching across sessions.
        import hashlib
        import importlib.util
        import os
        import sys
        import tempfile

        digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
        mod_name = f"fluidsim_snippet_{name}_{digest}"
        snip_dir = os.path.join(tempfile.gettempdir(), "fluidsim_warp_snippets")
        os.makedirs(snip_dir, exist_ok=True)
        path = os.path.join(snip_dir, mod_name + ".py")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
        self._module_name = mod_name
        if mod_name in sys.modules:
            module = sys.modules[mod_name]
        else:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        self.kernel = module.__dict__[name]

    def launch(self, solver, time_ctx):
        if self.kernel is None:
            return
        import warp as wp

        sn = self.snippet
        extra_inputs = []
        for kind, binder in self._binders:
            if kind == "field":
                extra_inputs.append(binder(solver))
            else:
                extra_inputs.append(binder)
        common = [
            solver.grid.voxel_size,
            int(time_ctx["frame"]),
            float(time_ctx["time"]),
            float(time_ctx["real_time_step"]),
            float(time_ctx["solver_time_step"]),
            float(time_ctx["time_scale"]),
            int(time_ctx["substeps"]),
        ]
        if sn.run_over == CustomKernelRunOver.PARTICLES:
            particle_dim_fn = getattr(solver, "_np_dim", None)
            particle_dim = (particle_dim_fn() if callable(particle_dim_fn)
                            else solver.particles.capacity)
            wp.launch(
                self.kernel,
                dim=particle_dim,
                inputs=[
                    solver.particles.position,
                    solver.particles.velocity,
                    solver.particles.alive,
                    solver.particles.count,
                ] + common + extra_inputs,
                device=solver.device,
            )
        else:
            voxel_dim = max(1, int(getattr(solver, "_nvox", solver.grid.voxel_capacity)))
            wp.launch(
                self.kernel,
                dim=voxel_dim,
                inputs=[
                    solver.grid.leaf_coords,
                    solver.grid.num_leaves,
                    solver.voxel_type.data,
                    solver.grid.origin,
                ] + common + extra_inputs,
                device=solver.device,
            )


class SnippetManager:
    """Owns compiled snippets grouped by hook order."""

    def __init__(self):
        self._by_order = {order: [] for order in CustomKernelOrder}
        self.errors = []

    def set_snippets(self, snippets):
        self._by_order = {order: [] for order in CustomKernelOrder}
        self.errors = []
        for sn in snippets:
            if not sn.enabled:
                continue
            if sn.snippet_type != CustomKernelType.KERNEL:
                self.errors.append((sn.kernel_name, FUNCTION_UNSUPPORTED_MESSAGE))
                continue
            # Static validation gate: invalid snippets never reach codegen
            # or the running scene (Warp compile remains the final check).
            static_errors = [
                issue for issue in validate_snippet(
                    sn.code, sn.run_over.name, sn.argument_names(),
                    sn.field_permissions())
                if issue.level == "ERROR"
            ]
            if static_errors:
                self.errors.append(
                    (sn.kernel_name, "; ".join(str(i) for i in static_errors)))
                continue
            compiled = CompiledSnippet(sn)
            if compiled.error:
                self.errors.append((sn.kernel_name, compiled.error))
            else:
                self._by_order[sn.order].append(compiled)

    def run(self, order, solver, time_ctx):
        frame = int(time_ctx.get("frame", 0))
        for compiled in self._by_order.get(order, ()):
            if not compiled.snippet.active_on_frame(frame):
                continue
            compiled.launch(solver, time_ctx)

    def has_any(self):
        return any(self._by_order.values())


# ---------------------------------------------------------------------------
# Custom Effects v2 compatibility conversion
# ---------------------------------------------------------------------------


def legacy_snippet_to_effect(snippet: CustomSnippet, *, instance_id: str | None = None):
    """Wrap a legacy one-kernel snippet as a strict Custom Effects v2 instance.

    Source, constants, field bindings, frame window, enable state, and legacy
    execution semantics are preserved.  ``VOXEL_TILE`` never had distinct tile
    scheduling in the Warp port, so it migrates to ``ACTIVE_VOXELS`` with an
    explicit warning rather than retaining a misleading public mode.
    """
    if snippet.snippet_type != CustomKernelType.KERNEL:
        raise ValueError(FUNCTION_UNSUPPORTED_MESSAGE)

    from .effects.enums import (
        EffectCadence,
        EffectDomain,
        EffectStage,
        FieldAccess,
        ParameterKind,
    )
    from .effects.schema import (
        EffectDefinitionV2,
        EffectInstanceV2,
        FieldBindingV2,
        KernelDefinitionV2,
        ParameterDefinitionV2,
    )

    warnings = []
    domain = (
        EffectDomain.PARTICLES
        if snippet.run_over == CustomKernelRunOver.PARTICLES
        else EffectDomain.ACTIVE_VOXELS
    )
    if snippet.run_over == CustomKernelRunOver.VOXEL_TILE:
        warnings.append(
            "Legacy VOXEL_TILE used the same flat active-voxel launch as "
            "VOXEL_LINEAR and was migrated to ACTIVE_VOXELS."
        )
    stage_map = {
        CustomKernelOrder.FIRST_STEP: EffectStage.BEFORE_EMISSION,
        CustomKernelOrder.BEFORE_SOURCING: EffectStage.BEFORE_EMISSION,
        CustomKernelOrder.BEFORE_FORCES: EffectStage.AFTER_PARTICLE_TRANSFER,
        CustomKernelOrder.BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION: EffectStage.BEFORE_PRESSURE,
        CustomKernelOrder.FINAL_STEP: EffectStage.AFTER_PARTICLE_UPDATE,
    }
    fields = []
    parameters = []
    values = {}
    for argument in snippet.arguments:
        if argument.mode == CustomKernelArgumentMode.FIELD:
            access = {
                CustomKernelFieldPermission.READ: FieldAccess.READ,
                CustomKernelFieldPermission.WRITE: FieldAccess.WRITE_CURRENT,
                CustomKernelFieldPermission.READ_WRITE: FieldAccess.READ_WRITE_CURRENT,
            }[argument.field_permission]
            fields.append(FieldBindingV2(
                argument.field,
                access,
            ))
        else:
            kind = (
                ParameterKind.FLOAT
                if argument.constant_type == CustomKernelConstantType.FLOAT
                else ParameterKind.INT
            )
            value = (
                float(argument.constant_value)
                if kind is ParameterKind.FLOAT
                else int(round(argument.constant_value))
            )
            parameters.append(ParameterDefinitionV2(
                argument.constant_name,
                argument.constant_name.replace("_", " ").title(),
                kind,
                value,
                description="Migrated legacy constant.",
            ))
            values[argument.constant_name] = value

    slug = re.sub(r"[^a-z0-9]+", "-", snippet.kernel_name.lower()).strip("-") or "effect"
    digest = hashlib.sha256((snippet.code + repr(snippet.arguments)).encode("utf-8")).hexdigest()[:10]
    effect_id = f"legacy.{slug}-{digest}"
    kernel_id = re.sub(r"[^A-Za-z0-9_]+", "_", snippet.kernel_name).strip("_") or "legacy_kernel"
    if kernel_id[0].isdigit():
        kernel_id = "legacy_" + kernel_id
    definition = EffectDefinitionV2(
        effect_id=effect_id,
        version="1.0.0",
        name=snippet.kernel_name or "Legacy Custom Effect",
        summary="Migrated legacy FluidSim Warp snippet.",
        description=(
            "This effect was migrated without rewriting its source. Review and "
            "trust the exact content before enabling it."
        ),
        category="Legacy",
        author="Scene author",
        license="User supplied",
        parameters=tuple(parameters),
        kernels=(KernelDefinitionV2(
            kernel_id=kernel_id,
            label=snippet.kernel_name or "Legacy Kernel",
            domain=domain,
            stage=stage_map[snippet.order],
            cadence=EffectCadence.EVERY_SUBSTEP,
            source=snippet.code or "pass",
            fields=tuple(fields),
            parameter_keys=tuple(parameter.key for parameter in parameters),
            safe_mode=False,
        ),),
        built_in=False,
        package_origin="legacy-scene-snippet",
        cost_hint="UNKNOWN",
    )
    instance = EffectInstanceV2(
        instance_id=instance_id or f"legacy-{digest}",
        definition=definition,
        enabled=bool(snippet.enabled),
        parameter_values=values,
        frame_start=int(snippet.frame_start),
        frame_end=int(snippet.frame_end),
        frame_step=max(1, int(snippet.frame_step)),
        safe_mode=False,
        expert_acknowledged=True,
        display_name=snippet.kernel_name,
    )
    return instance, tuple(warnings)

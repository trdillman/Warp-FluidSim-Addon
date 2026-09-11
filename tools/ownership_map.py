"""Phase 0 of the settings PropertyGroup split: derive the ownership map.

`docs/PROPOSAL-split-settings-propertygroup.md` names Phase 0 as the gate for
every later phase: one row per annotated field, saying where the UI *writes*
it, where its consumer *reads* it, and what the code currently *declares*.
This script derives that table from the source rather than from belief, and
`tests/test_ownership_map.py` pins the result so the map cannot silently rot.

Why this is not a grep
----------------------
Three things defeat a textual search:

1.  ``_make_panel_mirrors`` in ``blender/ui_physics.py`` re-draws Scene panels
    under Physics Properties.  A control's datablock is whatever its draw
    binds -- ``context.scene.fluidsim`` or ``ui_scene._domain_owner`` -- in
    either tab; ``owner_redirect=True`` only hides the domain mirrors while no
    domain object is designated.  The map records which panels are mirrored.
2.  Consumers rarely name the datablock.  ``solver_settings.build_settings``
    resolves ``active_domain_settings(scene)`` and passes it as ``fs`` into
    ``apply_scene_settings``, so the read sites that matter are a parameter
    hop away from the expression that determines ownership.
3.  ``getattr(fs, "field", default)`` and ``fs.field`` are both reads, but the
    defaulted form is what makes a wrong-group read invisible today.

So this walks the AST, resolves settings-group expressions, and propagates
those kinds across call boundaries to a fixpoint.

Usage
-----
    python tools/ownership_map.py                 # rewrite docs/ownership-map.md
    python tools/ownership_map.py --check         # fail if the file is stale
    python tools/ownership_map.py --print-summary # counts only, no file write
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import sys
import types
from collections import defaultdict
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = "fluidSim_warp_blender51"
PKG_ROOT = os.path.join(REPO_ROOT, PKG_NAME)
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "ownership-map.md")

SCENE = "scene"
DOMAIN = "domain"

# Expressions that yield the domain-scoped group no matter how they are spelled.
DOMAIN_RESOLVERS = frozenset({
    "active_domain_settings", "_domain_owner", "_active_settings",
})

# Modules that draw UI. Everything else counts as a consumer for the read column.
UI_MODULE_PREFIXES = ("ui_",)
UI_MODULE_NAMES = frozenset({"effect_ui", "effect_workspace", "menus"})

# Draw helpers from blender/ui_common.py. Value = (data_argument_index,
# field_argument_index, field_arg_is_sequence).
DRAW_HELPERS = {
    "draw_property": (1, 2, False),
    "draw_property_grid": (1, 2, True),
}
# Read-only telemetry rendering: reads the group, never edits it.
READOUT_HELPERS = {"draw_readout_grid": (1, 2)}


# --------------------------------------------------------------------------
# bpy stub, so the shipped declarations can be imported and queried directly
# --------------------------------------------------------------------------

def _install_bpy_stub():
    """Import the add-on's pure-python declarations without a real Blender."""
    if "bpy" in sys.modules:
        return
    bpy = types.ModuleType("bpy")
    bpy.context = SimpleNamespace()
    bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(frame_change_post=[], persistent=lambda f: f)
    )
    bpy.types = SimpleNamespace(
        PropertyGroup=object, Text=object, Object=object, Material=object,
        Image=object, Operator=object, Panel=object, AddonPreferences=object,
        Scene=object,
    )
    bpy.utils = SimpleNamespace(
        register_class=lambda c: None, unregister_class=lambda c: None
    )
    prop = lambda **kw: kw.get("default")  # noqa: E731 - terse stub by design
    bpy.props = SimpleNamespace(**{
        name: prop for name in (
            "BoolProperty", "IntProperty", "FloatProperty", "StringProperty",
            "EnumProperty", "PointerProperty", "CollectionProperty",
            "FloatVectorProperty", "IntVectorProperty", "BoolVectorProperty",
        )
    })
    sys.modules["bpy"] = bpy
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda v: v
    mathutils.Matrix = object
    sys.modules["mathutils"] = mathutils


def declared_scopes(fields):
    """Query the shipped declarations for effective and authoring ownership."""
    _install_bpy_stub()
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from fluidSim_warp_blender51.blender import pending_settings as P
    from fluidSim_warp_blender51.presets import schema as preset_schema

    domain_group = SimpleNamespace(_label=DOMAIN)
    domain_object = SimpleNamespace(fluidsim_domain=domain_group)
    scene_group = SimpleNamespace(_label=SCENE, domain_object=domain_object)
    scene = SimpleNamespace(fluidsim=scene_group)

    effective, authoring = {}, {}
    for name in fields:
        effective[name] = getattr(P.setting_owner(scene, name), "_label", "?")
        authoring[name] = getattr(
            P._authoring_owner(scene_group, domain_group, name), "_label", "?"
        )
    tiers = {n: t.value for n, t in P.SETTING_TRANSITION_REGISTRY.items()}
    return effective, authoring, tiers, frozenset(preset_schema.SCENE_OWNED_SETTINGS)


# --------------------------------------------------------------------------
# Field inventory
# --------------------------------------------------------------------------

SETTINGS_CLASSES = ("FLUIDSIM_PG_scene", "FLUIDSIM_PG_domain")


def annotated_fields():
    """Every annotated property on the two settings groups, in source order."""
    path = os.path.join(PKG_ROOT, "properties.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    groups = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name in SETTINGS_CLASSES
    }
    return [
        node.target.id
        for name in SETTINGS_CLASSES
        for node in groups[name].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


# --------------------------------------------------------------------------
# AST model
# --------------------------------------------------------------------------

class Function:
    """One analyzed function, with its resolved settings-group bindings."""

    __slots__ = ("key", "module", "path", "name", "node", "params", "bindings",
                 "name_bindings", "writes", "reads", "calls", "call_names",
                 "is_ui")

    def __init__(self, key, module, name, node, is_ui, path=""):
        self.key = key
        self.module = module
        self.path = path
        self.name = name
        self.node = node
        self.is_ui = is_ui
        self.params = [a.arg for a in node.args.args] + [
            a.arg for a in node.args.kwonlyargs
        ]
        self.bindings = {}     # param name -> SCENE/DOMAIN learned from callers
        # param name -> field names it is called with, for helpers that take
        # the attribute name as an argument
        # (`_ensure_typed_whitewater_material(scene, obj, "foam_material", ...)`).
        self.name_bindings = defaultdict(set)
        self.writes = []       # (field, kind, lineno, via)
        self.reads = []        # (field, kind, lineno, defaulted)
        self.calls = []        # (callee_name, {param_name: kind})
        self.call_names = []   # (callee_name, {param_name: field_name})


def _module_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _is_ui_module(module):
    return module.startswith(UI_MODULE_PREFIXES) or module in UI_MODULE_NAMES


_MODULE_CACHE = {}


def _import_for(path):
    """Import an add-on module under the bpy stub, or None if it will not load.

    Field tables like ``DIRECT_SETTING_FIELDS`` and ``enum_setting_fields()``
    are the real read surface for most settings, and they are data, not syntax.
    Reading them as live objects is both simpler and more faithful than trying
    to constant-fold them out of the AST.
    """
    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]
    _install_bpy_stub()
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    relative = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
    dotted = relative[:-3].replace("/", ".")
    if dotted.endswith(".__init__"):
        dotted = dotted[: -len(".__init__")]
    try:
        module = importlib.import_module(dotted)
    except Exception:
        module = None
    _MODULE_CACHE[path] = module
    return module


def _evaluate_iterable(module, node):
    """Evaluate a loop's iterable to a live object, conservatively.

    Only names, zero-argument calls, and ``.items()/.keys()/.values()`` on
    those are evaluated -- enough for the declarative field tables, and small
    enough that nothing with side effects runs.
    """
    if module is None:
        return None
    try:
        if isinstance(node, ast.Name):
            return getattr(module, node.id, None)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and not node.args and not node.keywords:
                target = getattr(module, func.id, None)
                return target() if callable(target) else None
            if isinstance(func, ast.Attribute) and func.attr in (
                "items", "keys", "values"
            ):
                base = _evaluate_iterable(module, func.value)
                if base is None:
                    return None
                accessor = getattr(base, func.attr, None)
                return list(accessor()) if callable(accessor) else None
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return _string_items(node)
    except Exception:
        return None
    return None


def _bind_loop_names(target, value, sink):
    """Destructure one iterated value against a loop target, collecting strings."""
    if isinstance(target, ast.Name):
        if isinstance(value, str):
            sink[target.id].add(value)
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        try:
            parts = list(value)
        except TypeError:
            return
        if len(parts) != len(target.elts):
            return
        for element, part in zip(target.elts, parts):
            _bind_loop_names(element, part, sink)


def loop_name_bindings(func_node, module, fields):
    """Map loop variables to the field names they can take on.

    Covers ``for name in DIRECT_SETTING_FIELDS`` and the tuple-unpacking form
    ``for target, (source, enum_type) in enum_setting_fields().items()``, plus
    the comprehension equivalents.
    """
    sink = defaultdict(set)
    for node in ast.walk(func_node):
        pairs = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            pairs.append((node.target, node.iter))
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                               ast.GeneratorExp)):
            for generator in node.generators:
                pairs.append((generator.target, generator.iter))
        for target, iterable in pairs:
            values = _evaluate_iterable(module, iterable)
            if values is None:
                continue
            try:
                items = list(values)
            except TypeError:
                continue
            if len(items) > 4096:
                continue
            for item in items:
                _bind_loop_names(target, item, sink)
    return {
        name: {value for value in values if value in fields}
        for name, values in sink.items()
    }


def _iter_source_files():
    for dirpath, dirnames, filenames in os.walk(PKG_ROOT):
        parts = dirpath.replace("\\", "/").split("/")
        if "tests" in parts or "__pycache__" in parts:
            dirnames[:] = [d for d in dirnames if d not in ("tests", "__pycache__")]
            continue
        dirnames[:] = [d for d in dirnames if d not in ("tests", "__pycache__")]
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _call_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def resolve_kind(node, env):
    """Resolve an expression to SCENE, DOMAIN, or None (not a settings group)."""
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Attribute):
        if node.attr == "fluidsim":
            return SCENE
        if node.attr == "fluidsim_domain":
            return DOMAIN
        return None
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name in DOMAIN_RESOLVERS:
            return DOMAIN
        if name == "getattr" and len(node.args) >= 2:
            key = node.args[1]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value == "fluidsim":
                    return SCENE
                if key.value == "fluidsim_domain":
                    return DOMAIN
        return None
    if isinstance(node, ast.BoolOp):
        # `active_domain_settings(scene) or scene.fluidsim` -- the left operand
        # is canonical whenever it is truthy, so the domain answer wins.
        kinds = {resolve_kind(value, env) for value in node.values} - {None}
        if len(kinds) == 1:
            return next(iter(kinds))
        if kinds == {DOMAIN, SCENE}:
            return DOMAIN
        return None
    if isinstance(node, ast.IfExp):
        kinds = {resolve_kind(node.body, env), resolve_kind(node.orelse, env)} - {None}
        return next(iter(kinds)) if len(kinds) == 1 else None
    return None


def _string_items(node):
    """String constants from a literal tuple/list, else ()."""
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(
            item.value for item in node.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
    return ()


def analyze_function(func, fields, module=None):
    """Recompute a function's env, writes, reads and outbound call bindings."""
    env = dict(func.bindings)
    func.writes, func.reads, func.calls, func.call_names = [], [], [], []
    loop_fields = loop_name_bindings(func.node, module, fields)

    def table_names(node):
        """Field names a non-constant attribute-name expression can take."""
        if isinstance(node, ast.Name):
            return tuple(
                set(loop_fields.get(node.id, ()))
                | func.name_bindings.get(node.id, set())
            )
        return ()

    for node in ast.walk(func.node):
        # Local bindings: `fs = active_domain_settings(scene)` etc.
        if isinstance(node, ast.Assign):
            kind = resolve_kind(node.value, env)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if kind is None:
                        env.pop(target.id, None)
                    else:
                        env[target.id] = kind
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            continue

    for node in ast.walk(func.node):
        if isinstance(node, ast.Call):
            name = _call_name(node)

            # UI write: layout.prop(data, "field")
            if name == "prop" and len(node.args) >= 2:
                field = node.args[1]
                if isinstance(field, ast.Constant) and field.value in fields:
                    kind = resolve_kind(node.args[0], env)
                    func.writes.append((field.value, kind, node.lineno, "prop"))

            # UI write through the shared helpers
            elif name in DRAW_HELPERS:
                data_index, field_index, is_sequence = DRAW_HELPERS[name]
                if len(node.args) > max(data_index, field_index):
                    kind = resolve_kind(node.args[data_index], env)
                    target = node.args[field_index]
                    names = (
                        _string_items(target) if is_sequence
                        else ((target.value,) if isinstance(target, ast.Constant)
                              and isinstance(target.value, str) else ())
                    )
                    for field_name in names:
                        if field_name in fields:
                            func.writes.append(
                                (field_name, kind, node.lineno, name)
                            )

            # Read-only telemetry rendering
            elif name in READOUT_HELPERS:
                data_index, pairs_index = READOUT_HELPERS[name]
                if len(node.args) > max(data_index, pairs_index):
                    kind = resolve_kind(node.args[data_index], env)
                    pairs = node.args[pairs_index]
                    if isinstance(pairs, (ast.Tuple, ast.List)):
                        for item in pairs.elts:
                            head = (
                                item.elts[0]
                                if isinstance(item, (ast.Tuple, ast.List)) and item.elts
                                else None
                            )
                            if (isinstance(head, ast.Constant)
                                    and head.value in fields):
                                func.reads.append(
                                    (head.value, kind, item.lineno, True)
                                )

            # Read: getattr(fs, "field", default) / hasattr(fs, name) and the
            # table-driven forms getattr(fs, name) where `name` iterates a
            # declared field table.
            elif name in ("getattr", "hasattr") and len(node.args) >= 2:
                key = node.args[1]
                kind = resolve_kind(node.args[0], env)
                if kind is not None:
                    defaulted = name == "getattr" and len(node.args) >= 3
                    if isinstance(key, ast.Constant) and key.value in fields:
                        func.reads.append(
                            (key.value, kind, node.lineno, defaulted)
                        )
                    else:
                        for field_name in table_names(key):
                            func.reads.append(
                                (field_name, kind, node.lineno, defaulted)
                            )

            # Write: setattr(fs, "field", value), including the table form.
            elif name == "setattr" and len(node.args) >= 3:
                key = node.args[1]
                kind = resolve_kind(node.args[0], env)
                if isinstance(key, ast.Constant) and key.value in fields:
                    func.writes.append((key.value, kind, node.lineno, "setattr"))
                else:
                    for field_name in table_names(key):
                        func.writes.append(
                            (field_name, kind, node.lineno, "setattr")
                        )

            # Outbound call: propagate resolved kinds into the callee's params.
            if name:
                bound = {}
                bound_names = {}
                for index, arg in enumerate(node.args):
                    kind = resolve_kind(arg, env)
                    if kind is not None:
                        bound[index] = kind
                    if isinstance(arg, ast.Constant) and arg.value in fields:
                        bound_names[index] = arg.value
                for keyword in node.keywords:
                    kind = resolve_kind(keyword.value, env)
                    if kind is not None and keyword.arg:
                        bound[keyword.arg] = kind
                    if (keyword.arg
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value in fields):
                        bound_names[keyword.arg] = keyword.value.value
                func.calls.append((name, bound))
                if bound_names:
                    func.call_names.append((name, bound_names))

        # Read: fs.field  (attribute load)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if node.attr in fields:
                kind = resolve_kind(node.value, env)
                if kind is not None:
                    func.reads.append((node.attr, kind, node.lineno, False))

        # Write: fs.field = value
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            if node.attr in fields:
                kind = resolve_kind(node.value, env)
                func.writes.append((node.attr, kind, node.lineno, "assign"))


def build_functions(fields):
    """Parse the package and resolve group bindings to a fixpoint."""
    functions = {}
    module_files = {}
    by_name = defaultdict(list)

    for path in _iter_source_files():
        module = _module_name(path)
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        module_files[module] = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
        is_ui = _is_ui_module(module)

        stack = [("", tree)]
        while stack:
            prefix, parent = stack.pop()
            for child in ast.iter_child_nodes(parent):
                if isinstance(child, ast.ClassDef):
                    stack.append((f"{prefix}{child.name}.", child))
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = f"{module}:{prefix}{child.name}"
                    function = Function(key, module, child.name, child, is_ui, path)
                    functions[key] = function
                    by_name[child.name].append(function)
                    stack.append((f"{prefix}{child.name}.", child))

    # Fixpoint: analyze, push argument kinds into callees, repeat.
    module_objects = {}
    for _ in range(8):
        for function in functions.values():
            if function.path not in module_objects:
                module_objects[function.path] = _import_for(function.path)
            analyze_function(function, fields, module_objects[function.path])
        changed = False
        for function in functions.values():
            for callee_name, bound in function.calls:
                for callee in by_name.get(callee_name, ()):
                    for slot, kind in bound.items():
                        param = (
                            callee.params[slot]
                            if isinstance(slot, int) and slot < len(callee.params)
                            else (slot if isinstance(slot, str) else None)
                        )
                        if param is None or param not in callee.params:
                            continue
                        if callee.bindings.get(param) != kind:
                            # Conflicting kinds across call sites: the callee is
                            # genuinely polymorphic, so record the ambiguity
                            # rather than pretending one caller wins.
                            if param in callee.bindings:
                                callee.bindings[param] = "ambiguous"
                            else:
                                callee.bindings[param] = kind
                            changed = True
            for callee_name, bound_names in function.call_names:
                for callee in by_name.get(callee_name, ()):
                    for slot, field_name in bound_names.items():
                        param = (
                            callee.params[slot]
                            if isinstance(slot, int) and slot < len(callee.params)
                            else (slot if isinstance(slot, str) else None)
                        )
                        if param is None or param not in callee.params:
                            continue
                        if field_name not in callee.name_bindings[param]:
                            callee.name_bindings[param].add(field_name)
                            changed = True
        if not changed:
            break

    return functions, by_name, module_files


# --------------------------------------------------------------------------
# Mirror handling
# --------------------------------------------------------------------------

def redirected_source_panels():
    """Panel class names whose draw is re-run against the domain group.

    Parsed out of ``blender/ui_physics.py`` rather than hardcoded, so that
    adding or removing ``owner_redirect=True`` shows up in the map.
    """
    path = os.path.join(PKG_ROOT, "blender", "ui_physics.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    tuples = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(
                node.value, (ast.Tuple, ast.List)
            ):
                names = []
                for element in node.value.elts:
                    if isinstance(element, ast.Attribute):
                        names.append(element.attr)
                    elif isinstance(element, ast.Name):
                        names.append(element.id)
                tuples[target.id] = names

    redirected, mirrored = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "_make_panel_mirrors":
            continue
        source = node.args[0] if node.args else None
        if isinstance(source, ast.Name):
            names = tuples.get(source.id, [])
        elif isinstance(source, (ast.Tuple, ast.List)):
            names = [
                element.attr for element in source.elts
                if isinstance(element, ast.Attribute)
            ]
        elif isinstance(source, ast.Attribute):
            names = tuples.get(source.attr, [])
        else:
            names = []
        is_redirect = any(
            keyword.arg == "owner_redirect"
            and isinstance(keyword.value, ast.Constant)
            and bool(keyword.value.value)
            for keyword in node.keywords
        )
        mirrored.update(names)
        if is_redirect:
            redirected.update(names)
    return redirected, mirrored


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def build_map():
    fields = annotated_fields()
    field_set = frozenset(fields)
    functions, by_name, module_files = build_functions(field_set)
    effective, authoring, tiers, preset_scene_owned = declared_scopes(fields)
    redirected, mirrored = redirected_source_panels()

    writes = defaultdict(set)      # field -> {(kind, module, line, via)}
    reads = defaultdict(set)
    for function in functions.values():
        location = module_files.get(function.module, function.module)
        for field, kind, line, via in function.writes:
            writes[field].add((kind, function.module, location, line, via,
                               function.is_ui))
        for field, kind, line, defaulted in function.reads:
            reads[field].add((kind, function.module, location, line, defaulted,
                              function.is_ui))

    rows = []
    for field in fields:
        ui_write_kinds, ui_sites = set(), []
        other_write_kinds, other_write_sites = set(), []
        for kind, module, location, line, via, is_ui in writes.get(field, ()):
            if is_ui and via in ("prop", "draw_property", "draw_property_grid"):
                if kind:
                    ui_write_kinds.add(kind)
                    ui_sites.append((location, line, kind, via))
            elif kind:
                other_write_kinds.add(kind)
                other_write_sites.append((location, line, kind, via))

        read_kinds, read_sites, defaulted_reads = set(), [], 0
        ui_read_kinds, ui_read_sites = set(), []
        for kind, module, location, line, defaulted, is_ui in reads.get(field, ()):
            if kind not in (SCENE, DOMAIN):
                continue
            if is_ui:
                # A panel displaying the value -- the consuming end of the
                # telemetry direction, not of the authoring direction.
                ui_read_kinds.add(kind)
                ui_read_sites.append((location, line, kind, defaulted))
                continue
            read_kinds.add(kind)
            read_sites.append((location, line, kind, defaulted))
            if defaulted:
                defaulted_reads += 1

        rows.append({
            "field": field,
            "tier": tiers.get(field, "-"),
            "ui_write": ui_write_kinds,
            "ui_sites": sorted(ui_sites),
            "other_write": other_write_kinds,
            "other_write_sites": sorted(other_write_sites),
            "read": read_kinds,
            "read_sites": sorted(read_sites),
            "ui_read": ui_read_kinds,
            "ui_read_sites": sorted(ui_read_sites),
            "defaulted_reads": defaulted_reads,
            "effective": effective.get(field, "?"),
            "authoring": authoring.get(field, "?"),
            "preset_scene_owned": field in preset_scene_owned,
        })
    return rows, redirected, mirrored


def classify(row):
    """Verdict for one field: does its write side reach its read side?

    Two directions of flow exist and they are not symmetric:

    * **authoring** -- a panel control writes the field, non-UI consumer code
      reads it.  This is the direction the split proposal is about, and the one
      the ``BROKEN`` verdict polices.
    * **telemetry** -- the runtime writes the field (``scene_bridge.step``,
      ``playback.play_cache_frame``) and a panel *displays* it read-only via
      ``draw_readout_grid``.  The ``stat_*`` block flows this way.

    A field with a real UI control is judged on the authoring direction; the
    telemetry verdicts apply only to fields no panel control writes.
    """
    write = row["ui_write"]
    read = row["read"]
    effective_write = {w.replace("(mirror)", "") for w in write}
    runtime_write = row["other_write"]
    ui_read = row["ui_read"]

    if write and read:
        if effective_write & read:
            if len(effective_write) > 1 or len(read) > 1:
                return "OK_MULTI", "write and read meet, but more than one path exists"
            return "OK", "write lands where the consumer reads"
        if "domain(mirror)" in write and DOMAIN in read:
            return "MIRROR_ONLY", (
                "only the Physics mirror writes where the consumer reads; "
                "the Scene-tab control still writes the scene group"
            )
        return "BROKEN", (
            f"UI writes {sorted(write)} but the consumer reads {sorted(read)}"
        )

    if write:
        if ui_read:
            return "UI_STATE", (
                "a panel writes it and a panel reads it; no consumer outside "
                "the UI observes it"
            )
        return "WRITE_ONLY", "UI writes it; no consumer read resolved"

    # No panel control writes this field, so judge the telemetry direction.
    consumers = read | ui_read
    if runtime_write and consumers:
        if runtime_write & consumers:
            if len(runtime_write) > 1 or len(consumers) > 1:
                return "PUBLISHED_MULTI", (
                    "the runtime publishes it and a reader observes it, but "
                    "more than one path exists"
                )
            return "PUBLISHED", "the runtime publishes it where its reader looks"
        return "PUBLISHED_BROKEN", (
            f"the runtime writes {sorted(runtime_write)} but its reader looks "
            f"at {sorted(consumers)}"
        )
    if consumers:
        return "READ_ONLY", "something reads it; nothing resolved writes it"
    if runtime_write:
        return "INTERNAL", "written outside the UI; no resolved reader"
    return "UNOBSERVED", "no access of any kind resolved"


# Verdicts naming a defect rather than a shape; each is rendered in detail.
FAULT_VERDICTS = ("BROKEN", "MIRROR_ONLY", "PUBLISHED_BROKEN")


def render(rows, redirected, mirrored):
    verdicts = defaultdict(list)
    for row in rows:
        verdict, _reason = classify(row)
        verdicts[verdict].append(row["field"])

    lines = []
    lines.append("# Settings ownership map (Phase 0)")
    lines.append("")
    lines.append(
        "Generated by `tools/ownership_map.py` -- do not edit by hand. "
        "Regenerate with `python tools/ownership_map.py`; "
        "`tests/test_ownership_map.py` fails if this file is stale."
    )
    lines.append("")
    lines.append(
        "This is the Phase 0 deliverable of "
        "[PROPOSAL-split-settings-propertygroup.md](PROPOSAL-split-settings-propertygroup.md): "
        "one row per annotated field on `FLUIDSIM_PG_scene` and "
        "`FLUIDSIM_PG_domain`, derived from the "
        "AST rather than from the declarations, so the declarations themselves "
        "can be audited against it."
    )
    lines.append("")

    lines.append("## How to read the columns")
    lines.append("")
    lines.append("Two directions of flow exist, and a field uses one or the "
                 "other:")
    lines.append("")
    lines.append("- **authoring** -- a panel control writes the field and "
                 "non-UI consumer code reads it. This is the direction the "
                 "split is about, and the one `BROKEN` polices.")
    lines.append("- **telemetry** -- the runtime writes the field "
                 "(`scene_bridge.step`, `playback.play_cache_frame`) and a "
                 "panel displays it read-only. The `stat_*` block flows this "
                 "way.")
    lines.append("")
    lines.append("- **UI writes** -- datablock the panel control edits.")
    lines.append("- **Consumer reads** -- datablock the non-UI code reads. "
                 "This is the one that decides whether an edit takes effect.")
    lines.append("- **Runtime writes** -- datablock non-UI code *assigns*; the "
                 "producing end of the telemetry direction.")
    lines.append("- **UI reads** -- datablock a panel displays read-only "
                 "(`draw_readout_grid`); the consuming end of that direction.")
    lines.append("- **Declared: effective / authoring** -- what "
                 "`pending_settings.setting_owner()` and `_authoring_owner()` "
                 "currently claim.")
    lines.append("- **`getattr` defaults** -- how many consumer reads pass a "
                 "default and would therefore swallow the `AttributeError` the "
                 "split is meant to raise (Phase 3).")
    lines.append("")

    lines.append("## Verdicts")
    lines.append("")
    lines.append("| verdict | fields | meaning |")
    lines.append("|---|---:|---|")
    meanings = {
        "OK": "UI writes exactly where the consumer reads",
        "OK_MULTI": "write and read meet, but more than one path exists",
        "MIRROR_ONLY": "**only** the Physics mirror writes where the consumer "
                       "reads; the Scene-tab control edits the wrong datablock",
        "BROKEN": "**UI writes a datablock no consumer reads**",
        "WRITE_ONLY": "UI writes it; no consumer read resolved statically",
        "READ_ONLY": "something reads it; nothing resolved writes it",
        "PUBLISHED": "the runtime publishes it where its reader looks",
        "PUBLISHED_MULTI": "published and observed, but by more than one path",
        "PUBLISHED_BROKEN": "**the runtime writes one datablock and the panel "
                            "displays the other**",
        "UI_STATE": "a panel writes it and a panel reads it; nothing outside "
                    "the UI observes it",
        "INTERNAL": "written outside the UI; no resolved reader",
        "UNOBSERVED": "no access of any kind resolved -- a dead property",
    }
    order = ("OK", "OK_MULTI", "MIRROR_ONLY", "BROKEN", "PUBLISHED",
             "PUBLISHED_MULTI", "PUBLISHED_BROKEN", "UI_STATE", "WRITE_ONLY",
             "READ_ONLY", "INTERNAL", "UNOBSERVED")
    unknown = set(verdicts) - set(order)
    assert not unknown, f"unrendered verdicts: {sorted(unknown)}"
    # Every verdict is listed even at zero. The fault verdicts reaching zero is
    # the headline result of Phase 1, and a legend that simply omitted them
    # would make "no field is BROKEN" indistinguishable from "the analyzer no
    # longer looks for BROKEN".
    for verdict in order:
        count = len(verdicts.get(verdict, ()))
        emphasis = "**" if count and verdict in FAULT_VERDICTS else ""
        lines.append(
            f"| {verdict} | {emphasis}{count}{emphasis} | {meanings[verdict]} |"
        )
    lines.append(f"| **total** | **{len(rows)}** | |")
    lines.append("")

    redirect_note = ", ".join(sorted(redirected)) or "(none)"
    no_redirect = sorted(mirrored - redirected)
    lines.append("## Mirror redirection")
    lines.append("")
    lines.append(
        f"`owner_redirect=True` ({len(redirected)} source panels): {redirect_note}"
    )
    lines.append("")
    lines.append(
        f"Mirrored **without** redirect ({len(no_redirect)}): "
        + (", ".join(no_redirect) or "(none)")
    )
    lines.append("")
    lines.append(
        "A mirror re-runs its source panel's draw unchanged, so every control "
        "writes the same datablock in both tabs. `owner_redirect=True` only "
        "hides a mirror while no domain object is designated."
    )
    lines.append("")

    lines.append("## The map")
    lines.append("")
    lines.append(
        "| field | tier | UI writes | consumer reads | runtime writes | "
        "UI reads | declared eff/auth | getattr defaults | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---:|---|")
    for row in rows:
        verdict, _reason = classify(row)
        ui_write = ", ".join(sorted(row["ui_write"])) or "-"
        read = ", ".join(sorted(row["read"])) or "-"
        runtime_write = ", ".join(sorted(row["other_write"])) or "-"
        ui_read = ", ".join(sorted(row["ui_read"])) or "-"
        declared = f"{row['effective']}/{row['authoring']}"
        flag = "**" if verdict in FAULT_VERDICTS else ""
        lines.append(
            f"| `{row['field']}` | {row['tier']} | {ui_write} | {read} | "
            f"{runtime_write} | {ui_read} | {declared} | "
            f"{row['defaulted_reads']} | {flag}{verdict}{flag} |"
        )
    lines.append("")

    for verdict in FAULT_VERDICTS:
        subject = [row for row in rows if classify(row)[0] == verdict]
        if not subject:
            continue
        lines.append(f"## {verdict} in detail")
        lines.append("")
        for row in subject:
            lines.append(f"### `{row['field']}`  ({row['tier']})")
            lines.append("")
            lines.append(f"- {classify(row)[1]}")
            for location, line, kind, via in row["ui_sites"][:4]:
                lines.append(
                    f"- UI writes **{kind}** via `{via}` -- {location}:{line}"
                )
            for location, line, kind, via in row["other_write_sites"][:4]:
                lines.append(
                    f"- runtime writes **{kind}** via `{via}` -- {location}:{line}"
                )
            for location, line, kind, defaulted in row["read_sites"][:4]:
                suffix = " (defaulted getattr)" if defaulted else ""
                lines.append(
                    f"- consumer reads **{kind}** -- {location}:{line}{suffix}"
                )
            for location, line, kind, defaulted in row["ui_read_sites"][:4]:
                lines.append(f"- UI reads **{kind}** -- {location}:{line}")
            lines.append("")

    dead = [r for r in rows if classify(r)[0] in ("UNOBSERVED", "INTERNAL")]
    if dead:
        lines.append("## Fields with no resolved reader")
        lines.append("")
        lines.append(
            "These have a declared scope, but nothing in the shipped add-on "
            "observes them -- so the declaration is unfalsifiable and the "
            "split may assign them either way without changing behavior. They "
            "are candidates for deletion rather than migration."
        )
        lines.append("")
        lines.append("| field | verdict | written at |")
        lines.append("|---|---|---|")
        for row in dead:
            sites = ", ".join(
                f"`{loc}:{line}`"
                for loc, line, _kind, _via in row["other_write_sites"][:2]
            ) or "(never written either)"
            lines.append(f"| `{row['field']}` | {classify(row)[0]} | {sites} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the checked-in map is stale")
    parser.add_argument("--print-summary", action="store_true",
                        help="print verdict counts without writing the file")
    args = parser.parse_args(argv)

    rows, redirected, mirrored = build_map()
    rendered = render(rows, redirected, mirrored)

    counts = defaultdict(int)
    for row in rows:
        counts[classify(row)[0]] += 1

    if args.print_summary:
        for verdict in sorted(counts):
            print(f"{verdict:<12} {counts[verdict]:>4}")
        print(f"{'total':<12} {len(rows):>4}")
        return 0

    if args.check:
        if not os.path.exists(OUTPUT_PATH):
            print(f"missing {OUTPUT_PATH}; run python tools/ownership_map.py")
            return 1
        with open(OUTPUT_PATH, encoding="utf-8") as handle:
            existing = handle.read()
        if existing != rendered:
            print(f"{OUTPUT_PATH} is stale; run python tools/ownership_map.py")
            return 1
        print("ownership map is current")
        return 0

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
    for verdict in sorted(counts):
        print(f"{verdict:<12} {counts[verdict]:>4}")
    print(f"wrote {os.path.relpath(OUTPUT_PATH, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Settings declarations agree with each other and with their meaning.

A Blender property, the solver's ``SolverSettings`` field and every literal
fallback that reads the same setting must use one default; otherwise a scene
built headlessly, a partial settings object and the UI quietly disagree.
Property subtypes must also match the value's meaning (lengths show units,
FACTOR sliders stay within 0..1).
"""

import ast
import pathlib

from fluidSim_warp_blender51.blender import solver_settings
from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim import enums

PKG = pathlib.Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"

# Settings whose literal fallbacks must equal the declared default.
FALLBACK_CHECKED = ("surface_resolution_multiplier", "surface_leaf_capacity_scale",
                    "surface_max_dense_dim", "surface_smooth_iterations")
# Settings projected to the solver by a path other than DIRECT_SETTING_FIELDS or
# the enum table in blender/solver_settings.py.
ALSO_PROJECTED = {"gravity": "apply_gravity"}


def _tree(rel):
    return ast.parse((PKG / rel).read_text(encoding="utf-8"))


def _class(tree, name):
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == name)


def _names(tree):
    """Names a default expression may use: the constants module (as ``C`` and
    ``_C``) and the enum classes the file imports from ``sim/enums.py``."""
    names = {"C": C, "_C": C, "int": int, "float": float}
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.level
                and node.module in ("enums", "sim.enums")):
            for alias in node.names:
                names[alias.asname or alias.name] = getattr(enums, alias.name)
    return names


def _evaluate(node, names):
    try:
        return ast.literal_eval(node)
    except ValueError:
        return eval(compile(ast.Expression(node), "<default>", "eval"),
                    {"__builtins__": {}}, names)


def _solver_defaults():
    """Every ``SolverSettings`` field that has a plain default, evaluated: literals,
    ``C.DEFAULT_*`` constants and ``int(SomeEnum.MEMBER)`` expressions alike."""
    tree = _tree("sim/solver.py")
    names = _names(tree)
    defaults = {}
    for item in _class(tree, "SolverSettings").body:
        if (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                and item.value is not None):
            if isinstance(item.value, ast.Call) and getattr(item.value.func, "id", "") == "dc_field":
                continue  # a field(...) with a factory or hidden default, not a setting
            defaults[item.target.id] = _evaluate(item.value, names)
    return defaults


def _rna_properties():
    tree = _tree("properties.py")
    props = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("FLUIDSIM_PG_"):
            for item in node.body:
                if (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                        and isinstance(item.annotation, ast.Call)):
                    props[(node.name, item.target.id)] = item.annotation
    return props


def _keywords(call):
    values = {}
    for kw in call.keywords:
        try:
            values[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            pass
    return values


def _rna_defaults(group):
    """``{name: default}`` for every property of ``group`` that declares one,
    evaluated the same way as the solver side (constants and enum members too)."""
    names = _names(_tree("properties.py"))
    defaults = {}
    for (cls, name), call in _rna_properties().items():
        if cls != group:
            continue
        for kw in call.keywords:
            if kw.arg == "default":
                defaults[name] = _evaluate(kw.value, names)
    return defaults


def _same(a, b):
    if isinstance(a, str) or isinstance(b, str) or isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(float(x) == float(y) for x, y in zip(a, b))
    return float(a) == float(b)


def _projection():
    """``(rna name, solver field, enum class or None)`` for every setting that
    ``blender/solver_settings.py`` projects onto ``SolverSettings``, including the
    renamed ones (``SETTING_ALIASES``) and the enum-valued ones."""
    aliases = solver_settings.SETTING_ALIASES
    pairs = [(name, aliases.get(name, name), None)
             for name in solver_settings.DIRECT_SETTING_FIELDS]
    for target, (source, enum_type) in solver_settings.enum_setting_fields().items():
        pairs.append((source, aliases.get(target, target), enum_type))
    pairs.extend((name, name, None) for name in ALSO_PROJECTED)
    return pairs


def test_rna_defaults_match_solver_defaults():
    """The Blender property default is the one default: the solver dataclass must
    equal it for every projected setting, enum, string and constant-backed ones
    included, so a headless ``SolverSettings()`` bakes what the UI shows."""
    solver = _solver_defaults()
    rna = _rna_defaults("FLUIDSIM_PG_domain")
    missing, divergent = [], []
    for source, target, enum_type in _projection():
        if source not in rna or target not in solver:
            missing.append(f"{source}->{target}")
            continue
        value = solver[target]
        if enum_type is not None and not isinstance(value, str):
            value = enum_type(value).name
        if not _same(rna[source], value):
            divergent.append(f"{source}->{target}: rna={rna[source]!r} solver={solver[target]!r}")
    assert not missing, "not found: " + ", ".join(missing)
    assert not divergent, "default divergences:\n" + "\n".join(divergent)
    assert len(_projection()) > 100


def test_every_shared_setting_name_is_projected():
    """A field that exists on a settings group and on ``SolverSettings`` under the
    same name must be reached through the projection, so it cannot go unchecked
    by the default-agreement test above."""
    solver = _solver_defaults()
    projected = {source for source, _target, _enum in _projection()}
    shared = []
    for group in ("FLUIDSIM_PG_domain", "FLUIDSIM_PG_scene"):
        shared.extend(f"{group}.{name}" for name in _rna_defaults(group)
                      if name in solver and name not in projected)
    assert not shared, "same-name fields outside the projection: " + ", ".join(shared)


def test_literal_fallbacks_match_declared_defaults():
    solver = _solver_defaults()
    wrong = []
    seen = set()
    for path in sorted(PKG.rglob("*.py")):
        if "tests" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            pairs = []
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "_value") and len(node.args) == 3):
                pairs.append((node.args[1], node.args[2]))
            elif isinstance(node, ast.Dict):
                pairs.extend((k, v) for k, v in zip(node.keys, node.values) if k is not None)
            for key, value in pairs:
                if not (isinstance(key, ast.Constant) and key.value in FALLBACK_CHECKED):
                    continue
                try:
                    literal = ast.literal_eval(value)
                except ValueError:
                    continue
                seen.add(key.value)
                if not _same(literal, solver[key.value]):
                    wrong.append(f"{path.relative_to(PKG).as_posix()}:{getattr(node, 'lineno', 0)} "
                                 f"{key.value}={literal!r}")
    assert seen == set(FALLBACK_CHECKED)
    assert wrong == []


def test_quality_contract_and_extractor_defaults_match_solver():
    solver = _solver_defaults()
    authored = _class(_tree("sim/quality_contract.py"), "AuthoredQualitySettings")
    field = next(item for item in authored.body if isinstance(item, ast.AnnAssign)
                 and isinstance(item.target, ast.Name)
                 and item.target.id == "surface_resolution_multiplier")
    assert field.value is not None
    assert ast.literal_eval(field.value) == solver["surface_resolution_multiplier"]

    init = next(item for item in _class(_tree("sim/surface.py"), "SurfaceExtractor").body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__")
    names = [arg.arg for arg in init.args.args]
    defaults = dict(zip(names[len(names) - len(init.args.defaults):], init.args.defaults))
    assert ast.literal_eval(defaults["max_dense_dim"]) == solver["surface_max_dense_dim"]


def test_property_subtypes_match_meaning():
    props = {name: _keywords(call) for (_cls, name), call in _rna_properties().items()}
    for name in ("voxel_size", "surface_voxel_size", "height_base", "feather"):
        assert props[name].get("subtype") == "DISTANCE", name
    assert props["max_speed"].get("unit") == "VELOCITY"
    assert props["height_velocity"].get("subtype") == "VELOCITY"
    for (cls, name), call in _rna_properties().items():
        kw = _keywords(call)
        if kw.get("subtype") == "FACTOR":
            bound = kw.get("max", kw.get("soft_max"))
            assert bound is not None and bound <= 1.0, f"{cls}.{name}"


def test_surface_smooth_items_keep_their_saved_numbers():
    tree = _tree("properties.py")
    assign = next(node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "SURFACE_SMOOTH_ITEMS"
                          for t in node.targets))
    items = ast.literal_eval(assign.value)
    assert [(item[0], item[3]) for item in items] == [
        ("BOX", 0), ("MEAN_CURVATURE_FLOW", 1)]
    filter_members = {item.targets[0].id for item in _class(_tree("sim/enums.py"), "SdfFilterType").body
                      if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)}
    assert {item[0] for item in items} <= filter_members

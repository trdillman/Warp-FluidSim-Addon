"""Settings declarations agree with each other and with their meaning.

A Blender property, the solver's ``SolverSettings`` field and every literal
fallback that reads the same setting must use one default; otherwise a scene
built headlessly, a partial settings object and the UI quietly disagree.
Property subtypes must also match the value's meaning (lengths show units,
FACTOR sliders stay within 0..1).
"""

import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"

# Known RNA/solver divergences, carried to the 2.1 handoff: aligning the
# solver side changes headless viscosity and whitewater baselines.
KNOWN_RNA_SOLVER_DIVERGENCES = {"viscosity_constant", "ww_bubble_drag", "ww_spray_drag"}
# Settings whose literal fallbacks must equal the declared default.
FALLBACK_CHECKED = ("surface_resolution_multiplier", "surface_leaf_capacity_scale",
                    "surface_max_dense_dim", "surface_smooth_iterations")


def _tree(rel):
    return ast.parse((PKG / rel).read_text(encoding="utf-8"))


def _class(tree, name):
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == name)


def _solver_defaults():
    defaults = {}
    for item in _class(_tree("sim/solver.py"), "SolverSettings").body:
        if (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                and item.value is not None):
            try:
                defaults[item.target.id] = ast.literal_eval(item.value)
            except ValueError:
                pass
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


def _same(a, b):
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(float(x) == float(y) for x, y in zip(a, b))
    return float(a) == float(b)


def test_rna_defaults_match_solver_defaults():
    solver = _solver_defaults()
    divergent = set()
    compared = 0
    for (cls, name), call in _rna_properties().items():
        if cls not in ("FLUIDSIM_PG_domain", "FLUIDSIM_PG_scene") or name not in solver:
            continue
        default = _keywords(call).get("default")
        if default is None or isinstance(default, str) or isinstance(solver[name], str):
            continue
        compared += 1
        if not _same(default, solver[name]):
            divergent.add(name)
    assert compared > 100
    assert divergent == KNOWN_RNA_SOLVER_DIVERGENCES


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

"""ST-FLIP ships as Experimental and Variable Projection is off by default.

Release decision D6 for 2.0: the UI labels say Experimental (not Beta), and
both the Blender property and the solver's own default keep Variable
Projection off, so a scene that never touches it runs standard projection.
"""

import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"


def _class_annotations(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {node.target.id: node for node in cls.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}


def _property_keywords(name):
    call = _class_annotations(PKG / "properties.py", "FLUIDSIM_PG_domain")[name].annotation
    assert isinstance(call, ast.Call)
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
            if kw.arg in ("name", "default")}


def test_stflip_controls_are_labelled_experimental():
    for name in ("spacetime_enabled", "spacetime_variable_projection"):
        label = _property_keywords(name)["name"]
        assert label.endswith("(Experimental)"), label


def test_variable_projection_is_off_by_default_in_ui_and_solver():
    assert _property_keywords("spacetime_variable_projection")["default"] is False
    solver_default = _class_annotations(
        PKG / "sim" / "solver.py", "SolverSettings")["spacetime_variable_projection"]
    assert solver_default.value is not None
    assert ast.literal_eval(solver_default.value) is False

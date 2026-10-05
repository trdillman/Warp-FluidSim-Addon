"""CPU contract tests for the real workspace layout helper, without bpy imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def helper():
    source = Path(__file__).parents[2] / "fluidSim_warp_blender51/blender/ui_header.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in
              {"_screens", "_spaces_of", "prepare_workspace"}) or
             (isinstance(node, ast.Assign) and any(
              isinstance(target, ast.Name) and target.id == "_KEEP_AREA_TYPES"
              for target in node.targets))]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["prepare_workspace"]


def area(kind, subtype=None, width=500, height=100):
    return SimpleNamespace(type=kind, ui_type=subtype or kind, width=width,
                           height=height, spaces=[SimpleNamespace(
                               type=kind, show_region_ui=False)])


@pytest.mark.parametrize("subtype", ["DOPESHEET", "ACTION"])
def test_existing_dopesheet_subtype_becomes_a_real_timeline(subtype):
    viewport, editor = area("VIEW_3D"), area("DOPESHEET_EDITOR", subtype)
    workspace = SimpleNamespace(object_mode="SCULPT", screens=[
        SimpleNamespace(areas=[viewport, editor])])
    helper()(workspace)
    assert editor.type == "DOPESHEET_EDITOR"
    assert editor.ui_type == "TIMELINE"
    assert viewport.spaces[0].show_region_ui is True
    assert workspace.object_mode == "OBJECT"


def test_existing_timeline_does_not_repurpose_other_dopesheets():
    editors = [area("VIEW_3D"), area("DOPESHEET_EDITOR", "ACTION"),
               area("DOPESHEET_EDITOR", "TIMELINE")]
    helper()(SimpleNamespace(screens=[SimpleNamespace(areas=editors)]))
    assert editors[1].ui_type == "ACTION"
    assert editors[2].ui_type == "TIMELINE"


def test_missing_timeline_uses_smallest_spare_without_repurposing_properties():
    editors = [area("VIEW_3D"), area("OUTLINER", height=200),
               area("FILE_BROWSER", height=50), area("PROPERTIES")]
    helper()(SimpleNamespace(screens=[SimpleNamespace(areas=editors)]))
    assert editors[1].type == "OUTLINER"
    assert editors[2].type == "DOPESHEET_EDITOR"
    assert editors[2].ui_type == "TIMELINE"
    assert editors[3].type == "PROPERTIES"


def test_compact_layout_without_spare_remains_unresolved():
    editors = [area("VIEW_3D"), area("PROPERTIES")]
    helper()(SimpleNamespace(screens=[SimpleNamespace(areas=editors)]))
    assert [item.type for item in editors] == ["VIEW_3D", "PROPERTIES"]


def test_helper_prepares_each_screen():
    screens = [SimpleNamespace(areas=[area("VIEW_3D"),
               area("DOPESHEET_EDITOR", subtype)]) for subtype in ("ACTION", "DOPESHEET")]
    helper()(SimpleNamespace(screens=screens))
    assert all(screen.areas[1].ui_type == "TIMELINE" for screen in screens)

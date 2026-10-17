"""Every user setting is reachable from All Simulation Settings (Properties).

Basic/Advanced trims the 3D View sidebar only; Properties panels never hide a
setting by ``ui_mode``, and a row whose feature is off is greyed, not removed.
A new property must either be drawn somewhere in the add-on UI or be listed
below as an internal field, readout or legacy migration input.
"""

from __future__ import annotations

import ast
from pathlib import Path

ADDON = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
BLENDER = ADDON / "blender"

GROUPS = {
    ADDON / "properties.py": ("FLUIDSIM_PG_scene", "FLUIDSIM_PG_domain", "FLUIDSIM_PG_object"),
    BLENDER / "effects.py": ("FLUIDSIM_PG_effect_kernel",),
}

# Not settings: runtime readouts, bake/transport state, UI disclosure state,
# migration markers and legacy fields superseded by a drawn replacement.
EXEMPT = {
    # Scene: transport, readouts and disclosure state.
    "live", "status", "bake_status", "bake_message", "cache_format",
    "cache_enabled", "cache_playback_enabled", "setup_mode", "active_preset_id",
    "deck_shown", "deck_stage", "show_surface", "show_particles",
    "capacity_show_alternatives",
    # Domain: readouts; the whitewater type attribute is always published.
    "height_error", "whitewater_attribute_type",
    # Object: migration marker, readouts and legacy fields replaced by
    # Region (source/sink region kinds) and Body Type.
    "source_ui_schema_version", "mesh_error", "use_mesh_shape", "geometry_mode",
    "sink_region_mode", "source_mode", "collider_static", "collider_dynamic",
    # Effect kernel: identity, generated output and error position.
    "kernel_id", "fields", "parameter_keys_authored", "error_line",
    "error_column", "generated_source_digest", "generated_text_name",
}
EXEMPT_PREFIXES = ("stat_", "source_status_")
EXEMPT_SUFFIXES = ("_expanded",)


def _declared(path, class_names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    yield node.name, item.target.id


def _strings(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def _drawn_names():
    """String arguments of every UI drawing call in the add-on's blender package."""
    names = set()
    for path in BLENDER.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if "prop" in name or name.startswith(("draw", "_draw", "template_")):
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    names.update(_strings(arg))
    # The attribute panels draw their toggles from this table (ui_scene).
    from fluidSim_warp_blender51.sim.attributes import OUTPUT_TOGGLE_NAMES

    for toggles in OUTPUT_TOGGLE_NAMES.values():
        names.update(toggles.values())
    return names


def _exempt(name):
    return (name in EXEMPT or name.startswith(EXEMPT_PREFIXES)
            or name.endswith(EXEMPT_SUFFIXES))


def test_every_setting_is_drawn_in_some_panel():
    drawn = _drawn_names()
    missing = [f"{group}.{name}"
               for path, classes in GROUPS.items()
               for group, name in _declared(path, classes)
               if not _exempt(name) and name not in drawn]
    assert missing == []


def test_exempt_list_has_no_stale_entries():
    declared = {name for path, classes in GROUPS.items()
                for _, name in _declared(path, classes)}
    assert sorted(EXEMPT - declared) == []


def test_properties_panels_never_poll_on_ui_mode():
    offenders = []
    for name in ("ui_scene.py", "ui_physics.py", "ui_object.py", "effect_ui.py"):
        tree = ast.parse((BLENDER / name).read_text(encoding="utf-8"))
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "poll":
                    text = ast.unparse(fn)
                    if "ui_mode" in text or "_advanced_visible" in text:
                        offenders.append(f"{name}:{cls.name}")
    assert offenders == []


def test_properties_panels_start_expanded():
    # A collapsed panel hides every setting inside it until the user finds it,
    # so no Properties panel (or Physics mirror of one) may be DEFAULT_CLOSED.
    offenders = []
    for name in ("ui_scene.py", "ui_physics.py", "ui_object.py", "effect_ui.py",
                 "effect_workspace.py"):
        tree = ast.parse((BLENDER / name).read_text(encoding="utf-8"))
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            if "DEFAULT_CLOSED" in set(_strings(cls)):
                offenders.append(f"{name}:{cls.name}")
    assert offenders == []


def test_surface_resolution_rows_are_never_greyed():
    tree = ast.parse((BLENDER / "ui_scene.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "FLUIDSIM_PT_surface_resolution")
    text = ast.unparse(cls)
    for prop in ("surface_resolution_multiplier", "surface_voxel_size"):
        assert f"layout.prop(fs, '{prop}')" in text

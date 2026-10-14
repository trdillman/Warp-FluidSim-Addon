"""Liquid Deck parity: every Deck control is a thin twin of something a panel draws.

Spec trap 1 (docs/specs/4.0/SPEC-liquid-deck.md): the Deck never holds settings
state. A button calls an operator the panels also call; a dial writes a property
the panels also draw. The command palette indexes exactly the registered
FluidSim operators.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck
from settings_doubles import full_settings, split_settings

ROOT = Path(__file__).resolve().parents[2]
ADDON = ROOT / "fluidSim_warp_blender51"
PROPERTIES = ADDON / "properties.py"
# Modules that draw panels, headers and menus: the surfaces a Deck control twins.
UI_MODULES = (
    ADDON / "panels.py",
    *sorted((ADDON / "blender").glob("ui_*.py")),
    ADDON / "blender" / "demo_gallery.py",
    ADDON / "blender" / "menus.py",
)
GROUP_CLASSES = {
    "scene": "FLUIDSIM_PG_scene",
    "domain": "FLUIDSIM_PG_domain",
    "object": "FLUIDSIM_PG_object",
}


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def declared_operator_ids():
    """Every ``bl_idname = "fluidsim.*"`` an add-on class declares."""
    ids = set()
    for path in ADDON.rglob("*.py"):
        if "tests" in path.relative_to(ADDON).parts:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ClassDef):
                for statement in node.body:
                    if (isinstance(statement, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "bl_idname"
                                    for t in statement.targets)
                            and isinstance(statement.value, ast.Constant)
                            and str(statement.value.value).startswith("fluidsim.")):
                        ids.add(statement.value.value)
    return ids


def declared_fields(class_name):
    for node in _tree(PROPERTIES).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {s.target.id for s in node.body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    raise LookupError(class_name)


def panel_operator_ids():
    """Operator ids passed as the first argument of ``layout.operator(...)``."""
    ids = set()
    for path in UI_MODULES:
        for node in ast.walk(_tree(path)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"operator", "operator_menu_enum"}
                    and node.args and isinstance(node.args[0], ast.Constant)):
                ids.add(node.args[0].value)
    return ids


def panel_string_literals():
    """String literals in UI modules: ``prop`` names and the drawn ``*_PROPS`` tuples."""
    return {node.value for path in UI_MODULES for node in ast.walk(_tree(path))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}


@pytest.fixture
def deck(monkeypatch):
    install_deck_bpy(monkeypatch, deck_mode="ON")
    return reload_deck()


def _target_exists(control, operators):
    if control.kind == "operator":
        return control.target.startswith("screen.") or control.target in operators
    return control.target in declared_fields(GROUP_CLASSES[control.owner])


def test_every_deck_control_targets_an_existing_operator_or_property_a_panel_also_draws(deck):
    operators = declared_operator_ids()
    panel_ops = panel_operator_ids()
    panel_literals = panel_string_literals()
    controls = deck.controls.all_controls()
    assert controls, "the Deck exposes no controls"
    for control in controls:
        if control.optional and not _target_exists(control, operators):
            # Lands in another 4.0 lane (Liquid Preview, rigid coupling); the Deck
            # disables the control until its target exists.
            continue
        assert _target_exists(control, operators), control
        if control.kind == "operator":
            assert control.target in panel_ops, f"{control.target} has no panel twin"
        else:
            assert control.kind == "property", control
            assert control.target in panel_literals, f"{control.target} has no panel twin"


def test_optional_controls_are_exactly_the_targets_other_lanes_deliver(deck):
    optional = {c.target for c in deck.controls.all_controls() if c.optional}
    assert optional == {"fluidsim.toggle_liquid_preview", "collider_dynamic"}


def test_every_deck_button_tooltip_names_its_target(deck):
    for control in deck.controls.all_controls():
        assert control.target in control.tooltip, control


def test_deck_controls_use_the_real_names_not_the_spec_placeholders(deck):
    targets = {control.target for control in deck.controls.all_controls()}
    # The spec/mockup names differ from the add-on; the Deck must use the real ones.
    assert {"fluidsim.step", "fluidsim.apply_changes", "fluidsim.revert_pending",
            "fluidsim.restart_simulation", "fluidsim.bake_start",
            "initial_velocity", "collider_friction"} <= targets
    assert not targets & {"fluidsim.step_frame", "fluidsim.apply_pending_settings",
                          "fluidsim.revert_pending_settings", "fluidsim.bake",
                          "emitter_velocity", "emission_rate"}


def test_wheel_has_six_slots_for_each_stage_in_workflow_order(deck):
    assert tuple(deck.wheel.STAGE_SLOTS) == ("SETUP", "SIMULATE", "LOOK", "BAKE")
    for stage, slots in deck.wheel.STAGE_SLOTS.items():
        assert len(slots) == 6, stage


def _fake_ops(ids):
    def make(identifier):
        name = identifier.split(".", 1)[1]
        rna = SimpleNamespace(name=name.replace("_", " ").title(),
                              description=f"Run {identifier}")
        return SimpleNamespace(get_rna_type=lambda: rna, idname=lambda: identifier)

    return SimpleNamespace(**{identifier.split(".", 1)[1]: make(identifier)
                              for identifier in ids})


def test_palette_index_equals_the_registered_fluidsim_operator_set(deck):
    operators = declared_operator_ids()
    assert {"fluidsim.command_palette", "fluidsim.radial_wheel"} <= operators
    entries = deck.palette.operator_entries(_fake_ops(operators))
    indexed = {entry.target for entry in entries}
    assert indexed == operators - deck.palette.EXCLUDED_OPERATORS
    # Only the Deck's own launchers and its event router stay out of the index.
    assert deck.palette.EXCLUDED_OPERATORS == {
        "fluidsim.command_palette", "fluidsim.radial_wheel", "fluidsim.deck_modal"}
    assert all(entry.kind == "operator" for entry in entries)


def _scene():
    fs = full_settings(resolution_divisions=64, flip_ratio=0.95, viscosity_enabled=False,
                       liquid_view_mode="SURFACE", show_overlay=True)
    return SimpleNamespace(fluidsim=split_settings(fs), objects=[])


def test_palette_settings_show_current_values_and_edit_numbers_through_the_property(deck):
    scene = _scene()
    entries = {entry.target: entry for entry in deck.palette.setting_entries(scene)}
    divisions = entries["resolution_divisions"]
    assert divisions.kind == "setting" and divisions.value == "64" and divisions.editable
    deck.palette.commit_inline(scene, divisions, "96")
    assert scene.fluidsim.domain_object.fluidsim_domain.resolution_divisions == 96

    flip = entries["flip_ratio"]
    deck.palette.commit_inline(scene, flip, "0.5")
    assert scene.fluidsim.domain_object.fluidsim_domain.flip_ratio == pytest.approx(0.5)


def test_palette_booleans_toggle_through_the_property(deck):
    scene = _scene()
    entries = {entry.target: entry for entry in deck.palette.setting_entries(scene)}
    viscosity = entries["viscosity_enabled"]
    assert viscosity.value == "Off" and viscosity.editable
    deck.palette.commit_inline(scene, viscosity, "")
    assert scene.fluidsim.domain_object.fluidsim_domain.viscosity_enabled is True


def test_palette_enum_settings_are_read_only_and_jump_to_the_panel(deck):
    scene = _scene()
    entries = {entry.target: entry for entry in deck.palette.setting_entries(scene)}
    view = entries["liquid_view_mode"]
    assert not view.editable
    assert view.jump_operator == "fluidsim.open_scene_properties"
    with pytest.raises(ValueError):
        deck.palette.commit_inline(scene, view, "PARTICLES")
    assert scene.fluidsim.liquid_view_mode == "SURFACE"


def test_palette_rejects_a_non_number_without_writing(deck):
    scene = _scene()
    entries = {entry.target: entry for entry in deck.palette.setting_entries(scene)}
    with pytest.raises(ValueError):
        deck.palette.commit_inline(scene, entries["resolution_divisions"], "lots")
    assert scene.fluidsim.domain_object.fluidsim_domain.resolution_divisions == 64


def test_palette_leaves_presentation_and_status_fields_out(deck):
    targets = {entry.target for entry in deck.palette.setting_entries(_scene())}
    assert not targets & {"deck_shown", "deck_stage", "stat_particles", "bake_progress",
                          "status", "ui_mode"}


def test_palette_indexes_presets_and_demos_with_a_kind_column(deck):
    entries = deck.palette.demo_entries()
    assert {entry.kind for entry in entries} == {"demo"}
    assert {"dam_break", "paddle_stir", "crown_splash", "chromatic_vortex"} <= {
        entry.target for entry in entries}
    presets = deck.palette.preset_entries(records=[
        SimpleNamespace(source="builtin", preset=SimpleNamespace(
            id="calm", name="Calm Water", description="Slow"))])
    assert [(p.kind, p.target, p.label) for p in presets] == [
        ("preset", "builtin:calm", "Calm Water")]


def test_fuzzy_match_is_an_ordered_subsequence_and_ranks_tight_matches_first(deck):
    match = deck.palette.fuzzy_score
    assert match("rst", "Restart Simulation") is not None
    assert match("xyz", "Restart Simulation") is None
    assert match("", "anything") == 0
    assert match("bake", "Bake Start") > match("bake", "Big Area Kinetic Effect")

"""B57: a role picked from the Role dropdown gets the current source schema.

The batch "Set Selected as ..." operators stamp the current ``source_ui_schema_version``
when they tag an object. The Role dropdown on the Object panel wrote only
``role``, so the object stayed at schema 0: the sampling controls the panel
shows (Inherit / Multiplier) were ignored for a legacy absolute rate, and a
sink kept its legacy geometry instead of the Removal Region it shows.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender.scene_inputs import CURRENT_SOURCE_UI_SCHEMA
from test_artist_mode_callback_ownership import _load_properties

_PROPERTIES = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51" / "properties.py"


def _object_group(**overrides):
    values = dict(
        role="NONE", source_ui_schema_version=0,
        source_sampling_mode="LEGACY_ABSOLUTE", source_sampling_multiplier=1.0,
        emit_rate=8.0, source_region_kind="ANALYTIC",
        sink_region_kind="ANALYTIC", geometry_mode="ANALYTIC")
    values.update(overrides)
    return SimpleNamespace(**values)


def _scene():
    domain = SimpleNamespace(voxel_size=0.0625, particles_per_voxel=8)
    return SimpleNamespace(fluidsim=SimpleNamespace(
        quality="PREVIEW", domain_object=SimpleNamespace(fluidsim_domain=domain)))


def test_the_role_property_routes_through_the_schema_stamping_callback():
    tree = ast.parse(_PROPERTIES.read_text(encoding="utf-8"))
    group = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "FLUIDSIM_PG_object")
    role = next(statement for statement in group.body
                if isinstance(statement, ast.AnnAssign)
                and getattr(statement.target, "id", "") == "role")
    update = {kw.arg: kw.value for kw in role.annotation.keywords}["update"]
    assert ast.unparse(update) == "_update_object_role"


def test_a_dropdown_source_honours_the_sampling_the_panel_shows(monkeypatch):
    _fake_bpy, properties = _load_properties(monkeypatch)
    requested = []
    monkeypatch.setattr(properties, "_request_scene_update",
                        lambda _context, action: requested.append(action))
    from fluidSim_warp_blender51.blender import scene_inputs

    scene = _scene()
    pg = _object_group(role="SOURCE")
    properties._update_object_role(pg, SimpleNamespace(scene=scene))

    assert pg.source_ui_schema_version == CURRENT_SOURCE_UI_SCHEMA
    # A new source inherits the simulation's Liquid Sampling, as the
    # "Set Selected as Liquid Sources" operator gives it.
    assert pg.source_sampling_mode == "INHERIT"
    pg.source_sampling_mode = "MULTIPLIER"
    pg.source_sampling_multiplier = 0.0
    assert scene_inputs._effective_source_target(scene, pg) == 0
    assert [str(getattr(action, "value", action)) for action in requested] == ["RESET"]


@pytest.mark.parametrize("role", ["SINK", "FIELD"])
def test_a_dropdown_sink_or_field_uses_the_current_region_schema(monkeypatch, role):
    _fake_bpy, properties = _load_properties(monkeypatch)
    monkeypatch.setattr(properties, "_request_scene_update", lambda *_args: None)

    pg = _object_group(role=role, sink_region_kind="MESH_SOLID")
    properties._update_object_role(pg, SimpleNamespace(scene=_scene()))

    assert pg.source_ui_schema_version == CURRENT_SOURCE_UI_SCHEMA
    # Sampling choices belong to liquid sources only.
    assert pg.source_sampling_mode == "LEGACY_ABSOLUTE"


@pytest.mark.parametrize("role", ["NONE", "COLLIDER", "FORCE"])
def test_other_roles_and_current_schema_objects_are_left_alone(monkeypatch, role):
    _fake_bpy, properties = _load_properties(monkeypatch)
    monkeypatch.setattr(properties, "_request_scene_update", lambda *_args: None)

    untouched = _object_group(role=role)
    properties._update_object_role(untouched, SimpleNamespace(scene=_scene()))
    assert untouched.source_ui_schema_version == 0

    current = _object_group(role="SOURCE", source_ui_schema_version=4,
                            source_sampling_mode="MULTIPLIER",
                            source_sampling_multiplier=2.0)
    properties._update_object_role(current, SimpleNamespace(scene=_scene()))
    assert (current.source_sampling_mode, current.source_sampling_multiplier) == (
        "MULTIPLIER", 2.0)


def test_a_sampling_schema_object_keeps_its_sampling_when_restamped(monkeypatch):
    """Schema 5 (B47) changed no sampling rule: a schema-4 choice survives."""
    _fake_bpy, properties = _load_properties(monkeypatch)
    monkeypatch.setattr(properties, "_request_scene_update", lambda *_args: None)

    pg = _object_group(role="SOURCE", source_ui_schema_version=4,
                       source_sampling_mode="MULTIPLIER",
                       source_sampling_multiplier=2.0)
    properties._update_object_role(pg, SimpleNamespace(scene=_scene()))

    assert pg.source_ui_schema_version == CURRENT_SOURCE_UI_SCHEMA
    assert (pg.source_sampling_mode, pg.source_sampling_multiplier) == (
        "MULTIPLIER", 2.0)


# Regression (mesh-source acceptance): once the role stamps schema >= 3 the
# runtime reads only the region kinds, so a script that tags the role and then
# writes the legacy geometry_mode / sink_region_mode got an analytic box.

def _property_update(name):
    tree = ast.parse(_PROPERTIES.read_text(encoding="utf-8"))
    group = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "FLUIDSIM_PG_object")
    statement = next(statement for statement in group.body
                     if isinstance(statement, ast.AnnAssign)
                     and getattr(statement.target, "id", "") == name)
    return ast.unparse({kw.arg: kw.value for kw in statement.annotation.keywords}["update"])


def test_legacy_region_fields_route_through_their_alias_callbacks():
    assert _property_update("geometry_mode") == "_update_geometry_mode"
    assert _property_update("sink_region_mode") == "_update_sink_region_mode"


def test_a_scripted_mesh_source_after_the_role_reaches_the_mesh_region(monkeypatch):
    _fake_bpy, properties = _load_properties(monkeypatch)
    monkeypatch.setattr(properties, "_request_scene_update", lambda *_args: None)
    from fluidSim_warp_blender51.blender.scene_inputs import legacy_region_kinds

    context = SimpleNamespace(scene=_scene())
    pg = _object_group(sink_region_mode="SOLID")
    pg.role = "SOURCE"
    properties._update_object_role(pg, context)
    pg.geometry_mode = "MESH"
    properties._update_geometry_mode(pg, context)

    # The same mesh region the legacy schema simulated it with.
    assert (pg.source_region_kind, pg.sink_region_kind) == legacy_region_kinds(pg)
    assert pg.source_region_kind == "MESH_SHELL"

    pg.role = "SINK"
    properties._update_object_role(pg, context)
    pg.sink_region_mode = "SURFACE"
    properties._update_sink_region_mode(pg, context)
    assert pg.sink_region_kind == "MESH_SHELL"
    pg.sink_region_mode = "SOLID"
    properties._update_sink_region_mode(pg, context)
    assert pg.sink_region_kind == "MESH_SOLID"


def test_a_mesh_geometry_set_before_the_role_survives_the_stamp(monkeypatch):
    _fake_bpy, properties = _load_properties(monkeypatch)
    monkeypatch.setattr(properties, "_request_scene_update", lambda *_args: None)

    pg = _object_group(role="SINK", geometry_mode="MESH", sink_region_mode="SOLID")
    properties._update_object_role(pg, SimpleNamespace(scene=_scene()))
    assert pg.sink_region_kind == "MESH_SOLID"


def test_legacy_alias_never_overrides_a_chosen_region(monkeypatch):
    _fake_bpy, properties = _load_properties(monkeypatch)
    monkeypatch.setattr(properties, "_request_scene_update", lambda *_args: None)
    context = SimpleNamespace(scene=_scene())

    chosen = _object_group(role="SOURCE", source_ui_schema_version=CURRENT_SOURCE_UI_SCHEMA,
                           source_region_kind="MESH_SOLID", sink_region_kind="ANALYTIC",
                           geometry_mode="ANALYTIC", sink_region_mode="SOLID")
    properties._update_geometry_mode(chosen, context)       # Analytic write
    properties._update_sink_region_mode(chosen, context)    # analytic sink kind
    assert (chosen.source_region_kind, chosen.sink_region_kind) == (
        "MESH_SOLID", "ANALYTIC")
    chosen.geometry_mode = "MESH"
    properties._update_geometry_mode(chosen, context)
    assert chosen.source_region_kind == "MESH_SOLID"

    # Below schema 3 the runtime reads geometry_mode itself: nothing to forward.
    legacy = _object_group(role="SOURCE", source_ui_schema_version=0,
                           geometry_mode="MESH", sink_region_mode="SOLID")
    properties._update_geometry_mode(legacy, context)
    assert legacy.source_region_kind == "ANALYTIC"

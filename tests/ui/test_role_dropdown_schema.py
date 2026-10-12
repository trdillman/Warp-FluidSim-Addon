"""B57: a role picked from the Role dropdown gets the current source schema.

The batch "Set Selected as ..." operators stamp ``source_ui_schema_version = 4``
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

    assert pg.source_ui_schema_version == 4
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

    assert pg.source_ui_schema_version == 4
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

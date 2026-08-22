from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "blender" / "ui_object.py"


def _block(source: str, class_name: str) -> str:
    start = source.index(f"class {class_name}")
    end = source.find("\nclass ", start + 1)
    return source[start:] if end < 0 else source[start:end]


def test_object_panels_use_native_object_properties_and_stable_order():
    source = UI.read_text(encoding="utf-8")
    expected = (
        ("FLUIDSIM_PT_object_role", 0),
        ("FLUIDSIM_PT_object_region", 10),
        ("FLUIDSIM_PT_object_source", 20),
        ("FLUIDSIM_PT_object_blending", 30),
        ("FLUIDSIM_PT_object_collider", 40),
        ("FLUIDSIM_PT_object_force", 50),
        ("FLUIDSIM_PT_object_field", 60),
        ("FLUIDSIM_PT_object_volume", 70),
    )
    for class_name, order in expected:
        block = _block(source, class_name)
        assert "FluidSimObjectPanel" in block.splitlines()[0]
        assert f"bl_order = {order}" in block


def test_object_panel_polls_use_canonical_role_and_region_helpers():
    source = UI.read_text(encoding="utf-8")
    assert "def role_poll(context, *roles" in source
    assert "active_object_role(context)" in source
    assert "def volume_grid_poll(context)" in source
    assert "canonical_region_kind" in source
    assert 'role_poll(context, "SOURCE", "SINK", "FORCE", "FIELD")' in source
    assert 'role_poll(context, "SOURCE")' in source
    assert 'role_poll(context, "COLLIDER")' in source
    assert 'role_poll(context, "FORCE")' in source
    assert 'role_poll(context, "FIELD")' in source


def test_source_and_field_payloads_hide_inapplicable_controls():
    source = UI.read_text(encoding="utf-8")
    source_block = _block(source, "FLUIDSIM_PT_object_source")
    assert "source_sampling_mode" in source_block
    assert "source_sampling_multiplier" in source_block
    assert "emit_rate" in source_block
    assert "use_lifespan" in source_block
    field_block = _block(source, "FLUIDSIM_PT_object_field")
    assert "field_value_source" in field_block
    assert 'if settings.field_value_source == "CONSTANT"' in field_block
    assert "field_scalar_value" in field_block
    assert "field_vector_value" in field_block
    volume_block = _block(source, "FLUIDSIM_PT_object_volume")
    assert "volume_scalar_grid" in volume_block
    assert "volume_vector_grid" in volume_block
    assert "volume_vector_x_grid" in volume_block


def test_source_lifespan_is_a_conditional_child_panel():
    source = UI.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_object_lifespan")
    assert 'bl_parent_id = "FLUIDSIM_PT_object_source"' in block
    assert "use_lifespan" in block
    assert "life_min" in block
    assert "life_max" in block


def test_role_actions_hide_when_the_selected_set_already_has_that_role():
    source = UI.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_object_role")
    assert "selected_count" in block
    for role in ("SOURCE", "SINK", "COLLIDER", "FORCE", "FIELD"):
        assert f'counts["{role}"] < selected_count' in block
    assert "if tagged:" in block


def test_field_volume_payload_and_coverage_are_kept_coherent_by_property_updates():
    properties = (ROOT / "properties.py").read_text(encoding="utf-8")
    assert "def _update_field_region_kind" in properties
    assert "def _update_field_value_source" in properties
    region = properties.split("def _update_field_region_kind", 1)[1].split("def ", 1)[0]
    payload = properties.split("def _update_field_value_source", 1)[1].split("def ", 1)[0]
    assert 'self.field_value_source = "CONSTANT"' in region
    assert 'self.field_region_kind = "VOLUME_GRID"' in payload
    assert "update=_update_field_region_kind" in properties
    assert "update=_update_field_value_source" in properties

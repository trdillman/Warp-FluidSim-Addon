from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "fluidSim_warp_blender51" / "panels.py"
PROPERTIES = ROOT / "fluidSim_warp_blender51" / "properties.py"
OPERATORS = ROOT / "fluidSim_warp_blender51" / "operators.py"
SCENE_INPUTS = ROOT / "fluidSim_warp_blender51" / "blender" / "scene_inputs.py"


EXPECTED_PANEL_ORDER = [
    "FLUIDSIM_PT_liquid_emitters",
    "FLUIDSIM_PT_liquid_emitter_attributes",
    "FLUIDSIM_PT_liquid_sinks",
    "FLUIDSIM_PT_colliders",
    "FLUIDSIM_PT_forces",
    "FLUIDSIM_PT_domain_height_fill",
    "FLUIDSIM_PT_point_particle_injection",
    "FLUIDSIM_PT_advanced_field_sources",
]


def _source(path):
    return path.read_text(encoding="utf8")


def test_role_panels_exist_in_approved_order_without_mixed_legacy_registration():
    source = _source(PANELS)
    positions = [source.index(f"class {name}") for name in EXPECTED_PANEL_ORDER]
    assert positions == sorted(positions)
    classes_block = source[source.index("CLASSES = ("):]
    registered = [name for name in EXPECTED_PANEL_ORDER if name in classes_block]
    assert registered == EXPECTED_PANEL_ORDER
    assert "FLUIDSIM_PT_sources," not in classes_block
    assert "FLUIDSIM_PT_source_particle_attributes," not in classes_block


def test_presentation_rna_preserves_legacy_and_adds_only_approved_axes():
    source = _source(PROPERTIES)
    for legacy in (
        "role: EnumProperty", "geometry_mode: EnumProperty",
        "sink_region_mode: EnumProperty", "source_mode: EnumProperty",
        "emit_rate: FloatProperty", "emission_mode: EnumProperty",
        "height_back_mode: EnumProperty",
    ):
        assert legacy in source
    for approved in (
        "source_region_kind", "sink_region_kind", "point_source_representation",
        "point_input_object", "point_rasterize_radius", "volume_input_object",
        "volume_grid_type", "volume_mask_grid", "volume_sdf_grid",
        "field_operation", "field_region_kind", "field_value_source",
        "field_scalar_value", "field_vector_value", "source_ui_schema_version",
        "source_status_code", "source_status_message", "height_profile_image",
        "height_profile_scale", "height_emission_mode",
    ):
        assert f"{approved}:" in source
    assert '("FIELD", "Advanced Field Source"' in source


def test_ui_never_presents_height_infinite_or_legacy_source_mode_as_liquid():
    source = _source(PANELS)
    assert 'prop_enum(fs, "height_back_mode", "INFINITE"' not in source
    assert '.prop(pg, "source_mode"' not in source
    assert 'prop_enum(pg, "source_region_kind", "MESH_SOLID"' in source
    assert 'prop_enum(pg, "source_region_kind", "VOLUME_GRID"' in source
    assert 'prop_enum(pg, "source_region_kind", "POINTS"' in source


def test_role_specific_operators_and_plan_translator_are_real_entry_points():
    operators = _source(OPERATORS)
    for class_name in (
        "FLUIDSIM_OT_tag_liquid_emitter", "FLUIDSIM_OT_tag_liquid_sink",
        "FLUIDSIM_OT_tag_collider", "FLUIDSIM_OT_tag_force_region",
        "FLUIDSIM_OT_tag_field_source", "FLUIDSIM_OT_clear_object_role",
    ):
        assert f"class {class_name}" in operators
    inputs = _source(SCENE_INPUTS)
    assert "def gather_source_plans(" in inputs
    assert "VolumeRegionPlan" in inputs
    assert "PointRegionPlan" in inputs
    assert "SourceOperation.PRESSURE" in inputs
    assert "SourceOperation.STAGGERED_VELOCITY" in inputs


def test_panel_draw_code_has_no_compiler_or_gpu_probe_calls():
    tree = ast.parse(_source(PANELS))
    forbidden = {"compile", "gather_sources", "gather_source_plans", "wp", "openvdb"}
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    assert forbidden.isdisjoint(calls)

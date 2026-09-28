from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "fluidSim_warp_blender51" / "blender" / "ui_object.py"
PROPERTIES = ROOT / "fluidSim_warp_blender51" / "properties.py"
OPERATORS = ROOT / "fluidSim_warp_blender51" / "operator_domains" / "source_roles.py"
SCENE_INPUTS = ROOT / "fluidSim_warp_blender51" / "blender" / "scene_inputs.py"


EXPECTED_PANEL_ORDER = [
    "FLUIDSIM_PT_object_role",
    "FLUIDSIM_PT_object_status",
    "FLUIDSIM_PT_object_region",
    "FLUIDSIM_PT_object_source",
    "FLUIDSIM_PT_object_lifespan",
    "FLUIDSIM_PT_object_blending",
    "FLUIDSIM_PT_object_collider",
    "FLUIDSIM_PT_object_force",
    "FLUIDSIM_PT_object_field",
    "FLUIDSIM_PT_object_volume",
]


def _source(path):
    return path.read_text(encoding="utf8")


def test_role_panels_exist_in_approved_order_without_mixed_legacy_registration():
    source = _source(PANELS)
    positions = [source.index(f"class {name}") for name in EXPECTED_PANEL_ORDER]
    assert positions == sorted(positions)
    classes_block = source[source.index("OBJECT_PANEL_CLASSES = (") :]
    registered = [name for name in EXPECTED_PANEL_ORDER if name in classes_block]
    assert registered == EXPECTED_PANEL_ORDER
    assert "FLUIDSIM_PT_liquid_emitters," not in classes_block
    assert "FLUIDSIM_PT_advanced_field_sources," not in classes_block


def test_presentation_rna_preserves_legacy_and_adds_only_approved_axes():
    source = _source(PROPERTIES)
    for legacy in (
        "role: EnumProperty",
        "geometry_mode: EnumProperty",
        "sink_region_mode: EnumProperty",
        "source_mode: EnumProperty",
        "emit_rate: FloatProperty",
        "emission_mode: EnumProperty",
        "height_back_mode: EnumProperty",
    ):
        assert legacy in source
    for approved in (
        "source_region_kind",
        "sink_region_kind",
        "point_source_representation",
        "point_input_object",
        "point_rasterize_radius",
        "volume_input_object",
        "volume_grid_type",
        "volume_mask_grid",
        "volume_sdf_grid",
        "field_operation",
        "field_region_kind",
        "field_value_source",
        "field_scalar_value",
        "field_vector_value",
        "source_ui_schema_version",
        "source_status_code",
        "source_status_message",
        "height_profile_image",
        "height_profile_scale",
        "height_emission_mode",
    ):
        assert f"{approved}:" in source
    assert '("FIELD", "Advanced Field Source"' in source


def test_ui_never_presents_height_infinite_or_legacy_source_mode_as_liquid():
    source = _source(PANELS)
    assert 'prop_enum(fs, "height_back_mode", "INFINITE"' not in source
    assert '.prop(settings, "source_mode"' not in source
    assert 'layout.prop(settings, "source_region_kind")' in source
    properties = _source(PROPERTIES)
    for region in ("MESH_SOLID", "VOLUME_GRID", "POINTS"):
        assert f'("{region}",' in properties


def test_height_back_mode_offers_only_runnable_modes_with_stable_numbers():
    from fluidSim_warp_blender51.sim.enums import SourceHeightBackMode

    tree = ast.parse(_source(PROPERTIES))
    assign = next(node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name)
                          and target.id == "HEIGHT_BACK_MODE_ITEMS"
                          for target in node.targets))
    items = ast.literal_eval(assign.value)
    assert [(item[0], item[3]) for item in items] == [
        (member.name, int(member)) for member in SourceHeightBackMode
        if member is not SourceHeightBackMode.INFINITE]
    assert "items=HEIGHT_BACK_MODE_ITEMS" in _source(PROPERTIES)


def test_role_specific_operators_and_plan_translator_are_real_entry_points():
    operators = _source(OPERATORS)
    for class_name in (
        "FLUIDSIM_OT_tag_liquid_emitter",
        "FLUIDSIM_OT_tag_liquid_sink",
        "FLUIDSIM_OT_tag_collider",
        "FLUIDSIM_OT_tag_force_region",
        "FLUIDSIM_OT_tag_field_source",
        "FLUIDSIM_OT_clear_object_role",
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
